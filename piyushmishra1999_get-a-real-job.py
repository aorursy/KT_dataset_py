import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from fastai.text import *
from fastai import *
import seaborn as sns
import matplotlib.pyplot as plt
jobs  = pd.read_csv('/kaggle/input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
jobs.head()
jobs.fillna(value='NA',inplace=True)
jobs.fraudulent.value_counts()
sns.set_style('darkgrid')
plt.figure(1,figsize=(20,8))
sns.countplot(hue=jobs.fraudulent,x=jobs.employment_type)
plt.title('Fraudulence Distribution based on Type of Employment Opportunity')
plt.xlabel('Employment Type')
plt.ylabel('No. of Jobs')
path = Path('/kaggle/working/')
data_lm = (TextList.from_df(jobs,path,cols=['company_profile','description','requirements','benefits'])
                  .split_by_rand_pct(0.2)
                  .label_for_lm()
                  .databunch(bs=128))
data_lm.show_batch(rows=6)
learn = language_model_learner(data_lm,AWD_LSTM,metrics=[accuracy,Perplexity()],model_dir='/kaggle/working/',drop_mult=0.3).to_fp16()
learn.lr_find()
learn.recorder.plot()
import gc
gc.collect()
learn.fit_one_cycle(5, 1e-01, moms = (0.8,0.7))
learn.save_encoder('lm')
learn = None
gc.collect()
jobs0 = jobs[jobs['fraudulent']==0][:1400].copy()
jobs1 = jobs[jobs['fraudulent']==1].copy()
train = pd.concat([jobs0,jobs1])
label_cols = ['department','employment_type','required_experience','industry','function','required_education','title','company_profile','description','requirements','benefits']
data_cls = (TextList.from_df(train,path,cols=label_cols,vocab=data_lm.vocab)
                    .split_by_rand_pct(0.2,seed=64)
                    .label_from_df(cols='fraudulent')
                    .databunch(bs=128))
data_cls.show_batch(rows=6)
clf = None
gc.collect()
f_score = FBeta()
f_score.average = 'macro'
kappa = KappaScore()
kappa.weights = "quadratic"

clf = text_classifier_learner(data_cls,AWD_LSTM,metrics=[accuracy, f_score, kappa],drop_mult=0.3).to_fp16()
clf.load_encoder('/kaggle/working/lm');
gc.collect()
clf.lr_find()
clf.recorder.plot()
clf.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))
interp = TextClassificationInterpretation.from_learner(clf)
interp.plot_confusion_matrix()
