# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai.text import *
train = pd.read_csv("/kaggle/input/encoded-train/encoded_train.csv")
train.head()
train['text'] = train['text'].str.replace('([“”¨«»®´·º½¾¿¡§£₤‘’])', '')
test = pd.read_csv("/kaggle/input/test-data/Test_BNBR.csv")

test_id = test['ID']
test['text'] = test['text'].str.replace('([“”¨«»®´·º½¾¿¡§£₤‘’])', '')
data = (TextList.from_df(train, cols='text')
                .split_by_rand_pct(0.2)
                .label_for_lm()  
                .databunch(bs=48))
data.show_batch()
learn = language_model_learner(data,AWD_LSTM, drop_mult=0.3)
learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.save_encoder('fine_tuned_enc')
label_cols = ['Depression', 'Alcohol' , 'Suicide' , 'Drugs']
test_datalist = TextList.from_df(test, cols='text', vocab=data.vocab)

data_clas = (TextList.from_df(train, cols='text', vocab=data.vocab)
             .split_by_rand_pct(0.2)
             .label_from_df(cols= label_cols , classes=label_cols)
             .add_test(test_datalist)
             .databunch(bs=32))

data_clas.show_batch()
learn_classifier = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn_classifier.load_encoder('fine_tuned_enc')
learn_classifier.freeze()

learn_classifier.lr_find()
learn_classifier.recorder.plot()

learn_classifier.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))

learn_classifier.freeze_to(-2)
learn_classifier.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn_classifier.freeze_to(-3)
learn_classifier.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn_classifier.show_results()

preds, target = learn_classifier.get_preds(DatasetType.Test, ordered=True)
labels = preds.numpy()

labels
submission = pd.DataFrame({'ID': test_id})
submission = pd.concat([submission, pd.DataFrame(preds.numpy(), columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)
submission.head()