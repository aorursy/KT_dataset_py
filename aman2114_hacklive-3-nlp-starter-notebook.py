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

from google.colab import files
!pip install simpletransformers
from simpletransformers.classification import MultiLabelClassificationModel
import logging

from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
train = pd.read_csv('../input/hacklive-3-nlp/Train.csv')
test = pd.read_csv('../input/hacklive-3-nlp/Test.csv')
ss = pd.read_csv('../input/hacklive-3-nlp/SampleSubmission_Uqu2HVA.csv')
import re

train.loc[:,"ABSTRACT"] = train.ABSTRACT.apply(lambda x : " ".join(re.findall('[\w]+',x)))
test.loc[:,"ABSTRACT"] = test.ABSTRACT.apply(lambda x : " ".join(re.findall('[\w]+',x)))
ID_COL = 'id'

TARGET_COLS = ['Analysis of PDEs', 'Applications',
               'Artificial Intelligence', 'Astrophysics of Galaxies',
               'Computation and Language', 'Computer Vision and Pattern Recognition',
               'Cosmology and Nongalactic Astrophysics',
               'Data Structures and Algorithms', 'Differential Geometry',
               'Earth and Planetary Astrophysics', 'Fluid Dynamics',
               'Information Theory', 'Instrumentation and Methods for Astrophysics',
               'Machine Learning', 'Materials Science', 'Methodology', 'Number Theory',
               'Optimization and Control', 'Representation Theory', 'Robotics',
               'Social and Information Networks', 'Statistics Theory',
               'Strongly Correlated Electrons', 'Superconductivity',
               'Systems and Control']

TOPIC_COLS = ['Computer Science', 'Mathematics', 'Physics', 'Statistics']
def get_best_thresholds(true, preds):
    
    thresholds = [i/100 for i in range(100)]
    best_thresholds = []
    for idx in range(25):
        
        f1_scores = [f1_score(true[:, idx], (preds[:, idx] > thresh) * 1) for thresh in thresholds]
        best_thresh = thresholds[np.argmax(f1_scores)]
        best_thresholds.append(best_thresh)
    return best_thresholds
l1 = train[TARGET_COLS].values.tolist()

for col in TOPIC_COLS:
    train[col] = train[col].map({1:col, 0:'.'})
    test[col] = test[col].map({1:col, 0:'.'})
    
for col in TOPIC_COLS:
    train['ABSTRACT'] = train['ABSTRACT'] +'.'+ train[col]
    test['ABSTRACT'] = test['ABSTRACT'] +'.'+ test[col]
train['labels'] = l1
train.rename(columns={'ABSTRACT':'text'}, inplace=True)
train_data = train[['text','labels']]
train_df = train_data.iloc[:12000, :]
eval_df = train_data.iloc[12000:, :]
train_df.head()
test_df = test['ABSTRACT']
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model = MultiLabelClassificationModel('roberta', 'roberta-base', num_labels=25, use_cuda=True, args={'reprocess_input_data':True,
                                                                                      'overwrite_output_dir':True,
                                                                                      'num_train_epochs':8,
                                                                                      'early_stopping_consider_epochs':True,
                                                                                      'early_stopping_patience':True})

model.train_model(train_df, acc=f1_score)
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
print(result)
print(model_outputs)
predictions, raw_outputs = model.predict(test_df)
#print(predictions)
print(raw_outputs)
ss[TARGET_COLS] = raw_outputs
ss.to_csv('transmore_epochs.csv', index = False)
sub = pd.read_csv('../input/hacklive-3-nlp/transmore_epochs.csv')