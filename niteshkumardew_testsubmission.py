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
final_training_data = pd.read_csv('/kaggle/input/final-training-data/final_training_data.csv')

final_test_data = pd.read_csv('/kaggle/input/final-test-data/final_test_data.csv')
final_training_data.head()
final_test_data.head()
import xgboost as xgb

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import time

from sklearn.metrics import make_scorer
# reference : https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps



def calculate_QWK(actual_label,predicted_label):

    '''

    this function will calculate quadratic weighted kappa given actual 

    and predicted label array.

    '''

    N = 4 # unique labels

    hist_actual_label = np.zeros(N)

    hist_predicted_label = np.zeros(N)

    w = np.zeros((N,N))

    numerator = 0       # w and O

    denominator = 0     # w and E

    

    conf_mat = confusion_matrix(actual_label,predicted_label)



    for i in actual_label:               # this part will calculate histogram for actual and predicted label

        hist_actual_label[i]+=1

    for j in predicted_label:

        hist_predicted_label[j]+=1



    E = np.outer(hist_actual_label, hist_predicted_label)  # E is N-by-N matrix which is outer product of 

                                                           # histogram of actual and predicted label    

    for i in range(N):                   # w is N-by-N matrix which is calculated by the given expression

        for j in range(N):

            w[i][j] = (i-j)**2/((N-1)**2)



    E = E/E.sum()

    O = conf_mat/conf_mat.sum()  # normalize confusion matrix and E



    for i in range(N):

        for j in range(N):                # this section calculates numerator and denominator 

            numerator+=w[i][j]*O[i][j]

            denominator+=w[i][j]*E[i][j]



    kappa = 1-numerator/denominator

    

    return kappa
import pandas as pd

import numpy as np

import seaborn as sns

from tqdm import tqdm

from collections import Counter

import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
## testing code for function calculate_QWK()

actual_label_temp = np.array([0,3,2,3,1,0,2,1,2,1,0])

predicted_label_temp = np.array([0,3,2,3,1,0,2,1,2,1,0])

print('QWK when actual_label and predicted_label are same is :',calculate_QWK(actual_label_temp,predicted_label_temp))



actual_label_temp = np.array([0,3,0,3,2,0,3,1,2,3,0])

predicted_label_temp = np.array([0,3,2,3,1,0,2,1,2,1,0])

print('QWK when actual_label and predicted_label are different is :',calculate_QWK(actual_label_temp,predicted_label_temp))
final_training_data.head()
X = final_training_data.copy()

X_test = final_test_data.copy()

y = X['accuracy_group'].values

y_test = X_test['accuracy_group'].values
## if we include features 'correct_count','incorect_count' and 'accuracy' to train a model then

## it will become a trvial task like if-else condition to predict the label that we dont want. 

## we calculated 'correct_count','incorect_count' and 'accuracy' to get the label of training  and test data but

## we want our model to predict the label without those feature thats why we will remove those feature.



X = X.drop(['correct_count','incorrect_count','accuracy','accuracy_group','installation_id'], axis=1)

X_test = X_test.drop(['correct_count','incorrect_count','accuracy','accuracy_group','installation_id'],axis =1)



X_train, X_cv, y_train, y_cv = train_test_split(X, y,stratify=y,test_size=0.2)

X_train = X_train.values

X_cv = X_cv.values

X_test = X_test.values



print('size of training data and labels :',X_train.shape,y_train.shape)

print('size of cv data and labels :',X_cv.shape,y_cv.shape)

print('size of test data and labels :',X_test.shape,y_test.shape)
## train a very simple XGBClassifier base madel with default parameter

start = time.time()

model = XGBClassifier()

model.fit(X_train,y_train)



actual_label = y_test

predicted_label = model.predict(X_test)



print('Quadratic weighted kappa with simple base model :',calculate_QWK(actual_label,predicted_label))

print('time: ',time.time() - start)
temp_submission = {'installation_id':final_test_data['installation_id'],'accuracy_group':predicted_label}

submission = pd.DataFrame(temp_submission)
submission['accuracy_group'] = submission['accuracy_group'].astype(int)
type(submission['accuracy_group'][0])
sample = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
type(submission['accuracy_group'][0])
k = 0

for i in range(len(submission['installation_id'])):

    if submission['installation_id'][i]==sample['installation_id'][i]:

        k+=1

    else:

        print(submission['installation_id'][i])

        print(sample['installation_id'][i])

        print('='*100)



sample.to_csv('submission.csv',index=False)