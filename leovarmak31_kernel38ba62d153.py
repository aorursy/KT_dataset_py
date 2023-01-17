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
df = pd.read_csv('/kaggle/input/summeranalytics2020/test.csv')

d = pd.read_csv('/kaggle/input/summeranalytics2020/train.csv')

data = pd.read_csv('/kaggle/input/summeranalytics2020/Sample_submission.csv')
df.head()
d.head()
data.head()




# precision-recall curve and f1

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from matplotlib import pyplot
df = df.drop('EmployeeNumber', axis=1)

d = d.drop('EmployeeNumber', axis=1) 
from sklearn.preprocessing import LabelEncoder



for column in df.columns:

        if df[column].dtype == np.number:

            continue

        df[column] = LabelEncoder().fit_transform(df[column])

for column in d.columns:

        if d[column].dtype == np.number:

            continue

        d[column] = LabelEncoder().fit_transform(d[column])
xtest=df



ytrain=d['Attrition']

xtrain=d.drop('Attrition',axis = 1)

from sklearn.linear_model import LogisticRegression  

Lr=LogisticRegression(solver='lbfgs',max_iter=5000,C=0.5,penalty='l2',random_state=1)

Lr.fit(xtrain,ytrain)

a=Lr.predict_proba(xtest)

b=a[:,1]

data=data.drop('Attrition',axis=1)

data['Attrition']=b

data.to_csv('submission.csv',index=False)


# predict class values



lr_precision, lr_recall, _ = precision_recall_curve(ytest, lr_probs)

lr_f1, lr_auc = f1_score(ytest, yhat), auc(lr_recall, lr_precision)

# summarize scores

print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

# plot the precision-recall curves

no_skill = len(ytest[ytest==1]) / len(testy)

pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')

# axis labels

pyplot.xlabel('Recall')

pyplot.ylabel('Precision')