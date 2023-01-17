# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

from sklearn.tree import tree

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split # Create training and test sets

from sklearn.metrics import roc_curve # ROC Curves

from sklearn.metrics import auc # AUC 

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/heart.csv")
df.head()
df.isnull().sum().sum()
df.describe()
df.info()
print (df['ca'].unique())

print (df['thal'].unique())
for col in df.columns.tolist():

    print (col, len(df[col].unique()))
#separate categorical variables from continuous

cat_var = [col for col in df.columns.tolist() if len(df[col].unique()) <=5]

print (len(cat_var))

cont_var = [col for col in df.columns.tolist() if len(df[col].unique()) > 5]

print (len(cont_var))
#explore distributions of continuous variables

fig, axes = plt.subplots(3,2, figsize=(12,10))

for i, ax in enumerate(axes.flatten()):

    column_name = cont_var[i]

    ax.hist(df[column_name])

    ax.set_title(column_name)



plt.tightlayout()
fig, axes = plt.subplots(2, 5, figsize=(12,10))

for i, ax in enumerate(axes.flatten()):

    column_name = cat_var[i]

    ax.hist(df[column_name])

    ax.set_title(column_name)



plt.tightlayout()
df.head()
#closer look at predictor column for class imbalance

float(df[df['target'] > 0].shape[0]) / df['target'].shape[0]
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 4))

ax1.hist(df['target'])

ax1.set_title('Predictor Column')



#change predictor ('num') col to boolean

df['target'] = df['target'] > 0

df['target'] = df['target'].map({False: 0, True: 1})



ax2.hist(df['target'])

ax2.set_title('Predictor Column Cleaned')

plt.xticks([0,1], ['No Heart Disease', 'Heart Disease'])

plt.tight_layout()

plt.savefig('predictor_column.png')
#covariance matrix for looking into dimensionality reduction

from sklearn.preprocessing import scale

cdf = df.copy()

cdf.pop('target')

cdf = pd.DataFrame(scale(cdf.values), columns=cdf.columns.tolist())

cdf.cov()
#identify higher correlation values

for col in cdf.columns.tolist():

    mask = cdf.cov()[col].argsort()

    print (col, cdf.cov()[col][mask][-2])

mask = cdf.cov() > 0.3

mask
df.corr()
#Model Estimation

#Training and Testing

#Let's play with some algorithms. First, split the data into training and test sets.



train, test = train_test_split(df, test_size = 0.20, random_state = 42)

# Create the training test omitting the diagnosis



training_set = df.ix[:, df.columns != 'target']

# Next we create the class set 

class_set = train.ix[:, train.columns == 'target']



# Next we create the test set doing the same process as the training set

test_set = df.ix[:, df.columns != 'target']

test_class_set = df.ix[:, df.columns == 'target']
#Decision Trees

#Decision trees have a hierarchical structure, where each leaf of the tree represents a class label while the branches represent represent the process the tree used to deduce the class labels.



#Decision Trees

#Decision trees have a hierarchical structure, where each leaf of the tree represents a class label while the branches represent represent the process the tree used to deduce the class labels.



dt = tree.DecisionTreeClassifier()

dt = dt.fit(train[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']], train['target'])

predictions_dt = dt.predict(test[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']])

predictright = 0

predictions_dt.shape[0]

for i in range(0,predictions_dt.shape[0]-1):

    if (predictions_dt[i]== test.iloc[i][10]):

        predictright +=1

accuracy = predictright/predictions_dt.shape[0]

accuracy