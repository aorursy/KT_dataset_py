import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
train_df.head()
print(train_df.info())
print('_'*80)
print(test_df.info())
train_df.head()
train_df.drop(['Unnamed: 0', 'objid', 'rerun'], axis = 1, inplace = True)
test_df.drop(['Unnamed: 0', 'objid', 'rerun'], axis = 1, inplace = True)
print(train_df.info())
print('_'*80)
print(test_df.info())
X = train_df.drop('class', axis = 1)
y = train_df['class']

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X, y)
y_pred = model.predict(test_df)
subm = pd.read_csv('sample_submission.csv')
subm.head()
subm['class'] = y_pred
subm['class'].value_counts()
subm.to_csv('template_submission.csv', index= False)