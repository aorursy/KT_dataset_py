import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0']
#Removing the columns which aren't of importance for the prediction. Code for plotting the feature importances with the features is towards the end of the notebook



a= pd.get_dummies(df_train['V2'], drop_first=True, prefix = 'v2i')

b= pd.get_dummies(df_train['V3'], drop_first=True, prefix = 'v3i')

c= pd.get_dummies(df_train['V4'], drop_first=True, prefix = 'v4i')

d= pd.get_dummies(df_train['V9'], drop_first=True, prefix = 'v9i')

e= pd.get_dummies(df_train['V16'], drop_first=True, prefix = 'v16i')



df_train.drop(['V2','V3','V4','V5', 'V9','V16','Unnamed: 0'], axis=1, inplace=True)

df_train = pd.concat([df_train,a,b,c,d,e], axis =1)
print(df_train.shape)
a= pd.get_dummies(df_test['V2'], drop_first=True, prefix = 'v2i')

b= pd.get_dummies(df_test['V3'], drop_first=True, prefix = 'v3i')

c= pd.get_dummies(df_test['V4'], drop_first=True, prefix = 'v4i')

d= pd.get_dummies(df_test['V9'], drop_first=True, prefix = 'v9i')

e= pd.get_dummies(df_test['V16'], drop_first=True, prefix = 'v16i')



df_test.drop(['V2','V3','V4','V5', 'V9','V16','Unnamed: 0'], axis=1, inplace=True)

df_test = pd.concat([df_test,a,b,c,d,e], axis =1)
print(df_test.shape)
pd.set_option('display.max_columns', 100)

df_train.head()
y = df_train['Class']

X = df_train.drop('Class', axis=1)

y.tail()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=101, stratify=y)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedKFold



#Applying StratifiedKFold since the label has a moajority of 0.

cv = StratifiedKFold(n_splits=10, random_state=101, shuffle=True)

lr = LogisticRegression(random_state=621)



for (train, test), i in zip(cv.split(X, y), range(10)):

    lr.fit(X.iloc[train], y.iloc[train])



print(lr.score(X_test,y_test))

print('The classification report for Logistic Regresssion training class is: ')

print(classification_report(y_train, lr.predict(X_train)))



print('The classification report for for Logistic Regression test class is: ')

print(classification_report(y_test, lr.predict(X_test)))
'''

Ran the following code for Random Forests optimized with GridSearchCV. The features of lesser importances were removed which is done at the start of the notebook.



print(m2.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(m2.feature_importances_, index=X.columns)

feat_importances.nlargest(16).plot(kind='barh')

plt.show()



'''
predicted = lr.predict_proba(df_test)

print(predicted.shape)

result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(predicted[:,1])

result.head()

result.to_csv('outputRF1.csv', index=False)