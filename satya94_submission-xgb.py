#Download all data libraries

import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Download all model Libraries

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from numpy import loadtxt

from xgboost import XGBClassifier
#Read Data

df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
df_train.info()
df_train.drop('Unnamed: 0',axis=1,inplace=True)
df_test.info()
test_index = df_test['Unnamed: 0']
df_test.drop('Unnamed: 0',axis=1,inplace=True)
X=df_train.drop('Class',axis=1)

y=df_train['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=162)
model1 = XGBClassifier()

model1.fit(X, y)
model1.score(X_test,y_test)
error_rate = [] 



for k in range(1,150):

    

    knn = XGBClassifier(random_state=k)

    knn.fit(X_train, y_train)

    pred_k = knn.predict(X_test)

    

    error_rate.append(np.mean(pred_k != y_test))
plt.figure(figsize=(10,6))

plt.style.use('ggplot')



plt.plot(range(1,150), error_rate, color='blue',

        linestyle='dashed', marker='o',

        markerfacecolor='red')



plt.title('Error Rate vs K-value')

plt.xlabel('K')

plt.ylabel('Error Rate')
print(classification_report(y_test, model1.predict(X_test)))
X.head()
df2=pd.get_dummies(df_train['V2'],drop_first=True)

df3=pd.get_dummies(df_train['V3'],drop_first=True)

df4=pd.get_dummies(df_train['V4'],drop_first=True)

df5=pd.get_dummies(df_train['V5'],drop_first=True)

df7=pd.get_dummies(df_train['V7'],drop_first=True)

df8=pd.get_dummies(df_train['V8'],drop_first=True)

df9=pd.get_dummies(df_train['V9'],drop_first=True)

df16=pd.get_dummies(df_train['V16'],drop_first=True)
df2.head()
df2.columns = ['V2_1', 'V2_2','V2_3','V2_4','V2_5','V2_6','V2_7','V2_8','V2_9','V2_10','V2_11']

df3.columns = ['V3_1', 'V3_2']

df4.columns = ['V4_1', 'V4_2','V4_3']

df5.columns = ['V5_1']

df7.columns = ['V7_1']

df8.columns = ['V8_1']

df9.columns = ['V9_1', 'V9_2']

df16.columns = ['V16_1', 'V16_2','V16_3']
df2.head()
df_train_simp=df_train.drop(['V2','V3','V4','V5','V7','V8','V9','V16'],axis=1)
df_train_simp.head()
df_train_simp=pd.concat([df_train_simp,df3, df4,df2,df5,df7,df8,df9,df16], axis=1)
df_train_simp.head()
#The simplified train data

#i.to_csv('traindata.csv',index=False)
g2=pd.get_dummies(df_test['V2'],drop_first=True)

g3=pd.get_dummies(df_test['V3'],drop_first=True)

g4=pd.get_dummies(df_test['V4'],drop_first=True)

g5=pd.get_dummies(df_test['V5'],drop_first=True)

g7=pd.get_dummies(df_test['V7'],drop_first=True)

g8=pd.get_dummies(df_test['V8'],drop_first=True)

g9=pd.get_dummies(df_test['V9'],drop_first=True)

g16=pd.get_dummies(df_test['V16'],drop_first=True)
g2.columns = ['V2_1', 'V2_2','V2_3','V2_4','V2_5','V2_6','V2_7','V2_8','V2_9','V2_10','V2_11']

g3.columns = ['V3_1', 'V3_2']

g4.columns = ['V4_1', 'V4_2','V4_3']

g5.columns = ['V5_1']

g7.columns = ['V7_1']

g8.columns = ['V8_1']

g9.columns = ['V9_1', 'V9_2']

g16.columns = ['V16_1', 'V16_2','V16_3']
df_test_simp=df_test.drop(['V2','V3','V4','V5','V7','V8','V9','V16'],axis=1)
df_test_simp=pd.concat([df_test_simp,g3, g4,g2,g5,g7,g8,g9,g16], axis=1) 

#simplified test data
df_test_simp.head()
#Simplified data set

A=df_train_simp.drop('Class',axis=1)

b=df_train_simp['Class']
#Simplified data set

C=df_train_simp.drop('Class',axis=1)

d=df_train_simp['Class']
#Split the data to check accuracy

X_train, X_test, y_train, y_test = train_test_split(A, b, test_size=0.3, random_state=162)
# fit model no training data

model2 = XGBClassifier()

model2.fit(A, b)
model2.score(X_test,y_test)
model3 = XGBClassifier()

model3.fit(C, d)
pred = model3.predict_proba(df_test_simp)

result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.head()
result.to_csv('output.csv',index=False)