import pandas as pd

import numpy as np



df_train=pd.read_csv('../input/predictive-equipment-failures/equip_failures_training_set.csv')



df_train.head()
df_train.describe()
df_train.isna().sum()
df_train=df_train.replace('na',0)

df_train.head()
df_test=pd.read_csv('../input/predictive-equipment-failures/equip_failures_test_set.csv')



df_test=df_test.replace('na',0)



df_test.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



model = RandomForestClassifier(n_estimators=200,bootstrap=True,max_features='sqrt')

X = df_train.iloc[:, 2:].values

y = df_train.iloc[:, 1].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



y_pred
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred,digits=4))



train = pd.read_csv('../input/predictive-equipment-failures/equip_failures_training_set.csv')
type(train)
train.head()
print(train.count())

print(train.info())

print(train.describe())
descriptive_stats = train.describe()

descriptive_stats
new_train = train.replace({'na':0})
X_train, X_test, y_train, y_test = train_test_split(new_train.drop('target',axis=1), new_train['target'], test_size=0.10, 

random_state=0)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, y_train)



y_pred = logreg.predict(X_test)



print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(logreg.score(X_test, y_test)))

X = pd.read_csv('../input/predictive-equipment-failures/equip_failures_training_set.csv')

y = pd.read_csv('../input/predictive-equipment-failures/equip_failures_test_set.csv')
from sklearn.utils import resample

#separate training

new_X = X.replace('na',0)

new_y = y.replace('na',0)

df_major = new_X[new_X.target==0]

df_minor = new_X[new_X.target==1]



df_major_downsampled = resample(df_major, replace=False,n_samples=1000, random_state=123)



#combine minor class with downsampled majority class

df_downsampled = pd.concat([df_major_downsampled, df_minor])







df_downsampled

#print(df_downsampled.target.value_counts())

#X_train = df_downsampled.iloc[:,2:]

#y_train = df_downsampled.iloc[:,1]

#X_test = new_y.iloc[:,1:]



X_train, X_test, y_train, y_test = train_test_split(df_downsampled.drop('target',axis = 1), df_downsampled['target'], test_size = 0.20, random_state = 0)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



classifier = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=1),

    n_estimators=250

)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))


gb = GradientBoostingClassifier(n_estimators=300, max_depth = 1)

gb.fit(X_train, y_train)

predictions = gb.predict(X_test)



print("Classification Report")

print(classification_report(y_test, y_pred,digits=4))
prediction = model.predict(df_test.iloc[:, 1:].values)



df_final=pd.DataFrame(df_test['id'])

df_final['target']=prediction



df_final.to_csv('final.csv')