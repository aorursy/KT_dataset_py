import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix
dt= pd.read_csv('../input/minor-project-2020/train.csv')
dt.head()
dt.isna()
dt.info()
dt.isna().any()
dt.isna().sum()
print(dt[dt.target== 1].shape[0])



print(dt['target'].value_counts() / len(dt))
clmn = list(dt)

for col in clmn:

  dt[col].fillna(dt[col].median(),inplace=True);
dt.info()
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 33)

del dt['id']

X = np.array(dt.loc[:, dt.columns != 'target'])

Y= np.array(dt.loc[:, dt.columns == 'target']).reshape(-1, 1)
print(Y)


scaler = StandardScaler()

X = scaler.fit_transform(X)

print(X.shape)

print(Y.shape)
dt.head()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.22, random_state = 2, shuffle = True, stratify = Y)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

clf = LogisticRegression(solver = 'lbfgs')

X_train.shape
X_train_new, y_train_new = sm.fit_sample(X_train, Y_train.ravel())



X_train_new.shape
pd.Series(y_train_new).value_counts().plot.bar()
clf.fit(X_train_new, y_train_new)

train_pred_sm = clf.predict(X_train_new)

# prediction for Testing data

test_pred_sm = clf.predict(X_test)
print('Accuracy score for Training Dataset = ', accuracy_score(train_pred_sm, y_train_new))

print('Accuracy score for Testing Dataset = ', accuracy_score(test_pred_sm, Y_test))

print ("Classification Report")

results = classification_report(Y_test,test_pred_sm)

print(results)
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 

gnb.fit(X_train_new, y_train_new)



train_pred_sm = gnb.predict(X_train_new)

# prediction for Testing data

test_pred_sm = gnb.predict(X_test)

print('Accuracy score for Training Dataset = ', accuracy_score(train_pred_sm, y_train_new))

print('Accuracy score for Testing Dataset = ', accuracy_score(test_pred_sm, Y_test))

print ("Classification Report")

results = classification_report(Y_test,test_pred_sm)

print(results)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_new, y_train_new)

train_pred = model.predict(X_train_new)



# prediction for Testing data

test_pred= model.predict(X_test)
print('Accuracy score for Training Dataset = ', accuracy_score(train_pred, y_train_new))

print('Accuracy score for Testing Dataset = ', accuracy_score(test_pred, Y_test))

results = classification_report(Y_test,test_pred)

print(results)

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier()

tree.fit(X_train_new, y_train_new)

test_pred = tree.predict(X_test)

print ("Classification Report")

results = classification_report(Y_test,test_pred)

print(results)
dt_1 = pd.read_csv('../input/minor-project-2020/test.csv')
dt_1.info()
dt_1.isna()
dt_1.head()
dts=dt_1['id']

del dt_1['id']

dt_1.head()

label=model.predict_proba(dt_1)


print(label)
print(type(label))

column_one = [row[1] for row in label]

final_cs= pd.DataFrame()
#print(column_one)

print(type(column_one))

final_cs['id']=dts
type(final_cs)


final_cs['target']=pd.Series(column_one)
final_cs.head()
count = final_cs['target'].value_counts()
print(count)
final_cs.to_csv('/kaggle/working/Test_score1.csv')