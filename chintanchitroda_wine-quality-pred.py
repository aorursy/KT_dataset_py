import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/predict-red-wine-quality/train.csv")

dft = pd.read_csv("/kaggle/input/predict-red-wine-quality/test.csv")
df
df['quality'].value_counts()
df.info()
plt.subplots(figsize=(15,10))

sns.heatmap(df.corr(),annot=True)
df.columns
df.drop(["id"],axis = 1,inplace = True)
ids = dft['id']

dft.drop(["id"],axis = 1,inplace = True)
df
dft
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score

import statsmodels.api as sm

from sklearn.metrics import accuracy_score
X = df.drop( "quality",axis=1)

y = df["quality"]
# use 70% of the data for training and 30% for testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=50)
lr = LogisticRegression(random_state = 101,class_weight='balanced')

lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
print("F1 Score :",f1_score(y_pred,y_test,average = "weighted"))

print('Report:\n',classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
test = lr.predict(dft)
Predf = pd.DataFrame(columns=["id",'quality'])

Predf['id']= ids

Predf['quality']= test
Predf
#Predf.to_csv('Solution.csv',index=False)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50,max_depth=10, random_state=101,class_weight='balanced')

rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
print("F1 Score :",f1_score(y_pred,y_test,average = "weighted"))

print('Report:\n',classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
test = rfc.predict(dft)

Predf = pd.DataFrame(columns=["id",'quality'])

Predf['id']= ids

Predf['quality']= test
Predf.to_csv('Solution1.csv',index=False)
### Using Rfe

from sklearn.feature_selection import RFE

rfe = RFE(rfc, 4)

rfe.fit(X_train,y_train)
X_train[list(X_train.columns[rfe.support_])]
rfc.fit(X_train[list(X_train.columns[rfe.support_])],y_train)
y_pred = rfc.predict(X_test[list(X_test.columns[rfe.support_])])
print("F1 Score :",f1_score(y_pred,y_test,average = "weighted"))

print('Report:\n',classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, average_precision_score

from xgboost import XGBClassifier

import xgboost as xgb
model = XGBClassifier(random_state=7,class_weight='balanced')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("F1 Score :",f1_score(y_pred,y_test,average = "weighted"))

print('Report:\n',classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)