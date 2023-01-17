import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")

test.head(5)
test.describe()
train.describe()
train.head(10)
train.shape, test.shape
test.isna().sum()
import seaborn as sns

sns.countplot(train.target)
trainP = train.hist(figsize = (20,20))
testP = test.hist(figsize = (20,20))
import matplotlib

import matplotlib.pyplot as plt



plt.figure(figsize=(16,6))

features = train.columns.values[2:202]

plt.title("Distribution")

sns.distplot(train[features].mean(axis=1),color="red", hist = True, label='train')

sns.distplot(test[features].mean(axis=1),color="purple", hist = True, label='test')

plt.tight_layout()

plt.legend()

plt.show()
# train.drop(['ID_code'], axis=1, inplace=True)
train.head(5)
X = train.iloc[:,2:]

y = train.iloc[:,1]



X_t = test.iloc[:,1:]
X
y
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = y)
knn = KNeighborsClassifier()
# knn.fit(X, y)
# y_preds = knn.predict(X_t)
# sub_df = pd.DataFrame({"ID": test["ID_code"], "target": pd.Series(y_preds) }) sub_df.to_csv("knn_submission.csv", index=False)
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
logRes = LogisticRegression(solver='liblinear')
logRes.fit(X, y)
y_preds = logRes.predict(X_t)
sub_df = pd.DataFrame({"ID_code": test["ID_code"],

                       "target": pd.Series(y_preds)

                      })

sub_df.to_csv("logres_submission.csv", index=False)
from sklearn.svm import SVC
svc = SVC(kernel='linear')
#svc.fit(X, y)
#svc_y_preds = svc.predict(X_t)
#sub_df = pd.DataFrame({"ID_code": test["ID_code"], "target": pd.Series(svc_y_preds)}) sub_df.to_csv("svc_submission.csv", index=False)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier(n_estimators=100)
dtc.fit(X, y)

rfc.fit(X, y)
dtc_y_preds = dtc.predict(X_t)

rfc_y_preds = rfc.predict(X_t)
sub_df = pd.DataFrame({"ID_code": test["ID_code"],

                       "target": pd.Series(dtc_y_preds)

                      })

sub_df.to_csv("dtc_submission.csv", index=False)
sub_df = pd.DataFrame({"ID_code": test["ID_code"],

                       "target": pd.Series(rfc_y_preds)

                      })

sub_df.to_csv("rfc_submission.csv", index=False)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_X.fit(X)
sc_X.mean_
X_train = sc_X.transform(X)

X_test = sc_X.transform(X_t)
print(X_train)
import xgboost as xgb
xg_cl = xgb.XGBClassifier(objective = 'binary:logistic', n_estimators = 10, seed = 123)
xg_cl.fit(X, y)
xgboost_y_preds = xg_cl.predict(X_t)
sub_df = pd.DataFrame({"ID_code": test["ID_code"],

                       "target": pd.Series(xgboost_y_preds)

                      })

sub_df.to_csv("xgboost_submission.csv", index=False)
from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_X.fit(X)

X_train = sc_X.transform(X)

X_test = sc_X.transform(X_t)
X_train = np.absolute(X_train)

X_test = np.absolute(X_test)
gnb = GaussianNB() 

bnb = BernoulliNB() 

mnb = MultinomialNB() 
gnb.fit(X_train, y)

bnb.fit(X_train, y)

mnb.fit(X_train, y)
gnb_y_preds = gnb.predict(X_test)

bnb_y_preds = bnb.predict(X_test)

mnb_y_preds = mnb.predict(X_test)
sub_df = pd.DataFrame({"ID_code": test["ID_code"],

                       "target": pd.Series(gnb_y_preds)

                      })

sub_df.to_csv("gnb_y_preds_submission.csv", index=False)
sub_df = pd.DataFrame({"ID_code": test["ID_code"],

                       "target": pd.Series(bnb_y_preds)

                      })

sub_df.to_csv("bnb_y_preds_submission.csv", index=False)
sub_df = pd.DataFrame({"ID_code": test["ID_code"],

                       "target": pd.Series(mnb_y_preds)

                      })

sub_df.to_csv("mnb_y_preds_submission.csv", index=False)