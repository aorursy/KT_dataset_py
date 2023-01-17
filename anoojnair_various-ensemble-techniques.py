import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
dataset = pd.read_csv('../input/train.csv')
dataset.describe()
dataset = dataset.fillna(dataset.mean())
X = dataset.iloc[:,1:-1]
X.head()
y = dataset.loc[:,'Target']
X.loc[:,'dependency'].replace(to_replace=dict(yes=1, no=0), inplace=True)
X.loc[:,'edjefe'].replace(to_replace=dict(yes=1, no=0), inplace=True)
X.loc[:,'edjefa'].replace(to_replace=dict(yes=1, no=0), inplace=True)
X.head()
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X.loc[:,'idhogar'] = labelEncoder_X.fit_transform(X.loc[:,'idhogar'])
from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train = train_test_split(X,y,train_size = 0.2,random_state=0)
X_train.head()
# feature scaling the Test and train set
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(X_train, y_train)
print(model.score(X_test,y_test))
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import seaborn as sn
sn.heatmap(cm, annot=True)
from sklearn.ensemble import RandomForestClassifier
model_1= RandomForestClassifier(random_state=1)
model_1.fit(X_train, y_train)
model_1.score(X_test,y_test)
from sklearn.ensemble import AdaBoostClassifier
model_2 = AdaBoostClassifier(random_state=1)
model_2.fit(X_train, y_train)
model_2.score(X_test,y_test)
from sklearn.ensemble import GradientBoostingClassifier
model_3= GradientBoostingClassifier(learning_rate=0.01,random_state=1)
model_3.fit(X_train, y_train)
model_3.score(X_test,y_test)
import xgboost as xgb
model_4=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
model_4.fit(X_train, y_train)
model_4.score(X_test,y_test)
dataset_test = pd.read_csv('../input/test.csv')
dataset_test = dataset_test.fillna(dataset_test.mean())
X_validate = dataset_test.iloc[:,1:143]
X_validate.loc[:,'dependency'].replace(to_replace=dict(yes=1, no=0), inplace=True)
X_validate.loc[:,'edjefe'].replace(to_replace=dict(yes=1, no=0), inplace=True)
X_validate.loc[:,'edjefa'].replace(to_replace=dict(yes=1, no=0), inplace=True)
from sklearn.preprocessing import LabelEncoder
labelEncoder_XVal = LabelEncoder()
X_validate.loc[:,'idhogar'] = labelEncoder_XVal.fit_transform(X_validate.loc[:,'idhogar'])
from sklearn.preprocessing import StandardScaler
sc_XVal = StandardScaler()
X_validate = sc_XVal.fit_transform(X_validate)
y_val = model.predict(X_validate)
sub_val = pd.DataFrame()
sub_val['Id'] = dataset_test['Id']
sub_val['Target'] = y_val
sub_val.to_csv('submission.csv', index=False)
sub_val.head()