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
# Load the data to start.
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
# Percentages of null values.
print(100*train_data.isnull().sum()/train_data.shape[0])
# Convert cabin information to categorical
def get_cabin_prefix(cabin):
    if pd.isnull(cabin):
        return 'N'
    else:
        return cabin[0]
train_data['CabinInfo']=train_data.Cabin.apply(lambda x:get_cabin_prefix(x))
train_data.CabinInfo.value_counts(dropna=False)
# Total direct family members
train_data['nfamily'] = train_data['SibSp'] + train_data['Parch']
# Derive the title of each passenger
def get_title(name):
    lastname = name.split(',')[1]
    lastname_split = lastname.split('.')
    if len(lastname_split) > 1:
        return lastname_split[0].strip()
    else:
        return "None"
train_data['Title'] = train_data.Name.apply(lambda x:get_title(x))
train_data.Title.unique()
train_data.Age.describe()
# Impute missing ages with mean age. Which is 28 years.
train_data.loc[train_data.Age.isnull(),'Age']=np.mean(train_data.Age)
train_data.Embarked.value_counts(dropna=False)
# Impute with mode of the column "S"
# train_data.loc[train_data.Embarked.isnull(),'Embarked'] = train_data.Embarked.mode()
train_data.drop(train_data[train_data.Embarked.isnull()].index,axis=0,inplace=True)
# Build Logistic regression model
X = train_data[['Pclass','Sex','Age','nfamily','Fare','Embarked','Title','CabinInfo']]
y = train_data['Survived']
# Create dummy columns for category Embarked.
X = pd.get_dummies(X,columns=['Embarked'],drop_first=True)
X = pd.get_dummies(X,columns=['Title'],drop_first=True)
X = pd.get_dummies(X,columns=['CabinInfo'],drop_first=True)
X.Sex = X.Sex.apply(lambda x:1 if x=='male' else 0)
from sklearn.preprocessing import StandardScaler
cols = X.columns
sca = StandardScaler() 
X = sca.fit_transform(X)
from sklearn.linear_model import LogisticRegression
RAND_STATE = 42
mdl = LogisticRegression(random_state=RAND_STATE).fit(X, y)
from sklearn.metrics import confusion_matrix
from sklearn import metrics

print(confusion_matrix(y,mdl.predict(X)))
print(metrics.accuracy_score(y,mdl.predict(X)))
# Improve the scores by running RFE to eliminate features not needed.
from sklearn.feature_selection import RFE
rfe = RFE(mdl, n_features_to_select=10).fit(X,y)
cols_supp = cols[rfe.support_]
cols_supp
# Rerun prediction
mdl = LogisticRegression(random_state=RAND_STATE).fit(X[:,rfe.support_], y)
confusion_matrix(y,mdl.predict(X[:,rfe.support_]))
preds = mdl.predict_proba(X[:,rfe.support_])
tpr, tpr, thresholds = metrics.roc_curve(y,preds[:,1])
score_list = [metrics.accuracy_score(y, np.where(preds[:,1]>x,1,0)) for x in thresholds]
thr = thresholds[score_list>=max(score_list)][0]
thr
y_pred = np.where(preds[:,1]>thr,1,0)
print(confusion_matrix(y,y_pred))
print(metrics.accuracy_score(y,y_pred))
# Now use it on test data.
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.loc[test_data.Age.isnull(),'Age'] = np.mean(test_data.Age)
test_data.loc[test_data.Fare.isnull(),'Fare'] = np.mean(test_data.Fare)
test_data['Title']=test_data.Name.apply(lambda x:get_title(x))
test_data['CabinInfo']=test_data.Cabin.apply(lambda x:get_cabin_prefix(x))
test_data['nfamily'] = test_data['SibSp'] + test_data['Parch']
print(test_data.isnull().sum()/test_data.shape[0])
test_data = pd.get_dummies(test_data,columns=['Embarked'],drop_first=True)
test_data = pd.get_dummies(test_data,columns=['Title'],drop_first=True)
test_data = pd.get_dummies(test_data,columns=['CabinInfo'],drop_first=True)

for c in cols:
    if not c in test_data.columns:
        test_data[c] = 0
X_test = test_data[cols]
X_test.Sex = X_test.Sex.apply(lambda x:1 if x=='male' else 0)

X_test = sca.transform(X_test)
preds = mdl.predict_proba(X_test[:,rfe.support_])
y_test_pred = np.where(preds[:,1]>thr,1,0)
final_df = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':y_test_pred})
final_df.to_csv('/kaggle/working/output.csv',index=False)
