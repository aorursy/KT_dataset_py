# import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn
train = pd.read_csv('../input/train.csv')
train.describe()
train.shape
test = pd.read_csv('../input/test.csv')
test.describe()
test.shape
train.columns[train.dtypes=='object']
train['dependency'].unique()
train.loc[(train['dependency']=='yes')]['SQBdependency'].unique()
train.loc[(train['dependency']=='no')]['SQBdependency'].unique()
train.loc[train['dependency']=='yes', 'dependency'] = 1
train.loc[train['dependency']=='no', 'dependency'] = 0
train['dependency'].unique()
train['dependency'] = train['dependency'].astype(float)
train['edjefe'].unique()
train.loc[train['edjefe']=='yes', 'edjefe'] = 1
train.loc[train['edjefe']=='no', 'edjefe'] = 0
train['edjefa'].unique()
train.loc[train['edjefa']=='yes', 'edjefa'] = 1
train.loc[train['edjefa']=='no', 'edjefa'] = 0
train['edjefa'].unique()
train['edjefe'] = train['edjefe'].astype(int)
train['edjefa'] = train['edjefa'].astype(int)
train.isnull().sum().sort_values(ascending=False)[0:10]
train['rez_esc'].unique()
train_null = train.loc[train['rez_esc'].isnull()]
train_non_null = train.loc[train['rez_esc'].notnull()]
sns.distplot(train_null['age'], color='blue')
sns.distplot(train_non_null['age'], color='red')
train['rez_esc'] = train['rez_esc'].fillna(0)
train['v18q1'].unique()
train['v18q1'] = train['v18q1'].fillna(0)
train['v2a1'] = train['v2a1'].fillna(0)
meaneduc_null = train.loc[train['meaneduc'].isnull()]
meaneduc_null
meaneduc_null[['Id', 'idhogar', 'escolari']]
meaneduc_null[['Id', 'idhogar', 'escolari', 'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9']]
# for the household with id=1b31fd159
instlevel4_one = train.loc[(train['instlevel4']==1) & (train['meaneduc'].notnull())]
sns.distplot(instlevel4_one['meaneduc'])
# find mean
instlevel4_one['meaneduc'].mean()
# for the household with id=a874b7ce7
instlevel2_one = train.loc[(train['instlevel2']==1) & (train['meaneduc'].notnull())]
instlevel3_one = train.loc[(train['instlevel3']==1) & (train['meaneduc'].notnull())]
(instlevel2_one['meaneduc'].mean() + instlevel3_one['meaneduc'].mean())/2
# for the household with id=faaebf71a
instlevel7_one = train.loc[(train['instlevel7']==1) & (train['meaneduc'].notnull())]
instlevel7_one['meaneduc'].mean()
# replace
train.loc[train['idhogar']=='faaebf71a', 'meaneduc'] = instlevel7_one['meaneduc'].mean()
train.loc[train['idhogar']=='faaebf71a', 'SQBmeaned'] = instlevel7_one['meaneduc'].mean()**2
# replace
train.loc[train['idhogar']=='1b31fd159', 'meaneduc'] = train.loc[train['idhogar']=='1b31fd159', 'escolari']
train.loc[train['idhogar']=='1b31fd159', 'SQBmeaned'] = train.loc[train['idhogar']=='1b31fd159', 'escolari']**2
train.loc[train['idhogar']=='faaebf71a', 'meaneduc'] = train.loc[train['idhogar']=='faaebf71a', 'escolari']
train.loc[train['idhogar']=='faaebf71a', 'SQBmeaned'] = train.loc[train['idhogar']=='faaebf71a', 'escolari']**2
train.loc[train['idhogar']=='a874b7ce7', 'meaneduc'] = train.loc[train['idhogar']=='a874b7ce7', 'escolari']
train.loc[train['idhogar']=='a874b7ce7', 'SQBmeaned'] = train.loc[train['idhogar']=='a874b7ce7', 'escolari']**2
# test = test.drop(['Id', 'idhogar'], axis=1)
test.isnull().sum().sort_values(ascending=False)[0:10]
test['rez_esc'] = test['rez_esc'].fillna(0)
test['v18q1'] = test['v18q1'].fillna(0)
test['v2a1'] = test['v2a1'].fillna(0)

test_meaneduc_null = test.loc[test['meaneduc'].isnull()]
test_meaneduc_null_ids = test_meaneduc_null['idhogar'].tolist()
for idhogar in test_meaneduc_null_ids:
    test.loc[test['idhogar']==idhogar, 'meaneduc'] = test.loc[test['idhogar']==idhogar, 'escolari']
    test.loc[test['idhogar']==idhogar, 'SQBmeaned'] = test.loc[test['idhogar']==idhogar, 'escolari']**2
    # print(test.loc[test['idhogar']==idhogar][['escolari', 'meaneduc', 'SQBmeaned']])
test.columns[test.dtypes=='object']
test.loc[test['dependency']=='yes', 'dependency'] = 1
test.loc[test['dependency']=='no', 'dependency'] = 0
test.loc[test['edjefe']=='yes', 'edjefe'] = 1
test.loc[test['edjefe']=='no', 'edjefe'] = 0
test.loc[test['edjefa']=='yes', 'edjefa'] = 1
test.loc[test['edjefa']=='no', 'edjefa'] = 0
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
]
# remove the ids from the dataset
train = train.drop(['idhogar', 'Id'], axis=1)
# preprocess dataset, split into training and test part
y = train['Target']
X = train.drop(columns=['Target'])
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=40)

for clf in classifiers:
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
# # create the predictions
# clf = SVC(kernel="linear", C=0.025)
# clf.fit(X_train, y_train)

# test_ids = test['Id']
# test = test.drop(['idhogar', 'Id'], axis=1)
# y_pred = clf.predict(test)

# results = pd.DataFrame(columns=['Id', 'Target'])
# results['Id'] = test_ids
# results['Target'] = y_pred

# results.to_csv('submission.csv', index=False)

