# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Calculate information value
def calc_iv(df, feature, target, pr=False):
    """
    Set pr=True to enable printing of output.
    
    Output: 
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())


    iv = data['IV'].sum()
    # print(iv)

    return iv, data
def add_log_transform(df, feat_list):
    # Generate log transform
    for feat in feat_list:
        df[feat+'_log'] = np.log1p(df[feat])
    return df
def add_age_bin(df):
    # Generate Age_bin
    df.loc[ df['Age'] <= 12, 'Age_bin'] = 0
    df.loc[(df['Age'] > 12) & (df['Age'] <= 24), 'Age_bin'] = 1
    df.loc[(df['Age'] > 24) & (df['Age'] <= 36), 'Age_bin'] = 2
    df.loc[(df['Age'] > 36) & (df['Age'] <= 48), 'Age_bin'] = 3
    df.loc[(df['Age'] > 48) & (df['Age'] <= 60), 'Age_bin'] = 4
    df.loc[ df['Age'] > 60, 'Age_bin'] = 5
    
    return df
def get_title(name):
    # Extract title from name
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def add_title(df):
    # Add new feature title
    df['Title'] = df['Name'].apply(get_title)
    
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)
    
    return df
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()
train_data.info()
test_data.info()
tgt = 'Survived'
train = train_data.copy()
train.drop('PassengerId',axis=1).groupby(tgt).hist()
train.describe(include=['O'])
raw_feat = train.columns.to_list()
raw_feat.pop(1) #remove Survived
ivs={}
for feat in raw_feat:
    iv, data = calc_iv(train,feat, tgt)
    ivs[feat] = iv
ivs
    
train = train_data.copy()

# Group attributes based on data type
num_attribs= ['Age','Fare','Age_median_by_Pclass']
cat_attribs = ['Pclass', 'Sex', 'Embarked']
omit_attribs = ['Cabin', 'Name', 'Ticket', 'PassengerId']

# Impute age with median values within its Pclass
age_imputer = train.groupby('Pclass').Age.median()
train['imputed_age'] = train.Pclass.apply(lambda x: age_imputer[x])
train['Age_median_by_Pclass'] = train.Age
train['Age_median_by_Pclass'].fillna(train['imputed_age'], inplace=True)
train = train.drop('imputed_age',axis=1) #drop the dummy column

# Alternate way to impute age by taking its median value
train.Age.fillna(train.Age.median(), inplace=True)

# Generate Age_bin
train = add_age_bin(train)

# Generate Log Transform of numerical attributes
train = add_log_transform(train, num_attribs)

# Generate title
train = add_title(train)

# Generate IsAlone and FamilySize
train['IsAlone'] = (train.Parch + train.SibSp == 0)*1
train['FamilySize'] = train.Parch + train.SibSp + 1

# Generate HasCabin
train['HasCabin'] = ~train.Cabin.isna()

# Impute Embarked with most frequent values
train.Embarked.fillna(train.Embarked.mode(), inplace=True)

# Onehot encode categorical variables
train_transformed = pd.get_dummies(train[cat_attribs]).join(train.drop(cat_attribs + omit_attribs,axis=1))
train_transformed[['Fare_log','Age_log','Age_median_by_Pclass_log']].hist()
# Feature correlation with target
train_transformed.corr()['Survived'].sort_values(ascending=False)
# Remove duplicate features or features which are less correlated with target
train_transformed.drop(['Sex_male', 'Age','Age_median_by_Pclass','Age_median_by_Pclass_log','Fare'],axis=1, inplace=True)
# Feature correlation with target
train_transformed.corr()['Survived'].sort_values(ascending=False)
# Draw correlation heatmap
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_transformed.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
X_train, X_test, y_train, y_test = train_test_split(train_transformed.drop('Survived',axis=1), train_transformed.Survived, test_size=0.30, random_state=42)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print('AUC', roc_auc_score(y_test,rfc_pred))
print('accuracy: ',accuracy_score(y_test,rfc_pred))

rfc_feat_imp = pd.DataFrame(rfc.feature_importances_, index=X_train.columns.to_list(), columns=['feat_imp'])
rfc_feat_imp.sort_values('feat_imp', ascending=False)
# SVC
from sklearn.svm import SVC

svc = SVC(random_state=42)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print('AUC', roc_auc_score(y_test,svc_pred))
print('accuracy: ',accuracy_score(y_test,svc_pred))

# # Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print('AUC', roc_auc_score(y_test,lr_pred))
print('accuracy: ',accuracy_score(y_test,lr_pred ))

lr_coef = pd.DataFrame(lr.coef_[0], index=X_train.columns.to_list(), columns=['coef'])
lr_coef.sort_values('coef', ascending=False)
# # KNeighborClassifier
from sklearn.neighbors import KNeighborsClassifier

knc = KNeighborsClassifier()
knc.fit(X_train, y_train)
knc_pred = knc.predict(X_test)
print('AUC', roc_auc_score(y_test,knc_pred))
print('accuracy: ',accuracy_score(y_test,knc_pred ))
# LightGBM
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)
print('AUC', roc_auc_score(y_test,lgbm_pred))
print('accuracy: ',accuracy_score(y_test,lgbm_pred))
param_grid = [
     {'kernel':('linear', 'poly','rbf'), 'C':[1, 10], 'degree':[3,4]}
  ]

svc = SVC(random_state=42)
grid_search = GridSearchCV(svc, param_grid, cv=5,
                           scoring='accuracy',
                           return_train_score=True)
grid_search.fit(train_transformed.drop('Survived',axis=1), train_transformed.Survived)
estimator = grid_search.best_estimator_
test = test_data.copy()

# Impute missing values and onehot encode categorical variable
num_attribs= ['Age','Fare']
cat_attribs = ['Pclass', 'Sex', 'Embarked']
omit_attribs = ['Cabin', 'Name', 'Ticket', 'PassengerId']

# Impute age by taking its median value
test.Age.fillna(test.Age.median(), inplace=True)

# Generate Age_bin
test = add_age_bin(test)

# Impute Fare with median value of its Pclass
fare_imputer = train.groupby('Pclass').Fare.median()
test['imputed_fare'] = test.Pclass.apply(lambda x: fare_imputer[x])
test['Fare'].fillna(test['imputed_fare'], inplace=True)
test = test.drop('imputed_fare',axis=1) #drop the dummy column

# Generate log transform of numerical attributes
test = add_log_transform(test, num_attribs)

# Generate title
test = add_title(test)

# Generate new features IsAlone and FamilySize
test['IsAlone'] = (test.Parch + test.SibSp == 0)*1
test['FamilySize'] = test.Parch + test.SibSp + 1

# Generate new features HasCabin
test['HasCabin'] = ~test.Cabin.isna()

# Impute Embarked with most frequent values
test.Embarked.fillna(train.Embarked.mode(), inplace=True)

# Onehot encode categorical variables
test_transformed = pd.get_dummies(test[cat_attribs]).join(test.drop(cat_attribs + omit_attribs,axis=1))

# Remove duplicate or less correlated features
test_transformed.drop(['Sex_male', 'Age','Fare'],axis=1, inplace=True)
test_pred = estimator.predict(test_transformed)
test_data['Survived'] = test_pred
test_data[['PassengerId', 'Survived']].to_csv('prediction.csv', index=False)