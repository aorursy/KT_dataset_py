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
train = pd.read_csv('../input/health-insurance-cross-sell-prediction/train.csv')
train.head()
len(train)
categorical_fields = ['Gender','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']

numerical_fields = [c for c in train.columns if c not in categorical_fields]

numerical_fields.remove('id')

categorical_fields.remove('Response')

train[numerical_fields].describe()
import matplotlib.pyplot as plt

import seaborn as sns

corr = train.corr()

f,ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr, square=True, linewidths=.5, cbar_kws={"shrink":.5}, annot=True, fmt='.2f')
sns.countplot(train.Response)
df=train.groupby(['Previously_Insured','Response'])['id'].count().to_frame().rename(columns={'id':'count'}).reset_index()

g = sns.catplot(x="Previously_Insured", y="count",col="Response",

                data=df, kind="bar",

                height=4, aspect=.7);
sns.boxplot(x=train.Response,y=train.Annual_Premium)
saleschannel_counts = train['Policy_Sales_Channel'].value_counts()

top_saleschannels = saleschannel_counts[saleschannel_counts.values>5000].index

sns.catplot(x='Response',data=train[train['Policy_Sales_Channel'].isin(list(top_saleschannels))],

            col='Policy_Sales_Channel',kind='count',col_wrap=4)


sns.set()

cols = ['Response', 'Previously_Insured', 'Policy_Sales_Channel', 'Age', 'Annual_Premium']

sns.pairplot(train[cols], size = 2.5)

plt.show();
train.isnull().sum()
categorical_fields = ['Previously_Insured','Policy_Sales_Channel']

numerical_fields = ['Annual_Premium','Age']
def preprocess(data):

    

    features_dropped = [col for col in data.columns if ((col not in categorical_fields) and (col not in numerical_fields))]

    if 'Response' in features_dropped:

        features_dropped.remove('Response')

    print(features_dropped)

    data.drop(columns=features_dropped,inplace=True,errors='ignore')

    X,y = train.drop(columns=['Response'],errors='ignore'),train['Response'] if 'Response' in train.columns else []

    t = [('cat', OneHotEncoder(), categorical_fields), ('num', StandardScaler(), numerical_fields)]

    col_transform = ColumnTransformer(transformers=t)

    X = col_transform.fit_transform(X)

    return X,y

    
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.metrics import roc_auc_score
train_X,train_y = preprocess(train)
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.33, random_state=42)

grid = {

    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate

    'loss': ['log','hinge'], # logistic regression,

    'penalty': ['l2'],

    'n_jobs': [-1],

    'class_weight': ['balanced']

}



search = GridSearchCV(SGDClassifier(), grid, cv=5, scoring='roc_auc', verbose=1,n_jobs=-1).fit(train_X, train_y)
print(search.best_params_)

print(search.cv_results_)
model = SGDClassifier(alpha = 0.0001, loss = 'log', n_jobs = -1, penalty = 'l2',class_weight='balanced')

model.fit(train_X,train_y)
y_train_preds = model.predict(X_train)

y_test_preds = model.predict(X_test)
from sklearn.metrics import roc_auc_score

scores_train = roc_auc_score(y_train, y_train_preds)

scores_test = roc_auc_score(y_test, y_test_preds)

print(scores_train, scores_test)
test_set = pd.read_csv('../input/health-insurance-cross-sell-prediction/test.csv')
ids = test_set['id']
test_X,_ = preprocess(test_set)
test_y_predictions = model.predict(test_X)
submission = pd.concat([ids,pd.Series(test_y_predictions)],axis=1)
submission.to_csv('submission.csv',index=False)
!head submission.csv