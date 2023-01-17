# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train = df_train.set_index('PassengerId')
df_test = df_test.set_index('PassengerId')

df_all = pd.concat([df_train, df_test], keys=['train', 'test'])

df_train.describe()
df_train.head(100)
def trans_pclass(data):
    return data.replace({"Pclass": {1: 'A', 2: 'B', 3: 'C'}})
df_all = trans_pclass(df_all)
df_train = df_all.loc['train']
sns.countplot(df_train['Pclass'], hue=df_train['Survived'])
df_all['Family_name'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[1]
df_all.groupby(by=['Family_name']).size().sort_values(ascending=False)
family_name_count = df_all['Family_name'].value_counts() 
df_all['Family_name_count'] = df_all['Family_name'].apply(lambda x: x if family_name_count[x] > 1 else 'Small', 1)
df_all
df_train = df_all.loc['train']
sns.countplot(df_train['Family_name_count'], hue=df_train['Survived'])
df_all['Title'] = df_all['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df_all['Title'][df_all.Title=='Jonkheer'] = 'Master'
df_all['Title'][df_all.Title.isin(['Ms','Mlle'])] = 'Miss'
df_all['Title'][df_all.Title == 'Mme'] = 'Mrs'
df_all['Title'][df_all.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df_all['Title'][df_all.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
df_train = df_all.loc['train']
sns.countplot(df_train['Title'], hue=df_train['Survived'])
sns.countplot(df_train['Sex'], hue=df_train['Survived'])
sns.countplot(df_train[df_train['Sex'] == 'male']['Title'], hue=df_train['Survived'])
sns.countplot(df_train[df_train['Sex'] == 'female']['Title'], hue=df_train['Survived'])
g = sns.FacetGrid(df_train, col='Survived')
g.map(sns.distplot, 'Age', kde=False)
g = sns.FacetGrid(df_train, col='Survived')
g.map(sns.distplot, 'SibSp', kde=False)
g = sns.FacetGrid(df_train, col='Survived')
g.map(sns.distplot, 'Parch', kde=False)
df_all['Family_size'] = df_all['Parch'] + df_all['SibSp']
df_train = df_all.loc['train']
g = sns.FacetGrid(df_train, col='Survived')
g.map(sns.distplot, 'Family_size', kde=False)
df_all['Ticket_info'] = df_all['Ticket'].apply(lambda x : x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')
df_all['Ticket_info'].unique()
df_train = df_all.loc['train']
ticket_count = df_all['Ticket'].value_counts() 
df_all['Ticket_count'] = df_all['Ticket'].apply(lambda x: ticket_count[x], 1)
df_all
sns.countplot(df_train['Ticket_info'], hue=df_train['Survived'])
df_train['Ticket_info'].unique()
df_all['withFamily'] = df_all['Family_size'].apply(lambda x: 1 if x > 0 else 0, 1)
df_all
g = sns.FacetGrid(df_train, col='Survived')
g.map(sns.distplot, 'Fare', kde=False)
sns.countplot(df_train['Embarked'], hue=df_train['Survived'])
#sns.countplot(df_train['Cabin'], hue=df_train['Survived'])
df_all['Cabin_info'] = df_all['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin')
df_all['Cabin_info'].unique()
df_train = df_all.loc['train']
sns.countplot(df_train['Cabin_info'], hue=df_train['Survived'])
def get_na_summary(data):  
    na_count = data.isnull().sum().sort_values(ascending=False)
    na_rate = (na_count / len(data))
    na_data = pd.concat([na_count, na_rate], axis=1, keys=['count','ratio'])
    return na_data

na_data = get_na_summary(df_all)
na_data.head(20)
print (df_all.groupby(by=['Embarked']).size().sort_values(ascending=False))
df_all['Embarked'] = df_all['Embarked'].fillna('C')
df_all_fare_not_null = df_all[df_all['Fare'].isnull() == False]
df_all['Fare'] = df_all['Fare'].fillna(df_all_fare_not_null['Fare'].median())
df_all_age_not_null = df_all[df_all['Age'].isnull() == False]
df_all_age_not_null = df_all_age_not_null.drop(['Survived'], 1)

corrmat = df_all_age_not_null.corr()
k = 10
cols = corrmat.nlargest(k, 'Age')['Age'].index

cm = np.corrcoef(df_all_age_not_null[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
yticklabels=cols.values, xticklabels=cols.values)
plt.show()
age_all = df_all.drop(['Cabin', 'Ticket', 'Family_size', 'Ticket_info', 'Ticket_count', 'Cabin_info', 'Name', 'Sex', 'Embarked', 'Family_name', 'Family_name_count', 'withFamily'], 1)
age_all
age_all = pd.get_dummies(age_all)

df_all_age_not_null = age_all[age_all['Age'].isnull() == False]
df_all_age_not_null = df_all_age_not_null.drop(['Survived'], 1)

corrmat = df_all_age_not_null.corr()
k = 10
cols = corrmat.nlargest(k, 'Age')['Age'].index

cm = np.corrcoef(df_all_age_not_null[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
yticklabels=cols.values, xticklabels=cols.values)
plt.show()
import xgboost as xgb
from sklearn.cross_validation import cross_val_score

age_all = age_all.drop(['Survived'], 1)

train_x = age_all[age_all['Age'].isnull() == False]
train_y = train_x['Age']
train_x = train_x.drop(['Age'], 1)

test_y = age_all[age_all['Age'].isnull() == True]
from sklearn import datasets, linear_model

model_xgb = xgb.XGBRegressor(learning_rate=0.01, max_depth=3, n_estimators=1000)
print ('cross_val_score: {0}'.format(cross_val_score(model_xgb, train_x, train_y, scoring='neg_mean_squared_error').mean()))
model_xgb.fit(train_x, train_y)

test_x = age_all[age_all['Age'].isnull() == True]
test_x = test_x.drop(['Age'], 1)
predict_y = model_xgb.predict(test_x)
test_x['Age'] = predict_y
test_x[['Age']]
age = df_all[['Age']].merge(test_x[['Age']], how='outer', left_index=True, right_index=True)
age_combine = age['Age_x'].combine_first(age['Age_y'])
age['Age'] = age_combine
df_all['Age'] = age['Age']
df_all
# df_all['Age'].fillna(df_all['Age'].median(), inplace = True)
# df_all
train_all = df_all.drop(['Cabin', 'Ticket', 'SibSp', 'Parch', 'Name', 'Family_name', 'Family_name_count', 'withFamily'], 1)
train_all
train_all = pd.get_dummies(train_all)
from sklearn.ensemble import RandomForestClassifier
train_x = train_all.loc['train']

train_y = train_x['Survived']
train_x = train_x.drop(['Survived'], 1)
rf = RandomForestClassifier(criterion='gini', 
                            n_estimators=2000,
                            min_samples_split=12,
                            min_samples_leaf=1,
                            oob_score=True,
                            random_state=1,
                             n_jobs=-1) 
print ('cross_val_score: {0}'.format(cross_val_score(rf, train_x, train_y).mean()))
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(alpha=0.001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False)
print ('cross_val_score: {0}'.format(cross_val_score(clf, train_x, train_y).mean()))
import keras 
from keras.models import Sequential # intitialize the ANN
from keras.layers import Dense      # create layers

from sklearn.model_selection import train_test_split

train_data,test_data,train_labels,test_labels=train_test_split(train_x,train_y,random_state=7,train_size=0.7)

model = Sequential()

model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))
model.add(Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(train_data, train_labels, batch_size = 32, epochs = 300)
from sklearn.metrics import accuracy_score

predictions = model.predict(test_data)
for x in np.nditer(predictions,op_flags=['readwrite']):
    if x < 0.5:
        x[...] = 0
    else:
        x[...] = 1
  
result = np.array(predictions,dtype='int')
print(accuracy_score(test_labels, result))
from sklearn import metrics

rf.fit(train_x, train_y)

test_x = train_all.loc['test']
test_x = test_x.drop(['Survived'], 1)

predict_y = rf.predict(test_x)

submission = pd.DataFrame()
submission['PassengerId'] = df_test.index
submission['Survived'] = predict_y
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('submission.csv', index=False)
submission