# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings('ignore')







# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import math

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm



import string

import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_gender = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

test = pd.concat([df_test,df_gender['Survived']] , axis = 1)

def concat(train_data, test_data):

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

data = concat(df_train , test)

df_train.describe()

df_test.describe()
cabin_uniq = df_train['Cabin'].value_counts()

embark_uniq = df_train['Embarked'].value_counts()

embark_uniq
sex_s = df_train.groupby(['Sex', 'Survived']).Sex.count()

age_s = df_train.groupby(['Age', 'Survived']).Age.count()

sib_s = df_train.groupby(['SibSp', 'Survived']).SibSp.count()

parch_s = df_train.groupby(['Parch', 'Survived']).Parch.count()

fare_s = df_train.groupby(['Fare', 'Survived']).Fare.count()

cabin_s = df_train.groupby(['Cabin', 'Survived']).Cabin.count()

embark_s = df_train.groupby(['Embarked', 'Survived']).Embarked.count()

print("sex survived \n",sex_s)

print("age survived \n",age_s)
fig, axar = plt.subplots(2, 2 , figsize=(12, 8))



df_train.groupby(['Sex','Survived']).Survived.count().unstack('Sex').plot.bar( ax = axar [0][0])

df_train.groupby(['SibSp','Survived']).Survived.count().unstack('SibSp').plot.bar(ax = axar [0][1])

df_train.groupby(['Parch','Survived']).Survived.count().unstack('Parch').plot.bar(ax = axar [1][0])

df_train.groupby(['Embarked','Survived']).Survived.count().unstack('Embarked').plot.bar(ax = axar [1][1])
df_train_new = df_train.copy()

df_train.groupby('Survived').Fare.hist()
def univariate_kdeplots(dataframe, plot_features, cols=2, width=10, height=10, hspace=0.2, wspace=0.25):

    # define style and layout

    sns.set(font_scale=1.5)

    plt.style.use('seaborn-whitegrid')

    fig = plt.figure(figsize=(width, height))

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)

    rows = math.ceil(float(dataframe.shape[1]) / cols)

    # define subplots

    for i, feature in enumerate(plot_features):

        ax = fig.add_subplot(rows, cols, i + 1)

        g = sns.kdeplot(dataframe[plot_features[i]][(dataframe['Survived'] == 0)].dropna(), shade=True, color="red")

        g = sns.kdeplot(dataframe[plot_features[i]][(dataframe['Survived'] == 1)].dropna(), shade=True, color="blue")

        g.set(xlim=(0 , dataframe[plot_features[i]].max()))

        g.legend(['Died', 'Survived'])

        plt.xticks(rotation=25)

        ax.set_xlabel(plot_features[i], weight='bold')
univariate_kdeplots(data, ['Age'], cols=1, width=15, height=100, hspace=0.4, wspace=0.25)
df_train['Survived'].mean()
df_train.groupby(['Pclass']).mean()
df_pclass_sex = df_train.groupby(['Pclass','Sex']).mean()

df_pclass_sex

df_pclass_sex = df_pclass_sex['Survived']

df_pclass_sex.plot.bar()
df_train.describe()
fig, axar = plt.subplots(1, 2 , figsize=(12, 8))

df_age = pd.cut(df_train['Age'], np.arange(0, 90, 10))

df_mean_age = df_train.groupby(df_age).mean()

df_mean_age['Survived'].plot.bar(ax = axar[0])

df_age = pd.cut(df_train['Age'], np.arange(0, 90, 5))

df_mean_age = df_train.groupby(df_age).mean()

df_mean_age['Survived'].plot.bar(ax = axar [1])
data['family_size'] = data['Parch'] + data['SibSp'] +1
data['Title'] = data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

tt_count = data['Title'].value_counts()

data['Surname'] = data['Name'].str.split(', ', expand=True)[0]
miss_train = df_train.isnull().sum()

#print("missing values of train set\n" , miss_train)

miss_test = df_test.isnull().sum()

#print("missing values of test set\n" , miss_test)

miss_data = data.isnull().sum()

print("missing values in data \n" , miss_data)
mean_master = data.loc[data['Title']=='Master'].mean()['Age']

mean_master
data['Age'][(data['Title'] == 'Master') & (data['Age'].isnull())] = mean_master
data[data['Age'].isnull()].loc[data['Title']== 'Master']

data['Age'].isnull().sum()


data.groupby(['Title','Survived']).Survived.count().unstack('Title').plot.bar()
df_pclass_sex_age = data.groupby(['Pclass','Sex' ]).mean()



df_pclass_sex_age
data['Age'] = data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.mean()))

embarked_null = data[data['Embarked'].isnull()]

embarked_null
data['Embarked'] = data['Embarked'].fillna('S')
fare_null = data[data['Fare'].isnull()]

fare_null


mean_fare = data.groupby(['Pclass', 'Sex']).Fare.mean()[3][1]

data['Fare'] = data['Fare'].fillna(mean_fare)

cabin_null = data[data['Cabin'].isnull()]

cabin_null
data.groupby(['Cabin'], as_index=False)['Pclass'].count()
data['Cabin_code']=data.Cabin.str.extract('([a-zA-Z])' , expand=True)[0]

cabins = data['Cabin_code']
data.groupby(['Cabin_code','Pclass']).Pclass.count().unstack('Pclass').plot.bar()
data.groupby(['Cabin_code','Pclass', 'Surname']).count()

data.groupby(['Pclass']).median()




label = LabelEncoder()

data['Sex_code'] = label.fit_transform(data['Sex']) #male 1 female 0 

#data_new = data.drop(['Sex'] , axis=1)

data['Embarked_code'] = label.fit_transform(data['Embarked']) #s 2 c 0 q 1

bin_thresholds = [0, 15, 25, 40,60,80]

bin_labels = ['0-15', '16-25', '26-40', '41-60', '61-80']

data['Age_bin'] = pd.cut(data['Age'], bins=bin_thresholds, labels=bin_labels)

Age_bin_to_integer = {'0-15': 1,

                     '16-25': 2, 

                     '26-40': 3, 

                     '41-60': 4,

                     '61-80': 5}



data['Age_code'] = data['Age_bin'].map(Age_bin_to_integer)
feature_list = ['Fare' ,'Parch' , 'Pclass', 'SibSp' ,'family_size' ,'Sex_code','Embarked_code','Age_code']

x = data[feature_list].values

y = data['Survived'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score , confusion_matrix

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from xgboost import plot_importance
model = xgb.XGBClassifier()

kfold = KFold(n_splits=10, random_state=7)

results = cross_val_score(model, x, y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
data_dmatrix = xgb.DMatrix(data=x,label=y)

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}



cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

linreg = LogisticRegression()

results = cross_val_score(linreg, x, y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

tree = DecisionTreeClassifier()

tree = tree.fit(x_train, y_train)

y_pred_tree = tree.predict(x_test)

print(confusion_matrix(y_test,y_pred_tree))

print("Accuracy:",accuracy_score(y_test, y_pred_tree))

print(classification_report(y_test,y_pred_tree))
rfc = RandomForestClassifier()

rfc.fit(x_train, y_train)

y_pred_rf = rfc.predict(x_test)

#print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred_rf))

print('Mean Squared Error:', mean_squared_error(y_test, y_pred_rf))

print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred_rf)))





print(confusion_matrix(y_test,y_pred_rf))

print(classification_report(y_test,y_pred_rf))

print(accuracy_score(y_test, y_pred_rf))
clf_svm =  svm.SVC(kernel='rbf', C=1)

scores = cross_val_score(clf_svm, x, y, cv=5)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))