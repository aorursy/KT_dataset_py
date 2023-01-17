import numpy as np 

import pandas as pd

pd.plotting.register_matplotlib_converters()

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

np.random.seed(7)





# reading data into dataframe - train dataset, test dataset

train = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

test = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

train.head()
train.shape, test.shape
train.info()
# missing values in the train dataset



data_na = train.isna().sum()

data_na[data_na>0]
# missing values in test dataset



test_na = test.isna().sum()

test_na[test_na>0]
train.drop(['Name','Cabin','Ticket'], axis=1, inplace=True)

test.drop(['Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3,2, figsize=(12,8))

sns.set_style("darkgrid")

sns.distplot(train['Age'], color = 'green', kde=False, bins=30, ax=ax1)

sns.distplot(test['Age'], color = 'purple', kde=False, bins=30, ax=ax2)

sns.swarmplot(train['Sex'], train['Age'], palette='dark', ax=ax3)

sns.swarmplot(test['Sex'], test['Age'], palette ='rocket', ax=ax4)

sns.boxplot(train['Sex'], train['Age'], palette='dark', ax=ax5)

sns.boxplot(test['Sex'], test['Age'], palette ='rocket', ax=ax6)

ax1.set_title('Distribution of "Age" in train dataset', color='red')

ax2.set_title('Distribution of "Age" in test dataset', color='red')

ax3.set_title('"Age" in train dataset, by "Sex"', color='red')

ax4.set_title('"Age" in test dataset, by "Sex"', color='red')

plt.tight_layout()

plt.show()

# Concatenate train and test set to optimally fill missing values

df = pd.concat([train, test],sort=True).reset_index(drop=True)

df.index = df.index+1


for i, x in enumerate(df.columns):

    if (i%2 == 0):

        f,ax = plt.subplots(1,2, figsize=(12,5))

        sns.set_style('darkgrid')

        plt.figure(figsize=(6,4))

    sns.scatterplot(df[x], df['Age'], hue=df['Parch'], palette='dark', ax=ax[i%2])

    ax[i%2].set_ylabel('Age')

    ax[i%2].set_xlabel(str(x))

    if (i%2 == 1):

        plt.show()

sns.set_style('whitegrid')

plt.figure(figsize=(14,6))

sns.scatterplot(y=df['SibSp'], x=df['Age'], hue=df['Parch'], palette='Set1')

plt.legend()

plt.show()
df_new = df.groupby(['Parch','SibSp']).Age.describe().reset_index()



print('*'*75,'\nStatistical Insight on feature "Age"\n',  df.groupby(['Parch','SibSp']).Age.describe())

sns.set_style('darkgrid')

i=list(np.arange(60,220,20))

plt.figure(figsize=(12,5))

sns.scatterplot(x=df_new['mean'], y=df_new['50%'], hue=df_new['SibSp'], size=df_new['Parch'], sizes=i, palette='deep')

plt.show()

#

plt.figure(figsize=(12,5))

sns.boxplot(y=df['Age'], x=df['SibSp'], palette='deep')

plt.show()

#

plt.figure(figsize=(14,5))

sns.boxplot(y=df['Age'], x=df['Parch'], hue=df['SibSp'], palette='deep')

plt.show()

#

plt.figure(figsize=(14,5))

sns.boxplot(y=df['Age'], hue=df['Parch'], x=df['SibSp'], palette='deep')

plt.show()
df.columns
# Fill missing values



for i in ['Age', 'Fare']:

    df[i] = df.groupby(['Parch', 'SibSp'])[i].apply(lambda x: x.fillna(x.median()))

    df[i] = df.groupby(['SibSp'])[i].apply(lambda x: x.fillna(x.median()))          # to fill missing which cannot be grouped by Parch

df['Embarked'] = df.groupby(['SibSp'])['Embarked'].apply(lambda x: x.fillna(x.value_counts().index[0]))    # fill with modal value

# Split train and test data as original

train = df.loc[:891]

test = df.loc[892:].drop('Survived', axis=1)
train.shape, test.shape
train.isnull().sum().sum(), test.isnull().sum().sum()
from sklearn.model_selection import train_test_split

from category_encoders import TargetEncoder

from sklearn.feature_selection import SelectKBest, f_classif



# Split train data into feature data X and target data y

X = train.drop(['Survived'], axis=1)

y = train['Survived']



# Get 20% of validation data from train set

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

X_train.Age.astype('float64')
import itertools

num_col = ['Age', 'Fare']

cat_col = list(set(X.columns)-set(num_col))



# Generate interaction features with categorical columns

for c1, c2 in itertools.combinations(cat_col, 2):

    print(c1,c2)

    name = '_'.join([c1,c2])

    X_train[name] = X_train[c1].map(str)+'_'+X_train[c2].map(str)

    X_valid[name] = X_valid[c1].map(str)+'_'+X_valid[c2].map(str)

    test[name] = test[c1].map(str)+'_'+test[c2].map(str)



print(X_train.columns)

# Encode categorical variables

new_cat_col = [i for i in X_train.columns if X_train[i].dtype != 'float64']



enc = TargetEncoder(cols=new_cat_col)

X_train[new_cat_col] = enc.fit_transform(X_train[new_cat_col], y_train)

X_valid[new_cat_col] = enc.transform(X_valid[new_cat_col])

test[new_cat_col] = enc.transform(test[new_cat_col])

# Feature Selection

selector = SelectKBest(f_classif, k=15)

X_1 = selector.fit_transform(X_train, y_train)

X_2 = pd.DataFrame(selector.inverse_transform(X_1), index=X_train.index, columns = X_train.columns)

select_col = X_2.columns[X_2.var() != 0]



print(select_col)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix



model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)

model.fit(X_train[select_col], y_train)

prediction = model.predict(X_valid[select_col])



accuracy = accuracy_score(y_valid, prediction)

print('Validation Accuracy :', accuracy)

conf = confusion_matrix(y_valid, prediction)

print('Confusion matrix: ', conf)
# predicting test dataset

X_full = pd.concat([X_train,X_valid], sort=True)

y_full = pd.concat([y_train,y_valid], sort=True)

model_full = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)

model_full.fit(X_full[select_col], y_full)

final_pred = model_full.predict(test[select_col])

binary = final_pred.astype('int64')

output = pd.DataFrame({'PassengerId':test.index, 'Survived':binary})

output.to_csv('submission.csv', index=False)