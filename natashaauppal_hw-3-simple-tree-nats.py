# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')

print(df_train.shape)



df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')

print(df_test.shape)
df_train.head()
df_test['target'] = np.nan



df = pd.concat([df_train, df_test])



print(df.shape)



df_test.isnull().sum()

categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']



print('Categorical features - {}'.format(categorical_features))



numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']



print('Numerical features - {}'.format(numerical_features))



# Removing '?' from dataframe, some problem with dropna hence not using it.



for feature in categorical_features:

    df = df[df[feature] !='?']





features_with_na = [feature for feature in df.columns if df[feature].isnull().sum() > 0]

print(features_with_na)



## Ordinal - education

## Nominal - workclass, marital status, occupation, relationship, race, sex, native country
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()



for feature in df.columns:

    if df[feature].dtypes != 'O':

        df[feature].hist()

        plt.title(feature)

    plt.show()

    

    

def bar_chart(feature):

    # just calling rich - salary >= 50K, poor - salary < 50K

    rich = df_train[df_train['target'] == 1][feature].value_counts()

    poor = df_train[df_train['target'] == 0][feature].value_counts()

    sal_df = pd.DataFrame([rich, poor])

    sal_df.index = ['Rich', 'Poor']

    sal_df.plot(kind='bar', stacked=True, figsize=(10,5))

    

for feature in categorical_features:

    bar_chart(feature)

    
df.columns
df_tmp = df.loc[

    df['target'].notna()

].groupby(

    ['education']

)[

    'target'

].agg(['mean', 'std']).rename(

    columns={'mean': 'target_mean', 'std': 'target_std'}

).fillna(0.0).reset_index()



df_tmp.head()



df = pd.merge(

    df,

    df_tmp,

    how='left',

    on=['education']

)



df.shape

df['target_mean'] = df['target_mean'].fillna(0.0)

df['target_std'] = df['target_std'].fillna(0.0)
df.columns
features_with_na = [feature for feature in df.columns if df[feature].isnull().sum() > 0]

print(features_with_na)
# gender = pd.get_dummies(df['sex'], drop_first=True)

# df = pd.concat([df, gender], axis=1)

# df.drop(['sex'], axis=1, inplace=True)



df_train.shape



df_train['marital-status'].unique()



# # adding target mean encoding for native-country as it contains lot of categorical values

native_country_mean = df.loc[

    df['target'].notna()

].groupby(

    ['native-country'])['target'].mean()



df['native_country_mean_target'] = df['native-country'].map(native_country_mean)
# combining workclass in self emp, govt, private, others

print(df['workclass'].value_counts())



def get_workclass(x):

    if 'Private' in x:

        return 'private'

    elif 'Self' in x:

        return 'self'

    elif 'gov' in x:

        return 'gov'

    else:

        return 'others'



df['workclass_cat'] = df['workclass'].apply(lambda x: x.strip()).apply(lambda x: get_workclass(x))





# categorize marital status in married vs non married

df['marital_status_cat'] = df['marital-status'].apply(lambda x: 'married' if x.startswith('Married',1) else 'Single')



# Its evident from race target graph that white has majority so we can categorize them white vs others

df['race_cat']=df['race'].apply(lambda x: x.strip()).apply(lambda x: 'White' if x=='White' else 'Other')



print(df.info())
df.target.isna().sum()


df.drop(['marital-status', 'race', 'workclass', 'education', 'native-country'], inplace=True, axis=1)

categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']



print('Categorical features - {}'.format(categorical_features))



df.columns
for feature in ['occupation', 'relationship', 'sex', 'workclass_cat', 'marital_status_cat', 'race_cat']:

    new_col = pd.get_dummies(df[feature], drop_first=True)

    df = pd.concat([df, new_col], axis=1)

    df.drop([feature], axis=1, inplace=True)

    

features_with_na = [feature for feature in df.columns if df[feature].isnull().sum() > 0]

print(features_with_na)

df.shape

df.columns
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split, cross_val_score





# create training and testing vars

# y = df_train['target']

# df_train.drop('target', axis=1, inplace=True)

# X_train, X_holdout, y_train, y_holdout = train_test_split(df_train.values, y, test_size=0.4, random_state=17)

# print(df.shape)

# print(X_train.shape, y_train.shape)

# print(X_holdout.shape, y_holdout.shape)



from sklearn.model_selection import cross_val_score



from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier(

    criterion='gini',

    splitter='best',

    max_depth=7,

    min_samples_split=42,

    min_samples_leaf=17,

    random_state=14

)





model = model.fit(df.loc[df['target'].notna()].drop(columns=['target']), df.loc[df['target'].notna()]['target'])



#model = tree.fit(df.loc[df['target'].notna()][cols], df.loc[df['target'].notna()]['target'])



#tree.fit(X_train, y_train)



from sklearn.metrics import accuracy_score



#tree_pred = tree.predict(X_holdout)

#print(accuracy_score(y_holdout, tree_pred))



#scores = cross_val_score(tree, df.values, y, cv=10)

#print(scores)
df['target'].isna().sum()

df.loc[df['target'].isna()]
df.head()



model.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']).head())
p = model.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']))[:, 1]
%matplotlib inline

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("dark")
sns.distplot(p) # DECSION TREES



#sns.distplot(p[:, 1]) # knn
df_submit = pd.DataFrame({

    'uid': df.loc[df['target'].isna()]['uid'],

    'target': p

})
df_submit.to_csv('/kaggle/working/submit.csv', index=False)
!head /kaggle/working/submit.csv