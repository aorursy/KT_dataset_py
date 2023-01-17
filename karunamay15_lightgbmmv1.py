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
#Data Reading

train=pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

train_labels=pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

test=pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

spec=pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
print(train.shape)

print(test.shape)

print(train_labels.shape)
train.head( )

train_labels.head( )
#Count of unique installation ids

train.installation_id.unique( ).shape


#visualization

import matplotlib.pyplot as plt

import seaborn as sns
# accuracy group distribution across different titles

assesments=train_labels['title'].unique( ).tolist( )

plt.figure(figsize=(15,8))

for asset in assesments:

    # Subset by assesment type

    subset = train_labels[train_labels['title'] == asset]

    

    # Draw the density plot

    sns.distplot(train_labels['accuracy_group'], hist = False, kde = True,

                 kde_kws = {'shade': True, 'linewidth': 3},label =asset)



# Plot formatting

plt.legend(prop={'size': 16}, title = 'Assesment type')

plt.title('Density Plot with Assesment category')

plt.xlabel('accuracy_group')

plt.ylabel('Density')

plt.show( )

    
# accuracy group distribution of  incorrect answers

plt.figure(figsize=(15,8))

fg = sns.FacetGrid(data=train_labels,hue='accuracy_group',size=4,aspect=3)

fg.map(plt.scatter, 'accuracy_group','num_correct').add_legend()

plt.show( )
# accuracy group distribution of  incorrect answers

plt.figure(figsize=(15,8))

fg = sns.FacetGrid(data=train_labels,hue='accuracy_group',size=4,aspect=3)

fg.map(plt.scatter, 'accuracy_group','num_incorrect').add_legend()

plt.show( )
#Creating some new feature try some new features
class featureE:

    def __init__(self):

        self.count=0

    def get_time(self,df):

        df['timestamp'] = pd.to_datetime(df['timestamp'])

        df['date'] = df['timestamp'].dt.date

        df['month'] = df['timestamp'].dt.month

        df['hour'] = df['timestamp'].dt.hour

        df['dayofweek'] = df['timestamp'].dt.dayofweek

        return df

    def get_object_columns(self,df, columns):

        df = df.groupby(['installation_id', columns])['event_id'].count().reset_index()

        df = df.pivot_table(index = 'installation_id', columns = [columns], values = 'event_id')

        df.columns = list(df.columns)

        df.fillna(0, inplace = True)

        return df

    def get_numeric_columns(self,df, column):

        df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'std']})

        df.fillna(0, inplace = True)

        df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_std']

        return df

    def get_numeric_columns_2(self,df, agg_column, column):

        df = df.groupby(['installation_id', agg_column]).agg({f'{column}': ['mean', 'sum', 'std']}).reset_index()

        df = df.pivot_table(index = 'installation_id', columns = [agg_column], values = [col for col in df.columns if col not in ['installation_id', 'type']])

        df.fillna(0, inplace = True)

        df.columns = list(df.columns)

        return df

    def get_correct_incorrect(self,df):

        df = df.groupby(['title'])['num_correct', 'num_incorrect'].agg({'num_correct': ['mean', 'std'], 'num_incorrect': ['mean', 'std']}).reset_index()

        df.columns = ['title', 'num_correct_mean', 'num_correct_std', 'num_incorrect_mean', 'num_incorrect_std']

        return df
# columns for feature engineering

#calling the feature class

d=featureE( )

numerical_columns = ['game_time', 'event_count']

categorical_columns = ['type', 'world']

#creating features from time stamp

numerical_columns_single = ['hour', 'dayofweek', 'month', 'event_id_count', 'event_code_count']
# get time features

train =d.get_time(train)

test =d.get_time(test)   
#grouping by cout for getting unique pairs

def count_segments(train, test, cols):

    for col in cols:

        for df in [train, test]:

            df[f'{col}_count'] = df.groupby([col])['timestamp'].transform('count')

    return train, test
count_segments(train, test, ['event_id', 'event_code'])
#passing train and test for aggregation and feature creation

dftrain = pd.DataFrame({'installation_id': train['installation_id'].unique()})

dftrain.set_index('installation_id', inplace = True)

dftest = pd.DataFrame({'installation_id': test['installation_id'].unique()})

dftest.set_index('installation_id', inplace = True)
#Bringing numerical columns

for i in numerical_columns:

    dftrain = dftrain.merge(d.get_numeric_columns(train, i), left_index = True, right_index = True)

    dftest = dftest.merge(d.get_numeric_columns(test, i), left_index = True, right_index = True)
#Brining categorical columns

for i in categorical_columns:

        dftrain = dftrain.merge(d.get_object_columns(train, i), left_index = True, right_index = True)

        dftest = dftest.merge(d.get_object_columns(test, i), left_index = True, right_index = True)
#categorical columns grouping

for i in categorical_columns:

        for j in numerical_columns:

            dftrain = dftrain.merge(d.get_numeric_columns_2(train, i, j), left_index = True, right_index = True)

            dftest = dftest.merge(d.get_numeric_columns_2(test, i, j), left_index = True, right_index = True)
#getting columns related time stamp of operations

for i in numerical_columns_single:

        dftrain = dftrain.merge(d.get_numeric_columns(train, i), left_index = True, right_index = True)

        dftest = dftest.merge(d.get_numeric_columns(test, i), left_index = True, right_index = True)

         

dftrain.reset_index(inplace = True)

dftest.reset_index(inplace = True)  
# for grouping getting the mode accuracy group of titles--assuming title has unique map

labels_map = dict(train_labels.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))

# merge target

labels = train_labels[['installation_id', 'title', 'accuracy_group']]

# merge with correct incorrect

corr_inc = d.get_correct_incorrect(train_labels)

labels = labels.merge(corr_inc, how = 'left', on = 'title')

# replace title with the mode

labels['title'] = labels['title'].map(labels_map)

test.head( )
#For test set assiging  mode accuracy group to get correct_incorrect

dftest['title'] = test.groupby('installation_id').last()['title'].reset_index(drop = True)

dftest = dftest.merge(corr_inc, how = 'left', on = 'title')

dftest.head( )
# map title to convert title to numeric variable

dftest['title'] = dftest['title'].map(labels_map)

dftest.head( )
#preparing final train data as compatible to test data

# join train with labels

dftrain = labels.merge(dftrain, on = 'installation_id', how = 'left')

dftrain = dftrain[[col for col in dftest.columns] + ['accuracy_group']]

print('We have {} training rows'.format(dftrain.shape[0]))
#Distribution of accuracy group in data

plt.figure(figsize=(15,5))

sns.countplot(dftrain['accuracy_group'])

plt.ylabel("Count")

plt.title("Accuracy group counts", y=1, fontdict={"fontsize": 20});
#Formulating as a classification problem---
#Initial set of Xs

InFeature=dftrain.drop(['installation_id','accuracy_group'],axis=1).columns.tolist( )

print(InFeature)
#Will do first level feature selection using Random Forest method -recursively 
from sklearn.model_selection import train_test_split
#Train - test split- 80/20

X = dftrain[InFeature].values

y =dftrain['accuracy_group'].values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)

print(f"Original data shapes: {X_train.shape, X_test.shape}")
from sklearn.ensemble import RandomForestClassifier

# from sklearn.pipeline import make_pipline

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from sklearn.feature_selection import RFE
%%time

#RFE-recursive feture engineering

rfc = RandomForestClassifier(n_estimators=455,min_samples_split=10,min_samples_leaf=100,max_features='auto', max_depth=10, bootstrap=False,class_weight="balanced")

sel_rfe_tree = RFE(estimator=rfc, n_features_to_select=20, step=1)

X_train_rfe_tree = sel_rfe_tree.fit_transform(X_train, y_train)

print(sel_rfe_tree.get_support())
#top20 features

RFEtop20=[InFeature[i] for i in range(0,len(InFeature)) if(sel_rfe_tree.ranking_[i]==1)]

print(RFEtop20)
#Trying lightgbm
#train and test split

X = dftrain[RFEtop20].values

y = dftrain['accuracy_group'].values.flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)

print(f"Original data shapes: {X_train.shape, X_test.shape}")
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(X_train)

x_test = sc.transform(X_test)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(X_train)

x_test = sc.transform(X_test)

import lightgbm as lgb

d_train = lgb.Dataset(x_train, label=y_train)

params = {}

params['learning_rate'] = 0.001

params['boosting_type'] = 'gbdt'

params['objective'] = 'multiclass'

params['num_classes'] = 4

params['metric'] = 'multi_logloss'

params['sub_feature'] = 0.8

params['num_leaves'] = 50

params['min_data'] = 100

params['max_depth'] = 10

clf = lgb.train(params, d_train, 100)
#predictions on train data

predstrain = clf.predict(x_train)

predstest = clf.predict(x_test)

predictionsTR = []

for x in predstrain:

    predictionsTR.append(np.argmax(x))

predictionsTE=[ ]

for x in predstest:

    predictionsTE.append(np.argmax(x))
#scoring on train data

print("Train-F1-score-",f1_score(predictionsTR,y_train,average='weighted'))

print("Test-F1-score-",f1_score(predictionsTE,y_test,average='weighted'))
#predictions on test data for submission

xtestfinal=dftest[RFEtop20].values

predstest = clf.predict(xtestfinal)

predicttest=[ ]

for x in predstest:

    predicttest.append(np.argmax(x))
df=dftest[['installation_id']]

df['accuracy_group']=predicttest

df.head( )
sample_submission=pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

sample_submission=sample_submission[['installation_id']]

sample_submission.head( )
#submission result

submission= sample_submission.merge(df, on = 'installation_id')

submission.head( )
#submission

submission.to_csv("submission.csv",index=False)