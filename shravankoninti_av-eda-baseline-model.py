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
import pandas as pd

import numpy as np

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)  

from sklearn import metrics, preprocessing, model_selection

from sklearn.model_selection import train_test_split,cross_val_predict

from sklearn.ensemble import RandomForestClassifier

from xgboost.sklearn import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt                                      # to plot graph

%matplotlib inline

import xgboost as xgb

import lightgbm as lgb

SEED = 1



#To ignore warnings

import warnings

warnings.filterwarnings('ignore')
file = r'/kaggle/input/analytics-vidhya-janatahack-customer-segmentation/'

train_df = pd.read_csv(file+'Train_aBjfeNk.csv')

test_df = pd.read_csv(file+'Test_LqhgPWU.csv')

sub_df = pd.read_csv(file+'sample_submission_wyi0h0z.csv')
train_df.head()
test_df.head()
sub_df.head()
print(train_df.shape, test_df.shape,sub_df.shape)
train_df['Segmentation'].value_counts()
train_df.isnull().sum()
test_df.isnull().sum()
missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(8,6))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
missingvalues_prop = (train_df.isnull().sum()/len(train_df)).reset_index()

missingvalues_prop.columns = ['field','proportion']

missingvalues_prop = missingvalues_prop.sort_values(by = 'proportion', ascending = False)

# print(missingvalues_prop)

missingvaluescols = missingvalues_prop[missingvalues_prop['proportion'] > 0.10].field.tolist()

print(missingvaluescols)
# Normalise can be set to true to print the proportions instead of Numbers.

train_df['Segmentation'].value_counts(normalize=True)
train_df['Segmentation'].value_counts().plot.bar(figsize=(4,4),title='Segmentation - Split for Train Dataset')

plt.xlabel('ExtraTime')

plt.ylabel('Count')
plt.figure(1)

plt.subplot(221)

train_df['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), fontsize = 15.0)

plt.title('Gender', fontweight="bold", fontsize = 22.0)

plt.ylabel('Count %', fontsize = 20.0)





plt.subplot(222)

train_df['Ever_Married'].value_counts(normalize=True).plot.bar(figsize=(20,10), fontsize = 15.0)

plt.title('Ever_Married', fontweight="bold",fontsize = 22.0)

plt.ylabel('Count %', fontsize = 20.0)



plt.subplot(223)

train_df['Graduated'].value_counts(normalize=True).plot.bar(figsize=(20,10), fontsize = 15.0)

plt.title('Graduated', fontweight="bold", fontsize = 22.0)

plt.ylabel('Count %', fontsize = 20.0)



plt.subplot(224)

train_df['Work_Experience'].value_counts(normalize=True).plot.bar(figsize=(20,10), fontsize = 15.0)

plt.title('Work_Experience', fontweight="bold", fontsize = 22.0)

plt.ylabel('Count %', fontsize = 20.0)

plt.tight_layout()
plt.figure(1)

plt.subplot(221)

train_df['Spending_Score'].value_counts(normalize=True).plot.bar(figsize=(20,10), fontsize = 15.0)

plt.title('Spending_Score', fontweight="bold", fontsize = 22.0)

plt.ylabel('Count %', fontsize = 20.0)





plt.subplot(222)

train_df['Family_Size'].value_counts(normalize=True).plot.bar(figsize=(20,10), fontsize = 15.0)

plt.title('Family_Size', fontweight="bold",fontsize = 22.0)

plt.ylabel('Count %', fontsize = 20.0)



plt.subplot(223)

train_df['Var_1'].value_counts(normalize=True).plot.bar(figsize=(20,10), fontsize = 15.0)

plt.title('Var_1', fontweight="bold", fontsize = 22.0)

plt.ylabel('Count %', fontsize = 20.0)

plt.figure(1)

plt.subplot(121)

sns.distplot(train_df['Age'])



plt.subplot(122)

train_df['Age'].plot.box(figsize=(16,5))



plt.show()
train_df.columns
Gender=pd.crosstab(train_df['Gender'],train_df['Segmentation'])

Ever_Married=pd.crosstab(train_df['Ever_Married'],train_df['Segmentation'])

Graduated=pd.crosstab(train_df['Graduated'],train_df['Segmentation'])

Profession=pd.crosstab(train_df['Profession'],train_df['Segmentation'])







Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

Ever_Married.div(Ever_Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

Graduated.div(Graduated.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

Profession.div(Profession.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

plt.tight_layout()

Work_Experience=pd.crosstab(train_df['Work_Experience'],train_df['Segmentation'])

Ever_Married=pd.crosstab(train_df['Ever_Married'],train_df['Segmentation'])

Graduated=pd.crosstab(train_df['Graduated'],train_df['Segmentation'])

Profession=pd.crosstab(train_df['Profession'],train_df['Segmentation'])







Work_Experience.div(Work_Experience.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

Ever_Married.div(Ever_Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

Graduated.div(Graduated.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

Profession.div(Profession.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))



# * join the datasets

train_df['is_train']  = 1

test_df['Segmentation'] = -1

test_df['is_train'] = 0
full_df = train_df.append(test_df)
full_df.head()
full_df.dtypes
full_df.isnull().sum()
# append train and test data

testcount = len(test_df)

count = len(full_df)-testcount

print(count)



train = full_df[:count]

test = full_df[count:]

train_df = train.copy()

test_df = test.copy()
full_df.columns
cols = ['ID', 'Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession',

       'Work_Experience', 'Spending_Score', 'Family_Size', 'Var_1',

        'is_train' ]

for col in cols:

    if train_df[col].dtype==object:

        print(col)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))

        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))

        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
X = train_df.drop(['Segmentation', 'is_train' ,'ID'],axis=1)

y = train_df['Segmentation'].values



train_X = X.copy()

train_y = y.copy()



test_X = test_df.drop(['Segmentation', 'is_train' ,'ID'],axis=1)

print(X.shape, test_X.shape)
X.head()
test_X.head()
X.isnull().sum()
params = {}

params['learning_rate'] = 0.01

params['n_estimators'] = 10000

params['objective'] = 'multiclass'

params['boosting_type'] = 'gbdt'
feature_cols = X.columns
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                  stratify=y, 

                                                  random_state=1234, 

                                                  test_size=0.20, shuffle=True)
cat_cols = ['Gender','Ever_Married', 'Graduated', 'Profession','Family_Size',

            'Spending_Score','Var_1']

label_col = 'Segmentation'
clf = lgb.LGBMClassifier(**params)

    

clf.fit(X_train, y_train, early_stopping_rounds=200,

        eval_set=[(X_valid, y_valid)], 

        eval_metric='multi_error', verbose=False, categorical_feature=cat_cols)



eval_score = accuracy_score(y_valid, clf.predict(X_valid[feature_cols]))



print('Eval ACC: {}'.format(eval_score))
preds = clf.predict(test_X[feature_cols])
np.unique(preds, return_counts=True)
sub_df['Segmentation'] = preds



sub_df.head()
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



name = "baseline_lgb.csv"



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = name):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(sub_df)