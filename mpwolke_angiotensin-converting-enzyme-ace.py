#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT3hI0zL5ogpWGhM1kedDeZvdpRtewiXOh5ZIAaf7M-Ask5CM8-&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/cusersmarildownloadsangiotensincsv/angiotensin.csv', sep=';')

df
# First we will see Gender of our respondenets

sns.countplot(df['Genotyping'],linewidth=3,palette="Set2",edgecolor='black')

plt.show()
# First we will see Gender of our respondenets

sns.countplot(df['End_point'],linewidth=3,palette="Set2",edgecolor='black')

plt.show()
# First we will see Gender of our respondenets

sns.countplot(df['Country'],linewidth=3,palette="Set3",edgecolor='black')

plt.show()
sns.countplot(x=df['Selection'],palette='coolwarm',linewidth=2,edgecolor='black')
plt.figure(figsize=(18,6))

plt.subplot(1, 2, 1)

sns.countplot(x=df['LVH_GG'],hue=df['Outcomer'],palette='summer',linewidth=3,edgecolor='white')

plt.title('Outcomer')

plt.subplot(1, 2, 2)

sns.countplot(x=df['LVH_GG'],hue=df['Comparability'],palette='hot',linewidth=3,edgecolor='white')

plt.title('Comparability')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41368-020-0074-x/MediaObjects/41368_2020_74_Fig1_HTML.png?as=webp',width=400,height=400)
sns.countplot(x=df['LVH_GA'],hue=df['Selection'],palette='Oranges',linewidth=2,edgecolor='black')

plt.title('LVH GA')

plt.show()
fig = px.bar(df, x= "Country", y= "LVH_GG", color_discrete_sequence=['crimson'],)

fig.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41368-020-0074-x/MediaObjects/41368_2020_74_Fig2_HTML.png?as=webp',width=400,height=400)
import shap

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import random
df.isnull().sum()
df['LVH_GG'] = df['LVH_GG'].replace(['negative','positive'], [0,1])
SEED = 99

random.seed(SEED)

np.random.seed(SEED)
dfmodel = df.copy()



# read the "object" columns and use labelEncoder to transform to numeric

for col in dfmodel.columns[dfmodel.dtypes == 'object']:

    le = LabelEncoder()

    dfmodel[col] = dfmodel[col].astype(str)

    le.fit(dfmodel[col])

    dfmodel[col] = le.transform(dfmodel[col])
#change columns names to alphanumeric

dfmodel.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dfmodel.columns]
X = dfmodel.drop(['LVH_GG','Author_and_year'], axis = 1)

y = dfmodel['LVH_GG']
lgb_params = {

                    'objective':'binary',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.005,

                    'num_leaves': 20,

                    'max_depth':-1,

                    'subsample':0.9,

                    'n_estimators':2500,

                    'seed': SEED,

                    'early_stopping_rounds':100, 

                }
# choose the number of folds, and create a variable to store the auc values and the iteration values.

K = 5

folds = KFold(K, shuffle = True, random_state = SEED)

best_scorecv= 0

best_iteration=0



# Separate data in folds, create train and validation dataframes, train the model and cauculate the mean AUC.

for fold , (train_index,test_index) in enumerate(folds.split(X, y)):

    print('Fold:',fold+1)

          

    X_traincv, X_testcv = X.iloc[train_index], X.iloc[test_index]

    y_traincv, y_testcv = y.iloc[train_index], y.iloc[test_index]

    

    train_data = lgb.Dataset(X_traincv, y_traincv)

    val_data   = lgb.Dataset(X_testcv, y_testcv)

    

    LGBM = lgb.train(lgb_params, train_data, valid_sets=[train_data,val_data], verbose_eval=250)

    best_scorecv += LGBM.best_score['valid_1']['auc']

    best_iteration += LGBM.best_iteration



best_scorecv /= K

best_iteration /= K

print('\n Mean AUC score:', best_scorecv)

print('\n Mean best iteration:', best_iteration)
lgb_params = {

                    'objective':'binary',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.05,

                    'num_leaves': 20,

                    'max_depth':-1,

                    'subsample':0.9,

                    'n_estimators':round(best_iteration),

                    'seed': SEED,

                    'early_stopping_rounds':None, 

                }



train_data_final = lgb.Dataset(X, y)

LGBM = lgb.train(lgb_params, train_data)
print(LGBM)
# telling wich model to use

explainer = shap.TreeExplainer(LGBM)

# Calculating the Shap values of X features

shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[1], X, plot_type="bar")
shap.summary_plot(shap_values[1], X)