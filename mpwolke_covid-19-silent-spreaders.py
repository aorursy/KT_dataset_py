#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT8Ri42Jt4fEhipkUKQXeRT7OX5m-Y3tTN9XNsb2FeYbxMbWvpt&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/aipowered-literature-review-csvs/kaggle/working/TIE/Proportion of patients who were asymptomatic.csv")

df.head()
df = df.rename(columns={'Unnamed: 0':'unnamed', 'Asymptomatic Proportion': 'asymptomatic'})
dfcorr=df.corr()

dfcorr

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=False,cmap='summer')

plt.show()
import shap

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import random
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
X = dfmodel.drop(['unnamed','asymptomatic'], axis = 1)

y = dfmodel['unnamed']
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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRAtz0ePW_Aj_MqIU9BNukzr5wHk3kIuE3u5oBDY-5GvmciKYRc&usqp=CAU',width=400,height=400)
df1 = pd.read_csv("../input/aipowered-literature-review-csvs/kaggle/working/TIE/Pediatric patients who were asymptomatic.csv")

df1.head()
df1 = df1.rename(columns={'Unnamed: 0':'unnamed1', 'Asymptomatic Proportion': 'asymptomatic1', 'Age': 'age'})
fig=sns.lmplot(x="asymptomatic1", y="unnamed1",data=df1)
df1_age = pd.DataFrame({

    'Date': df1.Date,

    'age': df1.age

})
fig = px.line(df1_age, x="Date", y="age", 

              title="Pediatric Asymptomatic Patients: Silent Spreaders ")

fig.show()
fig = px.bar(df1, 

             x='Date', y='age', color_discrete_sequence=['#21bf73'],

             title='Pediatric Asymptomatic Patients: Silent Spreaders', text='age')

fig.show()
fig = px.line(df1, 

             x='Date', y='asymptomatic1', color_discrete_sequence=['#ff2e63'],

             title='Pediatric Asymptomatic Patients: Silent Spreaders', text='asymptomatic1')

fig.show()
labels = df1['asymptomatic1'].value_counts().index

size = df1['asymptomatic1'].value_counts()

colors=['#ff2e63','#3F3FBF']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('Pediatric Asymptomatic Patients: Silent Spreaders', fontsize = 20)

plt.legend()

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRUPdulinsmJdYByKBIO9rXzgSCESo9n03ND5-PLIBL2y-N8Bgf&usqp=CAU',width=400,height=400)