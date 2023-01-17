#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR6RPARAkN9rDfhrMjLsYO3V6p2TMI6dTnNayaKCraEM8ux7Uwl&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

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
df=pd.read_csv("../input/covid19-geography/mmsa-icu-beds.csv")

df.head()
fig = px.bar(df,

             y='MMSA',

             x='icu_beds',

             orientation='h',

             color='high_risk_per_ICU_bed',

             title='Covid-19 MMSA',

             opacity=0.8,

             color_discrete_sequence=px.colors.diverging.Armyrose,

             template='plotly_dark'

            )

fig.update_xaxes(range=[0,35])

fig.show()
corr=df[df.columns.sort_values()].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



fig = go.Figure(data=go.Heatmap(z=corr.mask(mask),

                                x=corr.columns.values,

                                y=corr.columns.values,

                                xgap=1, ygap=1,

                                colorscale="Rainbow",

                                colorbar_thickness=20,

                                colorbar_ticklen=3,

                                zmid=0),

                layout = go.Layout(title_text='Correlation Matrix', template='plotly_dark',              

                

                height=500,                            

                xaxis_showgrid=False,

                yaxis_showgrid=False,

                yaxis_autorange='reversed'))

fig.show()
fig = px.bar_polar(df, r="MMSA", theta="icu_beds", color="high_risk_per_ICU_bed", template="plotly_dark",

            color_discrete_sequence= px.colors.sequential.Plasma_r)

fig.show()
fig = px.bar(df, x= "MMSA", y= "icu_beds", color_discrete_sequence=['crimson'],)

fig.show()
fig = px.line(df, x="MMSA", y="icu_beds", color_discrete_sequence=['green'], 

              title="Covid-19 MMSA")

fig.show()
fig = px.scatter(df, x="MMSA", y="icu_beds", color="high_risk_per_ICU_bed",

                 color_continuous_scale=["red", "green", "blue"])



fig.show()
fig = px.parallel_coordinates(df, color="icu_beds",

                             color_continuous_scale=[(0.00, "red"),   (0.33, "red"),

                                                     (0.33, "green"), (0.66, "green"),

                                                     (0.66, "blue"),  (1.00, "blue")])

fig.show()
fig = px.pie(df, values=df['icu_beds'], names=df['MMSA'],

             title='Covid-19 MMSA',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
import itertools

columns=df.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in itertools.zip_longest(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    df[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
import shap

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import random
df.isnull().sum()
df['MMSA'] = df['MMSA'].replace(['negative','positive'], [0,1])
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
X = dfmodel.drop(['MMSA','icu_beds'], axis = 1)

y = dfmodel['MMSA']
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

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTzTCvvSDS4CJNkhjM5PCHwZ_U3seL7NzwWEhWlLTImqbl0b7mu&usqp=CAU',width=400,height=400)