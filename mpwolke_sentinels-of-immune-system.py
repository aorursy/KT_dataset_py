# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

import plotly.graph_objects as go

import plotly.offline as py

import plotly.express as px

from plotly.offline import iplot



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/ai4all-project/results/deconvolution/CIBERSORTx_Results_Krasnow_facs_droplet.csv')

df.head()
import plotly.express as px



# Grouping it by job title and country

plot_data = df.groupby(['viral_load', 'B cell'], as_index=False).Neutrophil.sum()



fig = px.bar(plot_data, x='viral_load', y='Neutrophil', color='B cell')

fig.show()
from plotly.subplots import make_subplots





fig= make_subplots(rows= 2,cols=2, 

                    specs=[[{'secondary_y': True},{'secondary_y': True}],[{'secondary_y': True},{'secondary_y': True}]],

                    subplot_titles=("Basophil/Mast","Dendritic","T cell","Goblet")

                   )

fig.add_trace(go.Bar(x=df['viral_load'],y=df['Basophil/Mast'],

                    marker=dict(color=df['Basophil/Mast'],coloraxis='coloraxis')),1,1)



fig.add_trace(go.Bar(x=df['viral_load'],y=df['Dendritic'],

                    marker=dict(color=df['Dendritic'],coloraxis='coloraxis1')),1,2)



fig.add_trace(go.Bar(x=df['viral_load'],y=df['T cell'],

                    marker=dict(color=df['T cell'],coloraxis='coloraxis2')),2,1)



fig.add_trace(go.Bar(x=df['viral_load'],y=df['Goblet'],

                    marker=dict(color=df['Goblet'],coloraxis='coloraxis3')),2,2)

fig= make_subplots(rows= 2,cols=2, 

                    specs=[[{'secondary_y': True},{'secondary_y': True}],[{'secondary_y': True},{'secondary_y': True}]],

                    subplot_titles=("Basal","Ciliated","Ionocyte","Monocytes/macrophages")

                   )

fig.add_trace(go.Bar(x=df['viral_load'],y=df['Basal'],

                    marker=dict(color=df['Basal'],coloraxis='coloraxis')),1,1)



fig.add_trace(go.Bar(x=df['viral_load'],y=df['Ciliated'],

                    marker=dict(color=df['Ciliated'],coloraxis='coloraxis')),1,2)



fig.add_trace(go.Bar(x=df['viral_load'],y=df['Ionocyte'],

                    marker=dict(color=df['Ionocyte'],coloraxis='coloraxis')),2,1)



fig.add_trace(go.Bar(x=df['viral_load'],y=df['Monocytes/macrophages'],

                    marker=dict(color=df['Monocytes/macrophages'],coloraxis='coloraxis')),2,2)
plt.rcParams['figure.figsize'] = (20.0, 10.0)

plt.rcParams['font.family'] = "serif"

fig, ax =plt.subplots(3,2)

sns.countplot(df['B cell'], ax=ax[0,0])

sns.countplot(df['viral_load'], ax=ax[0,1])

sns.countplot(df['Club'], ax=ax[1,0])

sns.countplot(df['Basal'], ax=ax[1,1])

sns.countplot(df['czb_id'], ax=ax[2,0])

sns.countplot(df['Monocytes/macrophages'], ax=ax[2,1])

fig.show();
import plotly.express as px



fig = px.histogram(df, x="viral_load", y="B cell", color = 'czb_id',

                   marginal="rug", # or violin, rug,

                   hover_data=df.columns,

                   color_discrete_sequence=['indianred','lightblue'],

                   )



fig.update_layout(

    title="Covid-19 Viral Load",

    xaxis_title="Viral Load",

    yaxis_title="B cells",

)

fig.update_yaxes(tickangle=-30, tickfont=dict(size=7.5))



fig.show();
fig = px.scatter_3d(df,z="B cell",x="viral_load",y="T cell",

    color = 'czb_id', size_max = 18,

    #color_discrete_sequence=['indianred','lightblue'] 

                    symbol='czb_id', opacity=0.7

    )

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig = px.parallel_categories(df, color="B cell", color_continuous_scale=px.colors.sequential.OrRd)

fig.show()
import plotly.figure_factory as ff

fig = make_subplots(rows=1, cols=5)

df_num = df[['B cell', 'T cell', 'Neutrophil', 'Club', 'Monocytes/macrophages']]



fig1 = ff.create_distplot([df_num['B cell']], ['B cell'])

fig2 = ff.create_distplot([df_num['T cell']], ['T cell'])

fig3 =  ff.create_distplot([df_num['Neutrophil']], ['Neutrophil'])

fig4 =  ff.create_distplot([df_num['Club']], ['Club'])

fig5 =  ff.create_distplot([df_num['Monocytes/macrophages']], ['Monocytes/macrophages'])



fig.add_trace(go.Histogram(fig1['data'][0], marker_color='blue'), row=1, col=1)

fig.add_trace(go.Histogram(fig2['data'][0],marker_color='red'), row=1, col=2)

fig.add_trace(go.Histogram(fig3['data'][0], marker_color='green'), row=1, col=3)

fig.add_trace(go.Histogram(fig4['data'][0],marker_color='yellow'), row=1, col=4)

fig.add_trace(go.Histogram(fig5['data'][0],marker_color='purple'), row=1, col=5)





fig.show()
fig = px.line(df, x="viral_load", y="czb_id", color_discrete_sequence=['darksalmon'], 

              title="Covid-19 Viral Load")

fig.show()
import shap

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import random
df.isnull().sum()
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
X = dfmodel.drop(['viral_load','czb_id'], axis = 1)

y = dfmodel['viral_load']
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
dfcorr=df.corr()

dfcorr
plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=False,cmap='summer')

plt.show()