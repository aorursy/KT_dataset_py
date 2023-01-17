#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib.animation as animation

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns

from plotly.subplots import make_subplots

%matplotlib inline

import missingno as msno



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/uncover/UNCOVER/esri_covid-19/esri_covid-19/cdcs-social-vulnerability-index-svi-2016-overall-svi-census-tract-level.csv', encoding='ISO-8859-2')

df.head()
df.plot.area(y=['st','stcnty','e_uninsur','objectid'],alpha=0.4,figsize=(12, 6));
plt.clf()

df.groupby('st').size().plot(kind='bar')

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQDwwP5BdGRGIgNy2W55et26fLmDVFvab88gA2MNCPJsn3oox1K&usqp=CAU',width=400,height=400)
#Let's visualise the evolution of results

ppe = df.groupby('st').sum()[['stcnty','objectid','e_daypop']]

#evolution['Expiration Rate'] = (evolution['Expired'] / evolution['Cumulative']) * 100

#evolution['Discharging Rate'] = (evolution['Discharged'] / evolution['Cumulative']) * 100

ppe.head()
plt.figure(figsize=(20,7))

plt.plot(ppe['objectid'], label='objectid')

plt.plot(ppe['stcnty'], label='stcnty')

plt.plot(ppe['e_daypop'], label='e_daypop')

plt.legend()

#plt.grid()

plt.title('Social Vulnerability Index ')

plt.xticks(ppe.index,rotation=45)

plt.xlabel('st')

plt.ylabel('Population')

plt.show()
plt.figure(figsize=(20,7))

plt.plot(ppe['stcnty'], label='St County')

plt.legend()

#plt.grid()

plt.title('Social Vulnerability Census')

plt.xticks(ppe.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
#This is another way of visualizing the evolution: plotting the increase evolution (difference from day to day)

diff_ppe = ppe.diff().iloc[1:]

plt.figure(figsize=(20,7))

plt.plot(diff_ppe['stcnty'], label='St County')

plt.legend()

plt.grid()

plt.title('Social Vulnerability Census')

plt.xticks(ppe.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
print('Statistics Social Vulnerability ')



diff_ppe.describe()
fig = go.Figure()

fig.add_trace(go.Scatter(x=df['st'], y=df['stcnty'],

                    mode='lines+markers',marker_color='blue'))

fig.update_layout(title_text = 'Social Vulnerability Census')

fig.update_layout(plot_bgcolor='rgb(275, 270, 273)',width=600, height=600)

fig.show()
df_grp = df.groupby(["st"])[["stcnty","objectid","e_daypop"]].sum().reset_index()

df_grp.head()
df_grp_r = df_grp.groupby("st")[["stcnty","objectid","e_daypop"]].sum().reset_index()
df_grp_rl20 = df_grp_r.tail(20)
fig = px.bar(df_grp_rl20[['st', 'stcnty']].sort_values('stcnty', ascending=False), 

             y="stcnty", x="st", color='st', 

             log_y=True, template='ggplot2', title='Social Vulnerability Census')

fig.show()
df_grp_d = df_grp.groupby("st")[["stcnty","objectid","e_daypop"]].sum().reset_index()
df_grp_dl20 = df_grp_d.tail(20)
f, ax = plt.subplots(figsize=(23,10))

ax=sns.scatterplot(x="st", y="stcnty", data=df_grp_dl20,

             color="black",label = "counties")

ax=sns.scatterplot(x="st", y="objectid", data=df_grp_dl20,

             color="red",label = "Object ID")

ax=sns.scatterplot(x="st", y="e_daypop", data=df_grp_dl20,

             color="blue",label = "Population")

plt.plot(df_grp_dl20.st,df_grp_dl20.stcnty,zorder=1,color="black")

plt.plot(df_grp_dl20.st,df_grp_dl20.objectid,zorder=1,color="red")

plt.plot(df_grp_dl20.st,df_grp_dl20.e_daypop,zorder=1,color="blue")
df1 = pd.read_csv('../input/uncover/UNCOVER/HDE_update/inform-covid-indicators.csv', encoding='ISO-8859-2')

df1.head()
corr = df1.corr(method='pearson')

sns.heatmap(corr)
#Let's visualise the evolution of vulnerability

vulnerability = df1.groupby('country').sum()[['inform_risk', 'population_density', 'inform_vulnerability', 'inform_epidemic_lack_of_coping_capacity', 'people_using_at_least_basic_sanitation_services']]

vulnerability.head()
plt.figure(figsize=(20,7))

plt.plot(vulnerability['inform_risk'], label='Risks')

plt.plot(vulnerability['population_density'], label='Population Density')

plt.plot(vulnerability['inform_vulnerability'], label='Inform Vulnerability')

plt.plot(vulnerability['inform_epidemic_lack_of_coping_capacity'], label='Inform Epidemic Lack of Coping Capacity')

plt.plot(vulnerability['people_using_at_least_basic_sanitation_services'], label='People Using at least Basic Sanitation Services')

plt.legend()

#plt.grid()

plt.title('Social Vulnerabilities Indicators')

plt.xticks(vulnerability.index,rotation=45)

plt.xlabel('Countries')

plt.ylabel('Vulnerabilities Indicators')

plt.show()
plt.figure(figsize=(20,7))

plt.plot(vulnerability['inform_risk'], label='Respiratory Diseases')

plt.legend()

plt.grid()

plt.title('')

plt.xticks(vulnerability.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
import shap

import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

import random
SEED = 99

random.seed(SEED)

np.random.seed(SEED)
dfmodel = df1.copy()



# read the "object" columns and use labelEncoder to transform to numeric

for col in dfmodel.columns[dfmodel.dtypes == 'object']:

    le = LabelEncoder()

    dfmodel[col] = dfmodel[col].astype(str)

    le.fit(dfmodel[col])

    dfmodel[col] = le.transform(dfmodel[col])
#change columns names to alphanumeric

dfmodel.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dfmodel.columns]
X = dfmodel.drop(['iso3','current_health_expenditure_per_capita'], axis = 1)

y = dfmodel['inform_risk']
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
cat = []

num = []

for col in df1.columns:

    if df1[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num 
plt.style.use('dark_background')

for col in df1[num].drop(['inform_risk'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df1[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('Social Vulnerability Factors ')

    plt.tight_layout()

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQ1IDyyWbqKXOX627hm1cIBSFSggg3RdhzxlQiBAQnSQvtGmkA1&usqp=CAU',width=400,height=400)