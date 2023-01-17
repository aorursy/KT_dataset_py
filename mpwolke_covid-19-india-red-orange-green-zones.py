#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
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

import seaborn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel('/kaggle/input/covid19-india-red-orange-and-green-zones/https___www.ndtv.com_india-new.xlsx')

df.head()
df = df.rename(columns={'Sr. No.':'SerReq'})
fig = px.bar(df[['State','SerReq']].sort_values('SerReq', ascending=False), 

                        y = "SerReq", x= "State", color='SerReq', template='ggplot2')

fig.update_xaxes(tickangle=45, tickfont=dict(family='Rockwell', color='crimson', size=14))

fig.update_layout(title_text="Covid-19 India Service Request Zones")



fig.show()
plt.figure(figsize=(20,10))

plt.bar(df.State, df.District,label="redzone")

plt.xlabel('redzone')

plt.ylabel("SerReq")

plt.legend(frameon=True, fontsize=25)

plt.title('Covid-19 India Redzones',fontsize=30)

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
# Grouping it by job title and country

plot_data = df.groupby(['SerReq', 'redzone'], as_index=False).State.sum()



fig = px.bar(plot_data, x='SerReq', y='State', color='redzone')

fig.show()
sns.countplot(df["State"])

plt.xticks(rotation=90)

plt.show()
fig = px.pie(df, values=df['SerReq'], names=df['redzone'],

             title='Covid-19 Indian ',

            )

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
fig = px.line(df, x="SerReq", y="State", 

              title="Covid-19 Indian States Zones")

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
X = dfmodel.drop(['SerReq','State'], axis = 1)

y = dfmodel['SerReq']
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
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('State').size()/df['District'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])

fig.show()
import plotly.express as px

fig = px.bar(df.sort_values('SerReq', ascending=False).sort_values('SerReq', ascending=True), 

             x="SerReq", y="State", title='Covid-19 Indian State Zones', text='SerReq', orientation='h',width=1000, height=700, range_x = [0, max(df['SerReq'])]) 

            

fig.update_traces(marker_color='#46cdcf', opacity=0.8, textposition='inside')



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
%matplotlib inline

sns.set_style("white")

sns.set_context({"figure.figsize": (24, 10)})





sns.barplot(x = df['SerReq'], y = df['State'], color = "red")





bottom_plot = sns.barplot(x = df['SerReq'], y = df['State'], color = "#0000A3", )





topbar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')

bottombar = plt.Rectangle((0,0),1,1,fc='#0000A3',  edgecolor = 'none')

l = plt.legend([bottombar, topbar], ['redzone', 'SerReq'], loc=1, ncol = 2, prop={'size':16})

l.draw_frame(False)





sns.despine(left=True)

bottom_plot.set_ylabel("States")

bottom_plot.set_xlabel("Service Request Zones")





for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +

             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):

    item.set_fontsize(16)
f, ax = plt.subplots(figsize=(15,10))



data = df[['greenzone','SerReq']]

data.sort_values(['SerReq'])

sns.set_color_codes("muted")

sns.barplot(x="SerReq", y="greenzone",data=data, label="SerReq", color="g")
fig = px.bar(df, x="SerReq", y="District", color='SerReq', orientation='h', height=800,

             title='SerReq', color_discrete_sequence = px.colors.cyclical.mygbm)



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
fig = px.scatter(df, x="SerReq", y="District", color="State", marginal_y="rug", marginal_x="histogram")

fig
fig = px.scatter(df, x="SerReq", y="redzone", color="District", marginal_y="rug", marginal_x="histogram")

fig
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR5P_2ImOD2cHocHMxljZv50tnlhzza0AbJtCLleT1RY3ry5Bgz&usqp=CAU',width=400,height=400)
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTwmz2ok5bEhuPy_Qh8RJcWWRyQKoqWv1zUD8-MtVLyWWytWfFF&usqp=CAU',width=400,height=400)