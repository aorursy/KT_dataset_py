# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

import plotly.graph_objects as go

import plotly.offline as py

plt.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadscampaigncsv/campaign.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'campaign.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.head()
df.isnull().sum()
import plotly.offline as pyo

import plotly.graph_objs as go

lowerdf = df.groupby('campaign_vaccine').size()/df['target_age_group'].count()*100

labels = lowerdf.index

values = lowerdf.values



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values,marker_colors = px.colors.sequential.speed, hole=.6)])

fig.show()
# Count Plot

plt.style.use("classic")

plt.figure(figsize=(10, 8))

sns.countplot(df['target_age_group'], palette='Accent_r')

plt.xlabel("Target Age Group")

plt.ylabel("Count")

plt.title("Target Age Group")

plt.xticks(rotation=45, fontsize=8)

plt.show()
# categorical features

categorical_feat = [feature for feature in df.columns if df[feature].dtypes=='O']

print('Total categorical features: ', len(categorical_feat))

print('\n',categorical_feat)
index_str = ['iso3', 'country', 'campaign_vaccine', 'planned_start_date', 'planned_end_date', 'status', 'target_age_group', 'est_target_pop', 'campaign_type', 'geographic_scale']



plt.figure(figsize=[40,20])

i = 1

for col in index_str :

    plt.subplot(4,10,i)

    sns.scatterplot(x=col, y = 'campaign_vaccine' ,data= df)

    sns.despine()

    i = i+1

plt.tight_layout()

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
X = dfmodel.drop(['campaign_vaccine','est_target_pop'], axis = 1)

y = dfmodel['campaign_vaccine']
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
fig = go.Figure(data=[go.Scatter(

    x=df['campaign_vaccine'][0:10],

    y=df['est_target_pop'][0:10],

    mode='markers',

    marker=dict(

        color=[145, 140, 135, 130, 125, 120,115,110,105,100],

        size=[100, 90, 70, 60, 60, 60,50,50,40,35],

        showscale=True

        )

)])

fig.update_layout(

    title='Vaccine Campaigns',

    xaxis_title="campaign_vaccine",

    yaxis_title="est_target_pop",

)

fig.show()
fig = go.Figure(data=[go.Bar(

            x=df['campaign_vaccine'][0:10], y=df['est_target_pop'][0:10],

            text=df['est_target_pop'][0:10],

            textposition='auto',

            marker_color='black'



        )])

fig.update_layout(

    title='Vaccine Campaigns',

    xaxis_title="campaign_vaccine",

    yaxis_title="est_target_pop",

)

fig.show()
cnt_srs = df['campaign_vaccine'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Blues',

        reversescale = True

    ),

)



layout = dict(

    title='Vaccine Campaigns',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="campaign_vaccine")
fig = px.bar(df, 

             x='planned_start_date', y='campaign_vaccine', color_discrete_sequence=['#27F1E7'],

             title='Vaccine Campaigns', text='campaign_type')

fig.show()
fig = px.bar(df, 

             x='planned_end_date', y='campaign_vaccine', color_discrete_sequence=['crimson'],

             title='Vaccine Campaigns', text='geographic_scale')

fig.show()
fig = px.bar(df, 

             x='status', y='campaign_vaccine', color_discrete_sequence=['magenta'],

             title='Vaccine Campaigns', text='campaign_type')

fig.show()
fig = px.line(df, x="planned_start_date", y="campaign_vaccine", color_discrete_sequence=['darkseagreen'], 

              title="Vaccine Campaigns")

fig.show()
fig = px.line(df, x="planned_end_date", y="campaign_vaccine", color_discrete_sequence=['darksalmon'], 

              title="Vaccine Campaigns")

fig.show()
import plotly.express as px



fig = px.scatter(df, x="planned_start_date", y="campaign_vaccine", color="campaign_type")

fig.show()
import plotly.express as px



fig = px.scatter(df, x="target_age_group", y="campaign_vaccine", color="status")

fig.show()
plt.figure(figsize=(20,4))

plt.subplot(131)

sns.countplot(x= 'campaign_vaccine', data = df, palette="cool",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(132)

sns.countplot(x= 'campaign_type', data = df, palette="ocean",edgecolor="black")

plt.xticks(rotation=45)

plt.subplot(133)

sns.countplot(x= 'status', data = df, palette="Greens_r",edgecolor="black")

plt.xticks(rotation=45)

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.status)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()