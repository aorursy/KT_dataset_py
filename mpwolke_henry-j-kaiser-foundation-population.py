#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTNd6YekNRShMNVCUlJxt0bnoKnTfxKaymo_ekXwWU2zogCB_xF&usqp=CAU',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

import plotly.graph_objs as go

import plotly.offline as py



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/us-population-breakdown/edited US popluation breakdown.xlsx')

df.head()
#Let's visualise the evolution of results

uspop = df.groupby('CTYNAME').sum()[['CENSUS2010POP','POPESTIMATE2019']]

#evolution['Expiration Rate'] = (evolution['Expired'] / evolution['Cumulative']) * 100

#evolution['Discharging Rate'] = (evolution['Discharged'] / evolution['Cumulative']) * 100

uspop.head()
plt.figure(figsize=(20,7))

plt.plot(uspop['POPESTIMATE2019'], label='POPESTIMATE2019')

plt.plot(uspop['CENSUS2010POP'], label='CENSUS2010POP')

#plt.plot(ppe['e_daypop'], label='e_daypop')

plt.legend()

plt.grid()

plt.title('US Census 2010 ')

plt.xticks(uspop.index,rotation=45)

plt.xlabel('POPESTIMATE2019')

plt.ylabel('CENSUS2010POP')

plt.show()
import plotly.express as px

fig = px.scatter(df, x= "POPESTIMATE2019", y= "CENSUS2010POP")

fig.show()
plt.figure(figsize=(20,7))

plt.plot(uspop['CENSUS2010POP'], label='POPESTIMATE2019')

plt.legend()

#plt.grid()

plt.title('US Census 2010')

plt.xticks(uspop.index,rotation=45)

plt.ylabel('Rate %')

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.STNAME)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
df1 = pd.read_excel('/kaggle/input/us-population-breakdown/raw_data.xlsx')

df1.head()
fig,axes = plt.subplots(1,1,figsize=(20,5))

sns.heatmap(df1.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
df1 = df1.rename(columns={'Unnamed: 1':'unnamed1', 'Unnamed: 2': 'unnamed2', 'Unnamed: 3': 'unnamed3', 'Unnamed: 5': 'unnamed5', 'Unnamed: 7': 'unnamed7', 'Unnamed: 8': 'unnamed8'})
#Codes Andre Sionek

import plotly.express as px

plot_data = df1.groupby(['unnamed1', 'unnamed2'], as_index=False).unnamed3.sum()



fig = px.bar(plot_data, x='unnamed1', y='unnamed3', color='unnamed2')

fig.show()
import plotly.express as px



plot_data = df1.groupby(['unnamed1'], as_index=False).unnamed2.sum()



fig = px.line(plot_data, x='unnamed1', y='unnamed2')

fig.show()
import plotly.express as px



#Codes Andre sionek  

plot_data = df1.groupby(['unnamed2', 'unnamed3'], as_index=False).unnamed8.sum()



fig = px.line(plot_data, x='unnamed2', y='unnamed8', color='unnamed3')

fig.show()
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
X = dfmodel.drop(['unnamed1','unnamed8'], axis = 1)

y = dfmodel['unnamed1']
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
cnt_srs = df1['unnamed1'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Purples',

        reversescale = True

    ),

)



layout = dict(

    title='White Population Distribution - The Henry J. Kaiser Family Foundation',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="unnamed1")
cnt_srs = df1['unnamed2'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Greens',

        reversescale = True

    ),

)



layout = dict(

    title='Black Population Distribution - The Henry J. Kaiser Family Foundation',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="unnamed2")
cnt_srs = df1['unnamed3'].value_counts().head()

trace = go.Bar(

    y=cnt_srs.index[::-1],

    x=cnt_srs.values[::-1],

    orientation = 'h',

    marker=dict(

        color=cnt_srs.values[::-1],

        colorscale = 'Oranges',

        reversescale = True

    ),

)



layout = dict(

    title='Hispanic Population Distribution - The Henry J. Kaiser Family Foundation',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="unnamed3")
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSkqkmGjAv9TcTbCqPuMcCX7nTP-91K7l7fymrL3Xo6V2QICnzy&usqp=CAU',width=400,height=400)