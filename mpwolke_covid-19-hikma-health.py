#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)
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
df = pd.read_csv("../input/hikma-health-covid19-us-county-policies/county_policies.csv")

df.head()
df_test = pd.DataFrame({

    'testing_date': df.testing_date,

    'testing': df.testing

})
fig = px.line(df_test, x="testing_date", y="testing", 

              title="Testing Cases")

fig.show()
df_fips = pd.DataFrame({

    'work_date': df.work_date,

    'fips': df.fips

})
fig = px.line(df_fips, x="work_date", y="fips", 

              title="Work Fips")

fig.show()
fig = px.bar(df, 

             x='work_date', y='fips', color_discrete_sequence=['#D63230'],

             title='Work Fips', text='fips')

fig.show()
fig = px.line(df, 

             x='work_date', y='fips', color_discrete_sequence=['#D63230'],

             title='Work Fips (linear scale)', text='fips')

fig.show()
import networkx as nx

df1 = pd.DataFrame(df['fips']).groupby(['fips']).size().reset_index()



G = nx.from_pandas_edgelist(df1, 'fips', 'fips', [0])

colors = []

for node in G:

    if node in df["fips"].unique():

        colors.append("red")

    else:

        colors.append("lightgreen")

        

nx.draw(nx.from_pandas_edgelist(df1, 'fips', 'fips', [0]), with_labels=True, node_color=colors)
plt.style.use('dark_background')

df["fips"].plot.hist()

plt.show()
fig=sns.lmplot(x="fips", y="testing",data=df)
sns.countplot(df["shelter"])

plt.xticks(rotation=90)

plt.show()
plt.style.use('dark_background')

sns.jointplot(df['testing'],df['fips'],data=df,kind='scatter')
fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.violinplot(x='testing',y='fips',data=df)
plt.style.use('dark_background')

#sns.set(style="darkgrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

fig = sns.swarmplot(x="testing", y="fips", data=df)
plt.style.use('dark_background')

#sns.set(style="whitegrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

ax = sns.violinplot(x="testing", y="fips", data=df, inner=None)

ax = sns.swarmplot(x="testing", y="fips", data=df,color="white", edgecolor="black")
df.plot.area(y=['testing','fips','transport','work'],alpha=0.4,figsize=(12, 6));
df.corr()

plt.figure(figsize=(10,4))

sns.heatmap(df.corr(),annot=True,cmap='summer')

plt.show()
corr = df.corr(method='pearson')

sns.heatmap(corr)
cnt_srs = df['testing'].value_counts().head()

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

    title='Testing Distribution',

    )

data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename="testing")
labels = df['testing'].value_counts().index

size = df['testing'].value_counts()

colors=['#BFBF3F','#44BF3F']

plt.pie(size, labels = labels, colors = colors, shadow = True, autopct='%1.1f%%',startangle = 90)

plt.title('Quantity of testing', fontsize = 20)

plt.legend()

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
X = dfmodel.drop(['fips','testing'], axis = 1)

y = dfmodel['fips']
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

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcT55oBGC8KwsYhYIbYNblJIMhs4CxZGLToRxY-i-T7s36gU4Baa&usqp=CAU',width=400,height=400)