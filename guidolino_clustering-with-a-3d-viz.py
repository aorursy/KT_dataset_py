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

import math

import seaborn as sns

import plotly.express as px

import matplotlib

import time

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use('seaborn-darkgrid')

import statsmodels.api as sm

matplotlib.rcParams['axes.labelsize'] = 20

matplotlib.rcParams['xtick.labelsize'] = 12

matplotlib.rcParams['ytick.labelsize'] = 12

matplotlib.rcParams['text.color'] = 'k'

import altair as alt
customer = pd.read_csv("/kaggle/input/mall-customers/Mall_Customers.csv")
customer.head()
customer.rename(columns={"Annual Income (k$)":"AnnualIncome","Spending Score (1-100)":"SScore"}, inplace=True)
source = customer



alt.Chart(source).mark_bar().encode(

    alt.X("Age:Q", bin=True),

    y='count()',

)
source = customer



alt.Chart(source).mark_bar().encode(

    alt.X("AnnualIncome:Q", bin=True),

    y='count()',

)
source = customer



alt.Chart(source).mark_bar().encode(

    alt.X("SScore:Q", bin=True),

    y='count()',

)
customer.Genre.value_counts()
model = customer.set_index("CustomerID")
model.info()
model
sns.pairplot(model, hue = 'Genre', diag_kind = 'kde',

             plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},

             size = 4)
import plotly.express as px



fig = px.scatter_3d(model, x='AnnualIncome', y='SScore', z='Age',

              color='Genre')

fig.show()
model.groupby(['Genre']).agg({"AnnualIncome": ['mean','median','std'],

                                    "SScore": ['mean','median','std']}).reset_index()
model_df = model.drop(columns=['Genre']).copy()
fig = px.scatter_3d(model, x='AnnualIncome', y='SScore', z='Age')

fig.show()
from sklearn.cluster import KMeans
def kmeans(numero_de_cluster,generos):

    modelo = KMeans(n_clusters=numero_de_cluster)

    modelo.fit(generos)

    return [numero_de_cluster,modelo.inertia_]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

variaveis_escaladas = scaler.fit_transform(model_df)
resultado = [kmeans(numero_de_grupos,variaveis_escaladas) for numero_de_grupos in range(1,41)]
resultado = pd.DataFrame(resultado,columns=['grupos','inertia'])
fig = px.line(resultado, x="grupos",y='inertia')

fig.show()
model_df
model_escalado = model_df.copy()

model_escalado[['Age','AnnualIncome','SScore']] = scaler.fit_transform(model_escalado[['Age','AnnualIncome','SScore']])
km = KMeans(n_clusters=11)

y_predicted = km.fit_predict(model_escalado[['Age','AnnualIncome','SScore']])

# clustering

model_df['cluster'] = y_predicted
fig = px.scatter_3d(model_df, x='AnnualIncome', y='SScore', z='Age',

              color='cluster')

fig.show()
km = KMeans(n_clusters=6)

y_predicted = km.fit_predict(model_escalado[['Age','AnnualIncome','SScore']])

# clustering

model_df['cluster'] = y_predicted
fig = px.scatter_3d(model_df, x='AnnualIncome', y='SScore', z='Age',

              color='cluster')

fig.show()
model_df.groupby(['cluster']).agg({"Age":["mean","median"],

                                   "AnnualIncome":["mean","median"],

                                   "SScore":["mean","median"]}).reset_index()
def bplot (df,variavel):

    plt.figure(figsize=(10,8))

    sns.boxplot(data=df,x="cluster",y=variavel)
bplot(model_df,"Age")
bplot(model_df,"AnnualIncome")
bplot(model_df,"SScore")