# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh
from bokeh.io import show, output_notebook
from bokeh.palettes import Spectral9
from bokeh.plotting import figure
output_notebook() # You can use output_file();

import plotly.graph_objects as go
import plotly.express as px

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected = True)
import plotly.offline as py
py.init_notebook_mode(connected = True)

# Special
import wordcloud, missingno
from wordcloud import WordCloud # wordcloud
import missingno as msno # check missing value
import networkx as nx
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')
data.info()
data.describe()
data.head(10)
data.columns
data = data.drop(['URL','ID'],axis = 1)
msno.matrix(data)
data.isnull().sum()
data.dtypes
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

fig,ax = plt.subplots(20,10, figsize = (12,12))

for i in range(200):
    r = requests.get(data['Icon URL'][i])
    image = Image.open(BytesIO(r.content))
    ax[i//10][i%10].imshow(image)
    ax[i//10][i%10].axis('off')
plt.show()

plt.figure(figsize = (12,12))
sns.barplot(x = data['Average User Rating'].value_counts().index,
           y=data['Average User Rating'].value_counts().values)
plt.xlabel('Average User Rating')
plt.ylabel('Frequency')
plt.title('Average User Rating Bar Plot')
plt.show()

plt.figure(figsize=(12,12))
sns.barplot(x=data['Price'].value_counts().index,
           y=data['Price'].value_counts().values)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Price Plot')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(14,28))
wordcloud = WordCloud(background_color='white',width=800, height=800).generate(' '.join(data['Name']))
wordcloud_sub = WordCloud(background_color='white',width=800, height=800).generate(' '.join(data['Subtitle'].dropna().astype(str)) )
ax[0].imshow(wordcloud)
ax[0].axis('off')
ax[0].set_title('Wordcloud(Name)')
ax[1].imshow(wordcloud_sub)
ax[1].axis('off')
ax[1].set_title('Wordcloud(Subtitle)')
plt.show()
data.corr()
f,ax= plt.subplots(figsize=(12,12))
sns.heatmap(data.corr(),annot= True,ax=ax)
plt.show()
data['Original Release Date'] = pd.to_datetime(data['Original Release Date'], format = '%d/%m/%Y')
date_aur = pd.DataFrame({'Average User Rating':data['Average User Rating']})
date_aur = date_aur.set_index(data['Original Release Date'])
date_aur = date_aur.sort_values(by=['Original Release Date'])
date_aur.head()

data.dtypes
plt.figure(figsize=(12,6))
sns.barplot(data['User Rating Count'][:20],data.Name[:20])
plt.title('Top 10 User Rates')
plt.show()
fig = go.Figure([go.Bar(x=data["Primary Genre"], y=data["Average User Rating"])])
fig.update_layout(title_text="Primary Genre")
py.iplot(fig, filename="test") 
fig = px.scatter(data, y="In-app Purchases", x="Size")
py.iplot(fig, filename="test")
data.drop(["Name","Icon URL","Subtitle","Icon URL","In-app Purchases","Description","Languages","Developer","Size","Genres","Original Release Date","Current Version Release Date","Original Release Date","Age Rating"],axis=1,inplace=True)
data.head()

data.columns=["average_user_rating","user_rating_count","price","genre"]
data.isnull().sum()
def nan_to_median(series):
    return series.fillna(series.median())
data['user_rating_count']=data['user_rating_count'].transform(nan_to_median)
data['average_user_rating']=data['average_user_rating'].transform(nan_to_median)
data['price']=data['price'].transform(nan_to_median)
data.isnull().sum()
data['user_rating_count'] = data['user_rating_count'].astype(int)
data['price'] = data['price'].astype(int)
data['average_user_rating'] = data['average_user_rating'].astype(int)
x = data.drop('average_user_rating',axis = 1)
y = data['average_user_rating']
x = pd.get_dummies(x, columns=['genre'],prefix = ['genre'])
x.head()
y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=42, stratify=y)
x.info()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV , StratifiedKFold
from sklearn.model_selection import cross_val_score

knn_model=KNeighborsClassifier().fit(X_train,y_train)
lr_model=LogisticRegression().fit(X_train,y_train)
rf_model=RandomForestClassifier().fit(X_train,y_train)
lgb_model=LGBMClassifier().fit(X_train,y_train)
xgb_model=XGBClassifier().fit(X_train,y_train)
gbm_model=GradientBoostingClassifier().fit(X_train,y_train)


modeller=[lr_model,rf_model,lgb_model,gbm_model,xgb_model,knn_model]

sc_fold=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

for model in modeller:
    isimler=model.__class__.__name__
    accuracy=cross_val_score(model,X_train,y_train,cv=sc_fold)
    print("{}s score:{}".format(isimler,accuracy.mean()))