import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

sns.set()

pd.set_option('display.expand_frame_repr',False)

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

 



#regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#preprocessing

from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder



#evaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
df.columns
df.head(10)
country_point = df['points'].value_counts()

place = df['country'].value_counts()

label = place.index

size = country_point.values



colors = ['green','blue','yellow','purple']

trace = go.Pie(labels=label,

               hoverinfo='label+value', textinfo='percent', 

               values=size,

               marker=dict(colors=colors,line=dict(color='#fff', width=.4)))









layout = go.Layout(

    title='Points')



data = [trace]

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
df=df.drop(columns=['Unnamed: 0', 'description'])

df=df.reset_index(drop=True)
df['price'].min()
df['points'].max()
df['country'].value_counts().plot(kind = 'bar',figsize=(13,8),fontsize=13)

plt.show()
df.boxplot(column='price',by='points')

plt.show()
v_map = pd.DataFrame(df['country'].value_counts())

v_map['Country'] = v_map.index

v_map.columns = ['Points','Country']

v_map.reset_index().drop('index',axis = 1)



data = [ dict(

        type = 'choropleth',

        locations = v_map['Country'],

        locationmode = 'country names',

        z = v_map['Points'],

        text = v_map['Country'],

        colorscale = 'Reds',

        marker = dict(

            line = dict (

                color = 'Black',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '$',

            title = 'Rate'))]



layout = dict(

    #title = 'Country',

    geo = dict(

        showframe = False,

        showcoastlines = True,

        projection = dict(

            type = 'Mercator')))



fig = dict(data=data, layout=layout )

py.iplot(fig, validate=False)
df.head(10)
df.info()
df.describe()
fig, ax = plt.subplots(figsize=(20,10))

sns.boxplot(x='points', y='price', data=df)

plt.xticks(fontsize=10) # X Ticks

plt.yticks(fontsize=10) # Y Ticks

ax.set_title('Description Length per Points', fontweight="bold", size=20) # Title

ax.set_ylabel('Price', fontsize = 25) # Y label

ax.set_xlabel('Points', fontsize = 25) # X label

plt.show()
fig, ax = plt.subplots(figsize=(20,10)) 

sns.violinplot(

    x='points',

    y='price',

    data=df

)
df['price'].describe() 

df['price'].dtypes
df['points'].astype(float)
df['price'].values
df['points'].values
df['points'].describe()
y_prices = df['price']

N_price = (y_prices - np.min(y_prices))/(np.max(y_prices)-np.min(y_prices))
x_points = df['points']

N_points = (x_points - np.min(x_points))/(np.max(x_points)-np.min(x_points))
type(X_points)

print(X_points.shape)
fig, ax = plt.subplots(figsize=(20,10)) 

plt.scatter(X_points,Y_price)
X = df['price']

X=X.fillna(df['price'].mean())

X = X.as_matrix()
X = X.reshape(-1,1)
X.shape
y= df['points']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
clf =  LogisticRegression()

clf.fit(X_train,y_train)

pred = clf.predict(X_test)

result = {'Model Regression\'s accuracy {}'.format(accuracy_score(pred,y_test))}
result