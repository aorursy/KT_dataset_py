# Importing the libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import plotly.graph_objects as go
# Importing the dataset

dataset = pd.read_csv('../input/fifa19/data.csv')
dataset.info()
# Viewing the top 10 rows of the dataset

dataset.head(10)
# Viewing statistics of all numerical columns in the dataset

dataset.describe()
# Fetching the maximum overall rating for all the given countries

df = dataset.groupby(['Nationality'], as_index=False)['Overall'].max()
df
data = dict(type = 'choropleth',

            locations = df['Nationality'],

            locationmode = 'country names',

            colorscale = 'Viridis',

            autocolorscale=False,

            z=df['Overall'],

            text = df["Nationality"],

            colorbar = {'title':'Overall rating'})
layout = dict(geo = {'scope':'north america'},title="Maximum overall rating for countries in North America")
import plotly.graph_objs as gobj

col_map = gobj.Figure(data = [data],layout = layout)
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

iplot(col_map)
layout = dict(geo = {'scope':'south america'},title="Maximum overall rating for countries in South America")

col_map = gobj.Figure(data = [data],layout = layout)

init_notebook_mode(connected=True)

iplot(col_map)
layout = dict(geo = {'scope':'europe'},title="Maximum overall rating for countries in Europe")

col_map = gobj.Figure(data = [data],layout = layout)

init_notebook_mode(connected=True)

iplot(col_map)
layout = dict(geo = {'scope':'asia'},title="Maximum overall rating for countries in Asia")

col_map = gobj.Figure(data = [data],layout = layout)

init_notebook_mode(connected=True)

iplot(col_map)
layout = dict(geo = {'scope':'africa'},title="Maximum overall rating for countries in Africa")

col_map = gobj.Figure(data = [data],layout = layout)

init_notebook_mode(connected=True)

iplot(col_map)
layout = dict(geo = {'scope':'world'},title="Maximum overall rating country-wise across the world")

col_map = gobj.Figure(data = [data],layout = layout)

init_notebook_mode(connected=True)

iplot(col_map)
#Visualisation - Age vs Overall

import plotly.express as px

fig = px.scatter(dataset, x='Age', y='Overall',title = "Age vs Overall Rating correlation")

fig.update_traces(marker=dict(size=8,line=dict(width=1.5,color='blue')),selector=dict(mode='markers'))

fig.update_xaxes(title_text='Age of FIFA players')

fig.update_yaxes(title_text='Overall rating of players')

fig.show()
#Visualisation - Age vs Potential

fig = px.scatter(dataset, x='Age', y='Potential',title = "Age vs Potential correlation")

fig.update_traces( marker=dict(color='lawngreen',line=dict(width=1.5,color='black'),size=8))              

fig.update_xaxes(title_text='Age of FIFA players')

fig.update_yaxes(title_text='Potential of players')

fig.show()
df1 = dataset.filter(['Preferred Foot'], axis=1)
df1 = df1.dropna()
df1 = df1.groupby(['Preferred Foot']).size().reset_index(name='count')
df1
#Visualisation - Preferred Foot of FIFA Players

fig = px.pie(df1, values='count',names = 'Preferred Foot',title='Preferred Foot of FIFA Players')

fig.show()
df2 = dataset.filter(['Position'], axis=1)
df2 = df2.dropna()
df2 = df2.groupby(['Position']).size().reset_index(name='count')
df2
# Visualising the positions played by different FIFA players

df2.loc[df2['count'] < 300, 'Position'] = 'Other positions'

fig = px.pie(df2, values='count',names = 'Position',title='Positions of FIFA Players')

fig.show()
# Visualising - Age vs shot power correlation

fig = px.box(dataset, x="ShotPower", y="Age",title = 'Age vs Shot correlation')

fig.update_traces( marker=dict(color='crimson'))

fig.show()
df3 = dataset.filter(['Age','Wage','Potential'], axis=1)
df3
df3['Wage'] = df3['Wage'].str.replace('€', '').str.replace('K','').astype(int)
df3.isnull().values.any()
df3
# Visualisation- Age vs wage correlation

fig = px.scatter(df3, x='Age', y='Wage',title = "Age vs Wage correlation")

fig.update_traces(marker=dict(size=8,color = '#EB89B5',line=dict(width=1.5,color='fuchsia')),selector=dict(mode='markers'))

fig.update_xaxes(title_text='Age of FIFA players')

fig.update_yaxes(title_text='Wage of players in Thousand Euros')

fig.show()
# Visualsing - Potential vs wage correlation

fig = px.scatter(df3, x='Potential', y='Wage',title = "Potential vs Wage correlation")

fig.update_traces(marker=dict(size=8,color = 'orange',line=dict(width=1.5,color='red')),selector=dict(mode='markers'))

fig.update_xaxes(title_text='Potential of FIFA players')

fig.update_yaxes(title_text='Wage of players in Thousand Euros')

fig.show()
# Total number of distinct clubs in the dataset

len(dataset['Club'].unique())
df = dataset.groupby(['Club'], as_index=False)['Overall'].mean()

df['Overall'] = round(df['Overall'],2)

top25Clubs = df.sort_values('Overall',ascending=False).head(25)
# Visualising top 25 clubs based on the average overall rating of the players

fig = px.bar(top25Clubs, x='Club', y='Overall', title="Top 25 Clubs based on the average Overall rating of the players",\

            color_discrete_sequence=['teal']) 

fig.update_yaxes(title_text='Average Overall rating')

fig.update_xaxes(title_text='Clubs')

fig.show()
temp = dataset[dataset['Value'].str.contains('M')]

temp = temp.filter(['Value','Club','Nationality'], axis=1)

temp
temp['Value'] = temp['Value'].str.replace('€', '').str.replace('M','').astype(float)

temp
# Visualising the richest clubs in the dataset

temp.loc[temp['Value'] < 50.0, 'Club'] = 'Other clubs'

result = temp.groupby('Club').sum().sort_values('Value',ascending=False)

vls = result['Value'].values.tolist()

totalCount = []

lbl = []

for i in range(0,len(vls)):

    totalCount.append(vls[i])

    total = sum(totalCount)

for j in range(0,len(vls)):

    percentage=float((vls[j]/total)*100)

    lbl.append(str(result.index.values[j] +" "+str(round(percentage,2))+"%"))

labels = lbl

layout = dict(title="Value (in million €) of various Clubs")

fig = go.Figure(data=[go.Pie(labels=labels,values=result.Value)],layout=layout)

fig.update_traces(hoverinfo='value+label',textinfo='none')

fig.show()
# Visualising- Aggregate national value in million Euros

temp.loc[temp['Value'] < 50.0, 'Nationality'] = 'Other nations'

result = temp.groupby('Nationality').sum().sort_values('Value',ascending=False)

vls = result['Value'].values.tolist()

totalCount = []

lbl = []

for i in range(0,len(vls)):

    totalCount.append(vls[i])

    total = sum(totalCount)

for j in range(0,len(vls)):

    percentage=float((vls[j]/total)*100)

    lbl.append(str(result.index.values[j] +" "+str(round(percentage,2))+"%"))

labels = lbl

layout = dict(title="Aggregate national value (in million €)")

fig = go.Figure(data=[go.Pie(labels=labels,values=result.Value)],layout=layout)

fig.update_traces(hoverinfo='value+label',textinfo='none')

fig.show()
#Dropping unnecessary columns for creating a heatmap

hm = dataset.drop(['Unnamed: 0','ID','Jersey Number','Contract Valid Until','GKDiving',

                  'GKHandling','GKKicking','GKPositioning','GKReflexes'],axis=1)

## Determining the relevancy of features using heatmap in calculating the outcome variable

corrmat = hm.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(28,28))

#Plotting heat map

g=sns.heatmap(hm[top_corr_features].corr(),annot=True,linewidths=.5)

b, t = plt.ylim() 

b += 0.5

t -= 0.5 

plt.ylim(b, t)

plt.title('Correlation Matrix',fontdict = {'fontsize' : 20})

plt.show() 
# Dropping all null values from the feature columns

dataset = dataset.dropna(axis=0, subset=['Reactions','Composure','International Reputation','ShortPassing'

            ,'Vision','LongPassing','BallControl'])
# Splitting the dataset into features and outcome variable(Overall)

y = dataset.iloc[:,7].values

# Picking up the top ten most useful features based on the above heatmap 

x = dataset[['Age','Special','Reactions','Composure','Potential','International Reputation','ShortPassing'

            ,'Vision','LongPassing','BallControl']].values

print(x)
# Splitting the dataset into training and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
## Training the multiple linear regression on the training set

from sklearn.linear_model import LinearRegression

regressor_MultiLinear = LinearRegression()

regressor_MultiLinear.fit(x_train,y_train)
## Predicting test results

y_pred = regressor_MultiLinear.predict(x_test)
# Calculating r2 score

from sklearn.metrics import r2_score

r2_MultiLinear = r2_score(y_test,y_pred)

print(r2_MultiLinear)
## Finding out the optimal degree of polynomial regression

from sklearn.preprocessing import PolynomialFeatures

sns.set_style('darkgrid')

scores_list = []

pRange = range(2,6)

for i in pRange :

    poly_reg = PolynomialFeatures(degree=i)

    x_poly = poly_reg.fit_transform(x_train)

    poly_regressor = LinearRegression()

    poly_regressor.fit(x_poly,y_train)

    y_pred = poly_regressor.predict(poly_reg.fit_transform(x_test))

    scores_list.append(r2_score(y_test,y_pred))

plt.plot(pRange,scores_list,linewidth=2)

plt.xlabel('Degree of polynomial')

plt.ylabel('r2 score with varying degrees')

plt.show()
## Training the polynomial regression on the training model

poly_reg = PolynomialFeatures(degree=3)

x_poly = poly_reg.fit_transform(x_train)

poly_regressor = LinearRegression()

poly_regressor.fit(x_poly,y_train)

y_pred = poly_regressor.predict(poly_reg.fit_transform(x_test))

r2_poly = r2_score(y_test,y_pred)

print(r2_poly)
## Finding the optimal number of neighbors for KNN regression

from sklearn.neighbors import KNeighborsRegressor

knnRange = range(1,21,1)

scores_list = []

for i in knnRange:

    regressor_knn = KNeighborsRegressor(n_neighbors=i)

    regressor_knn.fit(x_train,y_train)

    y_pred = regressor_knn.predict(x_test)

    scores_list.append(r2_score(y_test,y_pred))

plt.plot(knnRange,scores_list,linewidth=2,color='green')

plt.xticks(knnRange)

plt.xlabel('No. of neighbors')

plt.ylabel('r2 score of KNN')

plt.show()  
# Training the KNN model on the training set

regressor_knn = KNeighborsRegressor(n_neighbors=6)

regressor_knn.fit(x_train,y_train)

y_pred = regressor_knn.predict(x_test)

r2_knn = r2_score(y_test,y_pred)

print(r2_knn)
# Training the Decision Tree regression on the training model

from sklearn.tree import DecisionTreeRegressor

regressor_Tree = DecisionTreeRegressor(random_state=0)

regressor_Tree.fit(x_train,y_train)
# Predicting test results

y_pred = regressor_Tree.predict(x_test)
# Calculating r2 score

r2_tree = r2_score(y_test,y_pred)

print(r2_tree)
# Finding out the optimal number of trees for Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

forestRange=range(50,500,50)

scores_list=[]

for i in forestRange: 

    regressor_Forest = RandomForestRegressor(n_estimators=i,random_state=0)

    regressor_Forest.fit(x_train,y_train)

    y_pred = regressor_Forest.predict(x_test)

    scores_list.append(r2_score(y_test,y_pred))

plt.plot(forestRange,scores_list,linewidth=2,color='maroon')

plt.xticks(forestRange)

plt.xlabel('No. of trees')

plt.ylabel('r2 score of Random Forest Reg.')

plt.show()  
# Training the Random Forest regression on the training model

regressor_Forest = RandomForestRegressor(n_estimators=150,random_state=0)

regressor_Forest.fit(x_train,y_train)

y_pred = regressor_Forest.predict(x_test)

r2_forest = r2_score(y_test,y_pred)

print(r2_forest)
## Feature Scaling for SVR

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(np.reshape(y_train,(len(y_train),1)))

y_test = sc_y.transform(np.reshape(y_test,(len(y_test),1)))
## Training the Linear SVR model on the training set

from sklearn.svm import SVR

regressor_SVR = SVR(kernel='linear')

regressor_SVR.fit(x_train,y_train)
## Predicting test results

y_pred = regressor_SVR.predict(x_test)
## Calculating r2 score

r2_linearSVR = r2_score(y_test,y_pred)

print(r2_linearSVR)
## Training the Non-linear SVR model on the training set

from sklearn.svm import SVR

regressor_NonLinearSVR = SVR(kernel='rbf')

regressor_NonLinearSVR.fit(x_train,y_train)
## Predicting test results

y_pred = regressor_NonLinearSVR.predict(x_test)
## Calculating r2 score

r2_NonlinearSVR = r2_score(y_test,y_pred)

print(r2_NonlinearSVR)
## Applying the XGBoost Regression model on the training set

from xgboost import XGBRegressor

regressor_xgb = XGBRegressor()

regressor_xgb.fit(x_train,y_train)
## Predicting test results

y_pred = regressor_xgb.predict(x_test)
## Calculating r2 score

r2_xgb = r2_score(y_test,y_pred)

print(r2_xgb)
## Comparing the r2 scores of different models

labelList = ['Multiple Linear Reg.','Polynomial Reg.','K-NearestNeighbors','Decision Tree','Random Forest',

             'Linear SVR','Non-Linear SVR','XGBoost Reg.']

mylist = [r2_MultiLinear,r2_poly,r2_knn,r2_tree,r2_forest,r2_linearSVR,r2_NonlinearSVR,r2_xgb]

for i in range(0,len(mylist)):

    mylist[i]=np.round(mylist[i]*100,decimals=2)

print(mylist)
plt.figure(figsize=(14,8))

ax = sns.barplot(x=labelList,y=mylist)

plt.yticks(np.arange(0, 101, step=10))

plt.title('r2 score comparison among different regression models',fontweight='bold')

for p in ax.patches:

    width, height = p.get_width(), p.get_height()

    x, y = p.get_xy() 

    ax.annotate('{:.2f}%'.format(height), (x +0.25, y + height + 0.25))

plt.show()