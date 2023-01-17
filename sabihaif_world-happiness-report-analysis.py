# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

import plotly.plotly as py # visualization library

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True) 

import plotly.graph_objs as go # plotly graphical object

import warnings

warnings.filterwarnings('ignore')

%pylab inline







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data2015 = pd.read_csv("../input/world-happiness/2015.csv")

location = pd.read_csv("../input/world-capitals-gps/concap.csv")

data2015.head()
data2015.tail()
data2015.describe()
#columns of data

data2015.columns
data2015.shape
#information about data

data2015.info()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(data2015.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data_plot = data2015.loc[:,["Health (Life Expectancy)","Family", "Economy (GDP per Capita)","Happiness Score" ]]

data_plot.plot()
data_plot.plot(subplots = True)
data_plot.plot(kind = "scatter", x = "Economy (GDP per Capita)", y = "Happiness Score")

data2015.Region.unique()
region_list = list(data2015.Region.unique())

region_happiness_score_ratio = []

for i in region_list:

    x = data2015[data2015.Region == i]

    region_happiness_score_rate = sum(x["Happiness Score"])/len(x)

    region_happiness_score_ratio.append(region_happiness_score_rate)

#I want to sort my new data 

data_bar = pd.DataFrame({'region_list':region_list, 'region_happiness_score_ratio':region_happiness_score_ratio})

new_index = (data_bar['region_happiness_score_ratio'].sort_values(ascending = False)).index.values

sorted_data = data_bar.reindex(new_index)



#visualisation

plt.figure(figsize=(10,7))

sns.barplot(x=sorted_data['region_list'], y=sorted_data['region_happiness_score_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('Regions')

plt.ylabel('Happiness Score')

plt.title('Happiness Score Ratio by Regions')



    

region_economy_ratio = []

for i in region_list:

    y = data2015[data2015.Region == i]

    region_economy_rate = sum(y['Economy (GDP per Capita)'])/len(y)

    region_economy_ratio.append(region_economy_rate)

    

data_bar2 = pd.DataFrame({'region_list':region_list,'region_economy_ratio':region_economy_ratio})

new_index2=(data_bar2['region_economy_ratio'].sort_values(ascending = False)).index.values

sorted_data2 = data_bar2.reindex(new_index2)



plt.figure(figsize=(10,7))

sns.barplot(x=sorted_data2['region_list'], y=sorted_data2['region_economy_ratio'], palette = sns.color_palette("BuGn_r",15) )

plt.xticks(rotation= 60)

plt.xlabel('Regions')

plt.ylabel('Economy')

plt.title('Economy Ratio by Regions')
sorted_data.region_happiness_score_ratio = sorted_data.region_happiness_score_ratio/max(sorted_data.region_happiness_score_ratio)

sorted_data2.region_economy_ratio = sorted_data2.region_economy_ratio/max(sorted_data2.region_economy_ratio)

data = pd.concat([sorted_data,sorted_data2.region_economy_ratio],axis = 1)

data.sort_values("region_happiness_score_ratio",inplace = True)



#visualisation

f,ax1 = plt.subplots(figsize = (12,8))

sns.pointplot(x = "region_list",y = "region_economy_ratio",data = data,color = 'purple',alpha = 0.7)

sns.pointplot(x = "region_list",y = "region_happiness_score_ratio", data = data,color = 'lime',alpha = 0.7)

plt.text(7,0.6, "Happiness Score Ratio of Given Regions",color = 'lime',fontsize = 13,style = 'italic' )

plt.text(7,0.55, "Economy Ratio of Given Regions",color = 'purple',fontsize = 13,style = 'italic')

plt.xlabel('Regions', fontsize = 15, color = 'blue')

plt.ylabel('Values', fontsize = 15, color ='blue')

plt.xticks(rotation = 60)

plt.title('Happines Score vs. Economy Score', fontsize = 20, color = 'blue')

plt.grid()





g = sns.jointplot(data.region_economy_ratio, data.region_happiness_score_ratio, kind = "kde", height = 7)

g.annotate(stats.pearsonr)

plt.show()
g = (sns.jointplot('region_economy_ratio', 'region_happiness_score_ratio',data = data, color = 'lime', ratio = 3))
sns.lmplot('region_economy_ratio', 'region_happiness_score_ratio',data = data)

plt.show()
sns.kdeplot(data.region_economy_ratio, data.region_happiness_score_ratio, shade = True, cut = 3, color ='green')

plt.show()
f,ax1 = plt.subplots(figsize = (15,8))

sns.violinplot(x = data2015.Region, y = data2015['Economy (GDP per Capita)'])

plt.xticks(rotation = 90)

plt.show()
sns.pairplot(data2015.iloc[:,[0,5,6,7,8]])

plt.show()
data_new = pd.merge(location[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\

data2015,left_on='CountryName',right_on='Country')

data_new.shape
filter1 = data_new['Happiness Score']>=6.5

happy_countries = data_new[filter1]

happy_countries.Region.unique()
happy_countries.Region.value_counts()


labels = happy_countries.Region.value_counts().index

colors = ['purple','blue','red','yellow','green','orange','lightcoral']

explode = [0,0,0,0,0,0,0]

sizes = happy_countries.Region.value_counts().values



# visual

plt.figure(figsize = (10,10))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Disstribution of the Happiest Countries by Region',color = 'blue',fontsize = 15)

filter2 = data_new['Happiness Score']<4.5

unhappy_countries = data_new[filter2]

unhappy_countries.Region.value_counts()
labels = unhappy_countries.Region.value_counts().index

colors = ['blue','purple','red','yellow','green']

explode = [0,0,0,0,0]

sizes = unhappy_countries.Region.value_counts().values



# visual

plt.figure(figsize = (10,10))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Disstribution of the Least Happy Countries by Region',color = 'red',fontsize = 15)
happiness_score = data_new['Happiness Score'].astype(float)
data = [dict(

        type='choropleth',

        colorscale = 'Rainbow',

        locations = data_new['CountryName'],

        z = happiness_score,

        locationmode = 'country names',

        text = data_new['Country'],

        colorbar = dict(

        title = 'Happiness Score', 

        titlefont=dict(size=25),

        tickfont=dict(size=18))

)]

layout = dict(

    title = 'Happiness Score',

    titlefont = dict(size=40),

    geo = dict(

        showframe = True,

        showcoastlines = True,

        projection = dict(type = 'equirectangular')

        )

)

choromap = go.Figure(data = data, layout = layout)

iplot(choromap, validate=False)



        



import sklearn
#Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

x = data2015['Economy (GDP per Capita)'].values.reshape(-1,1)

y = data2015['Happiness Score'].values.reshape(-1,1)





from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33, random_state=0)



lin_reg.fit(x_train,y_train)

y_pred = lin_reg.predict(x_test)

plt.plot(x_test,y_pred)



b0 = lin_reg.intercept_

b1 = lin_reg.coef_

print('equation of the line is: ',b1,'x +',b0)
xtest = pd.DataFrame(x_test)

ypred = pd.DataFrame(y_pred)

prediction = pd.concat([xtest,ypred],axis=1)

prediction.columns = ['xtest','ypred']

prediction.sort_values(by='xtest', ascending=False, axis = 0, inplace = True)

prediction.head()





xtest = pd.DataFrame(x_test)

ytest = pd.DataFrame(y_test)

test = pd.concat([xtest,ytest],axis=1)

test.columns = ['xtest','ytest']

test.sort_values(by='xtest', ascending=False, axis = 0, inplace = True)

test.head()
data2015.head()
#Multiple linear regression

x1 = data2015[['Economy (GDP per Capita)','Health (Life Expectancy)','Family','Freedom']].values

y1 = data2015['Happiness Score'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y1, test_size = 0.33, random_state=0)

mlp = LinearRegression()

mlp.fit(x1_train,y1_train)

y1_predict = pd.DataFrame(mlp.predict(x1_test))

y1_test = pd.DataFrame(y1_test)



#comparing of test and prediction

comp  = pd.concat([y1_predict,y1_test],axis=1)

comp.columns = ['y1_predict','y1_test']

comp.sort_values(by='y1_test', ascending=False, axis = 0, inplace = True)

comp.sample(10)
data2015.head()
data2015.plot(kind = "scatter", x = "Economy (GDP per Capita)", y = "Happiness Score")
#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

pol_reg = PolynomialFeatures(degree = 2)

x_poly = pol_reg.fit_transform(x)

x.shape
