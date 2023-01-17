# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/BreadBasket_DMS.csv')
#After loading of data ,we are taking a look to check dataset briefly.
df.info()
#To be able to examine sales frequency, we'd better seperate of Date and Time columns and add new columns as months,days and hours.
df['Months']=[ int(each.split('-')[1]) for each in df.Date]
df['Days']=[ int(each.split('-')[2]) for each in df.Date]
df['Hours']=[ int(each.split(':')[0]) for each in df.Time]
#In hours section, there is only one feature (2017-01-01	01:21:05	4090	Bread	2017) which is less important if we consider all features .
#Thats why we are removing this feature.
df=df[df.Hours!=1]
#We made a column for Monts too but in this kernel we are not gonna use .
df.head()
#Now our data is ready to analyze.
all_items=list(df.Item.unique())

print('Total Number of Item: ',len(all_items))
#lets check 15 Items  which are most sold
from collections import Counter
all_sales=Counter(df.Item)
most_common_sales=all_sales.most_common(15)
x,y=zip(*most_common_sales)
x,y=list(x),list(y)
names=pd.DataFrame(x)
number_of_sales=pd.DataFrame(y)
number_of_sales=pd.DataFrame(y)
most_selling_items=pd.concat([names.iloc[:,0],number_of_sales.iloc[:,0]],axis=1)
most_selling_items.columns='Items','Number_of_Sales'

most_selling_items.head(15)
import plotly.plotly as py
from plotly.offline import init_notebook_mode, plot,iplot # plotly offline mode
init_notebook_mode(connected=True) 
import plotly.graph_objs as go # plotly graphical object
init_notebook_mode(connected=True)

labels1=most_selling_items.Items
fig={'data':[{'values':most_selling_items.iloc[:,1],'labels':labels1,'name':
    'Items','hoverinfo':'label+percent+name','hole':.4,'type':'pie'}]
    ,'layout':{
    'title':'Most Selling 15 Items','annotations':[{'font':{'size':20},'showarrow':False,'text':'Most Selling ','x':.5,'y':.5}]}}  
iplot(fig)



labels = x
values = y

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', textinfo='value', 
               textfont=dict(size=20),
               marker=dict(line=dict(color='#000000', width=2)))

iplot([trace], filename='styled_pie_chart')
#As it can be seen below with pie chart,most selling Item is Coffee within 5471 number of sales .
#Second one is Bread within 3324 number of sales
#Now we are going to examine sales frequency according to hours

hours_list=sorted(list(df.Hours.unique()))

item_list_by_hours=[]
for i in hours_list:
    s=df[df.Hours==i]
    item_list_by_hours.append(Counter(s.Item))
    
df_hours=pd.DataFrame(item_list_by_hours).set_index([hours_list])
sales_list=[]
sales_list_sum=[]
sales_list_sum_avarage=[]
for i in hours_list:
     sales_list.append(df_hours.loc[i,:].sum())
     counters=df[df.Hours==i]
     sales_list_sum.append(counters.Item.value_counts())
     sales_list_sum_avarage.append(((counters.Item.value_counts())/len(counters.Hours)*100))
sales_list=(pd.DataFrame(sales_list)).set_index([hours_list])#Sales_list is all number of sales
sales_list['Hours']=sales_list.index
sales_list.columns='Number_of_Sales','Hours'
sales_list_sum=(pd.DataFrame(sales_list_sum)).set_index([hours_list]) #Sales_list_sum is all Items within number of sales 
sales_list_sum_avarage=(pd.DataFrame(sales_list_sum_avarage)).set_index([hours_list])  
#Sales_list_sum_avarage is percentage of specif time and specific Item.
# For e.g   ((number of all sales  at 7 am )/(number of coffe sales at 7 am))*100= 52 


import matplotlib.pyplot as plt
import seaborn as sns 
f, ax = plt.subplots(figsize=(15, 6))
sns.set(style="whitegrid")
sns.barplot(x='Hours', y='Number_of_Sales', data=sales_list,  label='Sales Frequency',color='r')
sns.set_context("paper")
ax.legend(ncol=2, loc="upper right", frameon=True)

#Lets take a look to scatter plot to be able to see what we need as regression model 
trace = go.Scatter(
                    x = sales_list.Hours,
                    y = sales_list.Number_of_Sales,
                    mode = "markers",
                    name = "citations",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))
#
data = [trace]
layout = dict(title = 'Sales Frequency of Hours',
              xaxis= dict(title= 'Hours',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


#According to above plot , we are going to set Polynomial Linear Regression Model to our Data
#For regression models , we are using Sklearn library
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x=sales_list.Hours.values.reshape(-1,1)
y=sales_list.Number_of_Sales.values.reshape(-1,1)
pr=PolynomialFeatures(degree=7) #Setting of degree to polynomial 
x_pr=pr.fit_transform(x) #Fitting and transforming 

linear_r=LinearRegression()

linear_r.fit(x_pr,y)

y_head=linear_r.predict(x_pr)
y_head=pd.DataFrame(y_head)
# x_pr values are being predicted 
#Now lets see how much it is proper on our data


trace1 = go.Scatter(
                    x = sales_list.Hours,
                    y = sales_list.Number_of_Sales,
                    mode = "markers",
                    name = "Actual Values",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))
#Trace1 is our data values of sales
trace2=go.Scatter(
                    x = sales_list.Hours,
                    y = y_head.iloc[:,0],
                    mode = "lines+markers",
                    name = "Polynomial Regression",
                    marker = dict(color = 'rgba(66, 32, 111, 0.6)'))
#Trace2 is prediction values 
data = [trace1,trace2]
layout = dict(title = 'Polynomial Regression',
              xaxis= dict(title= 'Hours',ticklen= 10,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
day_list=sorted(df.Days.unique())

item_list_by_days=[]
for each in day_list:
    m=df[df.Days==each]
    item_list_by_days.append(Counter(m.Item))
df_days=(pd.DataFrame(item_list_by_days)).set_index([day_list])

sales_list1=[]
sales_list_sum1=[]
for i in day_list:
    sales_list1.append(df_days.loc[i,:].sum())
    counters1=df[df.Days==i]
    sales_list_sum1.append(counters1.Item.value_counts())
sales_list1=(pd.DataFrame(sales_list1)).set_index([day_list])
sales_list1['Number_of_sales']=sales_list1.iloc[:,0]
sales_list1['Days']=sales_list1.index
sales_list1.drop([0],axis=1,inplace=True)
sales_list_sum1=pd.DataFrame(sales_list_sum1).set_index([day_list])
sales_list1.head()
sales_list_sum1.head()

f, ax = plt.subplots(figsize=(15, 6))
sns.set(style="darkgrid")
ax = sns.kdeplot(sales_list1.Days,sales_list1.Number_of_sales,
                 cmap="Reds", shade=True, shade_lowest=False,cut=3)
red = sns.color_palette("Reds")[-2]
ax.text(25,1100, "Sales Frequency", size=20, color=red)
#In this plot , we can see that most selling days are happening between 0-10 days in Months.
#Lets take a look to our  3 Items  to see max,min ,median etc parameters with BoxPlot
trace0 = go.Box(
    y=sales_list_sum1.iloc[:,0],
    name = 'Coffe Sales',
    boxpoints = 'suspectedoutliers',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=sales_list_sum1.iloc[:,1],
    name = 'Bread Sales',
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = 'rgb(42, 22, 110)',
    )
)
trace2 = go.Box(
    y=sales_list_sum1.iloc[:,2],
    name = 'Tea Sales',
    boxpoints = 'suspectedoutliers',
    marker = dict(
        color = 'rgb(76,112,143)',
    )
)
trace3 = go.Box(
    y=sales_list_sum1.iloc[:,4],
    name = 'Cake Sales',
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'all',
    marker = dict(
        color = 'rgb(100, 100, 100)',
    )
)
layout = go.Layout(title='Most Selling 4 Item in Month ', 
                   yaxis=dict(title='Number Of Sales'))
data = [trace0,trace1,trace2,trace3]
fig=go.Figure(data=data,layout=layout)

iplot(fig)

x_coffee=sales_list_sum1.index.values.reshape(-1,1)
y=sales_list_sum1.iloc[:,0].values.reshape(-1,1)
x_bread=sales_list_sum1.index.values.reshape(-1,1)
y_1=sales_list_sum1.iloc[:,1].values.reshape(-1,1)

plr=PolynomialFeatures(degree=6)
x_coffee=plr.fit_transform(x_coffee)
x_bread=plr.fit_transform(x_bread)

lnr=LinearRegression()
lnr_1=LinearRegression()

lnr.fit(x_coffee,y)
lnr_1.fit(x_bread,y_1)

y_predicted_coffee=lnr.predict(x_coffee)
y_predicted_bread=lnr_1.predict(x_bread)
y_predicted_coffee=pd.DataFrame(y_predicted_coffee)
y_predicted_bread=pd.DataFrame(y_predicted_bread)



#Trace1 is values of coffee sales
trace1 = go.Scatter(
                    x = sales_list_sum1.index,
                    y = sales_list_sum1.Coffee,
                    mode = "markers",
                    name = "Coffe Sales",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))
#Trace2 is values of bread sales
trace2=go.Scatter(
                    x = sales_list_sum1.index,
                    y = sales_list_sum1.Bread,
                    mode = "markers",
                    name = "Bread Sales",
                    marker = dict(color = 'rgba(66, 32, 111, 0.6)'))
#Trace3 is predicted values of coffee sales
trace3=go.Scatter(
                    x = sales_list_sum1.index,
                    y = y_predicted_coffee.iloc[:,0],
                    mode = "lines+markers",
                    name = "Coffee Sales Predict",
                    marker = dict(color = 'rgba(44, 103, 177, 0.6)'))
#Trace4 is predicted values of bread sales
trace4=go.Scatter(
                    x = sales_list_sum1.index,
                    y = y_predicted_bread.iloc[:,0],
                    mode = "lines+markers",
                    name = "Bread Sales Predict",
                    marker = dict(color = 'rgba(103, 10, 73, 0.6)'))

data = [trace1,trace2,trace3,trace4]
layout = dict(title = 'PLR for Coffee and Bread Sales ',
              xaxis= dict(title= 'Days',ticklen= 10,zeroline=True)
             )
fig = dict(data = data, layout = layout)
iplot(fig)



