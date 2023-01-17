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
# Loading the dataset 



data = pd.read_csv("/kaggle/input/stackindex/MLTollsStackOverflow.csv")

print(data)
data = data.drop(columns=["Tableau"])

print(data)
# Checking null values



print(data.isnull().sum())
# Display number of rows in each column



print(data.count(axis=0))
# Getting column names



print(list(data.columns))
# Converting column names into lower form



columns = data.columns

columns = columns.str.lower()

print(list(columns))
# Splitting the month column



data[['year','months']] = data.month.str.split("-",expand=True)

print(data['year'], data['months'])
# Print new dataset



print(data.head())
# Operations for visualization



import matplotlib.pyplot as plt

included =  data.drop(columns=["month","year","months"])

print(included.head())
# Operations for visualization



question_count  = included.sum(axis = 0)

top_keys = question_count.sort_values(ascending = False)[0:11]

print(list(top_keys))
# Pie Chart



import matplotlib.pyplot as plt

  

# Creating plot 

#fig = plt.figure(figsize =(200, 7)) 

plt.pie(top_keys, labels = top_keys.index, radius=3) 

  

# show plot 

plt.show() 
# Line Plot



import matplotlib.pyplot as plt

# Function to plot x and y values

fig = plt.figure(figsize = (30, 10))

plt.plot(included.columns[0:25],question_count[0:25]) 

  

plt.xlabel("Tech Keys") 

plt.ylabel("Total Questions") 

plt.title("Key Vs Questions") 

plt.show() 
# Grouping data by year



count = data.groupby(['year']).count()

print(count)
# Bar Plot



import matplotlib.pyplot as plt

# Function to plot x and y values

fig = plt.figure(figsize = (30, 10))

plt.bar(data["year"],data["nltk"]) 

  

plt.xlabel("Year") 

plt.ylabel("Total Questions") 

plt.title("NLTk Questions / Year") 

plt.show() 
# Multiple Bar Plots

# set width of bar 

barWidth = 0.25

fig = plt.subplots(figsize =(20, 10)) 



# Make the plot 

plt.bar(data["year"] , data["python"], color ='r', width = barWidth, edgecolor ='grey', label ='Python')

plt.bar(data["year"], data["r"], color = 'g', width = barWidth, edgecolor ='grey', label ='R') 

plt.bar(data["year"], data["machine-learning"], color ='b', width = barWidth, edgecolor ='grey', label ='ML') 

   

# Adding Xticks  

plt.xlabel('Year', fontweight ='bold') 

plt.ylabel('Tech-Keys', fontweight ='bold') 



plt.show() 
# Bar Plot



import matplotlib.pyplot as plt

# Function to plot x and y values

fig = plt.figure(figsize = (30, 10))

plt.bar(data["months"],data["spacy"]) 

  

plt.xlabel("Month") 

plt.ylabel("Total Questions") 

plt.title("Spacy Questions / Month") 

plt.show() 
# 3D Scatter Plot



import plotly.express as px

df = px.data.iris()

a = ['r', 'python']

fig = px.scatter_3d(data, x='year', y='months', z='numpy', color='year')

fig.show()
# Heat Map



import seaborn as sns

correlation = data.corr()

plt.figure(figsize = (12 , 12))

sns.heatmap(correlation)
# Donut chart



import plotly.graph_objects as go



labels = list(top_keys.index)

values = list(top_keys)



# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
# Applying lambda function for year column



data['year'] = data['year'].apply( lambda x : int('20'+x))

print(data['year'])
# Adding values corresponding to year 



tech = data.groupby(['year']).sum()

print(tech)

a = tech.index

tech['year'] = a

print(tech)
#Listing out data for input features 

x = tech[['year']]

print(x)
#Listing out data for target feature 

y = tech[['nltk']]

print(y)
# splitting x and y into training and validation sets 

from sklearn.model_selection import train_test_split 

from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2)
#Training ML model

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

rfr=RandomForestRegressor()

rfr.fit(x_train,y_train)

y_pred=rfr.predict(x_test)

print(y_pred)

print(y_pred.shape)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_pred)