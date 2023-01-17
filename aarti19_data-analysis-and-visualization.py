#importing all the necessary libraries

#for mathematical operations
import numpy as np

#for data manipulation
import pandas as pd

#for data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#for ignoring warnings
import warnings
warnings.filterwarnings("ignore")
#Reading the dataset
data = pd.read_csv('../input/indian-food-101/indian_food.csv')
#Checking the head of the data
data.head()
#Checking the tail of the data
data.tail()
#Checking the shape of the dataset
data.shape
print("There are {0} rows and {1} columns in the dataset".format(data.shape[0],data.shape[1]))
#Replacing all the -1 values in the categorical and numerical columns with nan 
data= data.replace('-1',np.nan)
data = data.replace(-1,np.nan)
data['diet'].value_counts()
data['course'].value_counts()
data['flavor_profile'].value_counts()
data['state'].value_counts()
#Checking the head of only categorical variables
data.select_dtypes('object').head()
#Setting the style and background of plots
plt.rcParams['figure.figsize'] = (15,6)
plt.style.use('fivethirtyeight')

#Countplot of Veg/Non-Veg
plt.subplot(1,2,1)
sns.countplot(data['diet'])
plt.title("Count Plot of Veg/Non-Veg")

#Piechart for Veg/NonVeg
plt.subplot(1,2,2)
x = data['diet'].value_counts()
explode = [0.1,0]
labels = ['Veg','Non-veg']
plt.pie(x,explode=explode,labels=labels, autopct = '%.2f%%')
plt.title("Pie Chart showing the distribution of Veg/Non-Veg")
plt.show()
plt.rcParams['figure.figsize'] = (15,6)
plt.style.use('fivethirtyeight')

#Count plot of different flavours of dishes
sns.countplot(data['flavor_profile'])
plt.title("Count Plot of flavours")
plt.rcParams['figure.figsize'] = (15,6)
plt.style.use('fivethirtyeight')

#Countplot of dish courses
sns.countplot(data['course'])
plt.title("Count Plot of course")
plt.rcParams['figure.figsize'] = (18,6)
plt.style.use('fivethirtyeight')

#Countplot of states
sns.countplot(data['state'])
plt.title("Count Plot of state")
plt.xticks(rotation=90)
plt.show()
a = data['state'].value_counts().reset_index()
a.sort_values(by='state',ascending=False).head(6).style.background_gradient(cmap='copper')
plt.rcParams['figure.figsize'] = (15,6)
plt.style.use('fivethirtyeight')

#Countplot of region
sns.countplot(data['region'])
plt.title("Count Plot of region")
#Checking only the numerical columns of the dataset
data.select_dtypes('number').head()
#Area plot of column "preparation time"
data['prep_time'].plot(kind='area',color='brown')
#area plot of column "cooking time"
data['cook_time'].plot(kind='area')
sns.boxenplot(data['cook_time'], orient = 'v')
sns.countplot(data['diet'],hue=data['flavor_profile'])
plt.title("Count of Veg/Non-Veg based on flavor")
plt.show()
#Crosstab of flavour and course of the dishes
x = pd.crosstab(data['flavor_profile'],data['course']).style.background_gradient(cmap='copper')
x
y = pd.crosstab(data['course'],data['diet'])
y.style.bar(color=['gold'])
data[['course','prep_time']].sort_values(by='prep_time',ascending=False).reset_index().head()
display(data[['flavor_profile','prep_time']].groupby(['flavor_profile']).agg(['max','mean','min']))

#Plotting the same
x= data[['flavor_profile','prep_time']].groupby(['flavor_profile']).agg(['max','mean','min'])
x.plot(kind='line')
plt.legend()
plt.show()
#Total time taken to serve a dish can be calculated by adding the cook time and prep time.
data['total_time'] = data['prep_time']+data['cook_time']

x = data.loc[data.groupby(['course'])['total_time'].idxmax()][['total_time','course']]
x
display(data[['total_time','course']].groupby(['course']).agg(['max','min']))

#Lets plot this
data[['total_time','course']].groupby(['course']).agg(['max','min']).plot(kind='line', color=['red','blue'])
plt.title("Which course takes maximum time to be served?", fontsize=20)
plt.ylabel("Time",fontsize=15)
plt.show()
pd.crosstab(data['region'], data['diet']).style.bar(color='gold')
#Number of Veg and Non veg dishes in different states of the country.
pd.crosstab(data['state'], data['diet']).style.bar(color='gold')
pd.crosstab(data['state'], data['diet']).plot()
plt.show()
#Defining a function which displays all the dishes names for a particular state
def state_dish(x):
    return data[data['state']==x][['name','diet','flavor_profile']]
state_dish('Gujarat')
state_dish('Maharashtra')
#Defining a function which displays all the dishes of a particular region
def region_dish(x):
    return data[data['region']==x][['name','diet','flavor_profile','state']]
region_dish('Central')