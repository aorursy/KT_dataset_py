# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Read data
import numpy as np
import pandas as pd

# Visualisation Library
import seaborn as sns                # seaborn

import matplotlib.pyplot as plt      # matplotlib
%matplotlib inline

import plotly                        # plotly
import plotly.express as px
import plotly.graph_objs as go

# Style
plt.style.use("fivethirtyeight")
sns.set_style("darkgrid")
# Load train data
covid_train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
covid_train.head()
# Load test data
covid_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
covid_test.head()
covid_train.shape, covid_test.shape # Rows x Columns
print("Index of train:", covid_train.index)
print("Index of test:", covid_test.index)
print("Column names of Train:", covid_train.columns)
print("------------------------------------------------------------------------------------------------------/n")
print("Column names of Test:", covid_test.columns)
covid_train.count()
covid_test.count()
covid_train.size, covid_test.size
# Information of data
covid_train.info()  # Shows Data types, Null Values for each variable
covid_train.dtypes  # Gives data types of each variable
covid_train.describe()
covid_train['Country'].nunique()  # Number of uniques values in variable "Country_Region"
covid_train['Country'].unique()  # List of uniques categories in variable "Country_Region"
covid_train['Country'].value_counts()  # Number of data points in variable "Country_Region"
covid_train['ConfirmedCases'].nunique()  # Number of uniques values in variable "ConfirmedCases"
covid_train['ConfirmedCases'].unique()  # List of uniques categories in variable "Country_Region"
covid_train['ConfirmedCases'].value_counts()  # Number of data points in variable "Country_Region"
# In Fatalities, listed no of unique categories
covid_train['Fatalities'].nunique()
covid_train['Fatalities'].value_counts()  # Number of data points in variable "Country_Region"
# convert format DD-MM-YYYY to YYYY-MM-DD
import datetime
covid_train['Date'] = pd.to_datetime(covid_train['Date'])
covid_train['Date']
# Shows Starting date and Ending date
print(covid_train['Date'].min())
print(covid_train['Date'].max())
covid_train.rename(columns={'Country_Region':'Country'}, inplace=True)
covid_test.rename(columns={'Country_Region':'Country'}, inplace=True)

covid_train.rename(columns={'Province_State':'State'}, inplace=True)
covid_test.rename(columns={'Province_State':'State'}, inplace=True)
### Top 20 Countries of "ConfirmedCases" & "Fatalities"
Groupby = covid_train.groupby(by='Country')[['ConfirmedCases','Fatalities']].sum().reset_index()
Groupby_Sort = Groupby[Groupby['ConfirmedCases'] != 0].sort_values(by='ConfirmedCases',ascending=False).head(20)
Groupby_Sort.style.background_gradient(cmap='viridis_r')
# Largest 25 countries
Columns = covid_train[['Country', 'State', 'ConfirmedCases', 'Fatalities']]
Countrys_15 = Columns.groupby(['Country', 'State']).max().reset_index().nlargest(25, "ConfirmedCases")
Countrys_15.style.background_gradient(cmap='nipy_spectral')
# Showing top 20 data of China
covid_train_China = covid_train.loc[covid_train['Country'] == 'China', :].head(20)
covid_train_China.style.background_gradient(cmap='viridis_r')
# Listed Sum of "Canada" Confirmed Cases & Fatalities
List = {
    'X' : [covid_train[covid_train['Country']=='Canada'].groupby(by='Country').sum()],
    'Y' : [covid_train[covid_train['Country']=='China'].groupby(by='Country').sum()],
    'Z' : [covid_train[covid_train['Country']=='India'].groupby(by='Country').sum()]
}
Combined = pd.DataFrame(List)
Combined.style.background_gradient(cmap='viridis_r')
# Listing Number of missing values by feature column wise.
print(covid_train.isnull().sum())
# Missing value representation by Heatmap
plt.figure(figsize=(15,11))
sns.heatmap(covid_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
corr = covid_train.corr()
sns.heatmap(corr,vmax=1.,square=True)
g=sns.heatmap(covid_train[["Id","ConfirmedCases","Fatalities"]].corr(),annot=True,fmt=".2f",cmap="coolwarm")
# Box Plot used to find out the outliers in feature column of "ConfirmedCases"
plt.figure(figsize=(12,10))
sns.boxplot(data=covid_train['ConfirmedCases'], palette='winter')
# Box Plot used to find out the outliers in feature column of "Fatalities"
plt.figure(figsize=(12,10))
sns.boxplot(data=covid_train['Fatalities'], palette='winter')
# Bar Chart for showing count of County/Region wise
plt.figure(figsize=(15,11))

covid_train['Country'].value_counts()[0:40].plot(kind='bar')

plt.xlabel('Country', fontsize=17, fontweight = 'bold')
plt.ylabel('Count', fontsize=17, fontweight = 'bold')

plt.title('Country Vs Count', fontsize=20, fontweight = 'bold')

plt.show()
# Bar Chart for showing count of Fatalities
plt.figure(figsize=(15,11))

covid_train['Fatalities'].value_counts()[0:15].plot(kind='bar')

plt.xlabel('Fatalities', fontsize=17, fontweight = 'bold')
plt.ylabel('Count', fontsize=17, fontweight = 'bold')

plt.title('Fatalities Vs Count', fontsize=20, fontweight = 'bold')

plt.show()
# Bar Chart for showing count of Date wise Fatalities
plt.figure(figsize=(15,11))

covid_train.groupby('Date').mean().sort_values(by='Fatalities', ascending=False)['Fatalities'][0:50].plot(kind='bar')

plt.xlabel('Date / Fatalities', fontsize=17, fontweight = 'bold')
plt.ylabel('Count', fontsize=17, fontweight = 'bold')

plt.title('Date / Fatalities Vs Count', fontsize=20, fontweight = 'bold')

plt.show()
# Count map by using Seaborn
# Showing Count for Each Country from first 5000 rows
plt.figure(figsize=(20,10))
sns.countplot(covid_train['Country'].head(5000))

plt.xlabel('Country', fontsize=22, fontweight = 'bold')
plt.ylabel('Count', fontsize=22, fontweight = 'bold')

plt.title('Country Vs Count', fontsize=28, fontweight = 'bold')

plt.xticks(rotation = 90, fontsize=18)

plt.show()
# Bar plot between "Country_Region" & "ConfirmedCases"
plt.figure(figsize=(30,10))
sns.barplot(x='Country', y='ConfirmedCases', data=covid_train)

plt.xlabel('Country', fontsize=25, fontweight='bold')
plt.ylabel('ConfirmedCases', fontsize=25, fontweight='bold')

plt.title('Country Vs ConfirmedCases', fontsize=35, fontweight='bold')

plt.xticks(rotation=90)

plt.show()
# Scatter plot between "Country_Region" & "ConfirmedCases"
plt.figure(figsize=(30,15))
plt.scatter(covid_train['Country'], covid_train['ConfirmedCases'])

plt.xlabel('Country', fontsize=25, fontweight='bold')
plt.ylabel('ConfirmedCases', fontsize=25, fontweight='bold')

plt.title('Country Vs ConfirmedCases', fontsize=35, fontweight='bold')
plt.xticks(rotation = 90)

plt.show()
# Scatter plot between "ConfirmedCases" & "Fatalities"
plt.figure(figsize=(15,10))
plt.scatter(covid_train['ConfirmedCases'], covid_train['Fatalities'])

plt.xlabel('ConfirmedCases', fontsize=18, fontweight='bold')
plt.ylabel('Fatalities', fontsize=18, fontweight='bold')

plt.title('ConfirmedCases Vs Fatalities', fontsize=18, fontweight='bold')
plt.xticks(rotation = 45)

plt.show()
df_Cases = Countrys_15[Countrys_15['ConfirmedCases'] != 0].sort_values(by='ConfirmedCases',ascending=False).head(10)
# Scatter Plot For Country VS ConfirmedCases
plt.figure(figsize=(15,8))
plt.scatter(Groupby_Sort['Country'], Groupby_Sort['ConfirmedCases'], c='green', s=250)

plt.xlabel('Country', fontsize = 15)
plt.ylabel('ConfirmedCases', fontsize = 15)

plt.title("Total ConfirmedCases", fontsize = 20, fontweight='bold')

plt.xticks(rotation=90)

plt.show()
# Scatter Plot For Country VS ConfirmedCases
plt.figure(figsize=(15,8))
plt.scatter(Groupby_Sort['Country'], Groupby_Sort['Fatalities'], c='green', s=250)

plt.xlabel('Country', fontsize = 15)
plt.ylabel('Fatalities', fontsize = 15)

plt.title("Fatalities", fontsize = 20, fontweight='bold')

plt.xticks(rotation=90)

plt.show()
# Box plot created for feature columns of "ConfirmedCases" & "Fatalities"
plt.figure(figsize=(15,10))
sns.boxplot(data=covid_train[['ConfirmedCases','Fatalities']])
# bar plot showing Confirmed Cases as per Country / Region wise
plt.figure(figsize=(20,18))
Country = covid_train.groupby(by='Country')[['ConfirmedCases','Fatalities']].sum().reset_index()

sns.barplot(x='ConfirmedCases', y='Country', data = Country[Country['ConfirmedCases'] != 0].sort_values(by='ConfirmedCases',ascending=False).head(30))

plt.xlabel('ConfirmedCases', fontsize=21, fontweight = 'bold')
plt.ylabel('Country_Region', fontsize=21, fontweight = 'bold')

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.title('ConfirmedCases Vs Country', fontsize=25, fontweight = 'bold')

plt.show()
# Pie chart for Confirmed Cases

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
# autopct : formatting how the percentages appear on the pie chart

plt.figure(figsize=(5,4))
explode =(0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.pie(Groupby_Sort['ConfirmedCases'], labels=Groupby_Sort['Country'], radius=2, autopct='%.1f%%',
        shadow=True, startangle=90, explode = explode, wedgeprops={'edgecolor': 'black'})

plt.show()
# Pie chart for Fatalities(Death)

plt.figure(figsize=(7,6))

plt.axis('equal')

explode =(0,0.1,0.1,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

plt.pie(Groupby_Sort['Fatalities'], labels=Groupby_Sort['Country'], radius=2, autopct='%0.1f%%',
        shadow=True, startangle=90, explode = explode, center=(0, 0), wedgeprops={'edgecolor': 'black'})

plt.show()
# Pairplot used to show features on Country/Region basis

sns.pairplot(covid_train)
covid_x=covid_train.drop(['ConfirmedCases','Country','State','Date'],axis='columns')
covid_x.head()
covid_y=pd.DataFrame(covid_train.iloc[:,-2])
covid_y.head()
# Splitting X and y into training and testing sets

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(covid_x,covid_y)
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(X_train,Y_train)
print(X_train.shape)
print(X_test.shape)
X_test.head()
Y_test.head()
from sklearn.tree import DecisionTreeRegressor
tree_regressor=DecisionTreeRegressor()
tree_regressor.fit(X_train,Y_train)
y_pred_tree=tree_regressor.predict(X_test)
y_tree_pred_df=pd.DataFrame(y_pred_tree,columns=['Predict_tree'])
y_tree_pred_df.head()
DTCscore = tree_regressor.score(X_train,Y_train)
print("Decision Tree Score: ",DTCscore)
plt.figure(figsize=(5,5))
plt.title('Actual vs Prediction')
plt.xlabel('Fatalities')
plt.ylabel('Predicted')
plt.legend()
plt.scatter((X_test['Fatalities']),(Y_test['ConfirmedCases']),c='red')
plt.show()
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier()
RFC.fit(X_train,Y_train)
RFC.score(X_train,Y_train)
sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
sub.to_csv('submission.csv',index=False)