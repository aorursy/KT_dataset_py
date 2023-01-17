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
from matplotlib import pyplot as plt

import seaborn as sns

import datetime



from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



%matplotlib inline
data = "/kaggle/input/us-border-crossing-data/Border_Crossing_Entry_Data.csv"

df = pd.read_csv(data)

df.head()
#Drop all NaNs in the data

df.dropna(inplace=True)

df.head()
#converting Date to datetime instance

df['Date'] = pd.to_datetime(df['Date'])

years = df['Date'].dt.year.unique().tolist() #Extract years from the date

years.remove(2020) #Remove 2020 due to insufficient data

df.Date.dt.year.value_counts()
df.head()
#Investigating Number of people crossing different borders by Measure of Crossing from 2010 to 2020

Avg_Measure = df.groupby("Measure")["Value"].mean()

Avg_Measure



sns.set(rc={'figure.figsize':(12,10)})



ax = sns.barplot(y=Avg_Measure, x = Avg_Measure.axes[0]).set(title='Method of Crossing from 2010 to 2020', xlabel = 'Measure of Crossing', ylabel = 'Number of People Crossing Border')



plt.xticks(rotation=45, ha='right', fontweight='light', fontsize='small');
#Investigating Number of people crossing different borders by Border of Crossing from 2010 to 2020

Avg_Value = df.groupby("Border")["Value"].mean()

Avg_Value



plt.figure(figsize=(8,6))

ax = sns.barplot(y=Avg_Value, x = Avg_Value.axes[0]).set(title='Method of Crossing from 2010 to 2020', xlabel = 'Measure of Crossing', ylabel = 'Number of People Crossing Border')



plt.xticks(rotation=45, ha='right', fontweight='light', fontsize='small');
sns.pairplot(df, hue="Border");
Port_Pivot = df.pivot_table(index='Port Name', columns='Measure', values='Value')

Port_Pivot.head()
Port_Pivot.columns
Port_most = Port_Pivot[["Buses", "Personal Vehicles"]]

Port_most_6 = Port_most.nlargest(6, 'Personal Vehicles', keep='first')

Port_most_6.plot(kind='pie', subplots=True)

plt.gcf().set_size_inches(20,18)
Port_Pivot['Total Transport'] = Port_Pivot['Bus Passengers'] + Port_Pivot['Train Passengers'] + Port_Pivot['Personal Vehicle Passengers']

Port_most_tr = Port_Pivot[['Total Transport', "Pedestrians"]]
Port_most_tr6 = Port_most_tr.nlargest(6, 'Pedestrians', keep='first')

Port_most_tr6.plot(kind='pie', subplots=True)

plt.gcf().set_size_inches(20,18)
Port_most_tr20 = Port_most_tr.nlargest(20, 'Pedestrians')

ax = sns.barplot(x=Port_most_tr20.axes[0], y=Port_most_tr20['Pedestrians']).set(title='Investigating Top 20 Cities by Pedestrian Crossing', xlabel = 'Cities', ylabel='Number of People Crossing (2010 - 2020)')

plt.xticks(rotation=50, ha='right', fontweight='light', fontsize='small');
Port_most_tr20 = Port_most_tr.nlargest(20, 'Total Transport')

ax = sns.barplot(x=Port_most_tr20.axes[0], y=Port_most_tr20['Total Transport']).set(title='Investigating Top 20 Cities by Passengers Crossing by Buses, Personal Vehicles and Trains', xlabel = 'Cities', ylabel='Number of People Crossing (2010 - 2020)')

plt.xticks(rotation=50, ha='right', fontweight='light', fontsize='small');
Port_stat = df.pivot_table(index='Port Name', values='Value', aggfunc=[np.sum, np.mean])

Port_stat.head()
Port_stat.columns
#Port_stat.columns = Port_stat.columns.get_level_values(1)

Port_stat.columns = [' '.join(col).strip() for col in Port_stat.columns.values]

Port_stat.columns
Port_stat.head()
Port_stat_ten = Port_stat.nlargest(5, 'sum Value')

Port_stat_ten.plot(kind='pie', subplots=True)

plt.gcf().set_size_inches(20,6)

plt.legend(loc="upper left");
Port_stat_20 = Port_stat.nlargest(20, 'sum Value', keep="last")

Port_stat_20.plot(kind='bar')

plt.gcf().set_size_inches(20,8)



plt.xticks(rotation=45, ha='right', fontweight='light', fontsize='small');

plt.title("Top 20 Cities with highest total people crossing from 2010 to 2020", fontsize=14)

plt.ylabel("People Crossing Border")

plt.legend(["Maximum Value", "Mean Value"]);
ax = sns.barplot(x=Port_stat_20.axes[0], y=Port_stat_20['mean Value']).set(title='Investigating Top 20 Cities by average number of people crossing from 2010 to 2020', xlabel = 'Cities', ylabel='Average Number of People Crossing (2010 - 2020)')

plt.xticks(rotation=50, ha='right', fontweight='light', fontsize='small');
Avg_Year_Values = df.groupby(pd.DatetimeIndex(df['Date']).year)["Value"].mean()

Avg_Year_Values.head()
x = Avg_Year_Values.axes[0].values.reshape(-1,1)

y = Avg_Year_Values.values.reshape(-1,1)

print(x.shape)

print(y.shape)
Poly_reg=PolynomialFeatures(degree=4) 

x_poly=Poly_reg.fit_transform(x)

Lin_reg = LinearRegression()

Lin_reg.fit(x_poly,y)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
lin_reg = LinearRegression()

lin_reg.fit(x, y)



plt.figure(figsize=(14, 8))

plt.scatter(x,y)

plt.xlim(1995,2019)

plt.xlabel("Years",fontsize=14)

plt.ylabel("People Crossing Border",fontsize=14)

plt.title("People Crossing Border by Years",fontdict={'fontsize': 18, 'fontweight': 'medium'},color='m')

plt.grid(color='k', linestyle='dotted', linewidth=0.5)

y_pred=Lin_reg.predict(x_poly)

plt.plot(x,y_pred,color="r",label="Polynomial Regression Model")

plt.legend()

plt.show()
# Predicting total number of immigration in n with Polymonial Regression

n = int(input());

Prediction = int(Lin_reg.predict(Poly_reg.fit_transform([[n]])));

print("The Predicted number of people that will cross the border in ", n, "is", Prediction)