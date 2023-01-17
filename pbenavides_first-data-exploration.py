# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mlp

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



df = pd.read_csv('../input/BlackFriday.csv')
#Finding different values per column

for var in list(df.columns):

    print('# of different', var,": ", len(df[var].unique()))  

print('Pero hay', len(df), 'rows')
#May I eliminate the NaN values of the columns? What columns have ones?

for var in df.columns:

    if df[var].isnull().values.any() == True:

        print(var, 'have Null values')
df.head()
print("Mean value: ",df['Purchase'].mean(), "\nStandard deviation:", df['Purchase'].std(),

      "\nMaximun value ", df['Purchase'].max(), "\nMinimun value" , df['Purchase'].min())

plt.hist(df['Purchase'], bins = 700)

plt.xlabel('Compras')

plt.ylabel('Compras en $')

plt.title('Number of purchases vs spended money ')

plt.show()
#Creating a new array counting the number of times that a row repeats for one value of gender per each different User_ID 

count_products = df.groupby('User_ID')['Gender'].count()
#Group by the features wanted and sum the values of each purchase

df_st = df.groupby(['User_ID','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status'])['Purchase'].sum()



df_st = df_st.to_frame()



df_st = df_st.reset_index().set_index('User_ID')

#we agregate the column of new data: Number of products buyed

df_st['Number_of_products_buyed'] = count_products

df_st.head()

#We have our new dataframe!
#Describing our new column: 

print(df_st['Number_of_products_buyed'].mean(),": mean\n", df_st['Number_of_products_buyed'].std(),": standard deviation\n",

      df_st['Number_of_products_buyed'].max(), ": max value\n",

      df_st['Number_of_products_buyed'].min(), ": min value")
print("The measures before were:", df['Purchase'].mean(), ":mean, ", df['Purchase'].std(), "desv. std.")

print("Now the measures are:", df_st['Purchase'].mean(), ":mean, ", df_st['Purchase'].std(), "desv. std")
def automatise_plots(data1,data2,n,y_col='Purchase'): #n is the number of columns that i want to plot

    col = list(data1.columns[0:n-1])

    num = [x for x in range(1,n*2,2)]

    zipped = dict(list(zip(col,num)))

    fig = plt.figure(figsize = (14,24))

    fig.subplots_adjust(hspace=0.6, wspace=0.15)

    for key,value in zipped.items():

        ax = fig.add_subplot(n-1,2,value)

        ax.set_title("Grouped data")

        sns.barplot(x = key, y = y_col, data = data1)

        ax_2 = fig.add_subplot(n-1,2,value+1)

        ax_2.set_title("Not grouped data")

        sns.barplot(x = key, y = y_col, data = data2)
automatise_plots(df_st,df,7)
plt.title('Age vs number of purchases divided by gender')

sns.countplot(df_st['Age'],hue=df_st['Gender'], order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])

#From this graphic we have more clients from 26-35
#Relation btw amount purchase and number of products purchased

plt.title('Number of products vs Purchase')

sns.scatterplot(x='Number_of_products_buyed', y = 'Purchase', data=df_st)