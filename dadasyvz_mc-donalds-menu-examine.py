# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/nutrition-facts/menu.csv')

data.info()

data.columns=['Category', 'Item', 'Serving_Size', 'Calories', 'Calories_from_Fat',

       'Total_Fat', 'Total_Fat (% Daily Value)', 'Saturated_Fat',

       'Saturated Fat (% Daily Value)', 'Trans_Fat', 'Cholesterol',

       'Cholesterol (% Daily Value)', 'Sodium', 'Sodium (% Daily Value)',

       'Carbohydrates', 'Carbohydrates (% Daily Value)', 'Dietary Fiber',

       'Dietary Fiber (% Daily Value)', 'Sugars', 'Protein',

       'Vitamin A (% Daily Value)', 'Vitamin C (% Daily Value)',

       'Calcium (% Daily Value)', 'Iron (% Daily Value)']

data.corr()
f,ax = plt.subplots(figsize=(16, 16))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Calories.plot(kind = 'line', color = 'g',label = 'Calories',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Calories_from_Fat.plot(color = 'r',label = 'Calories_from_Fat',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 



data.plot(kind='scatter', x='Calories', y='Calories_from_Fat',alpha = 0.5,color = 'red')

plt.xlabel('Calories')              # label = name of label

plt.ylabel('Calories_from_Fat')

plt.title('Calories Calories_from_Fat Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.Calories.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.Calories_from_Fat.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
data = pd.read_csv('/kaggle/input/nutrition-facts/menu.csv')

data.columns=['Category', 'Item', 'Serving_Size', 'Calories', 'Calories_from_Fat',

       'Total_Fat', 'Total_Fat (% Daily Value)', 'Saturated_Fat',

       'Saturated Fat (% Daily Value)', 'Trans_Fat', 'Cholesterol',

       'Cholesterol (% Daily Value)', 'Sodium', 'Sodium (% Daily Value)',

       'Carbohydrates', 'Carbohydrates (% Daily Value)', 'Dietary Fiber',

       'Dietary Fiber (% Daily Value)', 'Sugars', 'Protein',

       'Vitamin A (% Daily Value)', 'Vitamin C (% Daily Value)',

       'Calcium (% Daily Value)', 'Iron (% Daily Value)']

series = data['Calories']        # data['Calories'] = series

print(type(series))

data_frame = data[['Calories_from_Fat']]  # data[['Calories_from_Fat']] = data frame

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['Calories']>200     

print(x)

data[x]
 #2 - Filtering pandas with logical_and



data[np.logical_and(data['Calories']>300, data['Calories_from_Fat']>200 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['Calories_from_Fat']>200) & (data['Calories']>100)]
threshold = sum(data.Calories_from_Fat)/len(data.Calories_from_Fat)

data["new_calories"] = ["high" if i > threshold else "middle" if  i>45 else "low" for i in data.Calories_from_Fat]

data.loc[:10,["new_calories","Calories_from_Fat"]] # we will learn loc more detailed later

data
data.info()
data.describe()
print(data['Calories'].value_counts(dropna =False))
data['Calories'].describe
data.dropna(inplace = True)  

data.describe()
data.boxplot(column='Calories_from_Fat',by = 'Total_Fat')
data_new = data.head()    # I only take 5 rows into new data

data_new
melted = pd.melt(frame=data_new,id_vars = 'Category', value_vars= ['Calories_from_Fat','Total_Fat'])

melted
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data3 = data['Category'].head()

data1 = data['Calories_from_Fat'].head()

data2= data['Total_Fat'].head()

conc_data_col = pd.concat([data3,data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
data['Category'] = data['Category'].astype('category')

data['Calories_from_Fat'] = data['Calories_from_Fat'].astype('category')

data.dtypes
data.info()
data.Category.value_counts()
miktar = ['Trans_Fat','Calories', 'Total_Fat', 'Cholesterol','Sodium', 'Sugars', 'Carbohydrates']



for x in miktar:   

    plot = sns.swarmplot(x="Category", y=x, data=data)

    plt.setp(plot.get_xticklabels(), rotation=45)

    plt.title(x)

    plt.show()
data.head()
sns.barplot(x=data.groupby('Category')['Calories'].mean().index,y=data.groupby('Category')['Calories'].mean().values)

plt.title("total calories of each menu")



plt.ylabel("calories")

plt.xlabel("menu")

plt.xticks(rotation=45)

plt.show()