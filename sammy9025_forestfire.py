import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
forestdata=pd.read_csv('../input/forest-fires-data-set/forestfires.csv')

forestdata
forestdata.describe(include='all') #its shows basic statistical characteristics of each numerical feature.

# include all ,consider categorical columns also.
forestdata.head(6)                                # gives top  6 rows of dataset.
forestdata.tail(6)                              #gives last 6 rows of dataset.
forestdata.info()                            # gives general information about dataset.
df1=pd.pivot_table(data=forestdata,values=['rain','temp','wind','RH','area','FFMC','DMC','DC','ISI'],index='month',aggfunc=['mean'])

df1
df1[('mean','rain')].sort_values(ascending=False).head(4)
df1[('mean','temp')].sort_values(ascending=False).head(4)
df1[('mean','wind')].sort_values(ascending=True).head(4)
df1[('mean','RH')].sort_values(ascending=True).head(4)
df1[('mean','DC')].sort_values(ascending=False).head(4)
df1[('mean','DMC')].sort_values(ascending=False).head(4)
df1[('mean','FFMC')].sort_values(ascending=False).head(4)
df1[('mean','ISI')].sort_values(ascending=False).head(4)
# analysis on burned area

plt.figure(figsize=(16,5))

print("Skew: {}".format(forestdata['area'].skew()))

print("Kurtosis: {}".format(forestdata['area'].kurtosis()))

ax = sns.kdeplot(forestdata['area'],shade=True,color='g')

plt.xlabel('Area in hectare',color='red',fontsize=15)

plt.ylabel('probability density of forest fire',color='red',fontsize=15)

plt.title('Forest Fire Probability Density  Vs Amount of Area Burnt',color='blue',fontsize=18)

plt.xticks([i for i in range(0,1200,50)])

plt.show()
dfa = forestdata.drop(columns='area')

cat_columns = dfa.select_dtypes(include='object').columns.tolist()  #seperating categorical columns from data set

num_columns = dfa.select_dtypes(exclude='object').columns.tolist()  #seperating numerical columns from data set
# Analysis of forest fire based on different months and days.

plt.figure(figsize=(16,10))

for i,col in enumerate(cat_columns,1):

    plt.subplot(2,2,i)             #indexing subplot using loop

    sns.countplot(data=dfa,y=col)  #countplot:count of each month/day in month/day columns

    plt.subplot(2,2,i+2)

    forestdata[col].value_counts().plot.bar() #freq of each month/day in month/day columns

    plt.ylabel(col)

    plt.xlabel('% distribution per category')

plt.show()
# Analysis of forest fire damage based on different months and days.

# Adding categorical variable  based on forest fire area as No damage, low, moderate, high, very high

def area_cat(area):            # grouping damage category based on amount of area burned.

    if area == 0.0:

        return "No damage"

    elif area <= 1:

        return "low"

    elif area <= 25:

        return "moderate"

    elif area <= 100:

        return "high"

    else:

        return "very high"



forestdata['damage_category'] = forestdata['area'].apply(area_cat)









for col in cat_columns:      

    cross = pd.crosstab(index=forestdata['damage_category'],columns=forestdata[col],normalize='index')

    cross.plot.barh(stacked=True,rot=40,cmap='plasma')

    plt.xlabel('% distribution per category')

    plt.xticks(np.arange(0,1.1,0.1))

    plt.title("Forestfire damage each {}".format(col))

plt.show()
# Analysis of Burnt area based on spatial cordinates(X,Y)

forestdata.plot(kind='scatter', x='X', y='Y', alpha=0.2, s=20*forestdata['area'],figsize=(10,6))

plt.xlabel('X cordinates of park',color='red',fontsize=15)

plt.ylabel('Y cordinates of park',color='red',fontsize=15)

plt.title('Burnt area in different regions of the park',color='blue',fontsize=18)
# monthly analysis of burnt area, where the condition is: area>0

areaburnt=forestdata[forestdata['area']>0]

areaburnt
areaburnt.groupby('month')['area'].agg('count').plot(kind='pie',title='Monthly analysis of burnt area',figsize=(9,9),explode=[0,0.1,0,0,0,0,0,0,0,0.1],autopct='%0.1f%%')

plt.show()