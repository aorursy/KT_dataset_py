# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))



df = pd.read_csv('../input/who-suicide-statistics/who_suicide_statistics.csv')
df.groupby(['country','age']).suicides_no.sum().nlargest(10).plot(kind='barh')
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from numpy import median

ax = sns.catplot(x="sex", y="suicides_no",col='age', data=df, estimator=median,height=4, aspect=.7,kind='bar')
df.groupby(by=['age','sex'])['suicides_no'].sum().unstack().reset_index().melt(id_vars='age')
df.groupby(by=['country'])['suicides_no'].sum().reset_index().sort_values(['suicides_no'],ascending=True).tail(15).plot(x='country',y='suicides_no',kind='barh')
df.groupby(['country','age']).suicides_no.sum().nlargest(10).plot(kind='barh')
from numpy import median

ax = sns.catplot(x="sex", y="suicides_no",col='age', data=df, estimator=median,height=4, aspect=.7,kind='bar')
df.shape
df.columns
def create_list_number_suicides(name_column, list_unique):

    # list_unique = df[name_column].unique()

    

    i = 0

    

    list_number = list()

    

    while i < len(list_unique):

        list_number.append(len(df.loc[df[name_column] == list_unique[i]]))

        i += 1

    

    return list_unique, list_number
def pie_plot(list_number, list_unique):

    plt.figure(figsize=(20,10))

    plt.pie(list_unique, 

        labels=list_number,

        autopct='%1.1f%%', 

        shadow=True, 

        startangle=140)

 

    plt.axis('equal')

    plt.show()

    return 0
def bar_chart(list_number, list_unique):

    objects = list_unique

    y_pos = np.arange(len(objects))

    performance = list_number

 

    plt.figure(figsize=(20,10))    

    plt.bar(y_pos, performance, align='center', alpha=0.5)

    plt.xticks(y_pos, objects)

    plt.ylabel('Number') 

    plt.show()

    

    return 0
list_unique_year, list_number_year = create_list_number_suicides('year',df['year'].unique())
pie_plot(list_unique_year, list_number_year)
s = pd.crosstab(index=df.country,columns=df.year,values=df.suicides_no,aggfunc='sum')



sns.heatmap(s.loc[:,2011:2015].sort_values(2015, ascending=False).dropna().head(5),annot=True)



ss = pd.crosstab(index=df.country,columns=df.year,values=df.population,aggfunc='sum')



sdivss = s/ss*10000



#sns.heatmap(sdivss.loc[:,2006:2015].sort_values(2015, ascending=False).dropna().head(5),annot=True)