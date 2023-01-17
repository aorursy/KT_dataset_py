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
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

data.info()

print(data)
data.describe()
data.corr()
f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

data.columns
s=data['math score']

a=s.sort_values()

t=0

for i in a:

    b=s.max()

    if i== b:

        t+=1

print(a)

print(b)

print(t)

data.head()
data = data.rename(columns={'gender':'gender', 'race/ethnicity':'race/ethnicity','parental level of education': 'parental_level_of_education', 

                        'test preparation course': 'test_preparation_course','math score':'math_score',

                       'reading score':'reading_score','writing score':'writing_score'})

# Or rename the existing DataFrame (rather than creating a copy) 

data.rename(columns={'gender':'gender', 'race/ethnicity':'race/ethnicity','parental level of education': 'parental_level_of_education', 

                        'test preparation course': 'test_preparation_course','math score':'math_score',

                       'reading score':'reading_score','writing score':'writing_score'}, inplace=True)

data.math_score.plot(kind = 'line', color = 'g',label = 'math_score',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.reading_score.plot(color = 'r',label = 'reading_score',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()

data.plot(kind='scatter', x='writing_score', y='math_score',alpha = 0.5,color = 'red')

plt.xlabel('writing_score')              # label = name of label

plt.ylabel('math_score')

plt.title('writing_score math_score Scatter Plot')            # title = title of plot

plt.show()
data.math_score.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
a=data[(data['gender']!='female') & (data['math_score']>85) & (data['writing_score']>85) & (data['reading_score']>85)]

print(a)

e=a.count()

print(e)

b=data[(data['gender']!='male') & (data['math_score']>85) & (data['writing_score']>85) & (data['reading_score']>85)]

k=b.count()

print(k)
threshold=sum(data.math_score)/len(data.math_score)

data["math_score_level"]=["successful" if i>threshold else "unsuccessful" for i in data.math_score]

data.loc[:10,["math_score_level","math_score"]]
print(data)
print(data['gender'].value_counts(dropna=False))
data.dropna(inplace=True)

data.describe()
data.boxplot(column='math_score',by = 'gender')




melted = pd.melt(frame=data, id_vars = 'race/ethnicity', value_vars = ['test_preparation_course','lunch'])

melted
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
gender_list=list(data['gender'].unique())

success_male=[]

for i in gender_list:

    x = data[data['gender'] == i]

    math_male = sum (x.math_score)/len(x)

    success_male.append(math_male)

print("erkeklerin matematik scorunun ortalamasi:", math_male)

datam = pd.DataFrame({'gender_list' : x.gender , 'success_gender': x.math_score , 'parental_level_of_education': x.parental_level_of_education , 

                      'race/ethnicity' : x['race/ethnicity']  })

new_index = (datam['success_gender'].sort_values(ascending = False).index.values)

sorted_datam = datam.reindex(new_index)

print(sorted_datam)

print(sorted_datam.parental_level_of_education.head(10))

print(sorted_datam['race/ethnicity'].head(10))



#sorted_datam.gender_list.value_counts()
gender_list=list(data['gender'].unique())

success_female=[]

for i in gender_list:

    if i == 'male' :

        continue

    x = data[data['gender'] == i]

    math_female = sum (x.math_score)/len(x)

    success_female.append(math_female)

    

print(math_female)

dataf = pd.DataFrame({'gender_list' : x.gender , 'success_gender': x.math_score , 'parental_level_of_education': x.parental_level_of_education , 

                      'race/ethnicity' : x['race/ethnicity']  })

new_index = (dataf['success_gender'].sort_values(ascending = False).index.values)

sorted_dataf = dataf.reindex(new_index)

print(sorted_dataf)

print(sorted_dataf.parental_level_of_education.head(10))

print(sorted_dataf['race/ethnicity'].head(10))

plt.subplot(2,1,1)

plt.plot(sorted_dataf.success_gender, color = 'red' , label = 'success of females')

plt.ylabel('female')

plt.subplot(2,1,2)

plt.plot(sorted_datam.success_gender, color = 'green' , label = 'success of males')

plt.ylabel('male')

plt.xlabel('female and male comparision')
#f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=sorted_dataf.success_gender,y=sorted_dataf.parental_level_of_education,color='green',alpha = 0.5,label='parent of female education' )

plt.show()

sns.barplot(x=sorted_dataf.success_gender,y=sorted_dataf['race/ethnicity'],color='blue',alpha = 0.7,label='race of female')

plt.show()

sns.barplot(x=sorted_datam.success_gender,y=sorted_datam.parental_level_of_education,color='cyan',alpha = 0.6,label='parent of male education')

plt.show()

sns.barplot(x=sorted_datam.success_gender,y=sorted_datam['race/ethnicity'],color='yellow',alpha = 0.6,label='race of male')

plt.show()





#ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu

#ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")
x=sorted_dataf.success_gender

y=sorted_dataf.parental_level_of_education

plt.bar(x,y, color='green')

plt.title('female parental education', weight='bold')

plt.xlabel('female',labelpad=10, size=12)

plt.ylabel('education',size=12)

plt.show()
x=sorted_datam.success_gender

y=sorted_datam.parental_level_of_education

plt.bar(x,y, color='green')

plt.title('male parental education', weight='bold')

plt.xlabel('male',labelpad=10, size=12)

plt.ylabel('education',size=12)

plt.show()