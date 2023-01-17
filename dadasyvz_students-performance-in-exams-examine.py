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

data.columns=['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch',

       'test_preparation_course', 'math_score', 'reading_score',

       'writing_score']



data.columns
data.corr()
f,ax = plt.subplots(figsize=(16, 16))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.math_score.plot(kind = 'line', color = 'g',label = 'math score',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.reading_score.plot(color = 'r',label = 'reading score',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 



data.plot(kind='scatter', x='math_score', y='reading_score',alpha = 0.5,color = 'red')

plt.xlabel('math_score')              # label = name of label

plt.ylabel('reading_score')

plt.title('math_score reading_score Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.math_score.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.reading_score.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

data.columns=['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch',

       'test_preparation_course', 'math_score', 'reading_score',

       'writing_score']

series = data['math_score']        # data['math_score'] = series

print(type(series))

data_frame = data[['math_score']]  # data[['math_score']] = data frame

print(type(data_frame))



# 1 - Filtering Pandas data frame

x = data['math_score']>90     # 

print(x)

data[x]
 #2 - Filtering pandas with logical_and

data[np.logical_and(data['math_score']>70, data['reading_score']>90 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

data[(data['math_score']>80) & (data['reading_score']>90)]
threshold = sum(data.math_score)/len(data.math_score)

data["new_math_score"] = ["high" if i > threshold else "middle" if  i>45 else "low" for i in data.math_score]

data.loc[:10,["new_math_score","math_score"]] # we will learn loc more detailed later

data
data.info
data.describe()
data['parental_level_of_education'].value_counts(dropna =False)
data.describe()
data['lunch'].describe()
data.dropna(inplace = True)  

data.describe()
data.boxplot(column='math_score',by = 'reading_score')
data_new = data.head()    # I only take 5 rows into new data

data_new
melted = pd.melt(frame=data_new,id_vars = 'parental_level_of_education', value_vars= ['math_score','reading_score'])

melted
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data3 = data['math_score'].head()

data1 = data['reading_score'].head()

data2= data['writing_score'].head()

conc_data_col = pd.concat([data3,data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
data['lunch'] = data['lunch'].astype('category')

data['math_score'] = data['math_score'].astype('category')

data.dtypes


plt.figure(figsize=(15,10))

p = sns.countplot(x="math_score", data = data, palette="muted")

_ = plt.setp(p.get_xticklabels(), rotation=90) 

plt.title("math score count")

passmark = 40 #we will set the minimum marks to 40 to pass in a exam

data.math_score=data.math_score.astype(float)

data['Math_PassStatus'] = np.where(data['math_score']<passmark, 'F', 'P')

data.Math_PassStatus.value_counts()

data.head()

plt.figure(figsize=(10,8))

p = sns.countplot(x='parental_level_of_education', data = data, hue='Math_PassStatus', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 

plt.figure(figsize=(15,10))

data.reading_score=data.reading_score.astype(float)

sns.countplot(x="reading_score", data = data, palette="muted")

plt.xticks(rotation=90)

plt.show()
data['Reading_PassStatus'] = np.where(data['reading_score']<passmark, 'F', 'P')

plt.figure(figsize=(10,8))

p = sns.countplot(x='parental_level_of_education', data = data, hue='Reading_PassStatus', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 
data.writing_score=data.writing_score.astype(float)

data['Writing_PassStatus'] = np.where(data['writing_score']<passmark, 'F', 'P')



#How many students passed in all the subjects ?

data['OverAll_PassStatus'] = data.apply(lambda x : 'F' if x['Math_PassStatus'] == 'F' or 

                                    x['Reading_PassStatus'] == 'F' or x['Writing_PassStatus'] == 'F' else 'P', axis =1)



data.OverAll_PassStatus.value_counts()

#Find the percentage of marks

data['Total_Marks'] = data['math_score']+data['reading_score']+data['writing_score']

data['Percentage'] = data['Total_Marks']/3
def GetGrade(Percentage, OverAll_PassStatus):

    if ( OverAll_PassStatus == 'F'):

        return 'F'    

    if ( Percentage >= 80 ):

        return 'A'

    if ( Percentage >= 70):

        return 'B'

    if ( Percentage >= 60):

        return 'C'

    if ( Percentage >= 50):

        return 'D'

    if ( Percentage >= 40):

        return 'E'

    else: 

        return 'F'



data['Grade'] = data.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)



data.Grade.value_counts()
# plot the grades obtained in a order

plt.figure(figsize=(10,8))

plt.subplot(2,1,1)

sns.countplot(x="Grade", data = data, order=['A','B','C','D','E','F'],  palette="muted")

plt.subplot(2,1,2)



p = sns.countplot(x='parental_level_of_education', data = data, hue='Grade', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 

plt.show()

data.head()
data.Grade.dropna(inplace = True)

labels = data.Grade.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = data.Grade.value_counts().values

# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title("According to grades, pie chart",color = 'blue',fontsize = 15)


f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='parental_level_of_education',y='reading_score',data=data,color='lime',alpha=0.8)

sns.pointplot(x='parental_level_of_education',y='writing_score',data=data,color='red',alpha=0.8)

sns.pointplot(x='parental_level_of_education',y='math_score',data=data,color='blue',alpha=0.8)



plt.text(40,0.6,'writing_score',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'reading_score',color='lime',fontsize = 18,style = 'italic')

plt.text(40,0.5,'math_score',color='lime',fontsize = 18,style = 'italic')



plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('Reading, writting and math scores According to parental level of education',fontsize = 20,color='blue')

plt.grid()




# Visualization of score on the Percentage



g = sns.jointplot(data.math_score, data.Percentage, kind="scatter", size=7)

g = sns.jointplot(data.reading_score, data.Percentage, kind="scatter", size=7)

g = sns.jointplot(data.writing_score, data.Percentage, kind="scatter", size=7)



plt.savefig('graph.png')

plt.show()



plt.figure(figsize=(10,8))

sns.swarmplot(x="gender", y="Percentage",hue="parental_level_of_education", data=data)

plt.show()