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
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
sns.set_style('darkgrid')
# reading the csv file
df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
#computing the mean score of the test and assigning it to a new column
df['mean test score'] = df[['math score','reading score','writing score']].mean(axis = 1)
df.describe(include = 'all')
#splitting the dataset to compare males with females
female_df = df[df.gender == 'female']
male_df = df[df.gender == 'male']
plt.figure(figsize = (10,8))
sns.distplot(female_df['mean test score'], kde= False, label = 'Females')
sns.distplot(male_df['mean test score'], kde = False, label = 'Males')
plt.title('Comparison of males mean test scores vs females mean test scores', size = 16, weight = 'bold')
plt.legend()

print('Females mean test scores (avg): {:.1f}'.format(female_df['mean test score'].mean()))
print('Males mean test scores (avg): {:.1f}'.format(male_df['mean test score'].mean()))
#comparing math scores between males and females
plt.figure(figsize = (10,8))
sns.distplot(female_df['math score'], kde= False, label = 'Females')
sns.distplot(male_df['math score'], kde = False, label = 'Males')
plt.title('Comparison of males math scores vs females math score', size = 16, weight = 'bold')
plt.legend()

print('Females mean math score: {:.1f}'.format(female_df['math score'].mean()))
print('Males mean math score: {:.1f}'.format(male_df['math score'].mean()))
#Comparing reading scores between males and females
plt.figure(figsize = (10,8))
sns.distplot(female_df['reading score'], kde= False, label = 'Females')
sns.distplot(male_df['reading score'], kde = False, label = 'Males')
plt.title('Comparison of males reading scores vs females reading score', size = 16, weight = 'bold')
plt.legend()

print('Females mean reading score: {:.1f}'.format(female_df['reading score'].mean()))
print('Males mean reading score: {:.1f}'.format(male_df['reading score'].mean()))
#Comparing writing scores between males and females
plt.figure(figsize = (10,8))
sns.distplot(female_df['writing score'], kde= False, label = 'Females')
sns.distplot(male_df['writing score'], kde = False, label = 'Males')
plt.title('Comparison of males writing scores vs females writing score', size = 16, weight = 'bold')
plt.legend()

print('Females mean writing score: {:.1f}'.format(female_df['writing score'].mean()))
print('Males mean writing score: {:.1f}'.format(male_df['writing score'].mean()))
#plotting test preparation course status: females vs males
plt.figure(figsize = (10,8))
sns.barplot(data = df.groupby(['gender','test preparation course'], as_index = False)['lunch'].\
count().rename(columns = {'lunch':'count'}),x = 'test preparation course', y= 'count', hue = 'gender')
plt.title("Test preparation course status:\ncomparison between genders", size = 16, weight = 'bold')

print("Precentage of females that took the test preparation course: {:.1f}%".\
      format(100*len(female_df[female_df['test preparation course']=='completed'])/len(female_df)))

print("Precentage of males that took the test preparation course: {:.1f}%".\
      format(100*len(male_df[male_df['test preparation course']=='completed'])/len(male_df)))
plt.figure(figsize = (10,8))
sns.boxplot(data = df, x = 'gender', y = 'mean test score', hue = 'test preparation course')
plt.title("Test preparation course status:\nimpact on mean test score", size = 16, weight = 'bold')
plt.figure(figsize = (10,8))
sns.barplot(data = df.groupby(['gender','lunch'], as_index = False)['test preparation course'].\
count().rename(columns = {'test preparation course':'count'}),x = 'lunch', y= 'count', hue = 'gender')
plt.title("Lunch status:\ncomparison between genders", size = 16, weight = 'bold')

print("Precentage of females that took the test preparation course: {:.1f}%".\
      format(100*len(female_df[female_df['lunch']=='standard'])/len(female_df)))

print("Precentage of males that took the test preparation course: {:.1f}%".\
      format(100*len(male_df[male_df['lunch']=='standard'])/len(male_df)))
plt.figure(figsize = (10,8))
sns.boxplot(data = df, x = 'gender', y = 'mean test score', hue = 'lunch')
plt.title("Lunch status:\nimpact on mean test score", size = 16, weight = 'bold')
plt.figure(figsize = (10,8))
sns.barplot(data = df.groupby(['gender','parental level of education'], as_index = False)['lunch'].\
count().rename(columns = {'lunch':'count'}),x = 'parental level of education', y= 'count', hue = 'gender')
plt.title("Parental level of education:\ncomparison between genders", size = 16, weight = 'bold')

print("Precentage of females with parents having an associate's degree: {:.1f}%".\
      format(100*len(female_df[female_df['parental level of education']=="associate's degree"])/len(female_df)))

print("Precentage of males with parents having an associate's degree: {:.1f}%".\
      format(100*len(male_df[male_df['parental level of education']=="associate's degree"])/len(male_df)))

print("Precentage of females with parents having a bachelor's degree: {:.1f}%".\
      format(100*len(female_df[female_df['parental level of education']=="bachelor's degree"])/len(female_df)))

print("Precentage of males with parents having a bachelor's degree: {:.1f}%".\
      format(100*len(male_df[male_df['parental level of education']=="bachelor's degree"])/len(male_df)))

print("Precentage of females with parents having high school level: {:.1f}%".\
      format(100*len(female_df[female_df['parental level of education']=="high school"])/len(female_df)))

print("Precentage of males with parents having high school level: {:.1f}%".\
      format(100*len(male_df[male_df['parental level of education']=="high school"])/len(male_df)))

print("Precentage of females with parents having a master's degree: {:.1f}%".\
      format(100*len(female_df[female_df['parental level of education']=="master's degree"])/len(female_df)))

print("Precentage of males with parents having a master's degree: {:.1f}%".\
      format(100*len(male_df[male_df['parental level of education']=="master's degree"])/len(male_df)))

print("Precentage of females with parents having some college: {:.1f}%".\
      format(100*len(female_df[female_df['parental level of education']=="some college"])/len(female_df)))

print("Precentage of males with parents having some college: {:.1f}%".\
      format(100*len(male_df[male_df['parental level of education']=="some college"])/len(male_df)))

print("Precentage of females with parents having some high school: {:.1f}%".\
      format(100*len(female_df[female_df['parental level of education']=="some high school"])/len(female_df)))

print("Precentage of males with parents having some high school: {:.1f}%".\
      format(100*len(male_df[male_df['parental level of education']=="some high school"])/len(male_df)))
plt.figure(figsize = (10,8))
sns.boxplot(data = df, x = 'gender', y = 'mean test score', hue = 'parental level of education')
plt.title("Parental level of education:\nimpact on mean test score", size = 16, weight = 'bold')
plt.figure(figsize = (10,8))
sns.barplot(data = df.groupby(['gender','race/ethnicity'], as_index = False)['lunch'].\
count().rename(columns = {'lunch':'count'}),x = 'race/ethnicity', y= 'count', hue = 'gender')
plt.title("Race/ethnicity:\ncomparison between genders", size = 16, weight = 'bold')

# print("Precentage of females that took the test preparation course: {:.1f}%".\
#       format(100*len(female_df[female_df['test preparation course']=='completed'])/len(female_df)))

# print("Precentage of males that took the test preparation course: {:.1f}%".\
#       format(100*len(male_df[male_df['test preparation course']=='completed'])/len(male_df)))
#Importing a model to detect clusters in the data
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

X = df[["gender","race/ethnicity","parental level of education","lunch","test preparation course"]]
X = pd.get_dummies(X)

dict_params = {'n_clusters':[8,9,10,11,12,13,14,15,16,17,18],
               'max_iter':[25,50,100,150],
               'algorithm':['full','auto'],
               'random_state':[1234]}
model = KMeans()

search = GridSearchCV(estimator = model,param_grid = dict_params,cv = 5,n_jobs = -1).fit(X)
model1 = search.best_estimator_
y=model1.predict(X)
print(search.best_params_, search.best_score_)
#Creating a new column to assign the labels of clusters
df['clusters'] = np.nan

for i in range(19):
    df.loc[df[y==i].index,'clusters'] = i
#Comparing the different clusters mean test scores

plt.figure(figsize = (10,8))
sns.violinplot(data = df,x = 'clusters', y = 'mean test score')
plt.title("Comparison of clusters' mean test scores and\nstandard deviations", size = 16, weight = 'bold')

for i in range(18):
    print('Cluster '+str(int(i))+' mean test score (avg): {:.2f}, and standard deviation: {:.2f}'.format(df[df.clusters == i]['mean test score'].mean(),
                                                                                               df[df.clusters == i]['mean test score'].std()))
#Creating lists that later will be used to create small dataframes
variables = ["gender","race/ethnicity","parental level of education","lunch","test preparation course"]
var0 = []
quant0 = []
var1 = []
quant1 = []

for k in range(5):
    for i in variables:
        for j in df[df.clusters==k][i]:
            if k == 0:
                if j not in var0:
                    var0.append(j)
                    quant0.append(1)
                else:
                    quant0[var0.index(j)]+=1
            elif k == 1:
                if j not in var1:
                    var1.append(j)
                    quant1.append(1)
                else:
                    quant1[var1.index(j)]+=1
#Creating dataframes according to clusters to plot their features for comparison
s0 = pd.Series(var0, quant0)
s0 = s0.reset_index(drop = False).rename(columns = {'index':'count',0:'variables'})

s1 = pd.Series(var1, quant1)
s1 = s1.reset_index(drop = False).rename(columns = {'index':'count',0:'variables'})

plt.style.use('default')
plt.figure(figsize = (18,11))

ax0 = plt.subplot(1,2,1)
bar0=s0.plot(kind = 'bar', x = 'variables', y = 'count', legend = False, ax = ax0, color = ['red','blue','blue','yellow','yellow','yellow','yellow',
                                                                                            'yellow','yellow','green','purple'])
plt.legend(handles = [Patch(facecolor = 'red',label = 'gender'),
                      Patch(facecolor = 'blue', label = 'ethnicity'),
                      Patch(facecolor = 'yellow', label = 'parents educ. level'),
                      Patch(facecolor = 'green',label = 'lunch'),
                      Patch(facecolor = 'purple', label = 'test prep.course')])

for k,i in enumerate(s0['count']):
    plt.annotate(s = str(round(100*(i/s0['count'].max())))+'%', xy = (k-0.3,i+0.1))
    
plt.title("Cluster 0, mean test score: {:.2f}".format(df[y==0]['mean test score'].mean()),size = 16, weight = 'bold')

ax1 = plt.subplot(1,2,2)
bar1=s1.plot(kind = 'bar', x = 'variables', y = 'count', legend = False, ax = ax1, color = ['red','blue','blue','blue','blue','yellow','yellow','yellow',
                                                                                            'yellow','yellow','yellow','green','purple'])

plt.legend(handles = [Patch(facecolor = 'red',label = 'gender'),
                      Patch(facecolor = 'blue', label = 'ethnicity'),
                      Patch(facecolor = 'yellow', label = 'parents educ. level'),
                      Patch(facecolor = 'green',label = 'lunch'),
                      Patch(facecolor = 'purple', label = 'test prep.course')])

for k,i in enumerate(s1['count']):
    plt.annotate(s = str(round(100*(i/s1['count'].max())))+'%', xy = (k-0.3,i+0.1))

plt.title("Cluster 1, mean test score: {:.2f}".format(df[y==1]['mean test score'].mean()),size = 16, weight = 'bold')
