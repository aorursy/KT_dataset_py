# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#dataiteam1@gmail.com adresine mail at!!!

#public ve commit yapman lazım oluşacak linki atacaksın
'''This data set includes the marks secured by the students in high school students from the United States.'''

'''There are 1000 student in this dataset which are represented with 8 different data columns.'''

'''All the students are grouped in five due to race/ethnicity. The data includes the parental level of education of the whole students. Also math, reading, writing scores are listed for each individual. '''



df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

df.info()

#correlation mapping of whole students:

df.corr()

f,ax = plt.subplots(figsize =(10,6))

sns.heatmap(df.corr(),annot = True, linewidth = 1.5, fmt='.2f', ax = ax)

plt.show()



'''The heatmapping gives the correlation between the math,reading and writing scores of the whole students. 

1 refers a max correlation(linear connection) between chosen data sets.

Here, the correlation between reading score and writing score is 0.95 which means there is a strong correlation between these two scores. 

Success in reading ability may also improve one's ability of writing effectively. '''
#The first 10 entries for the students.

df.head()
df.columns = ['gender', 'race', 'parentDegree', 'lunch', 'course', 'mathScore', 'readingScore', 'writingScore'] 

# LINE PLOTTING 

df.mathScore.plot(kind = 'line',color = 'g', label = 'mathScore', linewidth =' 1',

               alpha = 0.5,grid = True, linestyle ='--')



plt.legend(loc = 'lower right') # başlığı nereye yazacağını gösterir

plt.ylabel('MathScore')

plt.title('Line Plot')

plt.show()



df.readingScore.plot(kind = 'line',color = 'b', label = 'readingScore', linewidth =' 1',

               alpha = 0.5,grid = True, linestyle ='-.')

plt.legend(loc = 'lower right') # başlığı nereye yazacağını gösterir

plt.ylabel('ReadingScore')

plt.title('Line Plot')

plt.show()



df.writingScore.plot(kind = 'line',color = 'y', label = 'writingScore', linewidth =' 1',

               alpha = 0.5,grid = True, linestyle ='-.')

plt.legend(loc = 'lower right') # başlığı nereye yazacağını gösterir

plt.ylabel('WritingScore')

plt.title('Line Plot')

plt.show()

#averaging scores of all students



df['average'] = (df['mathScore']+df['readingScore']+df['writingScore'])/3



df.average.plot(kind = 'line',color = 'b', label = 'average', linewidth =' 1',

               alpha = 0.5,grid = True, linestyle ='solid')

plt.legend(loc = 'lower right') # başlığı nereye yazacağını gösterir

plt.ylabel('AverageScore')

plt.title('Line Plot')

plt.show()

#Scatter Plotting

df.plot(kind = 'scatter', x = 'readingScore', y = 'writingScore',

          color = 'r', linewidth =' 1')

plt.legend(loc = 'upper right') 

plt.ylabel('writingScore')

plt.xlabel('readingScore')

plt.title('readingScore vs writingScore')

plt.show()



df.plot(kind = 'scatter', x = 'mathScore', y = 'writingScore',

          color = 'b', linewidth =' 1')

plt.legend(loc = 'upper right') 

plt.ylabel('mathScore')

plt.xlabel('readingScore')

plt.title('readingScore vs mathScore')

plt.show()
df.groupby(['race','parentDegree']).mean()

#When we compare Parent's Degree due to Scores:

course_race = df.groupby(['race','parentDegree']).mean().reset_index()

sns.factorplot(x='race', y='average', hue='parentDegree', data=course_race, kind='bar', size = 8)



'''The following chart decribes the distribution of parental Degree in groups. The group E has highest average scores due to parental degree compared to others.'''
# Grouping due to gender of the students

df.groupby(['gender']).mean()



course_gender = df.groupby(['gender','course']).mean().reset_index()

sns.factorplot(x='gender', y='average', hue='course', data=course_gender, kind='bar')



'''The following chart gives the average of scores due to number of courses completed/ notfinished over gender. There is not so much difference due to gender on the average scores of the courses but the female students have slightly much successfull about the course completion'''

#When we compare due to Parent's Degree 

parent_degree = df.groupby(['race','parentDegree']).mean().reset_index()

sns.factorplot(x='parentDegree', y='average', hue='race', data=course_race, kind='bar', size = 10)   



'''The following plotting shows that the groups average scores distribution due to parental educational levels. Highest educational level is master degree and Group E has the highest average scores due to parental education level in all in all.

The lowest parental education level is high school and even at this range Group E has the highest scores due to parental degree. '''
#Sorting due to average values of students

final_df = df.groupby(['gender','parentDegree','course','lunch','race']).mean().reset_index()

after_sort = final_df.sort_values(by= ['average'],ascending = False)

after_sort.drop(columns=['mathScore','readingScore','writingScore','lunch'],inplace = True)

after_sort



'''This table shows the ranking of whole students due to average scores. The first three students with the highest scores are female and their Parental Degree is bachelor or masters. 

This conclusion coincides with the above plottings. These students also have completed the courses and two thirds of the first three students are from Group E.'''
#Sorting due to average values of students

final_df = df.groupby(['mathScore','readingScore','writingScore','lunch']).mean().reset_index()

after_sort = final_df.sort_values(by= ['average'],ascending = False)

after_sort



'''The following table indicates that the highest scores on each separate lecture is strongly related to the lunch.

The highest rank students in each individual scores have their lunch. The lowest ranks belong to the ones who do not have lunch. '''
print("In conclusion the top highest rank students Performance belong to the ones who completed their courses and have regular lunch: \n",after_sort[:10])
print("Just the Opposite,Bottom rank students Performance is related to the ones do not have lunch and the ones do not complete their courses. \n",after_sort[-10:][::-1])
final_df = df.groupby(['average','parentDegree','lunch']).mean().reset_index()

after_sort = final_df.sort_values(by= ['average'],ascending = False)

after_sort.drop(columns=['mathScore','readingScore','writingScore'],inplace = True)

#after_sort

"  "

print("The top highest rank students Performance : High ParentDegree +  lunch: \n",after_sort[:10])

print("The top lowest rank students Performance :  Low ParentDegree  +  reduced lunch: \n",after_sort[-10:][::-1])