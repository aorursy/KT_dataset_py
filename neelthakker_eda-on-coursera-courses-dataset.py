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
import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS
data = pd.read_csv('/kaggle/input/coursera-course-dataset/coursea_data.csv')
data.head()
data.info()
#Dropping unwanted columns

data.drop('Unnamed: 0', axis=1, inplace = True)
data['course_students_enrolled'] = data['course_students_enrolled'].str.replace('k','*1000')

data['course_students_enrolled'] = data['course_students_enrolled'].str.replace('m','*1000000')

data['course_students_enrolled'] = data['course_students_enrolled'].map(lambda x: eval(x))
data.head()
# TOP 10 ORGANISATIONS WITH MAXIMUM NUMBER OF COURSES



x= data['course_organization'].value_counts(ascending=False)

a=x[:10]

a.plot(kind="bar", figsize=(15,10))

plt.title('TOP 10 ORGANISATIONS WITH MAXIMUM NUMBER OF COURSES OFFERED')

plt.xlabel('Organisations Name')

plt.ylabel('Number of Courses')
plt.figure(figsize=(15,10)) 

sns.countplot(x='course_rating', data=data)

plt.xlabel('Course Rating')

plt.ylabel('Number of Courses')

plt.title('NUMBER OF COURSES FOR DIFFERENT RATINGS')  
colors =  ["#DF6589FF", "#76528BFF","#FC766AFF"]

ax= data['course_Certificate_type'].value_counts(ascending=False).plot.pie(colors=colors,

            autopct='%1.1f%%',

            figsize=(15, 10))

plt.title('COURSE CERTIFICATION TYPE') 

plt.show()
large=data.nlargest(10, ['course_students_enrolled'])





plt.figure(figsize=(30,15))

g = sns.barplot(x="course_title" ,y="course_students_enrolled",hue="course_rating",data=large)

plt.xlabel('Course Titles')

plt.ylabel('Number of Students Enrolled')

plt.title("COURSES WITH MOST STUDENTS ENROLLMENT AND THEIR RATINGS")





for p in g.patches:

    

    g.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.figure(figsize=(15,10)) 

sns.countplot(x='course_difficulty', data=data)

plt.xlabel('Course Difficulty')

plt.ylabel('Number of Courses')

plt.title("PLOTTING DIFFICULTY OF COURSE WITH THEIR COUNT")

figure, ax =plt.subplots(1,2, figsize= (15,10))





sns.countplot(data['course_difficulty'], ax=ax[0], hue=data['course_Certificate_type'])

ax[0].set_title("RELATION OF DIFFUCULTY OF COURSE WITH CERTIFICATION TYPE")





sns.countplot(data['course_Certificate_type'], ax=ax[1], hue=data['course_difficulty'])

ax[1].set_title("RELATION OF COURSE CERTIFICATION TYPE WITH DIFFICULTY OF COURSE")





figure.show()
wordcloud = WordCloud(width = 1000, height = 1000, 

            background_color ='white', 

            stopwords = STOPWORDS, 

            min_font_size = 10).generate(str(data['course_title']))



plt.figure(figsize = (15, 10))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()