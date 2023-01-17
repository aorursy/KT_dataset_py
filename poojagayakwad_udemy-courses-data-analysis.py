# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
courses = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')
courses.columns
# creating a new column count which we'll use later in aggregate functions
# created a copy of 'courses' dataframe in 'demo'
demo = courses.copy()
demo['count']=1
demo.head(3)
# list of subjects
subject_list = list(demo['subject'].unique())

# creating fontdictionary for title of plots
font_title={'fontfamily':'monospace','fontweight':'bold','fontsize':20}

# creating fontdictionary for labels of plots
font_label={'fontfamily':'monospace', 'fontsize':12}
# calculating no. of courses in each subject
course_count = demo.groupby(['subject']).count()['count']

subject_course_count = list(zip(subject_list, course_count))
#print(subject_course_count)

#calculating labels 
labels=[]
for course in subject_course_count:
    labels.append(str(course[0])+'  ('+str(course[1])+')')

#plotting a pie chart
explode=[0.1]*4
plt.pie(course_count, labels=labels, autopct='%.2f%%', radius=2, explode=explode)

plt.show()
subscriber_count = demo.groupby(['subject']).sum()['num_subscribers']

plt.pie(subscriber_count, labels=subject_list, radius=2, autopct='%.2f%%', explode=explode)
plt.show()
best_courses = pd.DataFrame(columns=demo.columns)

for subject in subject_list:
    data =demo.loc[(demo.subject==subject) & (demo.is_paid==False)]
    data = data.sort_values(by=['num_subscribers', 'num_reviews'], ascending=False).head(1)
    
    best_courses = pd.concat([best_courses, data])
    
# rearranging columns   
cols = list(demo.columns)
best_courses = best_courses[[cols[-2]]+cols[1:3]+cols[4:7]+[cols[9]]]

best_courses
# creating a 'year' column, which indicates year in which course was published
demo['year']=demo['published_timestamp'].str[0:4]

bins =[2011,2012,2013,2014,2015,2016,2017]

year_count = demo.groupby(['year']).sum()['count']

# plotting a line graph for this 'year' data
plt.figure(figsize=(8,5))
plt.plot(bins, year_count, 'go-', )

plt.title('courses uploaded each year between 2011-2017', fontdict =font_title)
plt.xlabel('Year')
plt.ylabel('courses launched')
plt.show()

# no. of subscribers in each level
subs_in_levels = demo.groupby('level').sum()['num_subscribers']
# types of levels
level_names = demo['level'].sort_values().unique()

# plotting pie chart 

explode =[0.1]*4

wedges,texts, autotexts =plt.pie(subs_in_levels, labels=level_names, radius=2, explode=explode, autopct='%.2f%%')

plt.legend(wedges, level_names, title="course_levels", loc="center left", bbox_to_anchor=(1, 0, 2, 2))
          
          
plt.setp(texts, size=12, weight="bold")
plt.show()

#print(subs_in_levels)
#print(subs)
demo['content_duration'].dtypes # checking datatype of content_duration (its 'float64')

demo.loc[demo.index == demo['content_duration'].idxmin()]['content_duration']

# max duration is 7.5
# min duration is 0

bins =[0,1,2,3,4,5,6,7,8]

#plotting distribution
plt.figure(figsize=(8,5))

plt.title('distribution of course duration (in hours)',fontdict=font_title )

plt.hist(demo['content_duration'], bins=bins )
plt.xticks(bins)

plt.xlabel("duration of content (in hours)", fontdict=font_label)
plt.ylabel("no. of courses", fontdict =font_label)

plt.show()

# a new dataframe to store courses having least(i.e. zero) reviews
least_reviewed =pd.DataFrame(columns = demo.columns)

for subject in subject_list:
    data = demo.loc[(demo.subject ==subject) & (demo.num_reviews ==0)]
    #data = data.loc[data.index == data['num_reviews'].idxmin()]
    least_reviewed = pd.concat([least_reviewed,data])
    
least_reviewed
best_courses = pd.DataFrame(columns=demo.columns)

years= demo['year'].sort_values().unique()

for year in years:
    data =demo.loc[demo.year== year ]
    data = data.sort_values(by=['num_subscribers', 'num_reviews'], ascending=False).head(1)
    best_courses = pd.concat([best_courses, data])
    
# rearranging columns   
cols = list(demo.columns)
best_courses = best_courses[[cols[-1]]+cols[10:12]+cols[1:4]+cols[5:7]]

best_courses
avg_data= demo[['price','num_subscribers','num_reviews','num_lectures','content_duration']].mean()
avg_data