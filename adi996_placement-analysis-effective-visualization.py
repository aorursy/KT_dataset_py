# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O(e.g. pd.read_csv)

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df =  pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()
def categorize_grades(marks):

    if (marks>50 and marks<60):

        return '50%-60%'

    elif (marks>60 and marks<70):

        return '60%-70%'

    elif (marks>70 and marks<80):

        return '70%-80%'

    elif (marks>80 and marks<90):

        return '80%-90%'

    elif (marks>90):

        return '90%-100%'
df['school_percent_category'] = df['ssc_p'].apply(lambda x:categorize_grades(x))
df['degree_percent_category'] = df['degree_p'].apply(lambda x:categorize_grades(x))
df['specialization_percent_category'] = df['mba_p'].apply(lambda x:categorize_grades(x))
df.head()
fig = px.bar(df['status'].value_counts(),

             title="Overall campus placement results",labels={"index":"Status","value":"Total Students"})

fig.show()
fig = px.bar(df['degree_t'].value_counts(),

             title="Degree taken by students",labels={"index":"Degree","value":"Total Students"})

fig.show()
fig = px.bar(df['hsc_s'].value_counts(),

             title="Stream taken by students in High school",labels={

                 'index': 'Stream',

                 'value':'Total Count'

             })

fig.show()
only_placed = df.loc[df['status']=='Placed']
fig = px.bar(only_placed['degree_t'].value_counts(), 

             title="Which degree has got the most number of students placed?" ,

             labels={'value':'Total students placed',

                    'index':'Degree'})

fig.show()
fig = px.bar(only_placed['school_percent_category'].value_counts(),

             title="Category of students who got placed(Percentage in high school)",

             labels={"value":"Total Students","index":"Students",

                    "school_percent_category":"Category of students"

                    })

fig.show()
fig = px.bar(df['degree_percent_category'].value_counts(),

             title="Category of students who got placed(Percentage in college)",

             labels={"value":"Total Students","index":"Students"})

fig.show()
fig = px.bar(df['specialization_percent_category'].value_counts(),

             title="Cateogry of student who got placed(Specialization)",

            labels={"value":"Total Students","index":"Students"})

fig.show()
salary_science = only_placed['salary'].loc[(only_placed['degree_t']=='Sci&Tech')]

salary_management = only_placed['salary'].loc[(only_placed['degree_t']=='Comm&Mgmt')]

salary_other = only_placed['salary'].loc[(only_placed['degree_t']=='Others')]

hist_data = [salary_science,salary_management,salary_other]

group_labels = ['Science & Technology',"Commerce and management","Others"]

colors=['blue',"green","orange"]

fig = ff.create_distplot(hist_data, group_labels,show_hist=True, 

                         colors=colors,bin_size=[10000,10000,10000])

fig.update_layout(title="Salary distribution of each degree")

fig.show()
fig = px.violin(only_placed,y=only_placed['salary'],x=only_placed['degree_t'], box=True,

                points='all', color=only_placed['gender'],

               hover_data={'gender':True ,

                           'Percentage in Boards' : only_placed['ssc_p'],

                           'Work experience' : only_placed['workex'],

                           'Etest' : only_placed['etest_p'],

                           'Highschool Stream' : only_placed['hsc_s'],

                           'Highschool Percent' : only_placed['hsc_p'],

                           'Undergraduate degree ' : only_placed['degree_t'],

                           'Degree percentage' :  only_placed['degree_p']

                          },

                labels={

                    'workex' : 'Work Experience',

                    'salary' : 'Package',

                    'gender':'Gender',

                    'degree_t' : 'Different degrees'

                    

                },

               title="Salary distribution of students who got placed (Degree-wise)")

fig.show()
fig = px.bar(only_placed['workex'].value_counts(), 

             title="Students who got placed w/wo work exp" ,

             labels={'value':'Total',

                    'index':'Work Experience','workex':'Work Experience'})

fig.show()
with_work_exp = only_placed['salary'].loc[(only_placed['workex']=='Yes')]

without_work_exp = only_placed['salary'].loc[(only_placed['workex']=='No')]

hist_data = [with_work_exp,without_work_exp]

group_labels = ['With work experience',"Without work experience"]

colors=['blue',"green"]

fig = ff.create_distplot(hist_data, group_labels,show_hist=True, # Set False to hide histogram bars

                         colors=colors,bin_size=[10000,10000,10000])

fig.update_layout(title="Salary distribution of students who have prior work experience")

fig.show()
fig = px.violin(only_placed,y=only_placed['salary'],x=only_placed['workex'], box=True,

                points='all', color=only_placed['degree_t'],

               hover_data={'gender':True ,

                           'Percentage in Boards' : only_placed['ssc_p'],

                           'Highschool Stream' : only_placed['hsc_s'],

                            'Etest' : only_placed['etest_p'],

                           'Highschool Percent' : only_placed['hsc_p'],

                           'Undergraduate degree ' : only_placed['degree_t'],

                           'Degree percentage' :  only_placed['degree_p']

                          },

                labels={

                    'workex' : 'Work Experience',

                    'salary' : 'Package',

                    'gender':'Gender',

                    'degree_t': 'Legend',               

                }

               ,title="Salary distribution of students who have/don't have work experience")

fig.show()
fig = px.violin(only_placed,y=only_placed['salary'],x=only_placed['specialization_percent_category'],

                points='all', color=only_placed['specialisation'],

               hover_data={'gender':True ,

                           'Percentage in Boards' : only_placed['ssc_p'],

                           'Highschool Stream' : only_placed['hsc_s'],

                           'Highschool Percent' : only_placed['hsc_p'],

                           'Undergraduate degree ' : only_placed['degree_t'],

                           'Degree percentage' :  only_placed['degree_p']

                          },

                labels={

                    'specialization_percent_category' : 'Category',

                    'salary' : 'Package',

                    'gender':'Gender',

                    'specialisation':'Specialization'

                    

                },

               title="Specialization percent of students")

fig.show()
fig = px.bar(only_placed['ssc_b'].value_counts(),

             title="School board selected by the student's" , 

             labels={'value':'Total',

                     'index':'Work Experience'})

fig.show()
fig = px.violin(only_placed,y=only_placed['salary'],x=only_placed['ssc_b'], box=True,

                points='all', color=only_placed['degree_t'],

               hover_data={'gender':True ,

                           'Work Experience' : only_placed['workex'],

                           'Percentage in Boards' : only_placed['ssc_p'],

                           'Highschool Stream' : only_placed['hsc_s'],

                           'Highschool Percent' : only_placed['hsc_p'],

                           'Undergraduate degree ' : only_placed['degree_t'],

                           'Degree percentage' :  only_placed['degree_p'],

                           'Specialization' : only_placed['specialisation'],

                           'Specialization percent' : only_placed['mba_p']

                          },

                labels={

                    'ssc_b' : 'Board',

                    'salary' : 'Package',

                    'gender':'Gender',

                    'degree_t' : 'Legends'

                    

                }

               ,title="Which board does the students who got placed belong to?")

fig.show()
fig = px.box(only_placed, x="gender" ,

             y="salary",points="all" ,

             labels={"gender":"Gender",  "salary" :"Package"} ,

            title="Packages of students who got placed (Gender-wise)",

             color=only_placed['gender'],

            hover_data = {'Work Experience': only_placed['workex'],

                         '10th percentage': only_placed['ssc_p'],

                         '12th percentage' : only_placed['hsc_p'],

                         'Undergrad percentage' : only_placed['degree_p'],

                       'Specialization': only_placed['specialisation'],

                        'Specialization percent' : only_placed['mba_p']})

fig.show()