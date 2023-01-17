# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import plotly as py
import plotly.graph_objects as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')
data
data=data.drop_duplicates()
print('NaN values')
print(data.isna().sum())
print('Null values')
print(data.isnull().sum())
data.info()
data['course_students_enrolled']=data['course_students_enrolled'].apply(lambda x: float(x.split('k')[0] if x[-1]=='k' else float(x.split('m')[0])*1000))
import plotly.express as px
fig=px.pie(data.groupby('course_Certificate_type').size().reset_index(), values=0, names='course_Certificate_type', color_discrete_sequence=px.colors.sequential.RdBu, 
          title='Amount of certificate types')
iplot(fig)
import plotly.express as px
fig=px.pie(data.groupby('course_difficulty').size().reset_index(), values=0, names='course_difficulty', color_discrete_sequence=px.colors.sequential.RdBu, 
          title='Amount of levels')
iplot(fig)
px.bar(data.groupby('course_Certificate_type').course_students_enrolled.sum().reset_index(), x='course_Certificate_type', y='course_students_enrolled', 
       hover_data=[data.groupby('course_Certificate_type').course_rating.min(),data.groupby('course_Certificate_type').course_rating.max(),
                                                                                    data.groupby('course_Certificate_type').course_rating.mean()], labels={
           'course_Certificate_type':'Certificate type', 'course_students_enrolled':'amount of students','hover_data_0':'Min rate','hover_data_1':'Max rate',
           'hover_data_2':'Mean rate'}, title='Certificate type: Amount of students, Min, Max and Mean rate')
a=data[data.course_Certificate_type=='COURSE'].groupby('course_difficulty').course_students_enrolled.sum().reset_index()
a=a.merge(data[data.course_Certificate_type=='COURSE'].groupby('course_difficulty').course_rating.min().reset_index(), on='course_difficulty')
a=a.merge(data[data.course_Certificate_type=='COURSE'].groupby('course_difficulty').course_rating.max(), on='course_difficulty')
a=a.merge(data[data.course_Certificate_type=='COURSE'].groupby('course_difficulty').course_rating.mean(), on='course_difficulty')
a=a.rename(columns={'course_difficulty':'Difficulty', 'course_students_enrolled':'Student Amount', 'course_rating_x':'Min rate',
            'course_rating_y':'Max rate', 'course_rating':'Mean rate' })
px.bar(a, x='Difficulty', y='Student Amount', hover_data=['Min rate', 'Max rate', 'Mean rate'], title='Courses: Student Amount and Rates')
a=data[data.course_Certificate_type=='SPECIALIZATION'].groupby('course_difficulty').course_students_enrolled.sum().reset_index()
a=a.merge(data[data.course_Certificate_type=='SPECIALIZATION'].groupby('course_difficulty').course_rating.min().reset_index(), on='course_difficulty')
a=a.merge(data[data.course_Certificate_type=='SPECIALIZATION'].groupby('course_difficulty').course_rating.max(), on='course_difficulty')
a=a.merge(data[data.course_Certificate_type=='SPECIALIZATION'].groupby('course_difficulty').course_rating.mean(), on='course_difficulty')
a=a.rename(columns={'course_difficulty':'Difficulty', 'course_students_enrolled':'Student Amount', 'course_rating_x':'Min rate',
            'course_rating_y':'Max rate', 'course_rating':'Mean rate' })
px.bar(a, x='Difficulty', y='Student Amount', hover_data=['Min rate', 'Max rate', 'Mean rate'], title='Specialization: Student Amount and Rates')
a=data[data.course_Certificate_type=='PROFESSIONAL CERTIFICATE'].groupby('course_difficulty').course_students_enrolled.sum().reset_index()
a=a.merge(data[data.course_Certificate_type=='PROFESSIONAL CERTIFICATE'].groupby('course_difficulty').course_rating.min().reset_index(), on='course_difficulty')
a=a.merge(data[data.course_Certificate_type=='PROFESSIONAL CERTIFICATE'].groupby('course_difficulty').course_rating.max(), on='course_difficulty')
a=a.merge(data[data.course_Certificate_type=='PROFESSIONAL CERTIFICATE'].groupby('course_difficulty').course_rating.mean(), on='course_difficulty')
a=a.rename(columns={'course_difficulty':'Difficulty', 'course_students_enrolled':'Student Amount', 'course_rating_x':'Min rate',
            'course_rating_y':'Max rate', 'course_rating':'Mean rate' })
px.bar(a, x='Difficulty', y='Student Amount', hover_data=['Min rate', 'Max rate', 'Mean rate'], title='Profesional certificate: Student Amount and Rates')
px.pie(data.course_organization.value_counts().reset_index().head(20), values='course_organization',names='index',
       labels={'course_organization':'amount of courses', 'index':'Company'}, title='Top 20 Course organizations')
a=data[(data.course_rating>4.6) & (data.course_students_enrolled>200.0) & (data.course_Certificate_type=='COURSE')]
px.pie(a[:20],values='course_students_enrolled', hover_data=['course_organization', 'course_title'], title='TOP 20 COURSES')
px.bar(a[:20], x='course_difficulty', y='course_students_enrolled', title='TOP 20 COURSES: level distribution', hover_data=['course_organization', 'course_title'])
a=data[(data.course_rating>4.6) & (data.course_students_enrolled>200.0) & (data.course_Certificate_type=='SPECIALIZATION')]
px.pie(a,values='course_students_enrolled', hover_data=['course_organization', 'course_title'], title='TOP 20 SPECIALIZATIONS')
px.bar(a, x='course_difficulty', y='course_students_enrolled', hover_data=['course_organization', 'course_title'],
       title='TOP 20 SPECIALIZATIONS: level distribution')