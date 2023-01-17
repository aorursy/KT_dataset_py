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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# interactive visualization
import plotly as py
from plotly.offline import iplot
import plotly.tools as tls

import cufflinks as cf

import plotly.graph_objs as go
data = pd.read_csv('../input/coursera-course-dataset/coursea_data.csv')
data.head()
df = data.copy()
df.drop("Unnamed: 0",axis=1, inplace=True)
df['course_students_enrolled']= df['course_students_enrolled'].str.replace('k', '*1000')
df['course_students_enrolled']= df['course_students_enrolled'].str.replace('m', '*1000000')
df['course_students_enrolled'] = df['course_students_enrolled'].map(lambda x: eval(x))
py.offline.init_notebook_mode(connected=True)
cf.go_offline()
# print(cf.getThemes())
cf.set_config_file(theme='ggplot')
df['course_Certificate_type'].iplot(kind='hist',title='Course Certificate Type',
                                   xTitle='Course Type', yTitle='Counts')
# print(cf.getThemes())
cf.set_config_file(theme='polar')
df['course_rating'].iplot(kind='hist',title='Course Rating ',bargap=0.2,
                                   xTitle='Rating', yTitle='Counts')
# print(cf.getThemes())
cf.set_config_file(theme='pearl')
df['course_difficulty'].iplot(kind='hist',title='Course Difficulty',
                             xTitle='Course Type',yTitle='Count')
df.head()
# print(cf.getThemes())
cf.set_config_file(theme='ggplot')
df.iplot(x='course_difficulty',y='course_students_enrolled',kind='bar')
# print(cf.getThemes())
cf.set_config_file(theme='pearl')
df.iplot(x='course_Certificate_type',y='course_students_enrolled',
        kind='bar')
course_org = df.groupby('course_organization').count().reset_index()
trace = go.Pie(labels = course_org['course_organization'], values =course_org['course_students_enrolled'] )
data = [trace]
fig = go.Figure(data = data)
iplot(fig)