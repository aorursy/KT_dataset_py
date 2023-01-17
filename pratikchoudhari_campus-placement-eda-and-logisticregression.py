# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(111)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_colwidth', None)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
whole = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
whole.gender = np.where(whole.gender=='M','Male','Female')
whole.salary = whole.salary.fillna(0)
whole.sample(5)
whole.shape
def tag(x):
    if x<246666:
        return 'low'
    elif 246666<x<493332:
        return 'medium'
    elif x>493332:
        return 'high'
whole['salary_level'] = whole.salary.apply(tag)
gender_count = whole.groupby(['gender','status'])['sl_no'].count().reset_index()
px.bar(gender_count, 'gender', 'sl_no', title='Gender count w.r.t placement status',
      width=600, height=600, labels={'sl_no':'Count','gender':'Gender'},color='status',template='seaborn')
salary = whole[whole.salary!=0].groupby(['specialisation','workex'])['salary'].mean().reset_index()
px.bar(salary, 'specialisation', 'salary', title='Salary for specialisations and workex',
      width=600, height=600,color='workex')
px.bar(whole[whole.status=='Placed'], y='salary',color='gender')
px.histogram(whole[whole.status=='Placed'],x='salary',nbins=10,color_discrete_sequence=['indianred'],opacity=0.8,
            title='Salary Distribution',marginal="box")
status_ls = list(whole.status)*4
cert = ['SSC']*215 + ['HSC']*215 + ['DEGREE']*215 + ['MBA']*215
scores = list(whole.ssc_p) + list(whole.hsc_p) + list(whole.degree_p) + list(whole.mba_p)
cmp = pd.DataFrame({'Qualification':cert,'Percentage':scores,'Placement status':status_ls})
fig1 = px.violin(cmp[cmp['Placement status']=='Placed'], y='Percentage', color='Qualification', box=True, 
                     points='all', template='plotly_white',title='Placed')
fig1.show()
fig2 = px.violin(cmp[cmp['Placement status']=='Not Placed'], y='Percentage', color='Qualification', box=True, 
                 points='all', template='plotly_white',title='Not placed')
fig2.show()
px.violin(cmp, x='Qualification', y='Percentage', color='Placement status', box=True, points='all', template='plotly_white',
                 title='Placed vs Not placed score comparision')
stream = whole[whole.status=='Placed']['degree_t'].value_counts()
fig3=px.pie(stream, names = stream.index, values = stream.values,color_discrete_sequence=px.colors.qualitative.T10,
          title='Bachelors Specialization demanded by corporates')
fig3.update_traces(textposition='inside', textinfo='percent+label',textfont_size=15)
fig3.show()
stream = whole[whole.status=='Placed']['specialisation'].value_counts()
fig3=px.pie(stream, names = stream.index, values = stream.values,color_discrete_sequence=px.colors.qualitative.T10,
          title='MBA Specialization demanded by corporates',hole=0.5)
fig3.update_traces(textposition='inside', textinfo='percent+label',textfont_size=15)
fig3.show()
fig4 = px.treemap(whole,path=('hsc_s','specialisation','status'),color_discrete_sequence=px.colors.qualitative.G10,
                 title='HSC stream->MBA specialisation->Placement status')
fig4.show()
fig5 = px.treemap(whole,path=('degree_t','specialisation','status'),color_discrete_sequence=px.colors.qualitative.D3,
                 title='Bachelor stream->MBA specialisation->Placement status')
fig5.show()
px.sunburst(whole,path=('specialisation','salary_level'),color_discrete_sequence=px.colors.qualitative.Set1,
                 title='MBA specialisation->Salary level')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
train = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
train.head()
train.status = np.where(train.status=='Placed',1,0)
train.gender = np.where(train.gender=='M',1,0)
train.ssc_b = np.where(train.ssc_b=='Central',1,0)
train.hsc_b = np.where(train.hsc_b=='Central',1,0)
train.workex = np.where(train.workex=='Yes',1,0)
train.salary = train.salary.fillna(0)
train.drop('sl_no',axis=1,inplace=True)
train = pd.get_dummies(train)
train.sample(5)
train.shape
X_cols = list(train.columns)
X_cols.remove('status')
X = train[X_cols]
y = train['status']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, random_state=0)
lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
y_train.value_counts()
np.unique(lr.predict(X_test), return_counts=True)