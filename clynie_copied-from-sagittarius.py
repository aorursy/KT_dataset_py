# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# This is my first time in Kaggle,the code is almost copied from 
# sagittarius:
# https://www.kaggle.com/bhavikapanara/data-preprocessing-and-analysis?scriptVersionId=5330336
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls

# Any results you write to the current directory are saved as output.



def check_missing(df):
    
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
    

def find_uni(df):
    col_list = df.columns
    redundant_col =[]
    for col in col_list:
        if df[col].nunique() == 1:
            redundant_col.append(col)
    return redundant_col




df = pd.read_csv('../input/restaurant-and-market-health-inspections.csv')
# print(df)
print ('Size of the health-inspections data : ' , df.shape)
# print(df.head(0))
cat_col = df.select_dtypes(include = 'object').columns.tolist()
num_col = df.select_dtypes(exclude='object').columns.tolist()

print ('categorical feature :', cat_col)
print ('numeric feature :' ,num_col)
print ('number of categorical feature : ' , len(cat_col))
print ('number of numeric feature : ' , len(num_col))

# Description of categorical features
df.describe(include=["O"])
# print(df.describe(include=["O"]))
# print(df)
missing_data_df = check_missing(df)
missing_data_df.head()

redundant_col = find_uni(df)
# So, these two features are redundant for the ML algorithm.
print ('Number of redundant features in data :',len(redundant_col))
print ('Redundant Feature :', redundant_col)



# Find Duplicate features
df.drop(redundant_col,axis=1,inplace =True)

# Here, service_description feature describes the code for inspection service. 
# For machine learning perspective both are the duplicate feature. 
# The duplicate feature doesn't help for machine learning algorithm. 
# hence, it's better to remove any one of the feature
print(df['service_description'].value_counts())
df.drop('service_code' , axis =1 , inplace=True)
# Let's check program_element_pe feature and pe_description feature.
# program_element_pe feature describe the code of the pe_description feature. 
print(df['program_element_pe'].value_counts())
print(df['pe_description'].value_counts())
# Again these two features are duplicate. 
# so, remove program_element_pe feature.
df.drop('program_element_pe' , axis =1, inplace =True)

df[['score',]].describe()
df['score'].hist().plot()
plt.show()
# 50% data have more than 90 score value.


# 50% data have more than 90 score value.

temp = df['grade'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='grade')
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='grade')
# 86% places got A grade.

le = LabelEncoder()
# encode categorical integer features using a one-hot aka one-of-K scheme.
# Usually le.fit() before.
df['grade'] = le.fit_transform(df['grade'])
df['grade'].corr(df['score'])
# grade feature is highly correlated with score feature. 
# Grade feature has been calculated from the score value range.

# Top 15 highly health-violated place.
top_violated_place = df["facility_name"].value_counts().head(15)
tmp = pd.DataFrame({'Count':top_violated_place.values},index = top_violated_place.index)
print(tmp)

temp = df["facility_name"].value_counts().head(25)

trace = go.Bar(
    x = temp.index,
    y = temp.values,
)
data = [trace]
layout = go.Layout(
    title = "Distribution of facility_name",
    xaxis=dict(
        title='facility_name',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='how many times health-violations occur',
        titlefont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='../input/facility_name.html')


temp1 = df[['facility_name','score']].sort_values(['score'],ascending = False).drop_duplicates()
print(temp1.head(10))

temp1 = df[['facility_name','score']].sort_values(['score']).drop_duplicates()
print(temp1.head(10))
