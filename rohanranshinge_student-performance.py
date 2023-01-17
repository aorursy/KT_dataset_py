# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/StudentsPerformance.csv")
df.head(5)
df.info()
df['Total score'] = df['math score']+ df['reading score'] + df['writing score']
import pandas as pd
import seaborn as sns 
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.graph_objs import Scatter, Layout
import matplotlib as plt
init_notebook_mode(connected=True)
%matplotlib inline
plot1 = df[['Total score', 'gender', 'parental level of education']]
g = sns.FacetGrid(plot1, row = "gender",col = 'parental level of education',size= 5,)
g.map(sns.distplot, 'Total score')
for df in [df]:
    df['Sex_binary']=df['gender'].map({'male':1,'female':0})
female_df = df.loc[df['Sex_binary'] == 0]
male_df = df.loc[df['Sex_binary']==1]
#female_df = female_df.drop(['Sex_binary'],axis =1)
#male_df = male_df.drop(['Sex_binary'],axis =1)
female_df.head(3)
male_df.head(3)
sns.heatmap(male_df.corr())
sns.heatmap(female_df.corr())
y= df['Total score']
x= df[['gender','race/ethnicity','parental level of education','test preparation course']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

