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

import numpy as np

import seaborn as sns
data_path = r'/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'
data = pd.read_csv(data_path)
data.head()
data.tail()
data.info()
data.describe()
data.isna().sum()
# Lets Assume if the students are not placed then salary = 0



data['salary'] = data['salary'].fillna(0)
data.isna().sum()
data['gender'].value_counts()
data['ssc_b'].value_counts()
data['hsc_b'].value_counts()
data['hsc_s'].value_counts()
data['degree_t'].value_counts()
data['workex'].value_counts()
data['specialisation'].value_counts()
data['status'].value_counts()
# Visualise The Data
sns.set(style="darkgrid")

data = data

ax = sns.countplot(x="gender", data=data)
import plotly.offline as ply

import plotly.express as px

import plotly.graph_objs as go

ply.init_notebook_mode(connected=True)
specialisation_gr = data.groupby('specialisation').sum()['salary'].reset_index()



fig = go.Figure()

fig = px.bar(specialisation_gr[['specialisation','salary']].sort_values('salary',ascending=False),

             y='salary',x='specialisation', color='specialisation')

ply.iplot(fig)
fig = px.histogram(data, x="ssc_p", y="hsc_p", color="gender")

fig.show()
corr = data.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
a = data.groupby(["hsc_s"])[["hsc_p"]].mean().reset_index()



fig = px.pie(a,values="hsc_p",names="hsc_s",template="seaborn")

fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")

fig.show()
# I use Label Encoder Because there are not more than 3 categories in each column



from sklearn import preprocessing

le = preprocessing.LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])

data['ssc_b'] = le.fit_transform(data['ssc_b'])

data['hsc_b'] = le.fit_transform(data['hsc_b'])

data['hsc_s'] = le.fit_transform(data['hsc_s'])

data['degree_t'] = le.fit_transform(data['degree_t'])

data['workex'] = le.fit_transform(data['workex'])

data['specialisation'] = le.fit_transform(data['specialisation'])

data['status'] = le.fit_transform(data['status'])
data.head(10)
data = data.drop('sl_no',axis=1)
from sklearn.preprocessing import StandardScaler

a = StandardScaler()

new_df = pd.DataFrame(a.fit_transform(data), columns=data.columns, index=data.index)
new_df.head(5)
X = new_df.drop(['salary'],axis=1)

y = new_df['salary']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=55)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
regressor.score(X,y)
predict = regressor.predict(X_test)

predict
regressor.coef_