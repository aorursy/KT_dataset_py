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
df = pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')

df.head()
df.shape
df.dtypes
df.isnull().sum()
import seaborn as sns
import plotly.express as px

fig = px.scatter_3d(df, x='mean_texture', y='mean_radius', z='diagnosis',

              color='mean_area')

fig.show()
import plotly.express as px

fig = px.scatter(df, x="mean_radius", y="mean_area", color="mean_perimeter",

                 size='mean_texture', hover_data=['diagnosis'])

fig.show()
import plotly.graph_objects as go

labels = df['diagnosis']

values = df['mean_area']

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

    autosize=False,

    width=300,

    height=300,)

fig.show()

labels = df['diagnosis']

values = df['mean_perimeter']

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

    autosize=False,

    width=300,

    height=300,)

fig.show()

labels = df['diagnosis']

values = df['mean_texture']

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

    autosize=False,

    width=300,

    height=300,)

fig.show()
import plotly.express as px

fig = px.sunburst(df.sample(frac = 0.1), path=['diagnosis', 'mean_texture'], values='mean_area')

fig.show()
sns.distplot( df["mean_area"] )
x = df.drop(columns=['diagnosis'])

print(x)
y = df['diagnosis']

y
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# random forest model creation

rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)

# predictions

rfc_predict = rfc.predict(x_test)



print("Accuracy:",accuracy_score(y_test, rfc_predict))
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(x_train,y_train)

#Predict the response for test dataset

y_pred = clf.predict(x_test)





print("Accuracy:",accuracy_score(y_test, y_pred))