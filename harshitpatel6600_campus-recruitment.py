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

import plotly.graph_objects as go
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data = df.copy()

df.head()
df.columns
df.drop(['sl_no'], axis=1, inplace=True)
df.head()
df.dtypes
df.isna().sum()
plt.figure(figsize=(10,7))

sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
df.drop(['salary'], axis=1, inplace=True)
df.head()
plt.figure(figsize=(10,7))

sns.heatmap(df.isnull(), cmap='coolwarm', cbar=False, yticklabels=False)
category_columns = [col for col in df.columns if df[col].dtypes == 'O']
category_columns
values = []

for col in category_columns:

    values.append(df[col].nunique())
x = category_columns

y = values

colors = ['orange',] * len(category_columns)

colors[3] = 'green'

colors[4] = 'green'



# Use textposition='auto' for direct text

fig = go.Figure(data=[go.Bar(

            x=x, y=y,

            text=y,

            textposition='auto',

            marker_color=colors

        )])



fig.update_layout(

    title="category_columns vs. values",

    xaxis_title="category_columns",

    yaxis_title="values",

    font=dict(

        family="Courier New, monospace",

        size=18,

        color="#7f7f7f"

    )

)



fig.show()
df.shape
df = pd.get_dummies(df, drop_first=True)
df.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

from sklearn import metrics
X=df.drop(['status_Placed'],axis=1)

y=df['status_Placed']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\033[94m{}\033[00m".format(classification_report(y_test, predictions)))
print("\033[94m Accuracy: {}%\033[00m" .format(round(accuracy_score(y_test, predictions),3)))
cm = confusion_matrix(y_test, predictions)

cm
import plotly.figure_factory as ff



z = cm



fig = ff.create_annotated_heatmap(z, colorscale='darkmint')



# add title

fig.update_layout(title_text='<i><b>Confusion matrix</b></i>')



# add custom xaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=0.5,

                        y=-0.15,

                        showarrow=False,

                        text="Predicted value",

                        xref="paper",

                        yref="paper"))



# add custom yaxis title

fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=-0.05,

                        y=0.5,

                        showarrow=False,

                        text="Real value",

                        textangle=-90,

                        xref="paper",

                        yref="paper"))



# adjust margins to make room for yaxis title

fig.update_layout(margin=dict(t=50, l=80))



# add colorbar

fig.show()
res = pd.DataFrame({'Actual': y_test, 'Predictions': predictions})

res = pd.DataFrame({'Index': res.index, 'Actual': y_test, 'Predictions': predictions})

res
import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Scatter(x=res['Index'], y=res['Actual'],

                    mode='markers',

                    name='Actual'))



fig.add_trace(go.Scatter(x=res['Index'], y=res['Predictions'],

                    mode='markers',

                    name='Predictions'))



fig.add_annotation(dict(font=dict(color="black",size=14),

                        x=0.5,

                        y=-0.15,

                        showarrow=False,

                        text="Index Values",

                        xref="paper",

                        yref="paper"))



fig.update_layout(title='<i><b>Actual & Prediction plot</b></i>')



fig.show()