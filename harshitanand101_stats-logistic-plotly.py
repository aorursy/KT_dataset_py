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
df=pd.read_csv('/kaggle/input/Smarket.csv')
df.head()
df.info()
df.describe()

df=df.drop(columns=['Unnamed: 0'])
import seaborn as sns



sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.heatmap(df.corr(),cmap=sns.diverging_palette(20, 220, n=200), linewidths=.5)
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import plotly.express as px

import datetime
sns.countplot(df['Year'])
checker=df['Year'][0]

counter=1

df['Date']=0

for i in range(len(df)):

    if df['Year'].iloc[i]==checker:

        df['Date'].iloc[i]=counter

        counter+=1

    else:

        counter=1

        checker=df['Year'].iloc[i]

        df['Date'].iloc[i]=counter

        
fig = px.line(df, x="Date", y="Volume", color='Year')

fig.show()
fig = px.line(df, x="Date", y="Lag1", color='Year')

fig.show()
vol_means=[]

sum=0

checker=df['Year'][0]

counter=0

for i in range(len(df)):

    if df['Year'].iloc[i]==checker:

        sum+=df['Volume'].iloc[i]

        counter+=1

    else:

        vol_means.append(sum/counter)

        checker=df['Year'].iloc[i]

        sum=0

        counter=0

vol_means.append(sum/counter)
print(vol_means)

print(df.Year.unique())
fig = px.bar(y=vol_means, x=df.Year.unique(),color=vol_means)

fig.show()
lag_means=[]

sum=0

checker=df['Year'][0]

counter=0

for i in range(len(df)):

    if df['Year'].iloc[i]==checker:

        sum+=df['Lag1'].iloc[i]

        counter+=1

    else:

        lag_means.append(sum/counter)

        checker=df['Year'].iloc[i]

        sum=0

        counter=0

lag_means.append(sum/counter)
lag_means
fig = go.Figure(data=[go.Bar(y=lag_means, x=df['Year'].unique())])

# Customize aspect



fig.update_traces(marker_color='rgb(152, 180, 212)',marker_line_width=1.5)

fig.update_layout(title_text='Return on shares Lag1 , avg. per Year',paper_bgcolor='rgb(0, 0, 0)',plot_bgcolor='rgb(0, 0, 0)')
lag_means=[]

sum=0

checker=df['Year'][0]

counter=0

for i in range(len(df)):

    if df['Year'].iloc[i]==checker:

        sum+=df['Today'].iloc[i]

        counter+=1

    else:

        lag_means.append(sum/counter)

        checker=df['Year'].iloc[i]

        sum=0

        counter=0

lag_means.append(sum/counter)
fig = go.Figure(data=[go.Bar(y=lag_means, x=df['Year'].unique())])

# Customize aspect

fig.update_traces(marker_color='rgb(149, 82, 81)', marker_line_color='rgb(155, 35, 53)',

                  marker_line_width=1.5)

fig.update_layout(title_text='Return on shares Today , avg. per Year',paper_bgcolor='rgb(223, 207, 190)',plot_bgcolor='rgb(223, 207, 190)')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X=df.drop(columns=['Direction'])

y=pd.get_dummies(df['Direction'])

y=y.drop(columns='Down')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
coef=clf.coef_.tolist()
coef=coef[0]
for i in range(len(list(coef))):

    print(X.columns[i],'has coefficient coeff -----------',coef[i])

    print("")

print('Intercept has the coeff------------- ',clf.intercept_)
import math



# (p/1-p)= e^0.827



#Let RHS = x



x=math.exp(0.827)

print(x)
from sklearn.metrics import classification_report, confusion_matrix



clf.score(X_train, y_train)

clf.score(X_test,y_test)
cf=confusion_matrix(y_test, clf.predict(X_test))
sns.heatmap(cf, annot=True)

print(classification_report(y_test, clf.predict(X_test)))
from sklearn import metrics

y_pred_proba = clf.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()