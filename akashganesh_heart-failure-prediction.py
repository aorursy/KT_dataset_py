import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head(10)
df.isnull().sum()

plt.rcParams['figure.figsize']=15,6

sns.set_style('darkgrid')



x=df.iloc[:,:-1]

y=df.iloc[:,-1]



from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier()

model.fit(x,y)

print(model.feature_importances_)

feat_imp=pd.Series(model.feature_importances_,index=x.columns)

feat_imp.plot(kind='barh')

plt.show()

# Box Plot for Ejection Fraction

sns.boxplot(df['ejection_fraction'])

plt.show()
df['ejection_fraction']=df[df['ejection_fraction']<70]

sns.boxplot(df['ejection_fraction'])

plt.show()
#Boxplot for age

sns.boxplot(df['age'])

plt.show()
#Distribution of Age



import plotly.graph_objects as go



fig = go.Figure()

fig.add_trace(go.Histogram(

    x = df['age'],

    xbins=dict( # bins used for histogram

        start=40,

        end=95,

        size=2

    ),

    marker_color='#e8aa60',

    opacity=1

))



fig.update_layout(

    title_text='Age Distribution',

    xaxis_title_text='Age',

    yaxis_title_text='Count', 

    bargap=0.05, # gap between bars of adjacent location coordinates

    plot_bgcolor='#000000',

    xaxis =  {'showgrid': False },

    yaxis = {'showgrid': False }

)

fig.show()
# Now lets categorize the above histogram by DEATH_EVENT



import plotly.express as px

fig = px.histogram(df, x="age", color="DEATH_EVENT", hover_data=df.columns)

fig.show()
#Distribution of Serum Creatinine



fig = go.Figure()

fig.add_trace(go.Histogram(

    x = df['serum_creatinine'],

    xbins=dict( # bins used for histogram

        start=0.5,

        end=9.4,

        size=0.2

    ),

    marker_color='#e8ab60',

    opacity=1

))



fig.update_layout(

    title_text='Serum Creatinine Distribution',

    xaxis_title_text='Serum Creatinine',

    yaxis_title_text='Count', 

    bargap=0.05, # gap between bars of adjacent location coordinates

    plot_bgcolor='#000000',

    xaxis =  {'showgrid': False },

    yaxis = {'showgrid': False }

)

fig.show()
#Histogram in comparison to DEATH_EVENT



fig = px.histogram(df, x="serum_creatinine", color="DEATH_EVENT",marginal='violin', hover_data=df.columns)

fig.show()
#Distribution of Platelets



fig = go.Figure()

fig.add_trace(go.Histogram(

    x = df['platelets'],

    xbins=dict( # bins used for histogram

        start=25000,

        end=850000,

        size=10000

    ),

    marker_color='#e8ab60',

    opacity=1

))



fig.update_layout(

    title_text='Platelets Distribution',

    xaxis_title_text='Platelets',

    yaxis_title_text='Count', 

    bargap=0.05, # gap between bars of adjacent location coordinates

    plot_bgcolor='#000000',

    xaxis =  {'showgrid': False },

    yaxis = {'showgrid': False }

)

fig.show()
#Histogram of platelets as a function of DEATH_EVENT



fig = px.histogram(df, x="platelets", color="DEATH_EVENT",marginal='violin', hover_data=df.columns)

fig.show()
df['time'].describe()
fig = go.Figure()

fig.add_trace(go.Histogram(

    x = df['time'],

    xbins=dict( # bins used for histogram

        start=4,

        end=285,

        size=5

    ),

    marker_color='#e8ab60',

    opacity=1

))



fig.update_layout(

    title_text='Time Distribution',

    xaxis_title_text='Time',

    yaxis_title_text='Count', 

    bargap=0.05, # gap between bars of adjacent location coordinates

    plot_bgcolor='#000000',

    xaxis =  {'showgrid': False },

    yaxis = {'showgrid': False }

)

fig.show()
fig1=px.pie(df, values='diabetes',names='DEATH_EVENT', title='Diabetes VS Death Event',width=600, height=400)

fig2=px.pie(df, values='DEATH_EVENT',names='diabetes',width=500, height=400)



fig1.show()



fig2.show()

df.head()
fig1=px.pie(df, values='smoking',names='DEATH_EVENT', title='Smoking VS Death Event',width=600, height=400)

fig2=px.pie(df, values='DEATH_EVENT',names='smoking',width=500, height=400)



fig1.show()



fig2.show()
fig1=px.pie(df, values='high_blood_pressure',names='DEATH_EVENT', title='High BP VS Death Event',width=600, height=400)

fig2=px.pie(df, values='DEATH_EVENT',names='high_blood_pressure',width=500, height=400)



fig1.show()



fig2.show()
#We select the following features



Features=['time','ejection_fraction','serum_creatinine','age']
df.head()

x=df.iloc[:,[0,4,7,11]].values

y=df.iloc[:,-1].values
df.head()
print(x)
#Splitting the data into train and test set



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_test
x_test=np.nan_to_num(x_test)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(max_iter=10000)

classifier.fit(x_train,y_train)
# Predicting the value for the test set



y_pred=classifier.predict(x_test)
#Making Confusion matrix and predicting accuracy score



mylist=[]

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_pred)

ac=accuracy_score(y_test,y_pred)

print(cm)

print(ac)
#Finding the optimum number of neighbors



from sklearn.neighbors import KNeighborsClassifier



list1=[]

for neighbors in range(1,10):

    classifier=KNeighborsClassifier(n_neighbors=neighbors,metric='minkowski')

    classifier.fit(x_train,y_train)

    y_pred=classifier.predict(x_test)

    list1.append(accuracy_score(y_test,y_pred))

plt.plot(list(range(1,10)),list1)

plt.show()
classifier=KNeighborsClassifier(n_neighbors=7,metric='minkowski')

classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
#Finding the confusion matrix and accuracy score



cm=confusion_matrix(y_test,y_pred)

ac=accuracy_score(y_test,y_pred)

print(cm)

print(ac)
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier



classifier = DecisionTreeClassifier()

classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)



mylist=[]

from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_pred)

ac=accuracy_score(y_test,y_pred)

print(cm)

print(ac)