# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
udemy_data = pd.read_csv(r'/kaggle/input/udemy-courses/udemy_courses.csv')

udemy_data.head()
print("Number of datapoints in the dataset : ",udemy_data.shape[0])

print("Number of features in the dataset : ",udemy_data.shape[1])

print("Features : ",udemy_data.columns.values)
udemy_data.info()
udemy_data.describe()
print("Number of duplicate values in this dataset are : ",udemy_data.duplicated().sum())
udemy_data = pd.DataFrame.drop_duplicates(udemy_data)

print("Number of duplicate values in this dataset after removal are : ",udemy_data.duplicated().sum())
udemyu_data = udemy_data.reset_index(drop=True)

udemy_data.head()
fig = plt.figure(figsize=(20,18))



plt.subplot(2,2,1)

sns.countplot('level',data=udemy_data)

plt.grid()



plt.subplot(2,2,2)

sns.countplot(x='is_paid',data=udemy_data)

plt.grid()



plt.subplot(2,2,3)

sns.countplot('subject',data=udemy_data)

plt.grid()



plt.show()
plt.figure(figsize=(12,20))

sns.countplot(y = udemy_data['content_duration'])

plt.title("Content durations and their count in the dataset")

plt.xticks(rotation=90)

plt.grid()

plt.show()
plt.figure(figsize=(15,8))

sns.countplot(x = udemy_data['price'])

plt.grid()

plt.show()
fig = plt.figure(figsize=(20,18))



plt.subplot(2,2,1)

sns.countplot('level',hue='is_paid',data=udemy_data,color='red')

plt.grid()



plt.subplot(2,2,2)

sns.countplot('subject',hue='is_paid',data=udemy_data,color='violet')

plt.grid()



plt.show()
fig,ax = plt.subplots(1,2,figsize=(20,10))

sns.countplot('is_paid',hue='subject',palette=['yellow','green','purple','brown'],data=udemy_data,ax=ax[0])

sns.countplot('is_paid',hue='level',palette=['pink','teal','darkblue','orange'],data=udemy_data,ax=ax[1])

plt.show()
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(20,10))

sns.countplot('subject',hue='level',data=udemy_data,ax=ax[0])

sns.countplot('level',hue='subject',data=udemy_data,ax=ax[1])

plt.show()
fig1 = px.histogram(udemy_data,x='level',y='price',color='level')

fig1.show()

fig2 = px.histogram(udemy_data,x='subject',y='price',color='subject')

fig2.show()
fig1 = px.histogram(udemy_data,x='level',y='num_subscribers',color='level')

fig2 = px.histogram(udemy_data,x='subject',y='num_subscribers',color='subject')

fig1.show()

fig2.show()
fig1 = px.histogram(udemy_data,x='level',y='num_lectures',color='level')

fig2 = px.histogram(udemy_data,x='subject',y='num_lectures',color='subject')

fig1.show()

fig2.show()
fig1 = px.histogram(udemy_data,x='level',y='content_duration',color='level')

fig2 = px.histogram(udemy_data,x='subject',y='content_duration',color='subject')

fig1.show()

fig2.show()
fig1 = px.histogram(udemy_data,x='level',y='num_subscribers',color='subject')

fig2 = px.histogram(udemy_data,x='subject',y='num_subscribers',color='level')

fig1.show()

fig2.show()
fig1 = px.histogram(udemy_data,x='level',y='num_subscribers',color='is_paid')

fig2 = px.histogram(udemy_data,x='subject',y='num_subscribers',color='is_paid')

fig1.show()

fig2.show()
fig = px.pie(udemy_data, names= 'level')

fig.update_traces(textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig.show()



fig1 = px.pie(udemy_data, names= 'subject')

fig1.update_traces(textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig1.show()
fig1 = px.pie(udemy_data,values='num_subscribers', names= 'level',template='seaborn')

fig1.update_traces(rotation=90, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig1.show()



fig2 = px.pie(udemy_data,values='num_subscribers', names= 'subject',template='seaborn')

fig2.update_traces(rotation=90, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig2.show()
fig1 = px.pie(udemy_data,values='num_lectures', names= 'level',color_discrete_sequence=px.colors.sequential.RdBu)

fig1.update_traces(rotation=90, pull=0.02, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig1.show()



fig2 = px.pie(udemy_data,values='num_lectures', names= 'subject',color_discrete_sequence=px.colors.sequential.RdBu)

fig2.update_traces(rotation=90, pull=0.02, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig2.show()
fig1 = px.pie(udemy_data,values='price', names= 'level',color_discrete_sequence=px.colors.sequential.YlOrBr)

fig1.update_traces(rotation=90, pull=0.02, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig1.show()



fig2 = px.pie(udemy_data,values='price', names= 'subject',color_discrete_sequence=px.colors.sequential.YlOrBr)

fig2.update_traces(rotation=90, pull=0.02, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig2.show()
len(np.unique(udemy_data['course_title']))   #number of unique courses
courses = (udemy_data.groupby('course_title')['num_subscribers'].max().reset_index()).sort_values(by='num_subscribers',ascending=False)

courses = (courses.head(20)).reset_index(drop=True)

print(courses)



fig = px.pie(courses, values='num_subscribers', names= 'course_title' , title='Max Subscribers different for Courses',color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_traces(rotation=90, pull=0.02, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig.show()
courses = (udemy_data.groupby('course_title')['num_lectures'].max().reset_index()).sort_values(by='num_lectures',ascending=False)

courses = (courses.head(20)).reset_index(drop=True)

print(courses)



fig = px.pie(courses, values='num_lectures', names= 'course_title' , title='Most Lectures for different Courses',color_discrete_sequence=px.colors.sequential.GnBu)

fig.update_traces(rotation=90, pull=0.02, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig.show()
courses = (udemy_data.groupby('course_title')['price'].max().reset_index()).sort_values(by='price',ascending=False)

courses = (courses.head(25)).reset_index(drop=True)

print(courses)



fig = px.pie(courses, values='price', names= 'course_title' , title='High price Courses',color_discrete_sequence=px.colors.sequential.RdPu)

fig.update_traces(rotation=90, pull=0.02, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig.show()
courses = (udemy_data.groupby('course_title')['content_duration'].max().reset_index()).sort_values(by='content_duration',ascending=False)

courses = (courses.head(25)).reset_index(drop=True)

print(courses)



fig = px.pie(courses, values='content_duration', names= 'course_title' , title='Max duration courses',color_discrete_sequence=px.colors.sequential.YlOrRd)

fig.update_traces(rotation=90, pull=0.02, textinfo="percent+label",marker=dict(line=dict(color='#000000', width=2)))

fig.show()
udemy_data.head()
labels = udemy_data['is_paid']

data = pd.DataFrame.drop(udemy_data,labels=['published_timestamp','course_id','course_title','url','is_paid'],axis=1)

data.head()
#dealing with categorical data

le = LabelEncoder()    

data['level'] = le.fit_transform(data['level'])

data['subject'] = le.fit_transform(data['subject'])

data.head()
X_train,X_test,Y_train,Y_test = train_test_split(data,labels,test_size=0.2)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
Y_test = Y_test.reset_index(drop=True)
def plot_conf_matrix(Y_test,Y_pred):

    conf = confusion_matrix(Y_test,Y_pred)

    sns.heatmap(conf,annot=True,cmap='YlOrBr',fmt=".3f")

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')
lr = LogisticRegression(max_iter=5000)

lr.fit(X_train,Y_train)

Y_pred = lr.predict(X_test)



r=0

for i in range(len(Y_test)):

    if Y_test[i] == Y_pred[i]:

        r=r+1

print("Accuracy : ",r/len(Y_test)*100)



plot_conf_matrix(Y_test,Y_pred)
dt = DecisionTreeClassifier()

dt.fit(X_train,Y_train)

Y_pred = dt.predict(X_test)



r=0

for i in range(len(Y_test)):

    if Y_test[i] == Y_pred[i]:

        r=r+1

print("Accuracy : ",r/len(Y_test)*100)



plot_conf_matrix(Y_test,Y_pred)