# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode (connected=True)

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Firstly,import the data.

data=pd.read_csv("../input/data.csv")

# Looking at the first four rows of the data. 

data.head()
# And also I look at the last four rows.

data.tail()
# When I look at the table, I saw that id and last column is not important for me so I drop these into my data.

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

data.head()

# Now, data is looking good :).
# looking the column names.

data.columns

# we see that column names are in the perfect form. So that we do not need to change formats.
# Now, start to explore data.

data.shape

# we have 31 columns and 569 rows.
data.info()

# Moreover, data has one object and thirty float.
# Now,look at is there any "na" in the data

data.isna().any()

# We see that there is no "na" in the data
# Let's make second check for "na"

data.isnull().sum()
data.describe()

# In there, we see that variables are in different ranges. Moreover, some variables have an outliers in first look.
# In data, we have only one object. Now, we look at the graph of diagnosis

f,ax=plt.subplots(figsize=(10,6))

sb.countplot(data.diagnosis)

plt.gca().invert_xaxis()

plt.title("Counting Plot of Diagnosis Types")

plt.show()

# We see that most of the diagnosis are benign.(b=benign,m=malignant)
# We also look at the pie chart to looking percentage of diagnosis.

plt.figure(figsize=(10,6))

plt.pie(data.diagnosis.value_counts().values,colors=("blue","red"),labels=data.diagnosis.value_counts().index

       ,explode=(0.01,0.01),autopct="%1.1f%%",startangle=67)

plt.title("Pie Chart of Diagnosis")

plt.show()

# We see that 62.7% of diagnosis are benign
# We can also show bar graph using the plotly.



trace=go.Bar(x=data.diagnosis.value_counts().index,

             y=data.diagnosis.value_counts().values,

             name="diagnosis",

             marker=dict(color="rgba(25,14,25,.8)",

             line=dict(color="rgb(0,0,0)",width=1.5)),

             text=data.diagnosis

             )

              

data1=[trace]     

        

layout1=dict(title="Bar Graph of Diagnosis",xaxis=dict(title="Diagnosis Type",ticklen=5,zeroline=False),

            yaxis=dict(title="Counts of Diagnosis",ticklen=5,zeroline=False))



fig=dict(data=data1,layout=layout1)

iplot(fig)
# We also look at the pie chart with plotly

trace=[go.Pie(labels=data.diagnosis.value_counts().index,values=data.diagnosis.value_counts().values

       ,hoverinfo="label+percent+value"

        ,hole=.5

      )]



layout2=dict(title="Pie Chart of Diagnosis")



iplot(trace,layout2)
# Now,we can make plotly then we procude with plotly plots.

# Before looking the describes of variable, we see that some of variables have an outlier or outliers. We can chech this information with box plots.

a=0

list1=[]

for i in data.columns[1:]:

    trace_a=go.Box(y=data.loc[:,i],name=i

                  )

    list1.append(trace_a)

    a=a+1



iplot(list1)

# Let's look at the some variables box_plots.

iplot(list1[5:13])

# In bottom graph, we see that variables values are more different than others.
# We cannot make a model with these values we need to normalize variables.

# Now,we make normalize the variables.

from sklearn import preprocessing

normalize_data=pd.DataFrame(preprocessing.normalize(data.iloc[:,1:]))

normalize_data.head()
# I need to change the column names with their true names.

normalize_data.columns=[data.columns[1:]]

normalize_data.head()
# We normalize the variables. We can look at their correlations.

f,ax=plt.subplots(figsize=(22,10))

sb.heatmap(normalize_data.corr(),annot=True,linewidths=.3,fmt=".2f",ax=ax)

plt.show()

# In the graph we see that area_worst has a high correlation almost every variables.
# Firstly,we made basic linear regression and predict the area_worst.

# To do that, we can use area_mean,radius_mean and perimeter_mean, because there are a high correlations with area_worst.

from sklearn.linear_model import LinearRegression

y=normalize_data.area_worst

x=normalize_data.loc[:,["area_mean","radius_mean","perimeter_mean"]]

lm=LinearRegression()

reg=lm.fit(x,y)

print("R^2 is :",reg.score(x,y))

# As we expected R^2 is very high that means we can explain the area_worst with area_mean,radius_mean and perimeter_mean.
# We also need to look at variables correlations.

normalize_data.loc[:,["area_mean","radius_mean","perimeter_mean"]].corr()

#The variables correlations are too high so there is a multicollinearty in our model.
#Now,let's classify the diagnosis with using logistic regression.But first

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

# To made a classification we need to add diagnosis. But first we need to change object 1 or 0 and after that concat with normalize_data.

binary_diagnosis=pd.DataFrame([1 if i=="B" else 0 for i in data.diagnosis],columns=["binary_diagnosis"])

new_data=pd.concat([binary_diagnosis,normalize_data],axis=1)

new_data.columns=[data.columns]

new_data.head()
# We also need to split our data into 80% train and 20% test.

from sklearn.model_selection import train_test_split

x=new_data.iloc[:,1:]

y=new_data.diagnosis

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print("x_train.shape :",x_train.shape)

print("x_test.shape :",x_test.shape)

print("y_train.shape :",y_train.shape)

print("y_test.shape :",y_test.shape)
# Let's fit the logistic regression.

lr.fit(x_train,y_train)

a=lr.predict(x_train).reshape(-1,1)

b=np.concatenate([a,y_train],axis=1)

combine_predict_actual=pd.DataFrame(b,columns=["Predict","Actual"])

print(combine_predict_actual.head(10))

print()

print("R^2 score :",lr.score(x_test,y_test))



# We see that our predictions give us almost same spliting with the actuals.

# Our logistic regression score is 78% that is also high R^2 score.

# However,we see that some predict values cannot same as actuals like fifth one.
#Let's make another classification that is KNN.

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

a=knn.predict(x_train).reshape(-1,1)

b=np.concatenate([a,y_train],axis=1)

combine_predict_actual=pd.DataFrame(b,columns=["Predict","Actual"])

print(combine_predict_actual.head(10))

print()

print('KNN (K=3) score is: ',knn.score(x_test,y_test))

# Now, we see that KNN predict the values better than logistic regression. 

# When logistic regression predict the fifth value wrong, KNN do not predict wrong the fifth value.

# We also see that KNN score is impressively high.
# Still we use the k=3, let's find out the which k values give us a best classification rate.



neig = np.arange(1, 25)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('Value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))



# That graph show us the best k value is 10.
# We find the best k values that is 10 then make our KNN with k=10.

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(x_train,y_train)

a=knn.predict(x_train).reshape(-1,1)

b=np.concatenate([a,y_train],axis=1)

combine_predict_actual=pd.DataFrame(b,columns=["Predict","Actual"])

print(combine_predict_actual.head(10))

print()

print('KNN (K=3) score is: ',knn.score(x_test,y_test))



# Now,we find the best prediction result with using KNN and k=10.