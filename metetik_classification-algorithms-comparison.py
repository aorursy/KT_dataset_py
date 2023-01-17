import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #count plot

import plotly.plotly as py #plotly library
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


import os
print(os.listdir("../input"))
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.info()

print()

data_test.info()
data_train.head()
data_test.head()

#As you see, there isn't "Survived" column in test data because it requested from us. 
#Drop unneed columns and save

data_train.drop(["Name","Cabin","Ticket","Embarked"],axis = 1,inplace = True)

data_test.drop(["Name","Cabin","Ticket","Embarked"],axis = 1,inplace = True)
#Split dataframe into 'survived' and 'not survived' so we will use these easily at data visualization

data_survived = data_train[data_train['Survived'] == 1].sort_values('Age') #dataframe that only has datas from survived peoples 

data_not_survived = data_train[data_train['Survived'] == 0].sort_values('Age')

#We will use this serie at line plot

survived_age_number = data_survived.Age.value_counts(sort = False,dropna = True)#How many survived people are from which age

not_survived_age_number = data_not_survived.Age.value_counts(sort = False,dropna = True)

display(survived_age_number)

not_survived_age_number
#0.42,0.67 .. values at tail of serie and this is a wrong sort.Lets fix it.

a = survived_age_number.tail(4)#put values into a.

survived_age_number.drop([0.42,0.67,0.83,0.92],inplace = True)#delete these values from tail of serie

survived_age_number = pd.concat([a,survived_age_number],axis=0)#attach a to head of serie

survived_age_number #Done
#trace1 is green line and trace2 is red line.

trace1 = go.Scatter(
    x = survived_age_number.index,
    y = survived_age_number,
    opacity = 0.75,
    name = "Survived",
    mode = "lines",
    marker=dict(color = 'rgba(0, 230, 0, 0.6)'))

trace2 = go.Scatter(
    x = not_survived_age_number.index,
    y = not_survived_age_number,
    opacity=0.75,
    name = "Not Survived",
    mode = "lines",
    marker=dict(color = 'rgba(230, 0, 0, 0.6)'))

data = [trace1,trace2]
layout = go.Layout(title = 'Age of Survived and not-Survived People in Titanic',
                   xaxis=dict(title='Age'),
                   yaxis=dict( title='Count'),)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
sns.countplot(data_survived.Pclass)
plt.title('Passenger Class of Survived People')
plt.show()
sns.countplot(data_not_survived.Pclass)
plt.title('Passenger Class of Not Survived People')
plt.show()
sns.countplot(data_survived.Sex)
plt.title('Gender of Survived People')
plt.show()
sns.countplot(data_not_survived.Sex)
plt.title('Gender of Not Survived People')
plt.show()
data_train.head()
data_train_x = data_train #We should prepare x and y data for train classification

data_train_x.Sex = [1 if i == 'male' else 0 for i in data_train_x.Sex] #Transform strings to integers

data_train_y = data_train_x.Survived #y is our output  

data_train_x.drop(['PassengerId','Survived'], axis = 1,inplace = True)#drop passengerıd and survived because they will not use while training

data_train_x.fillna(0.0,inplace = True) #fill NaN values with zero.We write '0.0' because we want to fill with float values 

#normalization :  i encountered 'to make conform to or reduce to a norm or standard' definition when i search normalization on google.
#But if you ask simply definition i say that : 'to fit values between 0 and 1'
#Normalization formula : (data - min)/(max-min) 

data_train_x = (data_train_x - np.min(data_train_x))/(np.max(data_train_x) - np.min(data_train_x)).values
#We repeat same process to test dataset

data_test.Sex = [1 if i == 'male' else 0 for i in data_test.Sex]

PassengerId = data_test['PassengerId'].values

data_test.drop(['PassengerId'], axis = 1,inplace = True)

data_test.fillna(0.0,inplace = True)

data_test = (data_test - np.min(data_test))/(np.max(data_test) - np.min(data_test)).values
#Split train data in order to reserve %80 of train data for test .You don't confuse this test data is for check.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_train_x,data_train_y,test_size = 0.2,random_state=1)

score_list = [] #to keep scores of algorithms
from sklearn.linear_model import LogisticRegression #importing logistic regression model

lr = LogisticRegression()

lr.fit(x_train,y_train)#fit or train data

print('Logistic Regression Score : ',lr.score(x_test,y_test))#Ratio of correct predictions

score_list.append(lr.score(x_test,y_test))
#this is our real prediction part

lr.fit(data_train_x,data_train_y)

lr_prediction = lr.predict(data_test)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

print('K-Nearest Neighbors Score : ',knn.score(x_test,y_test))

score_list.append(knn.score(x_test,y_test))
knn.fit(data_train_x,data_train_y)

knn_prediction = knn.predict(data_test)
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print('Support Vector Machine Score : ',svm.score(x_test,y_test))

score_list.append(svm.score(x_test,y_test))
svm.fit(data_train_x,data_train_y)

svm_prediction = svm.predict(data_test)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print('Naive Bayes Score : ',nb.score(x_test,y_test))

score_list.append(nb.score(x_test,y_test))
nb.fit(data_train_x,data_train_y)

nb_prediction = nb.predict(data_test)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print('Decision Tree Score : ',dt.score(x_test,y_test))

score_list.append(dt.score(x_test,y_test))
dt.fit(data_train_x,data_train_y)

dt_prediction = dt.predict(data_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 22,random_state = 40)

rf.fit(x_train,y_train)

print('Random Forest Score : ',rf.score(x_test,y_test))

score_list.append(rf.score(x_test,y_test))
rf.fit(data_train_x,data_train_y)

rf_prediction = rf.predict(data_test)
pr_dict = {'Logistic Regression' : lr_prediction,'KNN' : knn_prediction,'SVM' : svm_prediction,
           'Naive Bayes' : nb_prediction,'Decision Tree' : dt_prediction, 'Random Forest' : rf_prediction}

all_predictions = pd.DataFrame(pr_dict)

all_predictions
final_prediction = [] #final prediction list

#i : range columns , j : range rows

for i in all_predictions.values:
    sum_zero_score = 0 #summary of zero scores
    
    sum_one_score = 0 #summary of one scores
    
    for j in range(5):
        if i[j]==0:
            sum_zero_score += score_list[j]
        else:
            sum_one_score += score_list[j]
    
    if sum_zero_score >= sum_one_score:
        final_prediction.append(0)
    else:
        final_prediction.append(1)
    
output = {'PassengerId' : PassengerId,'Survived' : final_prediction}

submission = pd.DataFrame(output)

submission.to_csv('output.csv', index = False)