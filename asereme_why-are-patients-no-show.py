import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

from sklearn import metrics

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import auc

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.preprocessing import MinMaxScaler



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load dataset

data = pd.read_csv("../input/No-show-Issue-Comma-300k.csv")
data.count()
#Function

def convert_age_(age):

    if int(age) < 0:

        age = age * (-1)

        return age

    else:

        return age

    

def get_binary_status(status):

    status = status.strip()

    if status == 'Show-Up':

        return 1

    elif status == 'No-Show':

        return 0

    

def score_auc(model, X, y):

    pred = model.predict(X)

    fpr, tpr, thresholds = metrics.roc_curve(y, pred)

    return metrics.auc(fpr, tpr)
#get descriptive stats for the data

data.describe()
#get the proportion of no_show vs show-up

data.groupby('Status').count()
#Check for null values

data.isnull().any()
# Plot histogram for age

data['Age'].hist()

plt.show()
#Find records with negative age

neg_age = data[data['Age'] < 0]

neg_age.count()
#Plot a correlation matrix

# get the corr first

correlation = data.corr()

correlation



# plot the correlation

fig = plt.figure()

ax = fig.add_subplot(111)

cax= ax.matshow(correlation, vmin =-1, vmax = 1)

fig.colorbar(cax)

ticks = np.arange(0,10,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_yticklabels(data.columns)

plt.show()
# get binary data for status {1: Show-up, 0: no-show}

data['Status'] = data['Status'].map(lambda status: get_binary_status(status))



#convert wait time to positive

data['AwaitingTime'] = data['AwaitingTime'].map(lambda time: time*(-1))



# Convert all ages to positive numbers

data['Age'] = data['Age'].map(lambda age: convert_age_(age))





# drop the some variable i deem not necessary because they are already captued in the AwaitingTime

# variable

semi_final = data.drop(['AppointmentRegistration','ApointmentData'],axis =1)



# get dummy variable for the categorical variables

final = pd.get_dummies(semi_final, columns=['DayOfTheWeek','Gender'])



#split data between target variable and dependant variables

X = final.drop('Status', axis = 1)

y = final['Status']
# train and test split, we leave 30 of dataset for testing.



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 7)

model = LogisticRegression(class_weight = 'balanced')

model.fit(X_train, y_train)
results = model.score(X_test, y_test)

results
auc = score_auc(model, X_test, y_test)

auc
#use a random forest classifier. 

rf = RandomForestClassifier(n_jobs=2,max_depth = 5, class_weight='balanced', criterion='entropy', n_estimators=100)

rf.fit(X_train, y_train)
auc = score_auc(rf, X_test, y_test)

auc