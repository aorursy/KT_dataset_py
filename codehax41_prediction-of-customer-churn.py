# use these links to do so:

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from collections import defaultdict

from sklearn import metrics

from pylab import rcParams



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
import os

print(os.listdir('../input/telco-customer-churn'))
data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()
data.info()
data.isnull().sum()
data.dtypes
data.describe()
for item in data.columns:

    print(item)

    print (data[item].unique())
data.drop(['customerID'], axis=1, inplace=True)
data["gender"].replace(['Female','Male'],[0,1],inplace=True)

data["Partner"].replace(['No', 'Yes'], [0, 1], inplace=True)

data["Dependents"].replace(['No', 'Yes'], [0, 1], inplace=True)

data["PhoneService"].replace(['No', 'Yes'], [0, 1], inplace=True)

data["PaperlessBilling"].replace(['No', 'Yes'], [0, 1], inplace=True)

data["Churn"].replace(['No', 'Yes'], [0, 1], inplace=True)

data["StreamingMovies"].replace(['No', 'Yes'], [0, 1], inplace=True)



data["InternetService"].replace(['No','DSL', 'Fiber optic'],[0,1,2],inplace=True)

data["Contract"].replace(['Month-to-month','One year', 'Two year'],[0,1,2],inplace=True)



data = pd.get_dummies(data=data, columns=['PaymentMethod'])



data["MultipleLines"].replace(['No','Yes'],[0,1],inplace=True)

data["OnlineSecurity"].replace(['No','Yes'],[0,1],inplace=True)

data["OnlineBackup"].replace(['No','Yes'],[0,1],inplace=True)

data["DeviceProtection"].replace(['No','Yes'],[0,1],inplace=True)

data["TechSupport"].replace(['No', 'Yes'], [0, 1], inplace=True)

data["StreamingTV"].replace(['No', 'Yes'], [0, 1], inplace=True)
columns_to_convert = ['MultipleLines', 

                      'OnlineSecurity', 

                      'OnlineBackup', 

                      'DeviceProtection', 

                      'TechSupport',

                      'StreamingTV',

                     'StreamingMovies']



for item in columns_to_convert:

    data[item].replace(to_replace='No internet service',  value=0, inplace=True)

    data[item].replace(to_replace='No phone service',  value=0, inplace=True)

data.head()
#We can see TotalCharges is still an object. Fix TotalCharges as a float...

data['TotalCharges'] = data['TotalCharges'].replace(r'\s+', np.nan, regex=True)

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])



data = data.fillna(value=0)
data.dtypes
data.groupby('Churn').size()/len(data) # What is the percentage of churners
data.hist(bins=50, figsize=(20,15));
corr = data.corr()

corr
sns.countplot(data['Churn'],label = 'count')
# Data to plot

labels =data['Churn'].value_counts(sort = True).index

sizes = data['Churn'].value_counts(sort = True)





colors = ["whitesmoke","red"]

explode = (0.1,0)  # explode 1st slice

 

rcParams['figure.figsize'] = 8,8

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=270,)



plt.title('Percent of churn in customer')

plt.show()
sns.countplot(x='SeniorCitizen',data=data,hue='Churn')
plt.scatter(x='MonthlyCharges',y='TotalCharges',alpha=0.1, data=data)
#We plot the correlation matrix, the darker a box is, the more features are correlated

plt.figure(figsize=(12,10))

corr = data.apply(lambda x: pd.factorize(x)[0]).corr()

ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, linewidths=.2, cmap='Blues')
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score
data["Churn"] = data["Churn"].astype(int)

Y = data["Churn"].values

X = data.drop(labels = ["Churn"],axis = 1)

# Create Train & Test Data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
# Running logistic regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

result = model.fit(X_train, y_train)

from sklearn import metrics

prediction_test = model.predict(X_test)

# Print the prediction accuracy

print (metrics.accuracy_score(y_test, prediction_test))
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,

                                  random_state =50, max_features = "auto",

                                  max_leaf_nodes = 30)

model_rf.fit(X_train, y_train)



# Make predictions

prediction_test = model_rf.predict(X_test)

print (metrics.accuracy_score(y_test, prediction_test))
model.svm = SVC(kernel='linear') 

model.svm.fit(X_train,y_train)

preds = model.svm.predict(X_test)

metrics.accuracy_score(y_test, preds)
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train)

preds = model.predict(X_test)

metrics.accuracy_score(y_test, preds)
# AdaBoost Algorithm

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()

# n_estimators = 50 (default value) 

# base_estimator = DecisionTreeClassifier (default value)

model.fit(X_train,y_train)

preds = model.predict(X_test)

metrics.accuracy_score(y_test, preds)
# Create the Confusion matrix

from sklearn.metrics import classification_report, confusion_matrix  

print(confusion_matrix(y_test,preds))  
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, preds)

np.set_printoptions(precision=2)

class_names = ['Not churned','churned']

# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                      title='Normalized confusion matrix')



plt.show()



from sklearn.metrics import classification_report

eval_metrics = classification_report(y_test, preds, target_names=class_names)

print(eval_metrics)