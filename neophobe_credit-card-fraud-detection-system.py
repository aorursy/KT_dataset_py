# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as sns
df=pd.read_csv('../input/creditcard.csv')

df.info()
df.describe()
print(df.isnull().values.any())

for col in df:

    print(col)

    print(format(df[col].unique()))
#plt.rcParams['figure.figsize']= (200,200)

params = {'axes.titlesize':'32',

          'xtick.labelsize':'24',

          'ytick.labelsize':'24'}

plt.rcParams.update(params)

hist=df.hist(figsize=(50, 30))

#ax=fig.gca() 

#hist=df.hist()
plt.rcParams.update(plt.rcParamsDefault)

corr=df.corr()

sns.heatmap(corr)
plt.rcParams.update(plt.rcParamsDefault)

count_classes = pd.value_counts(df['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), ['Genuine','Fraud'])

plt.xlabel("Class")

plt.ylabel("Frequency");
df = df.sample(frac=1)

fraud_df = df.loc[df['Class'] == 1]

normal_df = df.loc[df['Class'] == 0][:int(2.5*len(fraud_df))]

ndf=pd.DataFrame()

ndf=ndf.append(fraud_df, ignore_index=True)

ndf=ndf.append(normal_df, ignore_index=True)

ndf=ndf.sample(frac=1)

y=ndf.Class

X=ndf.iloc[:,:-1]

X_train, X_test, y_train, y_test = train_test_split(

    X,y, test_size=0.33, random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

y_test=y_test.values

print("Total Fraud Cases in test dataset:",len(y_test[y_test==1]))
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

f.suptitle('Time of transaction vs Amount by class')

ax1.scatter(fraud_df.Time,fraud_df.Amount,color='red')

ax1.set_title('Fraud')

ax2.scatter(normal_df.Time, normal_df.Amount,color='green')

ax2.set_title('Normal')

plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharex=False)

f.suptitle('Amount per transaction by class')

bins = 50

ax1.hist(fraud_df.Amount, bins = bins,color='red')

ax1.set_title('Fraud')

ax2.hist(normal_df.Amount, bins = bins,color='green')

ax2.set_title('Normal')

plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.xlim((0, 2000))

plt.yscale('log')

plt.show();
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)

params = {}

params['learning_rate'] = 0.003

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'binary_logloss'

params['sub_feature'] = 0.5

params['num_leaves'] = 50

params['min_data'] = 50

params['max_depth'] = 100

params['num_iterations']=750

clf = lgb.train(params, d_train, 100)







y_pred=clf.predict(X_test)

#convert into binary values

for i in range(len(X_test)):

    if y_pred[i]>=.5:       # setting threshold to .5

       y_pred[i]=1

    else:  

       y_pred[i]=0

y_pred = y_pred.astype(np.int64)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:",cm)

#Accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_pred,y_test)

print("Accuracy:",accuracy)

print("Precision:",cm[1][1]/(cm[1][1]+cm[0][1]))# cm[1][1]-Actual and Predicted both Fraud

                                                # cm[0][1]-Actual Not Fraud Predicted Fraud

print("Recall:",cm[1][1]/(cm[1][1]+cm[1][0]))# cm[1][1]-Actual and Predicted both Fraud

                                             # cm[1][0]-Actual Fraud Predicted Not fraud
# Load libraries

from sklearn.ensemble import AdaBoostClassifier

from sklearn import datasets

# Import train_test_split function

from sklearn.model_selection import train_test_split

#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Create adaboost classifer object

abc = AdaBoostClassifier(n_estimators=200,learning_rate=0.03)

# Train Adaboost Classifer

model = abc.fit(X_train, y_train)

#Predict the response for test dataset

y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:",cm)

print("Precision:",cm[1][1]/(cm[1][1]+cm[0][1]))# cm[1][1]-Actual and Predicted both Fraud

                                                # cm[0][1]-Actual Not Fraud Predicted Fraud

print("Recall:",cm[1][1]/(cm[1][1]+cm[1][0]))# cm[1][1]-Actual and Predicted both Fraud

                                             # cm[1][0]-Actual Fraud Predicted Not fraud
#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:",cm)

print("Precision:",cm[1][1]/(cm[1][1]+cm[0][1]))# cm[1][1]-Actual and Predicted both Fraud

                                                # cm[0][1]-Actual Not Fraud Predicted Fraud

print("Recall:",cm[1][1]/(cm[1][1]+cm[1][0]))# cm[1][1]-Actual and Predicted both Fraud

                                             # cm[1][0]-Actual Fraud Predicted Not fraud