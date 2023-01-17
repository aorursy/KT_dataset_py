# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/input/pima-indians-diabetes-database/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

Pima_df =pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Pima_df.head()
Pima_df.shape
#Checking for Missing Values.

Pima_df.isnull().mean()

#Checking the  feature correlation with the OutCome (dependent variable).

corr =Pima_df.corr()

corr['Outcome'].sort_values(ascending =False)

plt.figure(figsize=(20,20))

top_corr_features = corr.index

#heat map

g=sns.heatmap(Pima_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#Replacing Os with NaN and Dropping NaN's

Pima_df[['Glucose', 'BMI']]=Pima_df[['Glucose', 'BMI']].replace(0,np.NaN)

Pima_df.dropna(inplace=True)
#Separating the Features and Target Variable

X=Pima_df.iloc[:,0:8]

y_actual=Pima_df.iloc[:,8]

# For Standarad Normal Distribution should have mean =0 and standard devaiation =1.So scale our selected columns for the model

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)



mean = np.mean(X, axis=0)

print('Mean: (%d, %d)' % (mean[0], mean[1]))

standard_deviation = np.std(X, axis=0)

print('Standard deviation: (%d, %d)' % (standard_deviation[0], standard_deviation[1]))

#Now training the model.Splitting into 80% Training Data and 20% Test Data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_actual, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

#Predicting using the Logistic Regression model

model.fit(X_train, y_train.ravel())

y_predicted = model.predict(X_test)

#Testing our model

y_actual

y_predicted
#Model Acuuracy on training data.

model.score(X_train,y_train)
#Model Accuracy on test data

model.score(X_test,y_test)
#Checking the classification metrics 

print("Accuracy:",metrics.accuracy_score(y_test,y_predicted))

print("Precision:",metrics.precision_score(y_test, y_predicted))

print("Recall:",metrics.recall_score(y_test, y_predicted))
#Confusion Matrix.

from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test, y_predicted)

print("Confusion Matrx",confusion_mat)
#Checking the ROC Curve

y_pred_proba = model.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()

#AUC curve is  0.86 which indicates a good Classifier.