# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import seaborn as sns

# For preprocessing the data

from sklearn.preprocessing import Imputer

from sklearn import preprocessing



# To split the dataset into train and test datasets

from sklearn.model_selection import train_test_split



# To calculate the accuracy score of the model

from sklearn.metrics import accuracy_score

from sklearn import metrics

from sklearn.metrics import classification_report,confusion_matrix



df = pd.read_csv("../input/Bank_Personal_Loan_Modelling.csv")
df.head()
df.dtypes
#5 point summary analysis

df.describe()
df.groupby(["Personal Loan"]).count()



# Class distribution among B and M is almost 2:1. The model will better predict B and M
df.columns[(df == 0).all()]
# The first column is id column which is customer ID and nothing to do with the model attriibutes. So drop it.

df =  df.drop(columns=['ID'], axis=1)

df
# Number of records(rows) in the dataframe

len(df)
df.isnull().values.any() # If there are any null values in data set
# Handling missing data

# Test whether there is any null value in our dataset or not. We can do this using isnull() method.

df.isnull().sum()
# Excluding Outcome column which has only 

df.drop(['Personal Loan'], axis=1).hist(stacked=False, bins=100, figsize=(30,45), layout=(14,4))
df.corr() # It will show correlation matrix
# However we want to see correlation in graphical representation so below is function for that

def plot_corr(df, size=11):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)
plot_corr(df)
sns.pairplot(df,diag_kind='kde')
from sklearn.model_selection import train_test_split



X = df.drop('Personal Loan',axis=1)     # Predictor feature columns (8 X m)

Y = df['Personal Loan']   # Predicted class (1=True, 0=False) (1 X m)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# 1 is just any random seed number



X_train.head()
# Checking if any column has any 0.

(df == 0).all()
from sklearn.linear_model import LogisticRegression



#Build the logistic regression model

logisticRegr = LogisticRegression()



logisticRegr.fit(X_train, y_train)
# Use score method to get accuracy of model

score = logisticRegr.score(X_test, y_test)

print(score)
#Predict for train set

pred_train = logisticRegr.predict(X_train)

mat_train = confusion_matrix(y_train,pred_train)



print("confusion matrix = \n",mat_train)
#Predict for test set

pred_test = logisticRegr.predict(X_test)



mat_test = confusion_matrix(y_test,pred_test)

print("confusion matrix = \n",mat_test)
cm = metrics.confusion_matrix(y_test, pred_test, labels=[1, 0])

cm
df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)
print("Classification Report for Logistic Regression")

print(metrics.classification_report(y_test, pred_test, labels=[1, 0]))
#AUC ROC curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logit_roc_auc = roc_auc_score(y_test, logisticRegr.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logisticRegr.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
auc_score = metrics.roc_auc_score(y_test, logisticRegr.predict_proba(X_test)[:,1])

round( float( auc_score ), 3 )
from sklearn.naive_bayes import GaussianNB # using Gaussian algorithm from Naive Bayes



# create the model

naive_model = GaussianNB()



naive_model.fit(X_train, y_train.ravel())
#Predict for train set

naive_train_predict = naive_model.predict(X_train)





print("Model Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, naive_train_predict)))

print()
#Predict for test set

naive_test_predict = naive_model.predict(X_test)



print("Model Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, naive_test_predict)))

print()
mat_test = confusion_matrix(y_test,naive_test_predict)

print("confusion matrix = \n",mat_test)
print("Confusion Matrix for Naive Bayes")

cm = metrics.confusion_matrix(y_test, naive_test_predict, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)
print("Classification Report for Naive Bayes")

print(metrics.classification_report(y_test, naive_test_predict, labels=[1, 0]))
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore

# convert the features into z scores as we do not know what units / scales were used and store them in new dataframe

# It is always adviced to scale numeric attributes in models that calculate distances.



XScaled  = X.apply(zscore)  # convert all attributes to Z scale 



XScaled.describe()
# Split X and y into training and test set in 75:25 ratio



X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(XScaled, Y, test_size=0.30, random_state=1)
NNH = KNeighborsClassifier(n_neighbors= 7 , weights = 'distance' )
# Call Nearest Neighbour algorithm



NNH.fit(X_train_knn, y_train_knn)
# For every test data point, predict it's label based on 5 nearest neighbours in this model. The majority class will 

# be assigned to the test data point



predicted_labels = NNH.predict(X_test_knn)

NNH.score(X_test_knn, y_test_knn)
mat_test = confusion_matrix(y_test_knn,predicted_labels)

print("confusion matrix for KNN = \n",mat_test)
cm = metrics.confusion_matrix(y_test_knn, predicted_labels, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)
print("Classification Report for KNN")

print(metrics.classification_report(y_test_knn, predicted_labels, labels=[1, 0]))