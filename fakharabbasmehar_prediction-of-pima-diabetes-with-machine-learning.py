# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt       # matplotlib.pyplot plots data





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pdata = pd.read_csv("../input/diabetes.csv")
pdata.shape # Check number of columns and rows in data frame
pdata.head() # To check first 5 rows of data set
pdata.tail() # Check last 5 rows of data
pdata.isnull().values.any() # If there are any null values in data set
columns = list(pdata)[0:-1] # Excluding Outcome column which has only 

pdata[columns].hist(stacked=False, bins=100, figsize=(12,30), layout=(14,2)); 

# Histogram of first 8 columns
pdata.corr() # It will show correlation matrix 
# However we want to see correlation in graphical representation so below is function for that

def plot_corr(df, size=11):

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    ax.matshow(corr)

    plt.xticks(range(len(corr.columns)), corr.columns)

    plt.yticks(range(len(corr.columns)), corr.columns)
plot_corr(pdata)
n_true = len(pdata.loc[pdata['Outcome'] == True])

n_false = len(pdata.loc[pdata['Outcome'] == False])

print("Number of true cases: {0} ({1:2.2f}%)".format(n_true, (n_true / (n_true + n_false)) * 100 ))

print("Number of false cases: {0} ({1:2.2f}%)".format(n_false, (n_false / (n_true + n_false)) * 100))
from sklearn.model_selection import train_test_split



features_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

predicted_class = ['Outcome']



X = pdata[features_cols].values      # Predictor feature columns (8 X m)

Y = pdata[predicted_class]. values   # Predicted class (1=True, 0=False) (1 X m)

split_test_size = 0.30



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=split_test_size, random_state=52)

# I took 52 as just any random seed number

print("{0:0.2f}% data is in training set".format((len(x_train)/len(pdata.index)) * 100))

print("{0:0.2f}% data is in test set".format((len(x_test)/len(pdata.index)) * 100))
print("Original Diabetes True Values    : {0} ({1:0.2f}%)".format(len(pdata.loc[pdata['Outcome'] == 1]), (len(pdata.loc[pdata['Outcome'] == 1])/len(pdata.index)) * 100))

print("Original Diabetes False Values   : {0} ({1:0.2f}%)".format(len(pdata.loc[pdata['Outcome'] == 0]), (len(pdata.loc[pdata['Outcome'] == 0])/len(pdata.index)) * 100))

print("")

print("Training Diabetes True Values    : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train)) * 100))

print("Training Diabetes False Values   : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train)) * 100))

print("")

print("Test Diabetes True Values        : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test)) * 100))

print("Test Diabetes False Values       : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test)) * 100))

print("")
pdata.head()
from sklearn.preprocessing import Imputer



rep_0 = Imputer(missing_values=0, strategy="mean", axis=0)



x_train = rep_0.fit_transform(x_train)

x_test = rep_0.fit_transform(x_test)
from sklearn.naive_bayes import GaussianNB # I am using Gaussian algorithm from Naive Bayes



# Lets creat the model

diab_model = GaussianNB()



diab_model.fit(x_train, y_train.ravel())
diab_train_predict = diab_model.predict(x_train)



from sklearn import metrics



print("Model Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, diab_train_predict)))

print()
diab_test_predict = diab_model.predict(x_test)



from sklearn import metrics



print("Model Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, diab_test_predict)))

print()
print("Confusion Matrix")



print("{0}".format(metrics.confusion_matrix(y_test, diab_test_predict, labels=[1, 0])))

print("")



print("Classification Report")

print(metrics.classification_report(y_test, diab_test_predict, labels=[1, 0]))
from sklearn.ensemble import RandomForestClassifier

diab_rf_model = RandomForestClassifier(random_state=52)

diab_rf_model.fit(x_train, y_train.ravel())
rf_train_predict = diab_rf_model.predict(x_train)

print("Model Accuracy: {0:.2f}".format(metrics.accuracy_score(y_train, rf_train_predict)))
rf_test_predict = diab_rf_model.predict(x_test)

print("Model Accuracy: {0:.2f}".format(metrics.accuracy_score(y_test, rf_test_predict)))
print("Confusion Matrix")

print(metrics.confusion_matrix(y_test, rf_test_predict, labels=[1, 0]))

print("")

print("Classification Report")

print(metrics.classification_report(y_test, rf_test_predict, labels=[1, 0]))
from sklearn.linear_model import LogisticRegression



diab_lr_model = LogisticRegression(C=0.7, random_state=52)

diab_lr_model.fit(x_train, y_train.ravel())

lr_test_predict = diab_lr_model.predict(x_test)



print("Model Accuracy: {0:.2f}".format(metrics.accuracy_score(y_test, lr_test_predict)))

print("")

print("Confusion Matrix")

print(metrics.confusion_matrix(y_test, lr_test_predict, labels=[1, 0]))

print("")

print("Classification Report")

print(metrics.classification_report(y_test, lr_test_predict, labels=[1, 0]))
from sklearn.linear_model import LogisticRegressionCV

diab_lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=52, Cs=3, cv=10, refit=True, class_weight="balanced")

# As this algorithm uses k-fold cross validation so I am using 10 folds. Also I am using class_weight as balanced so it will use balanced data for Outcome

diab_lr_cv_model.fit(x_train, y_train.ravel())
lr_cv_test_predict = diab_lr_cv_model.predict(x_test)



print("Model Accuracy: {0:.2f}".format(metrics.accuracy_score(y_test, lr_cv_test_predict)))

print("")

print("Confusion Matrix")

print(metrics.confusion_matrix(y_test, lr_cv_test_predict, labels=[1, 0]))

print("")

print("Classification Report")

print(metrics.classification_report(y_test, lr_cv_test_predict, labels=[1, 0]))