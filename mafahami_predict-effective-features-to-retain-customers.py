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
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import hamming_loss



telecom = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# *******************************************************************
# some initializations:
sns.set_palette("Set1")

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# 1) Gender Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'gender' )

pl.set_title('Gender Vs Churn', fontsize=12)
pl.set_ylabel('Number of each Gender')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('1 Gender_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'gender'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 2) SeniorCitizen Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'SeniorCitizen' )

pl.set_title('SeniorCitizen Vs Churn', fontsize=12)
pl.set_ylabel('Number of each SeniorCitizen Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('2 SeniorCitizen_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'SeniorCitizen'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 3) Partner Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'Partner' )

pl.set_title('Partner Vs Churn', fontsize=12)
pl.set_ylabel('Number of each Partner Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('3 Partner_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'Partner'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 4) Dependents Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'Dependents' )

pl.set_title('Dependents Vs Churn', fontsize=12)
pl.set_ylabel('Number of each Dependents Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('4 Dependents_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'Dependents'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 5) tenure Vs Churn:

telecom_average = telecom.groupby( [ 'Churn' ] )['tenure'].mean().reset_index(name='Average_of_Tenure')



plt.figure()
pl = sns.boxplot(x='Churn', y = 'tenure' , data=telecom)

pl.set_title('tenure Vs Churn', fontsize=12)
pl.set_ylabel('Tenure')
pl.set_xlabel('Churn')

fig = pl.get_figure()
fig.savefig('5 tenure_vs_Churn.png')
plt.show()
telecom_average

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 6) PhoneService Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'PhoneService' )

pl.set_title('PhoneService Vs Churn', fontsize=12)
pl.set_ylabel('Number of each PhoneService Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('6 PhoneService_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'PhoneService'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 7) MultipleLines Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'MultipleLines' )

pl.set_title('MultipleLines Vs Churn', fontsize=12)
pl.set_ylabel('Number of each MultipleLines Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('7 MultipleLines_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'MultipleLines'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 8) InternetService Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'InternetService' )

pl.set_title('InternetService Vs Churn', fontsize=12)
pl.set_ylabel('Number of each InternetService Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('8 InternetService_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'InternetService'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 9) OnlineSecurity Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'OnlineSecurity' )

pl.set_title('OnlineSecurity Vs Churn', fontsize=12)
pl.set_ylabel('Number of each OnlineSecurity Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('9 OnlineSecurity_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'OnlineSecurity'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 10) OnlineBackup Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'OnlineBackup' )

pl.set_title('OnlineBackup Vs Churn', fontsize=12)
pl.set_ylabel('Number of each OnlineBackup Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('10 OnlineBackup_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'OnlineBackup'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 11) DeviceProtection Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'DeviceProtection' )

pl.set_title('DeviceProtection Vs Churn', fontsize=12)
pl.set_ylabel('Number of each DeviceProtection Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('11 DeviceProtection_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'DeviceProtection'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 12) TechSupport Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'TechSupport' )

pl.set_title('TechSupport Vs Churn', fontsize=12)
pl.set_ylabel('Number of each TechSupport Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('12 TechSupport_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'TechSupport'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 13) StreamingTV Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'StreamingTV' )

pl.set_title('StreamingTV Vs Churn', fontsize=12)
pl.set_ylabel('Number of each StreamingTV Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('13 StreamingTV_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'StreamingTV'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 14) StreamingMovies Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'StreamingMovies' )

pl.set_title('StreamingMovies Vs Churn', fontsize=12)
pl.set_ylabel('Number of each StreamingMovies Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('14 StreamingMovies_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'StreamingMovies'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 15) Number of Online Services Vs Churn:
tel = telecom
#count of online services availed
tel['Number_of_Online_Services'] = (tel[['OnlineSecurity', 'DeviceProtection', 'StreamingMovies', 'TechSupport',
       'StreamingTV', 'OnlineBackup']] == 'Yes').sum(axis=1)

plt.figure()
pl = sns.countplot(x = 'Number_of_Online_Services' , data = telecom , hue = 'Churn' )

pl.set_title('Number_of_Online_Services Vs Churn', fontsize=12)
pl.set_ylabel('Number of Customers')
pl.set_xlabel('umber of Online Services')

fig = pl.get_figure()
fig.savefig('15 Number_of_Online_Services_vs_Churn.png')
plt.show()

g = tel.groupby( ['Number_of_Online_Services' , 'Churn' ] ).size().reset_index(name='Number_of_Customers')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 16) Contract Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'Contract' )

pl.set_title('Contract Vs Churn', fontsize=12)
pl.set_ylabel('Number of each Contract Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('16 Contract_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'Contract'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 17) PaperlessBilling Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'PaperlessBilling' )

pl.set_title('PaperlessBilling Vs Churn', fontsize=12)
pl.set_ylabel('Number of each PaperlessBilling Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('17 PaperlessBilling_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'PaperlessBilling'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 18) PaymentMethod Vs Churn:

plt.figure()
pl = sns.countplot(x = 'Churn' , data = telecom , hue = 'PaymentMethod' )

pl.set_title('PaymentMethod Vs Churn', fontsize=12)
pl.set_ylabel('Number of each PaymentMethod Type')
# pl.set_xlabel('')

fig = pl.get_figure()
fig.savefig('18 PaymentMethod_vs_Churn.png')
plt.show()

g = telecom.groupby( [ 'Churn' , 'PaymentMethod'] ).size().reset_index(name='Size')
g
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 19) MonthlyCharges Vs Churn:

# plt.figure()
telecom_average = telecom.groupby( [ 'Churn' ] )['MonthlyCharges'].mean().reset_index(name='Average_of_MonthlyCharges')

plt.figure()
pl = sns.boxplot(x='Churn', y = 'MonthlyCharges' , data=telecom)

pl.set_title('MonthlyCharges Vs Churn', fontsize=12)
pl.set_ylabel('MonthlyCharges')
pl.set_xlabel('Churn')

fig = pl.get_figure()
fig.savefig('19 MonthlyCharges_vs_Churn.png')
plt.show()

telecom_average
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# 20) TotalCharges Vs Churn:

# plt.figure()
# tel = telecom
#Replacing spaces with null values in total charges column
tel['TotalCharges'] = tel["TotalCharges"].replace(" ",np.nan)

#Dropping null values from total charges column which contain .15% missing data
tel = tel[tel["TotalCharges"].notnull()]
tel = tel.reset_index()[tel.columns]
tel["TotalCharges"] = tel["TotalCharges"].astype(float)


telecom_average = tel.groupby( [ 'Churn' ] )['TotalCharges'].mean().reset_index(name='Average_of_TotalCharges')

plt.figure()
pl = sns.boxplot(x='Churn', y = 'TotalCharges' , data=tel)

pl.set_title('TotalCharges Vs Churn', fontsize=12)
pl.set_ylabel('TotalCharges')
pl.set_xlabel('Churn')

fig = pl.get_figure()
fig.savefig('20 TotalCharges_vs_Churn.png')
plt.show()

telecom_average
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


## Classification of most effective features:

# telecom = tel

# First we should prepare the data for classification:
# We should convert the categorial features to numeric values

encodedTelecom = tel

telecomDict = {"Male":1,"Female":2,
               "Yes":3,"No":4,
               "No phone service":5,
               "DSL":6,"Fiber optic":7,
               "No internet service":8,
               "Month-to-month":9,"One year":10,"Two year":11,
               "Electronic check":12,"Mailed check":13,"Bank transfer (automatic)":14}

encodedTelecom.replace(telecomDict, inplace=True)


# create training and testing variables For classification:

x = encodedTelecom[['tenure','InternetService','Contract','OnlineSecurity']]
# x = encodedTelecom[['tenure','InternetService','Contract','OnlineSecurity',
#                     'OnlineBackup','TechSupport',
#                     'Number_of_Online_Services','MonthlyCharges']]
y = encodedTelecom['Churn']


X_train, X_test, y_train, y_test = train_test_split( x ,y ,  test_size=0.2)

# Fit the Classification Model:
# ######################################################################################
#  Gaussian Naive Bayes
print("*******************************************************************************")
print("*******************************************************************************")
print("Gaussian Naive Bayes")

clf_GaussianNB = GaussianNB()
clf_GaussianNB.fit(X_train, y_train)

predictedY_test = clf_GaussianNB.predict(X_test)  # predicted labels for test data

print(classification_report(y_test, predictedY_test))
print("accuracy: ", accuracy_score(y_test , predictedY_test)  )
print("Hamming loss: ", hamming_loss (y_test, predictedY_test))

print("*******************************************************************************")

# ######################################################################################
# ######################################################################################
# # Decision Tree Classifier
print("Decision Tree Classifier")

clf_DecisionTreeClassifier = DecisionTreeClassifier(random_state=0)
clf_DecisionTreeClassifier.fit(X_train, y_train)

predictedY_test = clf_DecisionTreeClassifier.predict(X_test)  # predicted labels for test data

print(classification_report(y_test, predictedY_test))
print("accuracy: ", accuracy_score(y_test , predictedY_test)  )
print("Hamming loss: ", hamming_loss (y_test, predictedY_test))

print("*******************************************************************************")

# ######################################################################################
# ######################################################################################
# # Random Forest Classifier
print("Random Forest Classifier")

clf_RandomForestClassifier = RandomForestClassifier()
clf_RandomForestClassifier.fit(X_train, y_train)

predictedY_test = clf_RandomForestClassifier.predict(X_test)  # predicted labels for test data

print(classification_report(y_test, predictedY_test))
print("accuracy: ", accuracy_score(y_test , predictedY_test)  )
print("Hamming loss: ", hamming_loss (y_test, predictedY_test))

print("*******************************************************************************")

# ######################################################################################
# ######################################################################################
# # K-Nearest Neighbor Classifier (KNN)
print("K-Nearest Neighbor Classifier")

clf_KNeighborsClassifier = KNeighborsClassifier()
clf_KNeighborsClassifier.fit(X_train, y_train)

predictedY_test = clf_KNeighborsClassifier.predict(X_test)  # predicted labels for test data

print(classification_report(y_test, predictedY_test))
print("accuracy: ", accuracy_score(y_test , predictedY_test)  )
print("Hamming loss: ", hamming_loss (y_test, predictedY_test))

print("*******************************************************************************")


print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print("Results with more features:")

# create training and testing variables For classification:

x = encodedTelecom[['tenure','InternetService','Contract','OnlineSecurity',
                    'OnlineBackup','TechSupport',
                    'Number_of_Online_Services','MonthlyCharges']]
y = encodedTelecom['Churn']


X_train, X_test, y_train, y_test = train_test_split( x ,y ,  test_size=0.2)

# Fit the Classification Model:
# ######################################################################################
#  Gaussian Naive Bayes
print("*******************************************************************************")
print("*******************************************************************************")
print("Gaussian Naive Bayes")

clf_GaussianNB = GaussianNB()
clf_GaussianNB.fit(X_train, y_train)

predictedY_test = clf_GaussianNB.predict(X_test)  # predicted labels for test data

print(classification_report(y_test, predictedY_test))
print("accuracy: ", accuracy_score(y_test , predictedY_test)  )
print("Hamming loss: ", hamming_loss (y_test, predictedY_test))

print("*******************************************************************************")

# ######################################################################################
# ######################################################################################
# # Decision Tree Classifier
print("Decision Tree Classifier")

clf_DecisionTreeClassifier = DecisionTreeClassifier(random_state=0)
clf_DecisionTreeClassifier.fit(X_train, y_train)

predictedY_test = clf_DecisionTreeClassifier.predict(X_test)  # predicted labels for test data

print(classification_report(y_test, predictedY_test))
print("accuracy: ", accuracy_score(y_test , predictedY_test)  )
print("Hamming loss: ", hamming_loss (y_test, predictedY_test))

print("*******************************************************************************")

# ######################################################################################
# ######################################################################################
# # Random Forest Classifier
print("Random Forest Classifier")

clf_RandomForestClassifier = RandomForestClassifier()
clf_RandomForestClassifier.fit(X_train, y_train)

predictedY_test = clf_RandomForestClassifier.predict(X_test)  # predicted labels for test data

print(classification_report(y_test, predictedY_test))
print("accuracy: ", accuracy_score(y_test , predictedY_test)  )
print("Hamming loss: ", hamming_loss (y_test, predictedY_test))

print("*******************************************************************************")

# ######################################################################################
# ######################################################################################
# # K-Nearest Neighbor Classifier (KNN)
print("K-Nearest Neighbor Classifier")

clf_KNeighborsClassifier = KNeighborsClassifier()
clf_KNeighborsClassifier.fit(X_train, y_train)

predictedY_test = clf_KNeighborsClassifier.predict(X_test)  # predicted labels for test data

print(classification_report(y_test, predictedY_test))
print("accuracy: ", accuracy_score(y_test , predictedY_test)  )
print("Hamming loss: ", hamming_loss (y_test, predictedY_test))

print("*******************************************************************************")

