# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler 

import xgboost

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings('ignore')
# Take a look at the data

df = pd.read_csv("../input/falldata/falldeteciton.csv", sep=",")

print(df.head(10))
# Data dimensionality [rows, columns]

print(df.shape)
# Check data quality

df.info()
# Describe the dataframe columns

# We will discard activity column as that is a nominal attribute

df.iloc[:,1:7].describe()
d = df["ACTIVITY"].value_counts().sort_index()

print(d)
# Pie chart, where the slices will be ordered and plotted counter-clockwise:

dict = {0:'Standing', 1:'Walking', 2:'Sitting', 3:'Falling', 4:'Cramps', 5:'Running'}

resp = list(dict.keys())

labels = list(dict.values())

sizes = [d[0], d[1], d[2], d[3], d[4], d[5]]

explode = (0, 0, 0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, explode = explode, autopct='%1.1f%%', startangle = 90, counterclock=False, shadow=False)

ax1.axis('equal')

plt.show()
# Visualize with Bar chart

#plt.bar(labels, d)

#plt.show()

sns.set(style="darkgrid")

ax = sns.countplot(y='ACTIVITY', data=df)

ax.set_yticklabels(labels);
# Histograms

df.iloc[:,1:7].hist(bins=10,figsize=(15, 15))

plt.show()
# Density

df.iloc[:,1:7].plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(15, 15))

plt.show()
# Create pivot_table

colum_names = ['TIME','SL','EEG','BP','HR','CIRCLUATION']

df_pivot_table = df.pivot_table(colum_names,

               ['ACTIVITY'], aggfunc='median')

print(df_pivot_table)
# Correlation matrix

tmp = df.drop('ACTIVITY', axis=1)

correlations = tmp.corr()

print(correlations)

# Plot figsize

fig, ax = plt.subplots(figsize=(15, 11))

# Generate Color Map

colormap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate Heat Map, allow annotations and place floats in map

sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")

ax.set_xticklabels(

    colum_names,

    rotation=45,

    horizontalalignment='right'

);

ax.set_yticklabels(colum_names);
#temp = df.iloc[:,[1,2,5,6]]

#temp.describe()

# Correlation matrix

temp = df.drop(['ACTIVITY', 'EEG', 'BP'], axis=1)

correlations = temp.corr()

print(correlations)

# Plot figsize

fig, ax = plt.subplots(figsize=(8, 6))

# Generate Color Map

colormap = sns.diverging_palette(220, 10, as_cmap=True)

# Generate Heat Map, allow annotations and place floats in map

sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")

ax.set_xticklabels(

    ['TIME','SL','HR','CIRCLUATION'],

    rotation=45,

    horizontalalignment='right'

);

ax.set_yticklabels(['TIME','SL','HR','CIRCLUATION']);
sns.set(style='ticks')

sns.pairplot(tmp)
# Use boxplot to do the outlier analysis for the dataset feature variables

# Boxplot for 'TIME'

sns.boxplot(y=df['TIME'], x=df['ACTIVITY'])
# Use boxplot to do the outlier analysis for the dataset feature variables

# Boxplot for 'SL'

ax = sns.boxplot(y=df['SL'], x=df['ACTIVITY'])
# Use boxplot to do the outlier analysis for the dataset feature variables

# Boxplot for 'EEG'

ax = sns.boxplot(y=df['EEG'], x=df['ACTIVITY'])
# Use boxplot to do the outlier analysis for the dataset feature variables

# Boxplot for 'HR'

ax = sns.boxplot(y=df['HR'], x=df['ACTIVITY'])
# Use boxplot tdfo do the outlier analysis for the dataset feature variables

# Boxplot for 'BP'

ax = sns.boxplot(y=df['BP'], x=df['ACTIVITY'])
# Use boxplot to do the outlier analysis for the dataset feature variables

# Boxplot for 'CIRCLUATION'

ax = sns.boxplot(y=df['CIRCLUATION'], x=df['ACTIVITY'])
# Remove outliers from dataset df



Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

df_out.shape
# Create a new column called Decision. This column will contain the values of 0 = 'No Fall', 1 = 'Fall' using following rule: 

# Activity Value : 3 --> Fall, else --> No Fall



decision = []

for i in df_out['ACTIVITY']:

    if i == 3:

        decision.append('1')

    else: 

        decision.append('0')

df_out['DECISION'] = decision

print(df_out.head(10))
df_out['DECISION'].value_counts().sort_index()
# Split the dataset into x and y : x -> Feature variables, y -> Class variable

X = df_out.iloc[:,1:7]

y = df_out['DECISION']

print(X.shape)

print(y.shape)
print(X.head(10))
print(y.head(10))
# Split dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)



# Apply standard scaling to get optimized result

sc = StandardScaler()

#sc = MinMaxScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# Proceed with Modelling. We will use the following algorithms and check for best accuracy:



#  1. Logistic Regression

#  2. Decision Trees

#  3. K-Nearest Neighbors

#  4. Naive Bayes

#  5. Random Forests

#  6. Support Vector Machines

#  7. Stochastic Gradient Decent Classifier
# Perform Logistic Regression Classifier

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_predict = lr.predict(X_test)



# Print confusion matrix and accuracy score

lr_conf_matrix = confusion_matrix(y_test, lr_predict)

lr_acc_score = accuracy_score(y_test, lr_predict)

lr_class_report = classification_report(y_test, lr_predict) 

print(lr_conf_matrix)

print('Accuracy Score :', '%.2f' %lr_acc_score)

print('Classification Report :')

print(lr_class_report)
# Perform Decision Trees Classifier

dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

dt_predict = dt.predict(X_test)



# Print confusion matrix and accuracy score

dt_conf_matrix = confusion_matrix(y_test, dt_predict)

dt_acc_score = accuracy_score(y_test, dt_predict)

dt_class_report = classification_report(y_test, dt_predict) 

print(dt_conf_matrix)

print('Accuracy Score :', '%.2f' %dt_acc_score)

print('Classification Report :')

print(dt_class_report)
# Perform K-Nearest Neighbors Classifier

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

knn_predict = knn.predict(X_test)



# Print confusion matrix and accuracy score

knn_conf_matrix = confusion_matrix(y_test, knn_predict)

knn_acc_score = accuracy_score(y_test, knn_predict)

knn_class_report = classification_report(y_test, knn_predict) 

print(knn_conf_matrix)

print('Accuracy Score :', '%.2f' %knn_acc_score)

print('Classification Report :')

print(knn_class_report)
# Perform Naive Bayes Classifier

nb = GaussianNB()

nb.fit(X_train,y_train)

nb_predict = nb.predict(X_test)



# Print confusion matrix and accuracy score

nb_conf_matrix = confusion_matrix(y_test, nb_predict)

nb_acc_score = accuracy_score(y_test, nb_predict)

nb_class_report = classification_report(y_test, nb_predict) 

print(nb_conf_matrix)

print('Accuracy Score :', '%.2f' %nb_acc_score)

print('Classification Report :')

print(nb_class_report)
# Perform Random Forest Classifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

rf_predict = rf.predict(X_test)



# Print confusion matrix and accuracy score

rf_conf_matrix = confusion_matrix(y_test, rf_predict)

rf_acc_score = accuracy_score(y_test, rf_predict)

rf_class_report = classification_report(y_test, rf_predict)

print(rf_conf_matrix)

print('Accuracy Score :','%.2f' %rf_acc_score)

print('Classification Report :')

print(rf_class_report)
# Perform SVM Classifier

svc = SVC()

svc.fit(X_train,y_train)

svc_predict = svc.predict(X_test)



# Print confusion matrix and accuracy score

svc_conf_matrix = confusion_matrix(y_test, svc_predict)

svc_acc_score = accuracy_score(y_test, svc_predict)

svc_class_report = classification_report(y_test, svc_predict)

print(svc_conf_matrix)

print('Accuracy Score :','%.2f' %svc_acc_score)

print('Classification Report :')

print(svc_class_report)
# Perform SGDC

sgdc = SGDClassifier()

sgdc.fit(X_train,y_train)

sgdc_predict = sgdc.predict(X_test)



# Print confusion matrix and accuracy score

sgdc_conf_matrix = confusion_matrix(y_test, sgdc_predict)

sgdc_acc_score = accuracy_score(y_test, sgdc_predict)

sgdc_class_report = classification_report(y_test, sgdc_predict)

print(sgdc_conf_matrix)

print('Accuracy Score :','%.2f' %sgdc_acc_score)

print('Classification Report :')

print(sgdc_class_report)
# From the above results it looks like KNN and Random Forest models are giving best accuracy result for our model

#

# We will implement following to get the mean accuracy of these two models:

#       a. K Fold Cross Validation

#       b. Stratified K Fold Cross Validation
# K Fold CV with Random Forest classifier



score = cross_val_score(rf, X, y, cv=10)

print(score)

print('Mean accuracy :')

print('%.2f' %score.mean())
# K Fold CV with KNN classifier



score = cross_val_score(knn, X, y, cv=10)

print(score)

print('Mean accuracy :')

print('%.2f' %score.mean())
# Set up Stratified K Fold Cross Validation with n_splits=10



skf = StratifiedKFold(n_splits=10, random_state=None)

skf.get_n_splits(X,y)
# SKFCV for Random Forest classifier



accuracy=[]



for train_index, test_index in skf.split(X, y):

    #print('Train :' , train_index, 'Test : ', test_index)

    X1_train, X1_test = X.iloc[train_index], X.iloc[test_index]

    y1_train, y1_test = y.iloc[train_index], y.iloc[test_index]

    

    rf.fit(X1_train, y1_train)

    prediction = rf.predict(X1_test)

    score = accuracy_score(prediction, y1_test)

    accuracy.append(score)

    

print(accuracy)

print('Mean accuracy :')

print('%.2f' %np.array(accuracy).mean())
# SKFCV for K Nearest Neighbour classifier



accuracy=[]



for train_index, test_index in skf.split(X, y):

    #print('Train :' , train_index, 'Test : ', test_index)

    X2_train, X2_test = X.iloc[train_index], X.iloc[test_index]

    y2_train, y2_test = y.iloc[train_index], y.iloc[test_index]

    

    knn.fit(X2_train, y2_train)

    prediction = knn.predict(X2_test)

    score = accuracy_score(prediction, y2_test)

    accuracy.append(score)

    

print(accuracy)

print('Mean accuracy :')

print('%.2f' %np.array(accuracy).mean())