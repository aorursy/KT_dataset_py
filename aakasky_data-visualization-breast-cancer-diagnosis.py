# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
cancer_data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
cancer_data_cleaned= cancer_data.drop(cancer_data.columns[32], axis=1)
cancer_data_cleaned= cancer_data.drop(["id","diagnosis",'Unnamed: 32'], axis=1)
cancer_data_labels= cancer_data["diagnosis"]
#print(cancer_data_labels.head())
cancer_data_cleaned.head()
cancer_data_cleaned.info()
cancer_data_cleaned.describe()
%matplotlib inline
import matplotlib.pyplot as plt
cancer_data_cleaned.hist(bins=50,figsize=(20,15))
plt.show()
cancer_data_labels.hist()
#lets plot better visualization 

ax= sns.countplot(cancer_data_labels,label="Count")
B,M = cancer_data_labels.value_counts()
print("Number of Benign: ", B)
print("Number of Malignant: ", M)

cancer = cancer_data_cleaned.copy()
data_dia = cancer_data_labels
data = cancer
data_n_2 = (data-data.mean()) / (data.std())
data = pd.concat([cancer_data_labels,data_n_2.iloc[:,0:10]],axis=1)
#print(data)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name="value")
#print(data)
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

#box plot

# box plots are also useful in terms of seeing outliers
# I do not visualize all features with box plot
# In order to show you lets have an example of box plot
# If you want, you can visualize other features as well.
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# Next 10 features
data = pd.concat([cancer_data_labels,data_n_2.iloc[:,10:20]],axis=1)
#print(data)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name="value")
#print(data)
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

#box Plot

# box plots are also useful in terms of seeing outliers
# I do not visualize all features with box plot
# In order to show you lets have an example of box plot
# If you want, you can visualize other features as well.
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


# Next 10 features
data = pd.concat([cancer_data_labels,data_n_2.iloc[:,20:31]],axis=1)
#print(data)
data = pd.melt(data,id_vars="diagnosis",var_name="features",value_name="value")
#print(data)
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=90)

#box Plot

# box plots are also useful in terms of seeing outliers
# I do not visualize all features with box plot
# In order to show you lets have an example of box plot
# If you want, you can visualize other features as well.
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)


#lets find out more

sns.set(style="white")
df= cancer.loc[:,['radius_worst','perimeter_worst','area_worst','concavity_worst']]
g =sns.PairGrid(df,diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)

# lets find out correaltion plot

f,ax =plt.subplots(figsize=(18,18))
sns.heatmap(cancer.corr(),annot= True, linewidths='.5', fmt ='.1f',ax=ax)

drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst',
              'compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst']

cancer_new = cancer.drop(drop_list1,axis=1)
cancer_new.head()


f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(cancer_new.corr(),annot= True, linewidths='.5', fmt ='.1f',ax=ax)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

full_pipeline = Pipeline([
    ('std_scaler',StandardScaler())
])

#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labeltencoder_Y = LabelEncoder()
cancer_data_labels_encoded = labeltencoder_Y.fit_transform(cancer_data_labels)
#print(cancer_data_labels_encoded)
print(cancer_data_labels)
print(cancer_new.info())
from sklearn.model_selection import train_test_split
X_train_set,X_test_set,Y_train_set,Y_test_set =train_test_split(cancer_new,cancer_data_labels_encoded,test_size=0.25,random_state=0)

X_train_prepared = full_pipeline.fit_transform(X_train_set)
X_test_prepared = full_pipeline.transform(X_test_set)

from sklearn.linear_model import LogisticRegression 
log_reg = LogisticRegression(random_state = 0)

log_reg.fit(X_train_set,Y_train_set)


cancer_predicted = log_reg.predict(X_test_set)
cancer_predicted


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test_set, cancer_predicted)
print(cm)

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

acc = accuracy(cm)
print(acc*100)
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train_set,Y_train_set)
cancer_predicted_rf = rf_classifier.predict(X_test_set)
print(accuracy(confusion_matrix(Y_test_set, cancer_predicted_rf)))
#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
DT_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DT_classifier.fit(X_train_set,Y_train_set)
cancer_predicted_DT = DT_classifier.predict(X_test_set)
print(accuracy(confusion_matrix(Y_test_set, cancer_predicted_DT)))

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'linear', random_state = 0)
svm_classifier.fit(X_train_set,Y_train_set)
cancer_predicted_svm = svm_classifier.predict(X_test_set)
print(accuracy(confusion_matrix(Y_test_set, cancer_predicted_svm)))

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
svm_classifier = SVC(kernel = 'rbf', random_state = 0)
svm_classifier.fit(X_train_set,Y_train_set)
cancer_predicted_svm = svm_classifier.predict(X_test_set)
print(accuracy(confusion_matrix(Y_test_set, cancer_predicted_svm)))

# logistic regression with C param
#The trade-off parameter of logistic regression that determines the strength of the regularization is called C, 
#and higher values of C correspond to less regularization (where we can specify the regularization function).
#C is actually the Inverse of regularization strength(lambda)

log_reg100 = LogisticRegression(C=100)
log_reg100.fit(X_train_set,Y_train_set)
cancer_predicted_logC100 = log_reg100.predict(X_test_set)
print(accuracy(confusion_matrix(Y_test_set, cancer_predicted_logC100)))
