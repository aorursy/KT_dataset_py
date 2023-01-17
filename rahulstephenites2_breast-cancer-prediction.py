# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt  # data visualization library  

import warnings

warnings.filterwarnings("ignore") #ignoring all the warnings
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head() #checking for the data
#shape of the data

data.shape
#datatypes of the columns

data.info()
# y includes our Target Labels and x includes our features

y = data.diagnosis                          # M or B 

list = ['Unnamed: 32','id','diagnosis']

x = data.drop(list,axis = 1 )

x.head()
x.isnull().sum() #Checking of null values in the data
sns.countplot(y)

B,M=y.value_counts()

print('Number of Patient with Malignant Tumor:', M)

print('Number of Patient with Benign Tumor:', B)
x.describe() #Since we dont know much about the features , we are getting to know about it by the mean and Standard deviation
# first ten features

data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=90)

plt.show()
# Second ten features

data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=90)

plt.show()
# Third ten features

data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=90)

plt.show()
# As an alternative of violin plot, box plot can be used

# box plots are also useful in terms of seeing outliers

# In order to show you lets have an example of box plot

# If you want, you can visualize other features as well.

plt.figure(figsize=(18,10))

data = pd.concat([y,data_n_2],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

sns.boxplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)

plt.show()
df = x.loc[:,['radius_worst','perimeter_worst','area_worst']]

g = sns.pairplot(df)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot, lw=3)

plt.show()
data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization

data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)



plt.xticks(rotation=60)

plt.show()
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=60)

plt.show()
data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)

plt.show()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
drop_list1 = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se',

              'perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst',

              'compactness_se','concave points_se','texture_worst','area_worst']

x_1 = x.drop(drop_list1,axis = 1 )        # do not modify x, we will use it later 

x_1.head()
#correlation map

f,ax = plt.subplots(figsize=(14, 14))

sns.heatmap(x_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix,classification_report

from sklearn.metrics import accuracy_score



# split data train 70 % and test 30 %

x_train, x_test, y_train, y_test = train_test_split(x_1, y, test_size=0.3, random_state=42)



#random forest classifier with n_estimators=10 (default)

clf_rf = RandomForestClassifier(random_state=43)      

clr_rf = clf_rf.fit(x_train,y_train)

pred=clf_rf.predict(x_test)

ac = accuracy_score(y_test,pred)

print('Accuracy is: ',ac)

print("Classification Report:\n",classification_report(y_test,pred))

cm = confusion_matrix(y_test,pred)

sns.heatmap(cm,annot=True,fmt="d")

plt.show()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

# find best scored 5 features

select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train)
print('Score list:', select_feature.scores_)

print('Feature list:', x_train.columns)
x_train_2 = select_feature.transform(x_train)

x_test_2 = select_feature.transform(x_test)

#random forest classifier with n_estimators=10 (default)

clf_rf_2 = RandomForestClassifier()      

clr_rf_2 = clf_rf_2.fit(x_train_2,y_train)

ac_2 = accuracy_score(y_test,clf_rf_2.predict(x_test_2))

print('Accuracy is: ',ac_2)

print("Classification Report:\n",classification_report(y_test,clf_rf_2.predict(x_test_2)))

cm_2 = confusion_matrix(y_test,clf_rf_2.predict(x_test_2))

sns.heatmap(cm_2,annot=True,fmt="d")

plt.show()
from sklearn.feature_selection import RFE

# Create the RFE object and rank each pixel

clf_rf_3 = RandomForestClassifier()      

rfe = RFE(estimator=clf_rf_3, n_features_to_select=5, step=1)

rfe = rfe.fit(x_train, y_train)
print('Chosen best 5 feature by rfe:',x_train.columns[rfe.support_])
from sklearn.feature_selection import RFECV



# The "accuracy" scoring is proportional to the number of correct classifications

clf_rf_4 = RandomForestClassifier() 

rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation

rfecv = rfecv.fit(x_train, y_train)
print('Optimal number of features :', rfecv.n_features_)

print('Best features :', x_train.columns[rfecv.support_])
x_train_2 = rfecv.transform(x_train)

x_test_2 = rfecv.transform(x_test)

#random forest classifier with n_estimators=10 (default)

rfecv_2 = RandomForestClassifier()      

rfecv_2 = rfecv_2.fit(x_train_2,y_train)

ac_3 = accuracy_score(y_test,rfecv_2.predict(x_test_2))

print('Accuracy is: ',ac_2)

print("Classification Report:\n",classification_report(y_test,rfecv_2.predict(x_test_2)))

cm_3 = confusion_matrix(y_test,rfecv_2.predict(x_test_2))

sns.heatmap(cm_2,annot=True,fmt="d")

plt.show()