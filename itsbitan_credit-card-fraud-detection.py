# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Import the Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # For Data Visualision

import seaborn as sns # For Data Visualision

import scipy as sp # For Staitical Calcultaion



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import the Dataset

df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#Let look our dataset

df.info()
#Lets check if there are any missing value or not

df.isnull().sum() 
#Lets look the statistical inference of the dataset

df.describe()
#Now lets check our target varibale first

sns.countplot(x = 'Class', data = df)

plt.show()
#Lets look the distribution of Amount 

sns.distplot(df['Amount'], color = 'red')

print('Skewness: %f', df['Amount'].skew())

print("Kurtosis: %f" % df['Amount'].kurt())

df['Amount_skew'] = df['Amount'] + 1e-9 

df['Amount_skew'], maxlog, (min_ci, max_ci) = sp.stats.boxcox(df['Amount_skew'], alpha=0.01)
#Lets see the distribution of Amount_skew

plt.figure(figsize=(10,4), dpi=80)

sns.distplot(df['Amount_skew'], color = 'red')

plt.xlabel('Transformed Amount')

plt.ylabel('Count')

plt.title('Transactions of Amount (Box-Cox Transformed)')

print('Skewness: %f', df['Amount_skew'].skew())

print("Kurtosis: %f" % df['Amount_skew'].kurt())
#Now drop the Amount Column

df.drop('Amount', axis = 1, inplace = True)
#Lets look the distribution of Time

#First we convert the time from seconds to hours

#to ease the interpretation.

df['Time'] = df['Time']/3600

df['Time'].max() / 24 #How many transactions in a days?

#Next plot a histogram of transaction times, with one bin per hour:

plt.figure(figsize=(10,4), dpi=80)

sns.distplot(df['Time'], color = 'green')

plt.xlim([0,48])

plt.xticks(np.arange(0,54,6))

plt.xlabel('Time After First Transaction (hr)')

plt.ylabel('Count')

plt.title('Transaction Times')

#Now lets see the correlation by plotting heatmap

corr = df.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (16,14))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.title('Correlation of df Features', y = 1.05, size=10)

#Lets look the correlation score

print (corr['Class'].sort_values(ascending=False), '\n')
#lets Take our dependent and independant variable

y_trial = df['Class']

x_trail = df.drop('Class', axis = 1, inplace = True)
##lets Take our matrices of features

y = y_trial[:].values

x = df.iloc[:,:].values
#Spliting the dataset into training set and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state =0)

#Feature Scaling the data

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.fit_transform(x_test)
#Fitting Logistic Regression to the traning set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0, class_weight='balanced', solver='newton-cg')

classifier.fit(x_train, y_train)

#Predicting the test set result

y_pred = classifier.predict(x_test)
#Making the Confusion Matrix, Accuracy score, Classification report

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

cr = classification_report(y_test, y_pred)

print(cm)

print(accuracy)

print(cr)
#Let see the ROC curve

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_pred)

print('AUC: %.3f' % auc)



fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.plot(fpr, tpr, marker='.')

plt.show()