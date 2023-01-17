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
#Import the dataset

df = pd.read_csv('../input/diabetes.csv')

#Lets check the dataset

df.info() 
#Check the statistical inferacne of the dataset

df.describe()
#We note that there are some zeros value in our dataset. It is 

#better to replace zeros with nan since after that counting them 

#would be easier and zeros need to be replaced with suitable values

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose',

  'BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
#Lets check the missing value of the dataset

df.isnull().sum() 
#To fill these Nan values the data distribution needs to be understood

p = df.hist(figsize = (20,20))
#Now we replace the missing value

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())

df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())

df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].median())

df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())

df['BMI'] = df['BMI'].fillna(df['BMI'].median())
#Lets Check our target variable

#Here zeros means not diabetic and one means diabetic

import seaborn as sns

sns.countplot(df['Outcome'], label = 'Count')
#Lets see the pair plot

p=sns.pairplot(df, hue = 'Outcome')
#Now lets see the correlation by plotting heatmap

import matplotlib.pyplot as plt

corr = df.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (8,6))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.title('Correlation of df Features', y = 1.05, size=10)
#Lets look the correlation score

print (corr['Outcome'].sort_values(ascending=False), '\n')

#lets creat the ml model for our problem

#Lets take our matrix of features and target variable

x = df.iloc[:, 0:8].values

y = df.iloc[:, -1].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0, solver = "lbfgs")

classifier.fit(x_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(x_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

print(cm)

print(accuracy)
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