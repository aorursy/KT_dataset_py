# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



#Import the Labraries for visualision 

import matplotlib.pyplot as plt

import seaborn as sns

# Any results you write to the current directory are saved as output.
#Import the Dataset

df = pd.read_csv("../input/pulsar_stars.csv")  
#lets check our dataset

df.info()
#At first we Renaming columns

df = df.rename(columns={' Mean of the integrated profile':"mean_profile",

       ' Standard deviation of the integrated profile':"std_profile",

       ' Excess kurtosis of the integrated profile':"kurtosis_profile",

       ' Skewness of the integrated profile':"skewness_profile", 

        ' Mean of the DM-SNR curve':"mean_dmsnr_curve",

       ' Standard deviation of the DM-SNR curve':"std_dmsnr_curve",

       ' Excess kurtosis of the DM-SNR curve':"kurtosis_dmsnr_curve",

       ' Skewness of the DM-SNR curve':"skewness_dmsnr_curve",

       })
#Now we see the statistical inference of the dataset

df.describe()
#Now check the if any missing value in our dataset

df.isnull().sum()
#Now we see out target varibale

sns.countplot(x ='target_class', data = df)

plt.show()
#Now lets see the correlation by plotting heatmap

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

print (corr['target_class'].sort_values(ascending=False), '\n')
#Lets see the pair plot between all variables

sns.pairplot(df,hue = 'target_class')

plt.title("pair plot for variables")

plt.show()
#Lets create ML model for our dataset

x = df.iloc[:, 0 : 8].values

y = df.iloc[:, - 1].values
#Spliting the dataset to the traning and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
#Feature Secaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
#Fitting Logistic Regression to the training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)
#Predicting the test set result

y_pred = classifier.predict(x_test)

#Making the Confussion Matrix and Print Accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression :")

print("Accuracy = ", accuracy)

print(cm)
#Let see the ROC curve

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_pred)

print('AUC : %.3f' % auc)



fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot([0, 1], [0, 1], linestyle = '--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.plot(fpr, tpr, marker = '.')

plt.show()