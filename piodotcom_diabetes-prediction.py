import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pylab

%matplotlib inline



from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

import keras

from keras.models import Sequential

from keras.layers import Dense

#from sklearn.cluster import KMeans



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
diab_df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

diab_df.head(20)
# Target Dictionary

result = { 1: 'Positive' , 0: 'Negative'}
plot = diab_df.hist(figsize = (20,20))
sns.set(style = "darkgrid")

sns.countplot(x = "Outcome", data = diab_df, palette = "bwr")
pd.crosstab(diab_df['Age'], diab_df['Outcome']).plot(kind = "bar", figsize = (20,5))
pd.crosstab(diab_df['Pregnancies'], diab_df['Outcome']).plot(kind = "bar", figsize = (20,5), color = ['#99A6BB','#AA4510' ])
countNoDisease = len(diab_df[diab_df['Outcome'] == 0])

countHaveDisease = len(diab_df[diab_df['Outcome'] == 1])

print("Percentage of Patients not having diabetes: {:.2f}%".format((countNoDisease / (len(diab_df['Outcome'])) * 100)))

print("Percentage of Patients having diabetes: {:.2f}%".format((countHaveDisease / (len(diab_df['Outcome'])) * 100)))
#get correlations of each features in dataset

corrmat = diab_df.corr()

top_corr_features = corrmat.index

plt.figure(figsize = (20,20))



#plot heat map

sns.heatmap(diab_df[top_corr_features].corr(), annot = True, cmap = "RdYlGn")

#Correlation with output variable

cor_diag = abs(corrmat["Outcome"])



#Selecting highly correlated features

relevant_features = cor_diag[cor_diag > 0.15]

relevant_features
X = diab_df.drop(['BloodPressure','SkinThickness','Insulin','Outcome'], 1)

Y = diab_df['Outcome']