#Import library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
diabetes = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

print(diabetes.columns)
diabetes.head()
print("dimension of diabetes data: {}".format(diabetes.shape))

#The diabetes dataset consists of 768 data points, with 9 features
print(diabetes.groupby('Outcome').size())
import seaborn as sns



sns.countplot(diabetes['Outcome'],label="Count")

#data has more No diabetic data as compared to diabetic data which would give a biased prediction towards no diabetic
diabetes.info()
diabetes.describe().transpose()
# Few Insights

# Min blood pressure of 0 is invalid, so impute it with appropriate values. Same with few other variables like BMI

# Mean and Median values of Insuline is very different

# Insuline has very high Standard deviation

# We will ignore all these issues for now to concentrate more on Model
colormap = plt.cm.viridis # Color range to be used in heatmap

plt.figure(figsize=(15,15))

plt.title('Pearson Correlation of attributes', y=1.05, size=19)

sns.heatmap(diabetes.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)

#There is no strong correlation between any two variables.

#There is no strong correlation between any independent variable and class variable.
spd = pd.plotting.scatter_matrix(diabetes, figsize=(20,20), diagonal="kde")
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=11)
X_train.shape
from sklearn.svm import SVC



svc = SVC()

svc.fit(X_train, y_train)



print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
#The model overfits substantially with a perfect score on the training set and only 65% accuracy on the test set.



#SVM requires all the features to be on a similar scale. We will need to rescale our data that all the features are approximately

#on the same scale and than see the performance
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)
svc = SVC()

svc.fit(X_train_scaled, y_train)



print("Accuracy on training set: {:.2f}".format(svc.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(X_test_scaled, y_test)))
svc = SVC(C=1000)

svc.fit(X_train_scaled, y_train)



print("Accuracy on training set: {:.3f}".format(

    svc.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))