#Import required python libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv("../input/data.csv", index_col=0)
data.head()
data.info()
data = data.drop(['Unnamed: 32'], axis =1)
data.head()
data.shape
data.isna().any().head() #Check for Missing data 
sns.countplot(x = 'diagnosis', data = data).set_title('Histogram plot for both type of diagnosis')
data.describe()
sns.pairplot(data, hue = 'diagnosis',palette='coolwarm')
dataM=data[data['diagnosis'] == "M"]

dataB=data[data['diagnosis'] == "B"]
sns.kdeplot(dataM.texture_mean, shade=True, label= "M");

sns.kdeplot(dataB.texture_mean, shade=True, label= "B");
sns.kdeplot(dataM.radius_mean, shade=True, label= "M");

sns.kdeplot(dataB.radius_mean, shade=True, label= "B");
sns.kdeplot(dataM.area_mean, shade=True, label= "M");

sns.kdeplot(dataB.area_mean, shade=True, label= "B");
sns.kdeplot(dataM.perimeter_mean, shade=True, label= "M");

sns.kdeplot(dataB.perimeter_mean, shade=True, label= "B");
sns.kdeplot(dataM.smoothness_mean, shade=True, label= "M");

sns.kdeplot(dataB.smoothness_mean, shade=True, label= "B");
sns.kdeplot(dataM.compactness_mean, shade=True, label= "M");

sns.kdeplot(dataB.compactness_mean, shade=True, label= "B");
sns.kdeplot(dataM.concavity_mean, shade=True, label= "M");

sns.kdeplot(dataB.concavity_mean, shade=True, label= "B");
sns.kdeplot(dataM['concave points_mean'], shade=True, label= "M");

sns.kdeplot(dataB['concave points_mean'], shade=True, label= "B");
sns.kdeplot(dataM['symmetry_mean'], shade=True, label= "M");

sns.kdeplot(dataB['symmetry_mean'], shade=True, label= "B");
sns.kdeplot(dataM['fractal_dimension_mean'], shade=True, label= "M");

sns.kdeplot(dataB['fractal_dimension_mean'], shade=True, label= "B");
train_data = data[0:400]

train_data.shape
test_data = data[400:]

test_data.shape
from scipy import stats

from sklearn import linear_model

logreg = linear_model.LogisticRegression(solver='liblinear')
data.columns
logreg.fit(train_data[['radius_mean','texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 

                         'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']],

         train_data['diagnosis']);

slopes_list = logreg.coef_

u = logreg.intercept_
print(slopes_list,u)
predicted_diag = logreg.predict(test_data[['radius_mean','texture_mean', 'perimeter_mean', 'area_mean','smoothness_mean', 'compactness_mean', 

                         'concavity_mean', 'concave points_mean','symmetry_mean', 'fractal_dimension_mean']]);
data_predicted = test_data.copy()

data_predicted ["Predicted_diagnosis"] = predicted_diag.tolist()

data_predicted[['diagnosis','Predicted_diagnosis']].head()
fig,ax =plt.subplots(1,2)

sns.countplot(data_predicted['Predicted_diagnosis'], ax=ax[0]).set_title('Predictive modelling using all 10 mean parameters')

sns.countplot(data_predicted['diagnosis'], ax=ax[1])
test_prediction_accuracy = (data_predicted["Predicted_diagnosis"] == data_predicted['diagnosis']).sum()*100/169

test_prediction_accuracy
data1 = data.copy()

logreg.fit(train_data[['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 

                         'concavity_mean', 'symmetry_mean']],

         train_data['diagnosis']);

slopes_list1 = logreg.coef_

u1 = logreg.intercept_

print(slopes_list1,u1)

prediction2= logreg.predict(test_data[['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean', 

                                       'concavity_mean', 'symmetry_mean']]);

data_predicted ["Predicted_diagnosis2"] = prediction2.tolist()

data_predicted[['diagnosis','Predicted_diagnosis','Predicted_diagnosis2']].head()
test_prediction_accuracy2 = (data_predicted["Predicted_diagnosis2"] == data_predicted['diagnosis']).sum()*100/169

test_prediction_accuracy2
fig,ax =plt.subplots(1,2)

sns.countplot(data_predicted['diagnosis'], ax=ax[0])

sns.countplot(data_predicted['Predicted_diagnosis2'], ax=ax[1])
