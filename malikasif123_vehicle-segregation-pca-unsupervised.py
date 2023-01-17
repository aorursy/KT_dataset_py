#importing necessary Libraries 



#working with data

import pandas as pd

import numpy as np



#visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



## Scikit-learn features various classification, regression and clustering algorithms

from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import preprocessing

from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score, classification_report,f1_score



## Scaling

from sklearn.preprocessing import StandardScaler



## Algo

from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



import warnings

warnings.filterwarnings('ignore')
#loading Data

Data = pd.read_csv('../input/vehicle/vehicle-1.csv')
#checking top 5 rows

Data.head()
#fetch all columns

Data.columns
#checking datatypes of each column

Data.dtypes
#shape of data 

shape_Data = Data.shape

print('Data set contains "{x}" number of rows and "{y}" number of columns' .format(x=shape_Data[0],y=shape_Data[1]))
#Data info 

#It gives the information about the number of rows, number of columns, data types , memory usage, 

#number of null values in each columns."

Data.info()

null_data = Data.isnull().sum()

null_data
sns.heatmap(Data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Null values percentage corresponding to the columns

Percentage_Null = (pd.DataFrame(Data.isnull().sum())/len(Data))*100

sns.set(rc={'figure.figsize':(12,6)})

Percentage_Null.plot.bar()

plt.xlabel("Columns with null values")

plt.ylabel("Null value percentage")
#Oveview of Data

Data.describe().T
#Replacing the missing values by mean

for i in Data.columns[:17]:

    mean_value = Data[i].mean()

    Data[i] = Data[i].fillna(mean_value)
#Again check for missing values

null_data = Data.isnull().sum()

null_data
#Skewness is computed for each row or each column , here we will check for column

skewValue = Data.skew(axis=0) # axis=0 for column

print("SKEW:")

print(skewValue)
f, axes = plt.subplots(1, 4,figsize=(15,5))

compactness = sns.distplot(Data['compactness'], color="green", kde=True,ax=axes[0])

circularity = sns.distplot(Data['circularity'], color="blue", kde=True,ax=axes[1])

distance_circularity = sns.distplot(Data['distance_circularity'], color="red",kde=True,ax=axes[2])

axis_aspect_ratio = sns.distplot(Data['pr.axis_aspect_ratio'], color="orange", kde=True,ax=axes[3])
f, axes = plt.subplots(1, 4,figsize=(15,5))

scatter_ratio = sns.distplot(Data['scatter_ratio'], color="orange", kde=True,ax=axes[0])

radius_ratio = sns.distplot(Data['radius_ratio'], color="pink", kde=True,ax=axes[1])

length_aspect_ratio = sns.distplot(Data['max.length_aspect_ratio'], color="magenta", kde=True,ax=axes[2])

elongatedness = sns.distplot(Data['elongatedness'], color="purple", kde=True,ax=axes[3])

f, axes = plt.subplots(1, 4,figsize=(15,5))

pr_axis_rectangularity  = sns.distplot(Data['pr.axis_rectangularity'], color="lime", kde=True,ax=axes[0])

max_length_rectangularity = sns.distplot(Data['max.length_rectangularity'], color="maroon", kde=True,ax=axes[1])

scaled_variance = sns.distplot(Data['scaled_variance'], color="olive",kde=True,ax=axes[2])

scaled_variance_1_ = sns.distplot(Data['scaled_variance.1'], color="LightBlue", kde=True,ax=axes[3])
f, axes = plt.subplots(1, 4,figsize=(15,5))

scaled_radius_of_gyration_1_  = sns.distplot(Data['scaled_radius_of_gyration.1'], color="DarkBlue", kde=True,ax=axes[0])

scaled_radius_of_gyration = sns.distplot(Data['scaled_radius_of_gyration'], color="Cyan", kde=True,ax=axes[1])

skewness_about = sns.distplot(Data['skewness_about'], color="Aquamarine",kde=True,ax=axes[2])

skewness_about_1_ = sns.distplot(Data['skewness_about.1'], color="Plum", kde=True,ax=axes[3])
f, axes = plt.subplots(1, 2,figsize=(15,5))

hollows_ratio = sns.distplot(Data['hollows_ratio'], color="Lime",kde=True,ax=axes[0])

skewness_about_2_ = sns.distplot(Data['skewness_about.2'], color="magenta", kde=True,ax=axes[1])
Data['class'].unique()
Data['class'].value_counts()
Class = sns.countplot(data = Data, x = 'class')

Class.set_xlabel('Types of Vehicle', fontsize=15)
# Label encoder 

from sklearn.preprocessing import LabelEncoder

#Encoding of categorical variable

labelencoder_X=LabelEncoder()

Data['class']=labelencoder_X.fit_transform(Data['class'])

Data['class'].value_counts()
corr= Data.corr()

corr
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 3.5})

plt.figure(figsize=(18,7))

# create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
corr[(corr>.85) | (corr <(-.85))]
#Variation of Vehicles w.r.t. Scatter Ratio

sns.kdeplot((Data[Data['class'] == 2].scatter_ratio), shade=False, label='Van') # for Van

sns.kdeplot((Data[Data['class'] == 1].scatter_ratio), shade=True ,label='Car') # for Car

sns.kdeplot((Data[Data['class'] == 0].scatter_ratio), shade=False , label='Bus') # for Bus



plt.title("Variation of Vehicles w.r.t. Scatter Ratio")
#Variation of Vehicles w.r.t. compactness

'''

0-- Bus

1-- Car

2 -- Van

'''

sns.boxplot(x='class' ,y= 'compactness' ,data=Data)
#Variation of Vehicles w.r.t. circularity

'''

0-- Bus

1-- Car

2 -- Van

'''

sns.boxplot(x='class' ,y= 'circularity' ,data=Data)
sns.violinplot(x="class", y="distance_circularity", data=Data,palette='rainbow')
#circularity and scaled_radius_of_gyration, as they are highly correlated.

'''

0-- Bus

1-- Car

2 -- Van

'''



a=sns.stripplot(x="circularity", y="scaled_radius_of_gyration", data=Data,jitter=True,hue='class',palette='Set1')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
#scatter_ratio and scaled_variance.1, as they are highly correlated.

'''

0-- Bus

1-- Car

2 -- Van

'''

plt.figure(figsize=(18,7))

a=sns.swarmplot(x="scatter_ratio", y="scaled_variance.1",hue='class',data=Data, palette="Set1", split=True)

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90,)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
#Check pairplot for class

sns.pairplot(Data, hue='class')
#Splitting the data between independent and dependent variables

X=Data.iloc[:,0:18]

y = Data['class']
#dropping correlated values which are have either more then 85% or less then -85%

X_new=X.drop(['circularity','scatter_ratio','scaled_variance'],axis=1)
#since there is lots of variety in the units of features let's scale it

scaler=StandardScaler().fit(X_new)

X_scaled=scaler.transform(X_new)
# Split X and y into training and test set in 70:30 ratio

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y, test_size = 0.3, random_state = 10)
#checking the dimensions of the train and test set

print(X_train.shape) # shape of train data

print(X_test.shape) # shape of test data
from sklearn.svm import SVC

svclassifier = SVC(gamma=0.05, C=3,random_state=0) 

svclassifier.fit(X_train, y_train) # To train the algorithm on training data

y_prediction = svclassifier.predict(X_test) #To make predictions
#check the accuracy on the training data

print('Accuracy on Training data: ',svclassifier.score(X_train, y_train))

# check the accuracy on the testing data

print('Accuracy on Testing data: ', svclassifier.score(X_test , y_test))
#measure the accuracy of this model's prediction

print("Confusion Matrix:\n",metrics.confusion_matrix(y_prediction,y_test))
#Evaluate Model Score

print("Classification Report:\n",metrics.classification_report(y_test,y_prediction))
#Using K fold to check how my algorighm varies throughout my data if we split it in 10 equal bins

models = []

models.append(('SVM before PCA', SVC(gamma=0.05, C=3)))



# evaluate each model

results = []

names = []

scoring = 'accuracy'

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=101)

	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	print("Name = %s , Mean Accuracy = %f, SD Accuracy = %f" % (name, cv_results.mean(), cv_results.std()))
fig = plt.figure()

fig.suptitle('Algorithm Variation')

ax = fig.add_subplot(111)

plt.plot(results[0],label='SVM before PCA')

plt.legend()
#Scaling with all Columns(leaving Target)

scaler=StandardScaler().fit(X)

X_scaled_PCA=scaler.transform(X)
#printing Covariance Matrix

covMatrix = np.cov(X_scaled_PCA,rowvar=False)

print(covMatrix)
#choosing n_component as 8 coz we saw there are 3-4 attributes having 2 hidden clusters

pca = PCA(n_components=8)

pca.fit(X_scaled_PCA)
#eigen values

print(pca.explained_variance_)
#eigen Vectors

print(pca.components_[0])
#percentage of variance explained by each vector

print(pca.explained_variance_ratio_)
#visualising it

plt.bar(list(range(1,9)),pca.explained_variance_ratio_,alpha=0.5, align='center')

plt.ylabel('Variation explained')

plt.xlabel('eigen Value')
#cummilative variance explained via each vector

plt.step(list(range(1,9)),np.cumsum(pca.explained_variance_ratio_), where='mid')

plt.ylabel('Cum of variation explained')

plt.xlabel('eigen Value')
#transforming and storing result in X_Train_PCA

pca_model_test = PCA(n_components=7)

pca_model_test.fit(X_scaled_PCA)

X_PCA= pca_model_test.transform(X_scaled_PCA)

X_PCA[0]
# Split X and y into training and test set in 70:30 ratio

X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(X_PCA,y, test_size = 0.3, random_state = 10)
#model Building

svclassifier_PCA = SVC(gamma=0.05, C=3,random_state=0) 

svclassifier_PCA.fit(X_train_PCA,y_train_PCA)
#To make predictions

y_prediction_PCA = svclassifier_PCA.predict(X_test_PCA) 
#check the accuracy on the training data

print('Accuracy on Training data after PCA: ',svclassifier_PCA.score(X_train_PCA, y_train_PCA))

#check the accuracy on the test data

print('Accuracy on Training data after PCA: ',svclassifier_PCA.score(X_test_PCA, y_test_PCA))
#measure the accuracy of this model's prediction

print("Confusion Matrix:\n",metrics.confusion_matrix(y_prediction_PCA,y_test_PCA))
#Evaluate Model Score

print("Classification Report:\n",metrics.classification_report(y_test_PCA,y_prediction_PCA))
#Using K fold to check how my algorighm varies throughout my data if we split it in 10 equal bins

models = []

models.append(('SVM After PCA', SVC(gamma=0.05, C=3)))



# evaluate each model

results_PCA = []

names = []

scoring = 'accuracy'

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=101)

	cv_results_PCA = model_selection.cross_val_score(model, X_train_PCA, y_train_PCA, cv=kfold, scoring=scoring)

	results_PCA.append(cv_results_PCA)

	names.append(name)

	print("Name = %s , Mean Accuracy = %f, SD Accuracy = %f" % (name, cv_results_PCA.mean(), cv_results_PCA.std()))
fig = plt.figure()

fig.suptitle("SVM Variation with 15 'Attributes' vs 7 'Components'")

ax = fig.add_subplot(111)

plt.plot(results[0],label='SVM with 15 attributes')

plt.plot(results_PCA[0],label='SVM with 7 components')

plt.legend()


Accuracy_df=pd.DataFrame([{'Model':'SVM with 15 attribute','Mean Accuracy': cv_results.mean(),'Standard Deviation':cv_results.std()},

                       {'Model':'SVM with 7 Components','Mean Accuracy':cv_results_PCA.mean(),'Standard Deviation':cv_results_PCA.std()}

                       ] ) 

Accuracy_df=Accuracy_df[['Model','Mean Accuracy','Standard Deviation']]

Accuracy_df
