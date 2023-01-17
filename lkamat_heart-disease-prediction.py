# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Visualize

import matplotlib.pyplot as plt #For visualization

from matplotlib import rcParams #add styling to the plots

from matplotlib.cm import rainbow #for colors

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



# Process the data 

from sklearn.model_selection import train_test_split #split the available dataset for testing and training

from sklearn.preprocessing import StandardScaler #To scale the features

#Machine Learning algorithms I will be using.



from sklearn.neighbors import KNeighborsClassifier #K Neighbors Classifier

from sklearn.svm import SVC #Support Vector Classifier

from sklearn.tree import DecisionTreeClassifier #Decision Tree Classifier

from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
# read the data file

dataset = pd.read_csv('../input/heart.csv')
dataset.shape # number of rows and columns
dataset.info()
dataset.describe()
dataset.sample(5)
# dataset.isnull().sum()

# dataset.isnull().values.any()
dataset.columns
#Rename the columns

dataset.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
dataset.columns
dataset.dtypes
# Understanding the data

# Visualizations to better understand data and do any processing if needed.

rcParams['figure.figsize'] = 20, 14

plt.matshow(dataset.corr())

plt.yticks(np.arange(dataset.shape[1]), dataset.columns)

plt.xticks(np.arange(dataset.shape[1]), dataset.columns)

plt.colorbar()
dataset.hist()
# Check for the target classes to evaluate if they are of approximately same size



rcParams['figure.figsize'] = 8,6

plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color = ['red', 'green'])

plt.xticks([0, 1])

plt.xlabel('Target Classes')

plt.ylabel('Count')

plt.title('Count of each Target Class')
# Male vs Female data

male = len(dataset[dataset.sex == 1])

female = len(dataset[dataset.sex == 0])

plt.pie(x=[male, female], explode=(0, 0), labels=['Male', 'Female'], autopct='%1.2f%%', shadow=True, startangle=90)

plt.show()
# Chest Pain Type

x = [len(dataset[dataset['chest_pain_type'] == 0]),len(dataset[dataset['chest_pain_type'] == 1]), len(dataset[dataset['chest_pain_type'] == 2]), len(dataset[dataset['chest_pain_type'] == 3])]

plt.pie(x, data=dataset, labels=['chest_pain_type(1) typical angina', 'chest_pain_type(2) atypical angina', 'chest_pain_type(3) non-anginal pain', 'chest_pain_type(4) asymptomatic'], autopct='%1.2f%%', shadow=True,startangle=90)

plt.show()
# Fasting Blood Sugar



x = [len(dataset[dataset['fasting_blood_sugar'] == 0]),len(dataset[dataset['fasting_blood_sugar'] == 1])]

plt.pie(x, data=dataset, labels=['fasting_blood_sugar(1) >120mg', 'fasting_blood_sugar(2) <120mg',], autopct='%1.2f%%', shadow=True,startangle=90)

plt.show()





#

plt.figure(figsize=(10,6))

count= dataset.sex.value_counts()

sns.barplot(x=count.index, y=count.values)

plt.ylabel("Number of ca")

plt.xlabel("Ca values")

plt.title("Ca values in data", color="black", fontsize="12")



# Heart disease frequency by age

plt.figure(figsize=(15, 15))

sns.countplot(x='age', hue='target', data=dataset, palette=['blue', 'red'])

plt.legend(["No Disease", "Have Disease"])

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
fig, axes = plt.subplots(3, 2, figsize=(12,12))

fs = ['chest_pain_type', 'fasting_blood_sugar', 'rest_ecg','exercise_induced_angina', 'st_slope', 'num_major_vessels']

for i, axi in enumerate(axes.flat):

    sns.countplot(x=fs[i], hue='target', data=dataset, palette='bwr', ax=axi) 

    axi.set(ylabel='Frequency')
# Data Processing

# Convert some categorical variables into dummy variables and scale all the values before training the Machine Learning models. 

# First,use the get_dummies method to create dummy columns for categorical variables.

dataset = pd.get_dummies(dataset, columns = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'st_slope', 'num_major_vessels', 'thalassemia'])
# Use the StandardScaler from sklearn to scale my dataset.

#dataset.columns

standardScaler = StandardScaler()

columns_to_scale = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate_achieved', 'st_depression']

dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

#dataset.columns
y = dataset['target']

X = dataset.drop(['target'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

knn_scores = []

for k in range(1,10):

    knn_classifier = KNeighborsClassifier(n_neighbors = k)

    knn_classifier.fit(X_train, y_train)

    knn_scores.append(knn_classifier.score(X_test, y_test))
# Scores for different neighbor values are in the array knn_scores. Plot and see for which value of K we get the best scores.



plt.plot([k for k in range(1, 10)], knn_scores, color = 'red')

for i in range(1,10):

    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))

plt.xticks([i for i in range(1, 10)])

plt.xlabel('Number of Neighbors (K)')

plt.ylabel('Scores')

plt.title('K Neighbors Classifier scores for different K values')
print("The score for K Neighbors Classifier is {:0.2f}% with {} neighbors.".format(knn_scores[7]*100, 8))
# Support Vector Classifier

# There are several kernels for Support Vector Classifier. We will test some of them and check which has the best score.





svc_scores = []

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for i in range(len(kernels)):

    svc_classifier = SVC(kernel = kernels[i])

    svc_classifier.fit(X_train, y_train)

    svc_scores.append(svc_classifier.score(X_test, y_test))
# Plot a bar plot of scores for each kernel and see which performed the best.



colors = rainbow(np.linspace(0, 1, len(kernels)))

plt.bar(kernels, svc_scores, color = colors)

for i in range(len(kernels)):

    plt.text(i, svc_scores[i], svc_scores[i])

plt.xlabel('Kernels')

plt.ylabel('Scores')

plt.title('Support Vector Classifier scores for different kernels')
print("The score for Support Vector Classifier is {:0.2f}% with {} kernel.".format(svc_scores[3]*100, 'sigmoid'))
# Decision Tree Classifier

# Use the Decision Tree Classifier. Vary between a set of max_features and see which returns the best accuracy.



dt_scores = []

for i in range(1, len(X.columns) + 1):

    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)

    dt_classifier.fit(X_train, y_train)

    dt_scores.append(dt_classifier.score(X_test, y_test))
# Select the maximum number of features from 1 to 30 for split and see the scores for each of those cases.



plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')

for i in range(1, len(X.columns) + 1):

    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))

plt.xticks([i for i in range(1, len(X.columns) + 1)])

plt.xlabel('Max features')

plt.ylabel('Scores')

plt.title('Decision Tree Classifier scores for different number of maximum features')
print("The score for Decision Tree Classifier is {:0.2f}% with {} maximum features.".format(dt_scores[9]*100, [10]))
# Random Forest Classifier

# Use the ensemble method, Random Forest Classifier, to create the model and vary the number of estimators to see their effect.

rf_scores = []

estimators = [10, 100, 200, 500,1000]

for i in estimators:

    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)

    rf_classifier.fit(X_train, y_train)

    rf_scores.append(rf_classifier.score(X_test, y_test))
rf_scores[0]
colors = rainbow(np.linspace(0, 1, len(estimators)))

plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)

for i in range(len(estimators)):

    plt.text(i, rf_scores[i], rf_scores[i])

plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])

plt.xlabel('Number of estimators')

plt.ylabel('Scores')

plt.title('Random Forest Classifier scores for different number of estimators')
print("The score for Random Forest Classifier is {:0.2f}% with {} estimators.".format(rf_scores[2]*100, [200]))
import xgboost as xgb

from sklearn.metrics import accuracy_score

rf_scores = []

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(X_train, y_train)

Y_pred_xgb = xgb_model.predict(X_test)

rf_scores.append(xgb_model.score(X_test, y_test))
score_xgb = round(accuracy_score(Y_pred_xgb,y_test)*100,2)



print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")

print("The score for Random Forest Classifier is {:0.2f}% with {} estimators.".format(rf_scores[0]*100, 2))
