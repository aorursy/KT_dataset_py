#import the neccessary modules

# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')

#Import all required libraries for reading data, analysing and visualizing data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder
#Read the training & test data

dataset = pd.read_csv('../input/indian_liver_patient.csv')
dataset.head()
# View dataset info

dataset.info()
#Describe gives statistical information about NUMERICAL columns in the dataset

dataset.describe(include='all')

#We can see that there are missing values for Albumin_and_Globulin_Ratio as only 579 entries have valid values indicating 4 missing values.

#Gender has only 2 values - Male/Female
dataset.columns
dataset['Dataset'][:20]
#Check for any null values

dataset.isnull().sum()
sns.countplot(data=dataset, x='Dataset', label='Count')

LD, NLD = dataset['Dataset'].value_counts()

print('Number of patients diagnosed with liver disease: ',LD)

print('Number of patients not diagnosed with liver disease: ',NLD)

sns.countplot(data=dataset, x = 'Gender', label='Count')



M, F = dataset['Gender'].value_counts()

print('Number of patients that are male: ',M)

print('Number of patients that are female: ',F)
sns.catplot(x="Age", y="Gender", hue="Dataset",kind='point', data=dataset);
dataset[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).count().sort_values(by='Dataset', ascending=False)

dataset[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).mean().sort_values(by='Dataset', ascending=False)

g = sns.FacetGrid(dataset, col="Dataset", row="Gender", margin_titles=True)

g.map(plt.hist, "Age", color="red")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Disease by Gender and Age');
g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=dataset, kind="reg")
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=dataset, kind="reg")
g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=dataset, kind="reg")
g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Total_Protiens", "Albumin", data=dataset, kind="reg")
g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=dataset, kind="reg")
g = sns.FacetGrid(dataset, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Albumin_and_Globulin_Ratio", "Total_Protiens",  edgecolor="w")

plt.subplots_adjust(top=0.9)
dataset.head(5)
pd.get_dummies(dataset['Gender'], prefix = 'Gender').head()
dataset = pd.concat([dataset,pd.get_dummies(dataset['Gender'], prefix = 'Gender')], axis=1)

dataset.head()
dataset.describe()
dataset[dataset['Albumin_and_Globulin_Ratio'].isnull()]
dataset["Albumin_and_Globulin_Ratio"] = dataset.Albumin_and_Globulin_Ratio.fillna(dataset['Albumin_and_Globulin_Ratio'].mean())

dataset.head()
X = dataset.drop(['Gender','Dataset'], axis=1)

X.head(3)
y = dataset['Dataset'] # 1 for liver disease; 2 for no liver disease
# Correlation

liver_corr = X.corr()
liver_corr
plt.figure(figsize=(30, 30))

sns.heatmap(liver_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           cmap= 'coolwarm')

plt.title('Correlation between features');
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

print (X_train.shape)

print (y_train.shape)

print (X_test.shape)

print (y_test.shape)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set

lr_classifier = LogisticRegression(random_state = 101)

lr_classifier.fit(X_train, y_train)
# Predicting the Test set results

y_pred = lr_classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
logreg_score = round(lr_classifier.score(X_train, y_train) * 100, 2)

logreg_score_test = round(lr_classifier.score(X_test, y_test) * 100, 2)

#Equation coefficient and Intercept

print('Logistic Regression Training Score: \n', logreg_score)

print('Logistic Regression Test Score: \n', logreg_score_test)

print('Coefficient: \n', lr_classifier.coef_)

print('Intercept: \n', lr_classifier.intercept_)

print('Accuracy: \n', accuracy_score(y_test,y_pred))

print('Confusion Matrix: \n', confusion_matrix(y_test,y_pred))

print('Classification Report: \n', classification_report(y_test,y_pred))



sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt="d")
gnb_classifier = GaussianNB()

gnb_classifier.fit(X_train, y_train)



# Predicting the Test set results

gnb_y_pred = gnb_classifier.predict(X_test)

# Making the Confusion Matrix

gnb_cm = confusion_matrix(y_test, gnb_y_pred)
gnb_score = round(gnb_classifier.score(X_train, y_train) * 100, 2)

gnb_score_test = round(gnb_classifier.score(X_test, y_test) * 100, 2)

#Equation coefficient and Intercept

print('Gaussian Score: \n', gnb_score)

print('Gaussian Test Score: \n', gnb_score_test)

print('Accuracy: \n', accuracy_score(y_test, gnb_y_pred))

print(confusion_matrix(y_test,gnb_y_pred))

print(classification_report(y_test,gnb_y_pred))



sns.heatmap(confusion_matrix(y_test,gnb_y_pred),annot=True,fmt="d")
rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 101)

rf_classifier.fit(X_train, y_train)



# Predicting the Test set results

rf_y_pred = rf_classifier.predict(X_test)
# Making the Confusion Matrix

rf_cm = confusion_matrix(y_test, rf_y_pred)
# Score the model

random_forest_score = round(rf_classifier.score(X_train, y_train) * 100, 2)

random_forest_score_test = round(rf_classifier.score(X_test, y_test) * 100, 2)

print('Random Forest Score: \n', random_forest_score)

print('Random Forest Test Score: \n', random_forest_score_test)

print('Accuracy: \n', accuracy_score(y_test,rf_y_pred))

#print(confusion_matrix(y_test,rf_predicted))

print(classification_report(y_test,rf_y_pred))
sns.heatmap(rf_cm,annot=True,fmt="d")
# Fitting SVM to the Training set

from sklearn.svm import SVC

ksvm_classifier = SVC(kernel='rbf',random_state=101)

ksvm_classifier.fit(X_train,y_train)



# Predicting the Test set results

ksvm_y_pred = ksvm_classifier.predict(X_test)
# Score Kernel SVM

ksvm_score = round(ksvm_classifier.score(X_train, y_train) * 100, 2)

ksvm_score_test = round(ksvm_classifier.score(X_test, y_test) * 100, 2)

print('SVM Score: \n', ksvm_score)

print('SVM Test Score: \n', ksvm_score_test)

print('Accuracy: \n', accuracy_score(y_test,ksvm_y_pred))

#print(confusion_matrix(y_test,rf_predicted))

print(classification_report(y_test,ksvm_y_pred))
# Making the confusion matrix

ksvm_cm = confusion_matrix(y_test, ksvm_y_pred)

sns.heatmap(ksvm_cm,annot=True,fmt="d")
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

knn_classifier.fit(X_train, y_train)



# Predicting the Test set results

knn_y_pred = knn_classifier.predict(X_test)

# Score K-NN

knn_score = round(knn_classifier.score(X_train, y_train) * 100, 2)

knn_score_test = round(knn_classifier.score(X_test, y_test) * 100, 2)

print('SVM Score: \n', knn_score)

print('SVM Test Score: \n', knn_score_test)

print('Accuracy: \n', accuracy_score(y_test,knn_y_pred))

#print(confusion_matrix(y_test,rf_predicted))

print(classification_report(y_test,knn_y_pred))
#We can now rank our evaluation of all the models to choose the best one for our problem. 

models = pd.DataFrame({

    'Model': [ 'Logistic Regression', 'Gaussian Naive Bayes','Random Forest', 'Kernel SVM','K-NN'],

    'Score': [ logreg_score, gnb_score, random_forest_score, ksvm_score, knn_score],

    'Test Score': [ logreg_score_test, gnb_score_test, random_forest_score_test, ksvm_score_test, knn_score_test]})

models.sort_values(by='Test Score', ascending=False)
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score

linear.fit(X_train, y_train)

#Predict Output

lin_predicted = linear.predict(X_test)



linear_score = round(linear.score(X_train, y_train) * 100, 2)

linear_score_test = round(linear.score(X_test, y_test) * 100, 2)

#Equation coefficient and Intercept

print('Linear Regression Score: \n', linear_score)

print('Linear Regression Test Score: \n', linear_score_test)

print('Coefficient: \n', linear.coef_)

print('Intercept: \n', linear.intercept_)



from sklearn.feature_selection import RFE

rfe =RFE(linear, n_features_to_select=3)

rfe.fit(X,y)
for i in range(len(rfe.ranking_)):

    if rfe.ranking_[i] == 1:

        print(X.columns.values[i])
finX = dataset[['Total_Protiens','Albumin', 'Gender_Male']]

finX.head(4)
X_train, X_test, y_train, y_test = train_test_split(finX, y, test_size=0.30, random_state=101)
#Logistic Regression

logreg = LogisticRegression()

# Train the model using the training sets and check score

logreg.fit(X_train, y_train)

#Predict Output

log_predicted= logreg.predict(X_test)



logreg_score = round(logreg.score(X_train, y_train) * 100, 2)

logreg_score_test = round(logreg.score(X_test, y_test) * 100, 2)

#Equation coefficient and Intercept

print('Logistic Regression Training Score: \n', logreg_score)

print('Logistic Regression Test Score: \n', logreg_score_test)

print('Coefficient: \n', logreg.coef_)

print('Intercept: \n', logreg.intercept_)

print('Accuracy: \n', accuracy_score(y_test,log_predicted))

print('Confusion Matrix: \n', confusion_matrix(y_test,log_predicted))

print('Classification Report: \n', classification_report(y_test,log_predicted))



sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")
ksvm_classifier = SVC(kernel='rbf',random_state=101)

ksvm_classifier.fit(X_train,y_train)



# Predicting the Test set results

ksvm_y_pred = ksvm_classifier.predict(X_test)



# Score Kernel SVM

ksvm_score = round(ksvm_classifier.score(X_train, y_train) * 100, 2)

ksvm_score_test = round(ksvm_classifier.score(X_test, y_test) * 100, 2)

print('SVM Score: \n', ksvm_score)

print('SVM Test Score: \n', ksvm_score_test)

print('Accuracy: \n', accuracy_score(y_test,ksvm_y_pred))

#print(confusion_matrix(y_test,rf_predicted))

print(classification_report(y_test,ksvm_y_pred))



# Making the confusion matrix

ksvm_cm = confusion_matrix(y_test, ksvm_y_pred)

sns.heatmap(ksvm_cm,annot=True,fmt="d")