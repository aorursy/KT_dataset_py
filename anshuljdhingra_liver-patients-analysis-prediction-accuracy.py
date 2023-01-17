#Import all required libraries for reading data, analysing and visualizing data

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder
#Read the training & test data

liver_df = pd.read_csv('../input/indian_liver_patient.csv')
liver_df.head()
liver_df.info()
#Describe gives statistical information about NUMERICAL columns in the dataset

liver_df.describe(include='all')

#We can see that there are missing values for Albumin_and_Globulin_Ratio as only 579 entries have valid values indicating 4 missing values.

#Gender has only 2 values - Male/Female
#Which features are available in the dataset?

liver_df.columns
#Check for any null values

liver_df.isnull().sum()
sns.countplot(data=liver_df, x = 'Dataset', label='Count')



LD, NLD = liver_df['Dataset'].value_counts()

print('Number of patients diagnosed with liver disease: ',LD)

print('Number of patients not diagnosed with liver disease: ',NLD)
sns.countplot(data=liver_df, x = 'Gender', label='Count')



M, F = liver_df['Gender'].value_counts()

print('Number of patients that are male: ',M)

print('Number of patients that are female: ',F)
sns.factorplot(x="Age", y="Gender", hue="Dataset", data=liver_df);
liver_df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).count().sort_values(by='Dataset', ascending=False)
liver_df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).mean().sort_values(by='Dataset', ascending=False)
g = sns.FacetGrid(liver_df, col="Dataset", row="Gender", margin_titles=True)

g.map(plt.hist, "Age", color="red")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Disease by Gender and Age');
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=liver_df, kind="reg")
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Aspartate_Aminotransferase", "Alamine_Aminotransferase",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=liver_df, kind="reg")
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=liver_df, kind="reg")
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Total_Protiens", "Albumin", data=liver_df, kind="reg")
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=liver_df, kind="reg")
g = sns.FacetGrid(liver_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Albumin_and_Globulin_Ratio", "Total_Protiens",  edgecolor="w")

plt.subplots_adjust(top=0.9)
liver_df.head(3)
pd.get_dummies(liver_df['Gender'], prefix = 'Gender').head()
liver_df = pd.concat([liver_df,pd.get_dummies(liver_df['Gender'], prefix = 'Gender')], axis=1)
liver_df.head()
liver_df.describe()
liver_df[liver_df['Albumin_and_Globulin_Ratio'].isnull()]
liver_df["Albumin_and_Globulin_Ratio"] = liver_df.Albumin_and_Globulin_Ratio.fillna(liver_df['Albumin_and_Globulin_Ratio'].mean())
#liver_df[liver_df['Albumin_and_Globulin_Ratio'] == 0.9470639032815201]
# The input variables/features are all the inputs except Dataset. The prediction or label is 'Dataset' that determines whether the patient has liver disease or not. 

X = liver_df.drop(['Gender','Dataset'], axis=1)

X.head(3)
y = liver_df['Dataset'] # 1 for liver disease; 2 for no liver disease
# Correlation

liver_corr = X.corr()
liver_corr
plt.figure(figsize=(30, 30))

sns.heatmap(liver_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           cmap= 'coolwarm')

plt.title('Correlation between features');
#The above correlation also indicates the following correlation

# Total_Protiens & Albumin

# Alamine_Aminotransferase & Aspartate_Aminotransferase

# Direct_Bilirubin & Total_Bilirubin

# There is some correlation between Albumin_and_Globulin_Ratio and Albumin. But its not as high as Total_Protiens & Albumin
# Importing modules

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
#2) Logistic Regression

# Create logistic regression object

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

coeff_df = pd.DataFrame(X.columns)

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

#Predict Output

gauss_predicted = gaussian.predict(X_test)



gauss_score = round(gaussian.score(X_train, y_train) * 100, 2)

gauss_score_test = round(gaussian.score(X_test, y_test) * 100, 2)

print('Gaussian Score: \n', gauss_score)

print('Gaussian Test Score: \n', gauss_score_test)

print('Accuracy: \n', accuracy_score(y_test, gauss_predicted))

print(confusion_matrix(y_test,gauss_predicted))

print(classification_report(y_test,gauss_predicted))



sns.heatmap(confusion_matrix(y_test,gauss_predicted),annot=True,fmt="d")
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

#Predict Output

rf_predicted = random_forest.predict(X_test)



random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)

random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)

print('Random Forest Score: \n', random_forest_score)

print('Random Forest Test Score: \n', random_forest_score_test)

print('Accuracy: \n', accuracy_score(y_test,rf_predicted))

print(confusion_matrix(y_test,rf_predicted))

print(classification_report(y_test,rf_predicted))
###Model evaluation

#We can now rank our evaluation of all the models to choose the best one for our problem. 

models = pd.DataFrame({

    'Model': [ 'Logistic Regression', 'Gaussian Naive Bayes','Random Forest'],

    'Score': [ logreg_score, gauss_score, random_forest_score],

    'Test Score': [ logreg_score_test, gauss_score_test, random_forest_score_test]})

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
#I'm considering seven important features based on recursive feature elimination

#finX = liver_df[['Age','Direct_Bilirubin','Total_Protiens','Albumin', 'Gender_Female', 'Gender_Male']]

finX = liver_df[['Total_Protiens','Albumin', 'Gender_Male']]

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
