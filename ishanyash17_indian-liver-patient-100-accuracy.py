import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder
liv_df = pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')

liv_df.head()
liv_df.info()
liv_df.describe(include='all')
liv_df.isnull().sum()
sns.countplot(data=liv_df, x = 'Dataset', label='Count')

LD, NLD = liv_df['Dataset'].value_counts()

print('Number of patients diagnosed with liver disease: ',LD)

print('Number of patients not diagnosed with liver disease: ',NLD)
sns.countplot(data=liv_df, x = 'Gender', label='Count')

M, F = liv_df['Gender'].value_counts()

print('Number of patients that are male: ',M)

print('Number of patients that are female: ',F)
sns.factorplot(x="Age", y="Gender", hue="Dataset", data=liv_df);
liv_df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).count().sort_values(by='Dataset', ascending=False)
liv_df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).mean().sort_values(by='Dataset', ascending=False)
g = sns.FacetGrid(liv_df, col="Dataset", row="Gender", margin_titles=True)

g.map(plt.hist, "Age", color="red")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Disease by Gender and Age');
g = sns.FacetGrid(liv_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Direct_Bilirubin", "Total_Bilirubin", edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=liv_df, kind="reg")
g = sns.FacetGrid(liv_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Aspartate_Aminotransferase", "Alamine_Aminotransferase",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=liv_df, kind="reg")
g = sns.FacetGrid(liv_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Alkaline_Phosphotase", "Alamine_Aminotransferase",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data=liv_df, kind="reg")
g = sns.FacetGrid(liv_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Total_Protiens", "Albumin",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Total_Protiens", "Albumin", data=liv_df, kind="reg")
g = sns.FacetGrid(liv_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Albumin", "Albumin_and_Globulin_Ratio",  edgecolor="w")

plt.subplots_adjust(top=0.9)
sns.jointplot("Albumin_and_Globulin_Ratio", "Albumin", data=liv_df, kind="reg")
g = sns.FacetGrid(liv_df, col="Gender", row="Dataset", margin_titles=True)

g.map(plt.scatter,"Albumin_and_Globulin_Ratio", "Total_Protiens",  edgecolor="w")

plt.subplots_adjust(top=0.9)
liv_df = pd.concat([liv_df,pd.get_dummies(liv_df['Gender'], prefix = 'Gender')], axis=1)

liv_df.head()
liv_df[liv_df['Albumin_and_Globulin_Ratio'].isnull()] #the columns having null values
liv_df["Albumin_and_Globulin_Ratio"] = liv_df.Albumin_and_Globulin_Ratio.fillna(liv_df['Albumin_and_Globulin_Ratio'].mean())
from sklearn.model_selection import train_test_split

Droop_gender = liv_df.drop(labels=['Gender' ],axis=1 )

X = Droop_gender

y = liv_df['Dataset']
# split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn import linear_model
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

#Predicting Output

rf_predicted = random_forest.predict(X_test)

random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)

random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)



print('Random Forest Score: \n', random_forest_score)

print('Random Forest Test Score: \n', random_forest_score_test)

print('Accuracy: \n', accuracy_score(y_test,rf_predicted))

print(confusion_matrix(y_test,rf_predicted))

print(classification_report(y_test,rf_predicted))
from sklearn.linear_model import LogisticRegression
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
plt.figure(figsize=(10, 10))

sns.heatmap(X.corr(), cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 8},

           cmap= 'coolwarm')

plt.title('Correlation between features')
models = pd.DataFrame({

    'Model': [ 'Logistic Regression','Random Forest'],

    'Score': [ logreg_score, random_forest_score],

    'Test Score': [ logreg_score_test, random_forest_score_test]})

models.sort_values(by='Test Score', ascending=False)