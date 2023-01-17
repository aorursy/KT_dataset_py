import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/indian_liver_patient.csv')
df.head(2)
import seaborn as sns
patient = df.Dataset.value_counts()
sns.barplot(x = patient.index, y = patient, palette='magma')
plt.xlabel('Dataset')
plt.ylabel('Count')
plt.show()
gender = df.Gender.value_counts()
sns.barplot(x = gender.index, y = gender, palette='magma')
plt.xlabel('Gender of the Patients')
plt.ylabel('Count')
plt.show()
sns.factorplot(x='Age', y='Gender', hue='Dataset', data=df, palette='magma')
df.groupby('Dataset').mean()
liver_sub = df.iloc[:,[2,10]]
liver_sub.head(2)
sns.boxplot(x='Dataset', y='Total_Bilirubin', data=liver_sub, palette='magma')
sns.boxplot(x='Dataset', y='Total_Bilirubin', data=liver_sub, palette='magma', showfliers=False) 
liver_sub1 = df.iloc[:,[3,10]]
liver_sub1.head(2)
sns.boxplot(x='Dataset', y='Direct_Bilirubin', data=liver_sub1, palette='magma')
sns.boxplot(x='Dataset', y='Direct_Bilirubin', data=liver_sub1, palette='magma', showfliers=False)
liver_sub2 = df.iloc[:,[4,10]]
sns.boxplot(x='Dataset', y='Alkaline_Phosphotase', data=liver_sub2, palette='magma')
sns.boxplot(x='Dataset', y='Alkaline_Phosphotase', data=liver_sub2, palette='magma', showfliers = False)
liver_sub3 = df.iloc[:,[5,10]]
sns.boxplot(x='Dataset', y='Alamine_Aminotransferase', data=liver_sub3, palette='magma')
plt.show()
sns.boxplot(x='Dataset', y='Alamine_Aminotransferase', data=liver_sub3, palette='magma', showfliers = False);
liver_sub4 = df.iloc[:,[6,10]]
sns.boxplot(x='Dataset', y='Aspartate_Aminotransferase', data=liver_sub4, palette='magma')
plt.show()
sns.boxplot(x='Dataset', y='Aspartate_Aminotransferase', data=liver_sub4, palette='magma', showfliers = False);
liver_sub5 = df.iloc[:,[7,10]]
sns.boxplot(x='Dataset', y='Total_Protiens', data=liver_sub5, palette='magma')
plt.show()
liver_sub6 = df.iloc[:,[8,10]]
sns.boxplot(x='Dataset', y='Albumin', data=liver_sub6, palette='magma')
plt.show()
liver_sub7 = df.iloc[:,[9,10]]
sns.boxplot(x='Dataset', y='Albumin_and_Globulin_Ratio', data=liver_sub7, palette='magma', showfliers = False)
plt.show()
df[df['Albumin_and_Globulin_Ratio'].isnull()]
df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())
pd.get_dummies(df['Gender'], prefix = 'Gender').head()
df = pd.concat([df, pd.get_dummies(df['Gender'], prefix = 'Gender')], axis=1)
df.head()
X = df.drop(['Gender','Dataset'], axis=1)
X.head(3)
y = df['Dataset']
df_corr = X.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(df_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
           cmap= 'magma')
plt.title('Correlation between features');
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
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
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
log_predicted= logreg.predict(X_test)
accuracy_score(y_test, log_predicted)
confusion_matrix(y_test,log_predicted)
print('Classification Report: \n', classification_report(y_test,log_predicted))
sns.heatmap(confusion_matrix(y_test,log_predicted), annot=True, fmt="d")
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
gauss_predicted = gaussian.predict(X_test)
accuracy_score(y_test, gauss_predicted)
print(classification_report(y_test,gauss_predicted))
sns.heatmap(confusion_matrix(y_test, gauss_predicted),annot=True, fmt="d")
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_predicted = rf.predict(X_test)
accuracy_score(y_test,rf_predicted)
print(classification_report(y_test,rf_predicted))
sns.heatmap(confusion_matrix(y_test, rf_predicted), annot = True, fmt = "d")
linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
lin_predicted = linear.predict(X_test)
from sklearn.feature_selection import RFE
rfe =RFE(linear, n_features_to_select=3)
rfe.fit(X,y)
for i in range(len(rfe.ranking_)):
    if rfe.ranking_[i] == 1:
        print(X.columns.values[i])
X_final = X[['Total_Protiens','Albumin','Gender_Male']]
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.30, random_state=101)
logreg.fit(X_train, y_train)
log_predicted= logreg.predict(X_test)
accuracy_score(y_test, log_predicted)
print('Classification Report: \n', classification_report(y_test,log_predicted))
sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")
gaussian.fit(X_train, y_train)
gauss_predicted = gaussian.predict(X_test)
accuracy_score(y_test, gauss_predicted)
print(classification_report(y_test,gauss_predicted))
sns.heatmap(confusion_matrix(y_test, gauss_predicted),annot=True, fmt="d")
rf.fit(X_train, y_train)
rf_predicted = rf.predict(X_test)
accuracy_score(y_test,rf_predicted)
