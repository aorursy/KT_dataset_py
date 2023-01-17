# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import feature_selection as fs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
# Load the data
adult = pd.read_csv('../input/adult.csv', sep=',', decimal='.', header=None, names=['Age', 'Work Class', 'Final Weight', 'Education', 'Education Number', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours Per Week', 'Native Country', 'Income'])
# Check the data
print(adult)
# Verify the data
adult.head()
# Check for Missing Values and its count
adult.isna().sum()
# Check for datatypes
adult.dtypes
# Check all values present in the column and its count
adult['Age'].value_counts()
# Check all values present in the column and its count
adult['Work Class'].value_counts()
# Remove the rows with incorrect values
adult = adult[adult['Work Class'] != ' ?']
# Re-check all values present in the column and its count
adult['Work Class'].value_counts()
# Check all values present in the column and its count
adult['Final Weight'].value_counts()
# Check all values present in the column and its count
adult['Education Number'].value_counts()
# Check all values present in the column and its count
adult['Marital Status'].value_counts()
# Check all values present in the column and its count
adult['Occupation'].value_counts()
# Remove the rows with incorrect values
adult = adult[adult['Occupation'] != ' ?']
# Re-check all values present in the column and its count
adult['Occupation'].value_counts()
# Check all values present in the column and its count
adult['Relationship'].value_counts()
# Check all values present in the column and its count
adult['Race'].value_counts()
# Check all values present in the column and its count
adult['Sex'].value_counts()
# Check all values present in the column and its count
adult['Capital Gain'].value_counts()
# Check all values present in the column and its count
adult['Capital Loss'].value_counts()
# Check all values present in the column and its count
adult['Hours Per Week'].value_counts()
# Check all values present in the column and its count
adult['Native Country'].value_counts()
# Remove the rows with incorrect values
adult = adult[adult['Native Country'] != ' ?']
# Re-check all values present in the column and its count
adult['Native Country'].value_counts()
# Check all values present in the column and its count
adult['Income'].value_counts()
#Plot Box plot for Age Column
adult['Age'].plot(kind='box', figsize=(10,10))
plt.title('Box plot for Age of the Population', fontsize=15)
plt.suptitle('')
plt.show()
#Plot Pie chart for Work Class Column
adult['Work Class'].value_counts().plot(kind='pie',autopct='%.2f%%', shadow=True, figsize=(10,10))
plt.ylabel('')
plt.title('Percentage of Population as per Work Class', fontsize=15)
plt.show()
#Plot Bar chart for Education column
adult['Education'].value_counts().sort_index().plot(kind='bar', figsize=(10,10))
plt.xlabel('Education')
plt.ylabel('Number of People')
plt.title('Population w.r.t. Education', fontsize=15)
plt.show()
#Plot Box plot for Capital Gain Column
adult['Capital Gain'].plot(kind='box', figsize=(10,10))
plt.title('Box plot for Capital Gain of the Population', fontsize=15)
plt.suptitle('')
plt.show()
#Plot Pie chart for Marital Status Column
adult['Marital Status'].value_counts().plot(kind='pie',autopct='%.2f%%', shadow=True, figsize=(10,10))
plt.ylabel('')
plt.title('Percentage of Population as per Marital Status', fontsize=15)
plt.show()
#Plot Bar chart for Occupation column
adult['Occupation'].value_counts().sort_index().plot(kind='bar', figsize=(10,10))
plt.xlabel('Occupation')
plt.ylabel('Number of People')
plt.title('Population w.r.t. Occupation', fontsize=15)
plt.show()
#Plot Box plot for Capital Loss Column
adult['Capital Loss'].plot(kind='box', figsize=(10,10))
plt.title('Box plot for Capital Loss of the Population', fontsize=15)
plt.suptitle('')
plt.show()
#Plot Pie chart for Race Column
adult['Race'].value_counts().plot(kind='pie',autopct='%.2f%%', shadow=True, figsize=(10,10))
plt.ylabel('')
plt.title('Percentage of Population as per Race', fontsize=15)
plt.show()
#Plot Bar chart for Sex column
adult['Sex'].value_counts().sort_index().plot(kind='bar', figsize=(10,10))
plt.xlabel('Sex')
plt.ylabel('Number of People')
plt.title('Population w.r.t. Sex', fontsize=15)
plt.show()
#Plot Box plot for Hours Per Week Column
adult['Hours Per Week'].plot(kind='box', figsize=(10,10))
plt.title('Box plot for Hours Per Week of the Population', fontsize=15)
plt.suptitle('')
plt.show()
#Plot Bar chart for Income w.r.t. Education
ax = sns.countplot(y='Education', hue='Income', data=adult)
ax.set_title('Population w.r.t. Education and Sex')
ax.set(xlabel='Number of People', ylabel='Education')
plt.show()
#Plot Box plot for Income w.r.t. Hours Per Week
adult.boxplot(column='Hours Per Week', by='Income', figsize=(10,10))
plt.ylabel('Hours Per Week')
plt.title('Box plot for Price of the Cars as per its Make', fontsize=15)
plt.suptitle('')
plt.show()
#Plot Bar chart for Income w.r.t. Gender
ax = sns.countplot(x='Sex', hue='Income', data=adult)
ax.set_title('Income w.r.t. Gender')
ax.set(xlabel='Gender', ylabel='Number of People')
plt.show()
#Plot Bar chart for Income w.r.t. Relationship
ax = sns.countplot(y='Relationship', hue='Income', data=adult)
ax.set_title('Income w.r.t. Relationship')
ax.set(ylabel='Relationship', xlabel='Number of People')
plt.show()
#Plot Box plot for Income w.r.t. Age
adult.boxplot(column='Age', by='Income', figsize=(10,10))
plt.ylabel('Age')
plt.title('Box plot for Age of Population as per their Age', fontsize=15)
plt.suptitle('')
plt.show()
#Plot Box plot for Capital Gain w.r.t. Work Class
adult.boxplot(column='Capital Gain', by='Work Class', figsize=(10,10))
plt.ylabel('Number of People')
plt.title('Box plot for Capital Gain of Population as per their Work Class', fontsize=15)
plt.suptitle('')
plt.show()
#Plot Box plot for Capital Loss w.r.t. Work Class
adult.boxplot(column='Capital Loss', by='Work Class', figsize=(10,10))
plt.ylabel('Number of People')
plt.title('Box plot for Capital Loss of Population as per their Work Class', fontsize=15)
plt.suptitle('')
plt.show()
# Cat Plot for Hours Per Week w.r.t. Native Country
ax = sns.catplot(x="Hours Per Week", y="Native Country", jitter=False, data=adult)
plt.title('Hours Per Week w.r.t. Native Country', fontsize=15)
plt.show()
#Plot Box plot for Hours Per Week w.r.t. Gender
adult.boxplot(column='Hours Per Week', by='Sex', figsize=(10,10))
plt.ylabel('Number of People')
plt.title('Box plot for Hours Per Week of Population as per their Gender', fontsize=15)
plt.suptitle('')
plt.show()
# Cat Plot for Hours Per Week w.r.t. Education
ax = sns.catplot(x="Hours Per Week", y="Occupation", jitter=False, data=adult)
plt.title('Hours Per Week w.r.t. Occupation', fontsize=15)
plt.show()
#Plot Scatter Matrix for all numerical columns
scatter_matrix(adult, alpha=0.2, figsize=(12,12))
plt.show()
# Drop the column Education as it is redundant (Education Number is enough to proceed further)
adult = adult.drop(columns=['Education'])
# Apply interger labels on data set instead of string values
adult = adult.apply(LabelEncoder().fit_transform)
# Verify the data set after labelling
adult.head()
# Divide the data set into factors and its results
X_data = adult.iloc[:, :13]
y_data = adult.iloc[:, 13]
# Verify the divided data sets
X_data.head()
# Verify the divided data sets
y_data.head()
# Feature Selection of 10 best features present in our data set
fs_selectKBest = fs.SelectKBest(fs.f_classif, k=10)
fs_selectKBest.fit_transform(X_data, y_data)
fs_index = np.argsort(fs_selectKBest.scores_)[::-1][0:10]

# Filter the data set with only the 10 best selected features
X_data = X_data[X_data.columns[fs_index][0:10].values]
X_data.head()
# Splitting of 50% Training and 50% Testing data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=2)

# Predicting results using K-Nearest Neighbour algorithm 
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion Matrix
print ('\nConfusion Matrix:')
print (confusion_matrix(y_test, y_pred))

# Classification Report
print ('\nClassification Report:')
print (classification_report(y_test, y_pred))

# Accuracy Rate
score_knn = knn.score(X_test,y_test)
print ('Accuracy of the K-Nearest Neighbour Model is: ', score_knn)
# Splitting of 60% Training and 40% Testing data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.4, random_state=2)

# Predicting results using K-Nearest Neighbour algorithm 
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion Matrix
print ('\nConfusion Matrix:')
print (confusion_matrix(y_test, y_pred))

# Classification Report
print ('\nClassification Report:')
print (classification_report(y_test, y_pred))

# Accuracy Rate
score_knn = knn.score(X_test,y_test)
print ('Accuracy of the K-Nearest Neighbour Model is: ', score_knn)
# Splitting of 80% Training and 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=2)

# Predicting results K-Nearest Neighbour algorithm 
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion Matrix
print ('\n', 'Confusion Matrix:')
print (confusion_matrix(y_test, y_pred))

# Classification Report
print ('\n', 'Classification Report:')
print (classification_report(y_test, y_pred))

# Accuracy Rate
score_knn = knn.score(X_test,y_test)
print ('Accuracy of the K-Nearest Neighbour Model is: ', score_knn)
# Splitting of 50% Training and 50% Testing data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.5, random_state=2)

# Predicting results Decision Tree algorithm 
clf = DecisionTreeClassifier()
dt = clf.fit(X_train, y_train)
y_pred = dt.predict(X_test)   

# Confusion Matrix
print ('\n', 'Confusion Matrix:')
print (confusion_matrix(y_test, y_pred))

# Classification Report
print ('\n', 'Classification Report:')
print (classification_report(y_test, y_pred))

# Accuracy Rate
score_dt = clf.score(X_test,y_test)
print ('Accuracy of the Decision Tree Model is: ', score_dt )
# Splitting of 60% Training and 40% Testing data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.4, random_state=2)

# Predicting results K-Nearest Neighbour algorithm 
clf = DecisionTreeClassifier()
dt = clf.fit(X_train, y_train)
y_pred = dt.predict(X_test)   

# Confusion Matrix
print ('\n', 'Confusion Matrix:')
print (confusion_matrix(y_test, y_pred))

# Classification Report
print ('\n', 'Classification Report:')
print (classification_report(y_test, y_pred))

# Accuracy Rate
score_dt = clf.score(X_test,y_test)
print ('Accuracy of the Decision Tree Model is: ', score_dt )
# Splitting of 80% Training and 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=2)

# Predicting results K-Nearest Neighbour algorithm 
clf = DecisionTreeClassifier()
dt = clf.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Confusion Matrix
print ('\n', 'Confusion Matrix:')
print (confusion_matrix(y_test, y_pred))

# Classification Report
print ('\n', 'Classification Report:')
print (classification_report(y_test, y_pred))

# Accuracy Rate
score_dt = clf.score(X_test,y_test)
print ('Accuracy of the Decision Tree Model is: ', score_dt )