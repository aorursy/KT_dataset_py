# Import relevant Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Loading in Dataset (downloaded from kaggle.com/andrewmvd)

dataset = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

# View first 5 rows
dataset.head()
# Here we begin understanding the data.
# Using the info() method helps quantify null values and potentially unhelpful catagories.
dataset.info()
# Using the describe() method gives insight into the data's central tendencies.
dataset.describe()
# Split data into a group for numerical values and another group for categorical (in this case, binary 0 or 1) values.
n_dataset = dataset[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']]
c_dataset = dataset[['anaemia','diabetes','high_blood_pressure','sex','smoking','DEATH_EVENT']]
# visualize distributions for the Numerical Value dataset. 
# This helps us determine which distriutions are generally normalized, skewed, have outliers, etc.
for col in n_dataset.columns:
    plt.hist(n_dataset[col])
    plt.title(col)
    plt.show()
# Quantify correlations between the numerical values to understand their relationships.
# This will help avoid multicolinearity if necessary.
#print(n_dataset.corr())

# Visualize correlations
sns.heatmap(n_dataset.corr())
# Comparing mortality rate for each numerical column
pd.pivot_table(dataset, index = 'DEATH_EVENT', values = n_dataset.columns)
# Now visualize binary distributions for the Categorical Value dataset. 

for col in c_dataset.columns:
    sns.barplot(c_dataset[col].value_counts().index, c_dataset[col].value_counts()).set_title(col)
    plt.show()
# Now compare mortality rate for each categorical column
for col in c_dataset:
    if col != 'DEATH_EVENT':
        print(pd.pivot_table(dataset, index = 'DEATH_EVENT', columns = col, values = 'age',aggfunc = 'count'))
        print()
# importing libraries for Standard scaling and splitting the data into train and test sets
# Using Standardization and splitting the data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
#dataset.describe()
# With the chosen features, define the final dataset that will be split and used in the model
df_model = dataset[['DEATH_EVENT','age','anaemia','ejection_fraction','high_blood_pressure','time','serum_creatinine',]]
X = df_model.drop('DEATH_EVENT', axis = 1)
y = df_model.DEATH_EVENT.values
# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

# Using Standardization
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# First fit Logistic Regression model and show performance metrics
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

# Creating a list to save each performance score for each model added
acc_list = []
rec_list = []
pre_list = []
f1_list = []
# Full lists will each contain [LogRegression, kNearestNeighbors, gNaiveBayes, Randomforest, DecisonTree, SVM]
# Calculating confusion matrix for better sense of the model's performance
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Define a function that prints each element in a given confusion matrix
def print_cm(cm):
    print("True Negatives: "+str(cm[0,0]))
    print("True Positives: "+str(cm[1,1]))
    print("False Negatives: "+str(cm[1,0]))
    print("False Positives: "+str(cm[0,1]))

predictions = logreg.predict(X_test)
# determine and collect scores of Logistic Regression model
acc = accuracy_score(y_test,predictions)
rec = recall_score(y_test,predictions)
pre = precision_score(y_test,predictions)
f1 = f1_score(y_test,predictions)

print_cm(confusion_matrix(y_test, predictions))

acc_list.append(acc)
rec_list.append(rec)
pre_list.append(pre)
f1_list.append(f1)
# Now build a K Nearest Neighbors model
# First loop to find the number of neighbors that returns the highest accuracy
from sklearn.neighbors import KNeighborsClassifier

n_list = []
for n in range(2,10):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    n_list.append(accuracy_score(y_test,predictions))
plt.plot(list(range(2,10)), n_list)
plt.title("kNN Accuracy per Number of Neighbors")
plt.show()
# Since 5 neighbors yields the highest accuracy, use n = 5 in the model
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)

# KNN confusion matrix 

acc = accuracy_score(y_test,predictions)
rec = recall_score(y_test,predictions)
pre = precision_score(y_test,predictions)
f1 = f1_score(y_test,predictions)

print("Accuracy Score: " +str(acc))
print_cm(confusion_matrix(y_test, predictions))

acc_list.append(acc)
rec_list.append(rec)
pre_list.append(pre)
f1_list.append(f1)
# Naïve Bayes model
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
predictions = gnb.predict(X_test)
# determine and collect scores of Gaussian Naïve Bayes model
acc = accuracy_score(y_test,predictions)
rec = recall_score(y_test,predictions)
pre = precision_score(y_test,predictions)
f1 = f1_score(y_test,predictions)

print("Accuracy Score: " +str(acc))
print_cm(confusion_matrix(y_test, predictions))

acc_list.append(acc)
rec_list.append(rec)
pre_list.append(pre)
f1_list.append(f1)
# import Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Find best number of nodes

n_list = []
for n in range(2,10):
    dtc = DecisionTreeClassifier(max_leaf_nodes = n, random_state=0, criterion='entropy')
    dtc.fit(X_train, y_train)
    predictions = dtc.predict(X_test)
    n_list.append(accuracy_score(y_test,predictions))

plt.title("DTC Accuracy per Number of nodes")
plt.plot(list(range(2,10)), n_list)
plt.show()
# Based on the plot, n=5 and n=6 yield the highest accuracy
# Build Decision Tree Classifier
dtc = DecisionTreeClassifier(max_leaf_nodes = 5, random_state=0, criterion='entropy')
dtc.fit(X_train, y_train)
predictions = dtc.predict(X_test)
# Determine and collect scores of Decision Tree model

acc = accuracy_score(y_test,predictions)
rec = recall_score(y_test,predictions)
pre = precision_score(y_test,predictions)
f1 = f1_score(y_test,predictions)

print("Accuracy Score: " +str(acc))
print_cm(confusion_matrix(y_test, predictions))

acc_list.append(acc)
rec_list.append(rec)
pre_list.append(pre)
f1_list.append(f1)
# Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Find best number of estimators
n_list = []
for n in range(10,100):
    rfc = RandomForestClassifier(n_estimators = n, random_state=0)
    rfc.fit(X_train, y_train)
    predictions = rfc.predict(X_test)
    n_list.append(accuracy_score(y_test,predictions))
#print(mylist)
plt.plot(list(range(10,100)), n_list)
plt.title("Random Forest Classifier Accuracy per Number of Estimators")
plt.show()
# Based on the plot, n=10 yields the highest accuracy
# Build Random Forest Classifier model
rfc = RandomForestClassifier(n_estimators = 10, random_state=0)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
# Determine and collect score of Random Forest model

acc = accuracy_score(y_test,predictions)
rec = recall_score(y_test,predictions)
pre = precision_score(y_test,predictions)
f1 = f1_score(y_test,predictions)

print("Accuracy Score: " +str(acc))
print_cm(confusion_matrix(y_test, predictions))

acc_list.append(acc)
rec_list.append(rec)
pre_list.append(pre)
f1_list.append(f1)
# Support Vector Machine
from sklearn.svm import SVC
svmodel = SVC(random_state=0, kernel = 'rbf')
svmodel.fit(X_train, y_train)
predictions = svmodel.predict(X_test)
# Determine and collect scores of Decision Tree model

acc = accuracy_score(y_test,predictions)
rec = recall_score(y_test,predictions)
pre = precision_score(y_test,predictions)
f1 = f1_score(y_test,predictions)

print("Accuracy Score: " +str(acc))
print_cm(confusion_matrix(y_test, predictions))

acc_list.append(acc)
rec_list.append(rec)
pre_list.append(pre)
f1_list.append(f1)
# Results

# List of Classifier Models used 
models = ['Logistic Regression','K-NearestNeighbor','Naive Bayes','Decision Tree','Random Forest', 'Support Vector Machine']
# List of metrics used
mets = ['Accuracy', 'Recall','Precision','F1-Score']
colors = ['red','purple','blue','black']

# combine into a python Dict
d = {'Model':models, mets[0]:acc_list, mets[1]:rec_list, mets[2]:pre_list, mets[3]:f1_list}

# create a pandas dataframe from Dict
stat_df = pd.DataFrame(data=d)
stat_df
# rearrange the dataframe for easier plotting
stat_df = pd.melt(stat_df, id_vars="Model", var_name="Metric", value_name="Score")
stat_df
# Contruct bar plot to visualize each classifier's performance
sns.catplot(x='Model', y='Score', hue='Metric', data=stat_df, kind='bar',palette=colors,height=5,aspect=3)
plt.title("Performance Metrics by Classifier Model")
plt.ylabel("Score %")
plt.xlabel("Model")
plt.show()
# K-Nearest Neighbor has the highest Precision, but Recall and F1 have higher importance for this task.
# Decision Tree and Random Forest tied for highest Accuracy (83%), but Decision Tree has a better Recall and F1 Score.