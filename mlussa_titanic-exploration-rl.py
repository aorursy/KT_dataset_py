# Load it from source
import pandas as pd
import numpy as np

train = pd.read_csv("../input/train.csv")
# Look at the first 10 records
train[:10]
# Check to make sure the data type for each column is correct
print(train.dtypes)
# Change survived and Pclass
train["Survived"] = train["Survived"].astype('category')
train["Pclass"] = train["Pclass"].astype('category',ordered=True)
train["Embarked"] = train["Embarked"].astype('category')
# Check again
print(train.dtypes)
# Extracting only the first letter from the cabin class
cabin_class = []
for index, row in train.iterrows():
    if str(row['Cabin']) != "nan":
        cabin_class.append(str(row['Cabin'])[:1])
    else:
        cabin_class.append(row['Cabin'])
#print(cabin_class[:10])
# Add the extracted info back to the original training df
train['cabin_class'] = cabin_class
# Take a look!
train[:1]
# Handling more NA here
for c in train.columns.values:
    print("At column",c,"# of NA records:",train[c].isnull().sum())
# Fill age NA here
print("Filling NA for age.")
from sklearn.preprocessing import Imputer
my_imputer = Imputer(copy=False)
my_imputer.fit_transform(train['Age'].reshape(-1, 1))
# Count how many NAs here
print("# of NA records:",train['cabin_class'].isnull().sum())
cabin_class_gb = train.groupby(['cabin_class'],sort=False).size()
val = cabin_class_gb[cabin_class_gb == cabin_class_gb.max()].index[0]
# Finally, fill NA here...
print("Filling NAs...")
train['cabin_class'].fillna(val,inplace=True)
print("NAs filled in placed.")
# One-hot vector handling here
newp = pd.get_dummies(train.drop(['Ticket','Name','Cabin','Survived','PassengerId'], axis=1))
print(newp[:1])
traind = train.drop(['Ticket','Name','Cabin','Survived'], axis=1)
print(traind[:1])
# check again
print(newp.dtypes)
import matplotlib.pyplot as plt
# survival rate vs. classes (only looking at those who survived)
svc = train[(train['Survived'] == 1)].groupby(['Pclass']).size()
print(svc)
plt.title("survival rate vs. classes")
plt.ylabel('Number of people survived')
svc.plot.bar()
# survival rate vs. gender (only looking at those who survived)
svs = train[(train['Survived'] == 1)].groupby(['Sex']).size()
print(svs)
plt.title("survival rate vs. gender")
plt.ylabel('Number of people survived')
svs.plot.bar(color="green")
# survival rate vs. age (only looking at those who survived)
bins = [10,20,30,40,50,60,70,80]
sva = pd.cut(train[(train['Survived'] == 1)]['Age'],bins=bins).value_counts(sort=False)
print(sva)
plt.title("survival rate vs. Age")
plt.ylabel('Number of people survived')
sva.plot.bar(color="pink")
# survival rate vs. sibsp and parch
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#svsp = train[['SibSp','Parch','Survived']]
ax.scatter(train['SibSp'],train['Parch'],train['Survived'])
# Construct a train/test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(newp, train['Survived'], test_size=0.2)
# Print out the size of the train/test set
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("Features used:",newp.columns.values)
print("Total # of survival records:",len(train[(train['Survived'] == 1)]))
print("Total # of deceased records:",len(train[(train['Survived'] == 0)]))

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(class_weight="balanced")
logisticRegr.fit(X_train, y_train)
logisticRegr.predict(X_test)
# print the score here
score = logisticRegr.score(X_test, y_test)
print("Accuracy score of the logistic regression:",score)
# Also view the confusion matrix
# Note from https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
predictions = logisticRegr.predict(X_test)

from sklearn import metrics

# Original confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

# also get precision recall
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_dt = clf.predict(X_test)
print(classification_report(y_test, y_dt))
print("KNN accuracy:", accuracy_score(y_test, y_dt))
# Decision tree visualization
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=X_train.columns.values,  
                         class_names=y_train.name,  
                         filled=True, rounded=True,  
                         special_characters=True, max_depth=10)  
graph = graphviz.Source(dot_data)  
graph 
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
feature_imp_val = clf.feature_importances_
feature_imp = {}
for i in range(len(feature_imp_val)):
    feature_imp[X_train.columns.values[i]] = feature_imp_val[i]
for k, v in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(k,v)
print("\n")
y_rf = clf.predict(X_test)
print(classification_report(y_test, y_rf))
print("Random Forest accuracy:", accuracy_score(y_test, y_rf))
# The following code is adapted from scikit-learn official website
from sklearn import svm
clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)
y_svm = clf.predict(X_test)
print(classification_report(y_test, y_svm))
print("SVM accuracy:", accuracy_score(y_test, y_svm))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_gnb = gnb.fit(X_train, y_train).predict(X_test)
print(classification_report(y_test, y_gnb))
print("Naive Bayes (Gaussian) accuracy:", accuracy_score(y_test, y_gnb))
# load the test files
test = pd.read_csv("../input/test.csv")
# Change survived and Pclass
test["Pclass"] = test["Pclass"].astype('category',ordered=True)
test["Embarked"] = test["Embarked"].astype('category')
# Extracting only the first letter from the cabin class
cabin_class = []
for index, row in test.iterrows():
    if str(row['Cabin']) != "nan":
        cabin_class.append(str(row['Cabin'])[:1])
    else:
        cabin_class.append(row['Cabin'])
# Add the extracted info back to the original test df
test['cabin_class'] = cabin_class
# fill age NAs
from sklearn.preprocessing import Imputer
my_imputer = Imputer(copy=False)
my_imputer.fit_transform(test['Age'].reshape(-1, 1))
# fill cabin class NAs
cabin_class_gb = test.groupby(['cabin_class'],sort=False).size()
val = cabin_class_gb[cabin_class_gb == cabin_class_gb.max()].index[0]
test['cabin_class'].fillna(val,inplace=True)
# fill fare # NAs
fare_gb = test.groupby(['Fare'],sort=False).size()
val2 = fare_gb[fare_gb == fare_gb.max()].index[0]
test['Fare'].fillna(val2,inplace=True)
# One-hot vector handling here
newp = pd.get_dummies(test.drop(['Ticket','Name','Cabin','PassengerId'], axis=1))
for c in newp.columns.values:
    print("At column",c,"# of NA records:",newp[c].isnull().sum())
#X_train = X_train.drop(['cabin_class_T'],axis=1)

# Actual RF
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_rf = clf.predict(newp)
print("PassengerId,Survived")
for v in range(len(newp)):
    print(str(test['PassengerId'][v])+","+str(y_rf[v]))