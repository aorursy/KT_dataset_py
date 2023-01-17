# Import Relevant Libraries
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#Read the Data File
theraDF = pd.read_csv("../input/thera-bank-personal-loan/Bank_Personal_Loan_Modelling.csv")
theraDF.head()
# Let us ensure that there are no Null Values
print("Null Values Check\n")
print(theraDF.isnull().sum())
print("\n\n NAN Values Check \n")
print(theraDF.isna().sum())

theraDF.dtypes
theraDF.describe()
theraDF.describe().T
sns.set(color_codes = True)
sns.distplot(theraDF["Age"])
plt.hist(theraDF["Age"],bins=5)
sns.boxplot(x = 'Age', data=theraDF )
# Take only values >0
expDF = theraDF[theraDF["Experience"] > 0]
print("Number of Experiences <= 0:", (theraDF.count() - expDF.count())[0])
# Find mean of all > than zero values
meanExp = int(expDF["Experience"].mean())
print("\n\nMean Experience:",meanExp)

# Replace 0 and negative Experience with the Mean value
theraDF.loc[(theraDF.Experience <= 0),'Experience'] = meanExp
#Print the data description to ensure minimum Experience is fixed
theraDF.describe().T

sns.boxplot(x = 'Experience', data=theraDF )
sns.distplot(theraDF["Income"])
plt.hist(theraDF["Income"], bins=5)
sns.boxplot(x = 'Income', data = theraDF)
print("Number of unique Zip Codes:", theraDF['ZIP Code'].nunique())
theraDF['ZIP Code'].value_counts()[:10].plot(kind='barh')
sns.countplot(theraDF['Family'])
sns.distplot(theraDF["CCAvg"])
sns.boxplot(x = 'CCAvg', data = theraDF)
sns.countplot(theraDF['Education'])
sns.swarmplot(theraDF['Education'],theraDF['Income'])
sns.distplot(theraDF["Mortgage"])
numcustNoMortgage = theraDF[theraDF["Mortgage"] == 0].Mortgage.count()
print("Customers with no Mortgage:\n", numcustNoMortgage)
print("Percentage of customers without Morgage:\n", (numcustNoMortgage * 100)/5000)
sns.boxplot(x = 'Mortgage', data = theraDF)
prodArr = pd.DataFrame()

products = ["Personal Loan","Securities Account","CD Account","Online","CreditCard"]
productCount = [len(theraDF[theraDF["Personal Loan"] == 1]),len(theraDF[theraDF["Securities Account"] == 1]),
                len(theraDF[theraDF["CD Account"] == 1]),len(theraDF[theraDF["Online"] == 1]),
                len(theraDF[theraDF["CreditCard"] == 1])]

# Build cccnt = len(theraDF[theraDF["CreditCard"] == 1])plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(products, productCount, align='center', alpha=0.5)
ax.set_ylabel('Number of customers')
ax.set_xticks(products)
ax.set_xticklabels(products)
ax.set_title('Banking Products')
ax.yaxis.grid(True)


# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot.png')
plt.show()


sns.pairplot(theraDF[["Age","Income","Experience","Mortgage"]])
numcustPersonalLoan = theraDF[theraDF["Personal Loan"] == 1].Mortgage.count()
print("Customers with Personal Loans:\n", numcustPersonalLoan)
print("Percentage of customers with Personal Loans:\n", (numcustPersonalLoan * 100)/5000)
sns.countplot(x='Family',hue='Personal Loan',data=theraDF)
sns.pairplot(theraDF[['Mortgage', 'Income', 'CCAvg', 'Age', 'Personal Loan']], hue = 'Personal Loan');
sns.swarmplot(theraDF['Personal Loan'],theraDF['Income'])
sns.catplot(x='Family', y='Income', hue='Personal Loan', data = theraDF, kind='swarm')
corr = theraDF.corr()
corr
plt.figure(figsize=(12,10))
sns.heatmap(corr,annot=True)
theraDF.drop(columns ='ID',inplace=True)
theraDF.drop(columns ='Experience',inplace=True)
theraDF.drop(columns ='ZIP Code',inplace=True)

theraDF.head()
X = theraDF.drop('Personal Loan', axis=1) # X axis without the Target Variable
y= theraDF['Personal Loan'] # target variable on y Axis

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
X_train.head()
# Let us scale training data set test data using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
# Suggested value of K should be sqrt(n)
K = round(math.sqrt(len(theraDF)))
print(len(theraDF))
print(K)
error_rate = []

for i in range(1,100):
 NNH = KNeighborsClassifier(n_neighbors=i)
 NNH.fit(X_train,y_train)
 KNN_predicted_labels = NNH.predict(X_test)
 error_rate.append(np.mean(KNN_predicted_labels != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color= 'blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
# From the 
K = 3

NNH = KNeighborsClassifier(n_neighbors= int(K) , weights = 'distance' )

# Call Nearest Neighbour algorithm
NNH.fit(X_train, y_train)
#Predict using the test data and score the result 

KNN_predicted_labels = NNH.predict(X_train)
train_acc = metrics.accuracy_score(y_train, KNN_predicted_labels)
print("Model Accuracy with Training Data: {0:.4f}".format(metrics.accuracy_score(y_train, KNN_predicted_labels)*100))
print()

KNN_predicted_labels = NNH.predict(X_test)
test_acc = metrics.accuracy_score(y_test, KNN_predicted_labels)
print("Model Accuracy with Testing Data: {0:.4f}".format(metrics.accuracy_score(y_test, KNN_predicted_labels)*100))
print()
from sklearn.naive_bayes import GaussianNB # using Gaussian algorithm from Naive Bayes

# creatw the model
GNB = GaussianNB()

GNB.fit(X_train, y_train)
GNB_predicted_labels = GNB.predict(X_train)

print("Model Accuracy with Training Data: {0:.4f}".format(metrics.accuracy_score(y_train, GNB_predicted_labels)*100))
print()

GNB_predicted_labels = GNB.predict(X_test)

print("Model Accuracy with Testing data: {0:.4f}".format(metrics.accuracy_score(y_test, GNB_predicted_labels)*100))
print()
# Fit the model on train
LR = LogisticRegression(solver="liblinear")
LR.fit(X_train, y_train)
LR_predicted_labels = LR.predict(X_train)

print("Model Accuracy with Training Data: {0:.4f}".format(metrics.accuracy_score(y_train, LR_predicted_labels)*100))
print()

LR_predicted_labels = LR.predict(X_test)

print("Model Accuracy with Testing data: {0:.4f}".format(metrics.accuracy_score(y_test, LR_predicted_labels)*100))
print()
from sklearn.tree import DecisionTreeClassifier

dTree = DecisionTreeClassifier(criterion = 'gini', random_state=1)
dTree.fit(X_train, y_train)
print(dTree.score(X_train, y_train))
print(dTree.score(X_test, y_test))
# If graphviz doesn't work, we can use plot_tree method from sklearn.tree

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fn = list(X_train)
cn = ['No', 'Yes']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4, 4), dpi=600)
plot_tree(dTree, feature_names = fn, class_names=cn, filled = True)

fig.savefig('tree.png')
dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, min_samples_split = 150, min_samples_leaf = 50, random_state=1)
DT_predicted_labels = dTreeR.fit(X_train, y_train)
print(dTreeR.score(X_train, y_train))
print(dTreeR.score(X_test, y_test))
#Function to print Confusion Matrix and Classification Report
def Print_CM_CR_AUC(y_test,y_predict, algoname):
    cm = confusion_matrix(y_test,y_predict)
    sns.heatmap(cm,annot=True, fmt='.2f', xticklabels=[0,1], yticklabels=[0,1])
    plt.ylabel('observed')
    plt.xlabel('Predicted')
    plt.show()
    # get accuracy of model
    acc_score = accuracy_score(y_test,y_predict)
    # get F1-score of model
    F1_score = f1_score(y_test,y_predict) 
    # get the classification report
    class_rep = classification_report(y_test,y_predict)

    print("Accuracy of ", algoname, " is {} %".format(acc_score*100))
    print("F1-score of ", algoname, " is {} %".format(F1_score*100))
    print("Classification report for ", algoname, " is: \n",class_rep)
    
    #AUC Calculations
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    print("AUC for ", algoname, ":", round(metrics.auc(fpr, tpr)*100,2))

#Draw Confusion Matrix and Classification Report for KNN
y_predictedNNH = NNH.predict(X_test)
Print_CM_CR_AUC(y_test,y_predictedNNH, "KNN")
#Draw Confusion Matrix and Classification Report for Naive Bayes
y_predictedGNB = GNB.predict(X_test)
Print_CM_CR_AUC(y_test,y_predictedGNB, "Naive Bayes")
#Draw Confusion Matrix and Classification Report for Logitic Regression
y_predictedLR = LR.predict(X_test)
Print_CM_CR_AUC(y_test,y_predictedLR,"Logistic Regression")
#Draw Confusion Matrix and Classification Report for Decision Tree
y_predictedDT = dTreeR.predict(X_test)
Print_CM_CR_AUC(y_test,y_predictedDT,"Decision Tree")
