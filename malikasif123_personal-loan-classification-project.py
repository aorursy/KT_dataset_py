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

## Algo
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')



#loading Data
Data = pd.read_csv('../input/Bank_Personal_Loan_Modelling.csv')
#Fetching First 5 col
Data.head()
Data.columns
#checking data Type of each attributes
Data.dtypes
shape_data=Data.shape
print('Data set contains "{x}" number of rows and "{y}" number of columns columns'.format(x=shape_data[0],y=shape_data[1]))
#checking for Null Values
Data.isnull().sum()
sns.heatmap(Data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#overview of data
Data.describe().transpose()
#Value Counts of all Binary Catagorical Data
Data[['Personal Loan','Securities Account','CD Account','Online','CreditCard']].apply(pd.value_counts)
#lets visualize data apart from target variables via count plot
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
sns.countplot(x=Data['Securities Account'],ax=axes[0,0])
sns.countplot(x=Data['CD Account'],palette='Set1',ax=axes[0,1])
sns.countplot(x=Data['Online'],palette='Set2',ax=axes[1,0])
sns.countplot(x=Data['CreditCard'],palette='Set3',ax=axes[1,1])
fig.tight_layout()
sns.countplot(x=Data['Personal Loan'])
#Watching the graphs above we can say there is a significant diffrenece among binary catagorical data encluding 'Online' feature.
#infact out Traget Variable i.e. 'Personal Loan' is also having significant difference.
#now lets see how our Target variable is forming up with our other Catagorical binary data.

fig, axes = plt.subplots(2, 2, figsize=(12, 6))
sns.countplot(x="Securities Account",data=Data, hue="Personal Loan",ax=axes[0,0])
sns.countplot(x="CD Account",data=Data, hue="Personal Loan",palette='Set1',ax=axes[0,1])
sns.countplot(x="Online",data=Data, hue="Personal Loan",palette='Set2',ax=axes[1,0])
sns.countplot(x="CreditCard",data=Data, hue="Personal Loan",palette='Set3',ax=axes[1,1])
fig.tight_layout()
#distribution of Family and Education
fig, axes = plt.subplots(2, figsize=(12, 6))
sns.countplot(x="Family",data=Data,ax=axes[0])
sns.countplot(x="Education",data=Data,palette='Set1',ax=axes[1])
fig.tight_layout()
#how this distribution is spread out or segregated over Personal Loan
fig, axes = plt.subplots(2, figsize=(12, 6))
sns.countplot(x="Family",data=Data, hue="Personal Loan",ax=axes[0])
sns.countplot(x="Education",data=Data, hue="Personal Loan",palette='Set1',ax=axes[1])
fig.tight_layout()
#getting summary of numerical data, over-viewing data
Data[['Age','Experience','Income','CCAvg','Mortgage']].describe().transpose()
#A Skewness value of 0 in the output denotes a symmetrical distribution
#A negative Skewness value in the output denotes tail is larger towrds left hand side of data so we can say left skewed
#A Positive Skewness value in the output denotes tail is larger towrds Right hand side of data so we can say Right skewed
Data[['Age','Experience','Income','CCAvg','Mortgage']].skew()
#The distplot shows the distribution of a univariate set of observations.
fig, axes = plt.subplots(2, 2, figsize=(14, 4))
sns.distplot(Data['Age'],ax=axes[0,0])
sns.distplot(Data['Experience'],ax=axes[0,1])
sns.distplot(Data['Income'],ax=axes[1,0])
sns.distplot(Data['CCAvg'],ax=axes[1,1])
axes[0,0].set_title('Age Distribution')
axes[0,1].set_title('Experience Distribution')
axes[1,0].set_title('Income Distribution')
axes[1,1].set_title('CCAvg Distribution')
plt.tight_layout()
sns.distplot(Data['Mortgage'])
#lets try correlation metrix as well to get more insight of the pairing of all data
#and visualise it via heatmap
fig,ax= plt.subplots(figsize=(10, 10))
sns.heatmap(Data.corr())
fig.tight_layout()
#pair plot to check the distribution and correlation of all variables
sns.pairplot(Data)
#checking the spread of Target Column
sns.countplot(x=Data['Personal Loan'])
#from the above observations lets check the spread of Target coloumn with correlated features.
#1.spread over CCAvg 
sns.kdeplot(Data[Data['Personal Loan'] == 0]['CCAvg'], shade=False)
sns.kdeplot(Data[Data['Personal Loan'] == 1]['CCAvg'], shade=True)
plt.title("CCAvg spread over Personal Loan")
#2.Spread over Mortage
sns.kdeplot(Data[Data['Personal Loan'] == 0]['Mortgage'], shade=False)
sns.kdeplot(Data[Data['Personal Loan'] == 1]['Mortgage'], shade=True)
plt.title("CCAvg spread over Personal Loan")
#checking the variance of Zip Code
#high variance means fearure does not affect the target variable
Data[['ZIP Code','ID']].var()

#Creating Dummy Variables for multi-Catagorical Column
Family_dummies= pd.get_dummies(Data['Family'],drop_first=True,prefix='Family_Size')
Education_dummies= pd.get_dummies(Data['Education'],drop_first=True,prefix='Education_Level')
#concatinating the dummy variables to Data
Data = pd.concat([Data,Family_dummies,Education_dummies],axis=1)
Cleaned_Data=Data.drop(['ID','ZIP Code','Family','Education','Experience'],axis=1)
Cleaned_Data.head()
#Split the data into training and test set in the ratio of 70:30 respectively
X = Cleaned_Data.drop('Personal Loan',axis=1)
y = Cleaned_Data['Personal Loan']

# split data into train subset and test subset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# checking the dimensions of the train & test subset
# to print dimension of train set
print(X_train.shape)
# to print dimension of test set
print(X_test.shape)
# Train and Fit model
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)
#predict the Personal Loan Values
y_Logit_pred = model.predict(X_test)
y_Logit_pred
predicted_Logit_probas = model.predict_proba(X_test)
predicted_Logit_probas
# Let's measure the accuracy of this model's prediction
print("confusion_matrix")
print(confusion_matrix(y_test,y_Logit_pred))
# And some other metrics for Test
print(classification_report(y_test, y_Logit_pred, digits=2))
# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform', metric='euclidean')
# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_Knn_pred = knn.predict(X_test)

# evaluate Model Score
print(classification_report(y_test, y_Knn_pred, digits=2))
# creating odd list of K for KNN
myList = list(range(3,30,2))

# empty list that will hold accuracy scores
ac_scores = []

# perform accuracy metrics for values from 3,5....29
for k in myList:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # predict the response
    y_pred = knn.predict(X_test)
    # evaluate F1 Score
    scores = f1_score(y_test, y_pred)
    ac_scores.append(scores)

# changing to misclassification error
MSE = [1 - x for x in ac_scores]

# determining best k
optimal_k = myList[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)
#propability prediction
predicted_Logit_probas = knn.predict_proba(X_test)
predicted_Logit_probas
naive_model = GaussianNB()
naive_model.fit(X_train, y_train)

predicted_probas_NB = naive_model.predict_proba(X_test)

predicted_probas_NB
prediction_NB = naive_model.predict(X_test)
print(classification_report(y_test, prediction_NB, digits=2))
#confusion matric via Heat Map
conf_mat = confusion_matrix(y_test, y_Logit_pred)
conf_mat

df_conf_mat = pd.DataFrame(conf_mat)
plt.figure(figsize = (7,4))
sns.heatmap(df_conf_mat, annot=True,cmap='Blues', fmt='g')
#determining false positive rate and True positive rate, threshold
fpr, tpr, threshold = metrics.roc_curve(y_test, y_Logit_pred)
roc_auc_logit = metrics.auc(fpr, tpr)
# print AUC
print("AUC : % 1.4f" %(roc_auc_logit)) 
#plotting ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_logit)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#confusion matric via Heat Map
conf_mat = confusion_matrix(y_test, y_Knn_pred)
conf_mat

df_conf_mat = pd.DataFrame(conf_mat)
plt.figure(figsize = (7,4))
sns.heatmap(df_conf_mat, annot=True,cmap='Blues', fmt='g')
#determining false positive rate and True positive rate, threshold
fpr, tpr, threshold = metrics.roc_curve(y_test, y_Knn_pred)
roc_auc_knn = metrics.auc(fpr, tpr)
# print AUC
print("AUC : % 1.4f" %(roc_auc_knn)) 
#plotting ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_knn)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#confusion matric via Heat Map
conf_mat = confusion_matrix(y_test, prediction_NB)
conf_mat
df_conf_mat = pd.DataFrame(conf_mat)
plt.figure(figsize = (7,4))
sns.heatmap(df_conf_mat, annot=True,cmap='Blues', fmt='g')
#determining false positive rate and True positive rate, threshold
fpr, tpr, threshold = metrics.roc_curve(y_test, prediction_NB)
roc_auc_NB = metrics.auc(fpr, tpr)
# print AUC
print("AUC : % 1.4f" %(roc_auc_NB)) 
#plotting ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc_NB)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#Using K fold to check how my algorighm varies throughout my data if we split it in 10 equal bins
models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('K-NN', KNeighborsClassifier()))
models.append(('Naive Bayes', GaussianNB()))

# evaluate each model
results = []
names = []
scoring = 'f1'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=101)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	print("Name = %s , Mean F1-Score = %f, SD F1-Score = %f" % (name, cv_results.mean(), cv_results.std()))
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.plot(results[0],label='Logistic')
plt.plot(results[1],label='KNN')
plt.plot(results[2],label='Naive Bayes')
plt.legend()
plt.show()