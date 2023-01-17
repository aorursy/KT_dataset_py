# Importing important libraries
import pandas as pd, numpy as np
import seaborn as sns

import os

import matplotlib.pyplot as plt
%matplotlib inline
# Current working directory
os.getcwd()
# The current working directory name
print(os.listdir("../input"))
#Load the data
health =  pd.read_csv("../input/pima-indians-diabetes/health care diabetes.csv")
#Check the nature of the datasets
print(health.shape)
#View the the first 10 instances
health.head(10)
#Column wise null values check
health.isna().sum()
#For the entire DataFrame null values check
health.isnull().any().any()
#The mean of the variable 'Insulin'
print(health['Insulin'].mean(), health['Insulin'].median())
#### Describe the data to get the various statistics excluding the 'missing values' for the entire DataFrame 
health.describe()
health.info()
#Check the value counts for each of the indexes
for col in health.columns:
    print('The value counts in '+col+' are:', health[col].value_counts())
# We further do a histogram plot (value_counts) for Glucose column to see if contain any 0 values.

plt.figure(figsize=(18,8))
health['Glucose'].value_counts().plot.bar(title='Frequency Distribution of Glucose')
#The distribution column of the Glucose shows a positively skewed distribution. A large number of values occurs on the left
#with the fewer number of data values on the right side.
plt.figure(figsize=(18,8))
health['Glucose'].value_counts().plot.hist(title='Frequency Distribution of Glucose', color='g')
# Data type conversions
health['BMI'] = health['BMI'].astype('int64')
health['DiabetesPedigreeFunction'] = health['DiabetesPedigreeFunction'].astype('int64')
# Show new data types
health.dtypes
health.head()
col_with_0 = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']   
# Exploring the variables using histogram
health.hist(column=col_with_0, rwidth=0.95, figsize=(15,8))
# We set an appropriate range to observe the 0 value counts
health.hist(column=col_with_0, bins=[0,10,15], rwidth=0.95, figsize=(15,8))
# Replace the all the zero values with NaN
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
health[cols] = health[cols].replace(0, np.nan)
health
health.fillna(health.median())
health = health.fillna(health.median())
health.dtypes
health['Pregnancies']=health['Pregnancies'].astype('int64')
health.hist(column=cols, rwidth=0.95, figsize=(15,8))
# Plotting the Histogram barplot to compare the frequency distribution of each indexes.
# We also create a range to compare and analyse different distribution.

bins=[40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]

plt.hist(health['Glucose'], rwidth=0.95, bins=bins, color='g')
plt.xlabel('Glucose range')
plt.ylabel('Total no of patients')
plt.title('Glucose Analysis')
bin_BP=[10,20,30,40,50,60,70,80,100,110,120,130]

plt.hist(health['BloodPressure'], bins=bin_BP, rwidth=0.9, color='orange', orientation='horizontal')
plt.title('Blood Presure Analysis')
plt.xlabel('No of patients')

plt.ylabel('BloodPressure range')
# Check the new data type
health.dtypes
sns.countplot(health.dtypes.map(str))
plt.show()
from matplotlib.pyplot import figure
figure(figsize=(16,8))
sns.countplot(data=health, x='Age', hue='Outcome')
fig, ax =plt.subplots(1,2)
figure(figsize=(12,6))
sns.countplot(health['Age'],  ax=ax[0])
sns.countplot(health['DiabetesPedigreeFunction'], ax=ax[1])
fig.show()
health=health.astype('int64')
health.dtypes
sns.countplot(health['Glucose'])
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(health, ax.flatten()):
    sns.countplot(health[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
health['Outcome'].value_counts()
sns.countplot(data=health, x='Outcome', palette='hls')
plt.show()
plt.savefig('count_plot')
sns.scatterplot(y=health['Age'], x=health['Outcome']);
# The scatter charts between the pair of variables to understand the relationship.
sns.pairplot(health)
from pandas.plotting import scatter_matrix
scatter_matrix(health, alpha=0.2, figsize=(16, 10), diagonal='kde')
#Important libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = health.iloc[:,0:8]
y = health.iloc[:,-1]
#Apply SeleckKBest class to extarct top 6 best features
bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#Concat two DataFrames for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['col','Score']
featureScores
# Print the 6 best features
print(featureScores.nlargest(6,'Score'))
#The correlation of each feauture in the dataset
corrmat = health.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(16,8))
#We plot the heatmap
sns.heatmap(health[top_corr_features].corr(), annot=True, cmap='RdYlGn')
# We can drop this two features since they are not correlated with target variable(Outcome)
health.drop(['BloodPressure','DiabetesPedigreeFunction'], axis=1, inplace=True)
health.head()
#split the dataset in features and target varaible
X = health.drop("Outcome", axis=1)
y = health["Outcome"]
# Split X and y into training and testing set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# import the class
from sklearn.linear_model import LogisticRegression
#instantiate the model
logreg = LogisticRegression()

#fit the model with train data
logreg.fit(X_train, y_train)

#fit the model with predictor test data(X_test)
y_pred = logreg.predict(X_test)
# import the metrics
from sklearn import metrics
cf_matrix = metrics.confusion_matrix(y_test, y_pred)
cf_matrix
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
# Import the libraries (pipeline and models)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
pipeline_dt=Pipeline([('dt_classifier',DecisionTreeClassifier(random_state=0))])
pipeline_rf=Pipeline([('rf_classifier',RandomForestClassifier())])
pipeline_knn=Pipeline([('kn_classifier',KNeighborsClassifier())])
pipeline_lr=Pipeline([('lr_classifier',LogisticRegression())])
#Make the list of pipelines
pipelines = [pipeline_dt,pipeline_rf,pipeline_knn,pipeline_lr]
best_accuracy=0.0
best_classifier=0
best_pipeline=""
#Dictionery of pipelines and classifier type for ease of reference
pipe_dict = {0: 'Decision Tree', 1: 'RandomForest', 2: 'KNeighbors', 3:'Logistic Regression'}

#Fit the pipilines
for pipe in pipelines:
    pipe.fit(X_train, y_train)
for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))
for i,model in enumerate(pipelines):
    if model.score(X_test,y_test)>best_accuracy:
        best_accuracy=model.score(X_test,y_test)
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))         
y_pred_0 = pipeline_dt.predict(X_test)
dt_cnf_matrix=metrics.confusion_matrix(y_test,y_pred_0)
dt_cnf_matrix
y_pred_1 = pipeline_rf.predict(X_test)
rf_cnf_matrix=metrics.confusion_matrix(y_test,y_pred_1)
rf_cnf_matrix
y_pred_3 = pipeline_knn.predict(X_test)
knn_cnf_matrix=metrics.confusion_matrix(y_test,y_pred_3)
knn_cnf_matrix
y_pred_4=pipeline_lr.predict(X_test)
lr_cnf_matrix=metrics.confusion_matrix(y_test,y_pred_4)
lr_cnf_matrix
# For KNeighbors
sensitivity, specificity= [113/(113+28), 34/(34+17)]
print("sensitivity:", sensitivity)
print("specificity:", specificity)
# For Logistic Regression
sensitivity, specificity= [117/(117+29), 33/(33+13)]
print("sensitivity:", sensitivity)
print("specificity:", specificity)
# Import the roc libraries and use roc_curve() to get the threshold, TPR, and FPR
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, pipeline_rf.predict_proba(X_test)[:,1])
fpr
tpr
thresholds
# For AUC we use roc_auc_score() function for ROC
rf_roc_auc1 = roc_auc_score(y_test, pipeline_rf.predict(X_test))
#Plot the ROC Curve
plt.figure()
plt.plot(fpr, tpr, label='Random Forest(Sensitivity = %0.3f)' % rf_roc_auc1)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# The predicted probabilities of class 1(diabetes patients)
y_pred_11 = pipeline_rf.predict_proba(X_test)[:,1]
y_pred_11 = y_pred_11.reshape(1,-1)
y_pred_11
# Set the threshold at 0.35
from sklearn.preprocessing import binarize
y_pred_11 = binarize(y_pred_11,0.6)[0]
y_pred_11
#Converting the array from float data type to integer data type
y_pred_11 = y_pred_11.astype(int)
y_pred_11
rf_cnf_matrix1 = metrics.confusion_matrix(y_test, y_pred_11)
rf_cnf_matrix1
rf_cnf_matrix
# Other performance matrics
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_11))
print(classification_report(y_test, y_pred_1))
rf_roc_auc2 = roc_auc_score(y_test, y_pred_11)
plt.figure()
plt.plot(fpr, tpr, label='Threshold=.5=>Sensitivity = %0.3f,\nThreshold=.6=>Sensitivity= %0.3f' % (rf_roc_auc1,rf_roc_auc2))
plt.plot([0,1], [0,1], 'r--')
plt.plot([0,1], [0,1], 'b--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="best")
plt.savefig('Log_ROC')         
plt.show()
# We use roc_curve() to get the threshold, TPR and FPR
fpr, tpr, thresholds = roc_curve(y_test, pipeline_lr.predict_proba(X_test)[:,1])
fpr
tpr
thresholds
# For AUC we use roc_auc_score() function for ROC
lr_roc_auc3 = roc_auc_score(y_test, pipeline_lr.predict(X_test))
#Plot the ROC Curve
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression(Sensitivity = %0.3f)' % lr_roc_auc3)
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()