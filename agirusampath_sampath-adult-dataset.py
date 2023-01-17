#importing Packages
import numpy as np # for arrays
import pandas as pd # for dataframes (rows,columns)
import matplotlib.pyplot as plt # for plotting
from sklearn.preprocessing import LabelEncoder #to convert categorical value in to numerical values
from sklearn.model_selection import train_test_split # to split data into train and test
from sklearn.metrics import accuracy_score # to find  accuracy on test data
from sklearn import metrics,tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report 
# importing models
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline
# importing the data
adult = pd.read_csv("../input/adult-dataset/adult.csv")
adult
adult.shape
adult.columns
adult.dtypes
adult.describe() # it gives statistics for numerical variables
for i,j in zip(adult.columns,(adult.values.astype(str) == '?').sum(axis = 0)):
    if j > 0:
        print(str(i) + ': ' + str(j) + ' records')
adult.head()#  display first 5 samples
adult.info()
adult.isnull().sum() # shows no of null values
display(adult.corr())
adult["income"].value_counts()
# Data Exploration and Visualization
#Separate categorical and numberical columns
cat_col = adult.dtypes[adult.dtypes == 'object']
num_col = adult.dtypes[adult.dtypes != 'object']
cat_col
num_col
for col in list(cat_col.index):
    print(f"--------------------{col.title()}-------------------------")
    total= adult[col].value_counts()
    percent = adult[col].value_counts() / adult.shape[0]
    df = pd.concat([total,percent],keys = ['total','percent'],axis = 1)
    print(df)
    print('\n')
#adult["workclass"].value_counts()
#adult["education"].value_counts()
#adult["marital-status"].value_counts()
#adult["occupation"].value_counts()
#adult["relationship"].value_counts()
#adult["race"].value_counts()
#adult["gender"].value_counts()
adult["native-country"].value_counts()
#adult["income"].value_counts()

edit_cols = ['native-country','occupation','workclass']
# Replace ? with Unknown
for col in edit_cols:
    adult.loc[adult[col] == '?', col] = 'unknown'
# Check if ? is present
for col in edit_cols:
    print(f"? in {col}: {adult[(adult[col] == '?')].any().sum()}")
import matplotlib.pyplot as plt
import seaborn as sns
fig= plt.figure(figsize=(12,6))
sns.heatmap(adult[list(num_col.index)].corr(),annot=True,square=True)
plt.show()
plt.figure(figsize =(12,6));
sns.countplot(x = 'income', data = adult);
plt.xlabel("Income",fontsize = 12);
plt.ylabel("Frequency",fontsize = 12);
adult[list(num_col.index)].hist(figsize = (12,12))
adult['workclass'].value_counts().plot(kind='bar')# univariate analysis

adult['gender'].value_counts().plot(kind='bar')# univariate analysis

adult['gender'].value_counts().plot(kind='bar')
# Education VS Income
# Creating a dictionary that contain the education and it's corresponding education level
edu_level = {}
for x,y in adult[['educational-num','education']].drop_duplicates().itertuples(index=False):
    edu_level[y] = x
#bivariate analysis
education = round(pd.crosstab(adult.education, adult.income).div(pd.crosstab(adult.education, adult.income).apply(sum,1),0),2)
education = education.reindex(sorted(edu_level, key=edu_level.get, reverse=False))

ax = education.plot(kind ='bar', title = 'Proportion distribution across education levels', figsize = (10,8))
ax.set_xlabel('Education level')
ax.set_ylabel('Proportion of population')

# Gender VS Income
# Creating a dictionary that contain the education and it's corresponding education level
edu_level = {}
for x,y in adult[['educational-num','education']].drop_duplicates().itertuples(index=False):
    edu_level[y] = x
education = round(pd.crosstab(adult.education, adult.income).div(pd.crosstab(adult.education, adult.income).apply(sum,1),0),2)
education = education.reindex(sorted(edu_level, key=edu_level.get, reverse=False))

ax = education.plot(kind ='bar', title = 'Proportion distribution across education levels', figsize = (10,8))
ax.set_xlabel('Education level')
ax.set_ylabel('Proportion of population')
#multivariate analysis
gender_workclass = round(pd.crosstab(adult.workclass, [adult.income, adult.gender]).div(pd.crosstab(adult.workclass, [adult.income, adult.gender]).apply(sum,1),0),2)
gender_workclass[[('>50K','Male'), ('>50K','Female')]].plot(kind = 'bar', title = 'Proportion distribution across gender for each workclass', figsize = (10,8), rot = 30)
ax.set_xlabel('Gender level')
ax.set_ylabel('Proportion of population')
fig = plt.figure(figsize = (12,6))

sns.heatmap(adult[list(num_col.index)].corr(),annot = True,square = True)
plt.show()
# To convert categorical data in to numerical  data
from sklearn.preprocessing import LabelEncoder 
number = LabelEncoder()
adult["workclass"]=number.fit_transform(adult["workclass"])
adult["education"]=number.fit_transform(adult["education"])
adult["marital-status"]=number.fit_transform(adult["marital-status"])
adult["occupation"]=number.fit_transform(adult["occupation"])
adult["relationship"]=number.fit_transform(adult["relationship"])
adult["race"]=number.fit_transform(adult["race"])
adult["gender"]=number.fit_transform(adult["gender"])
adult["native-country"]=number.fit_transform(adult["native-country"])
adult["income"]=number.fit_transform(adult["income"])
adult
#feature engineering
del_cols = ['relationship','fnlwgt','educational-num']
adult.drop(labels = del_cols,axis = 1,inplace = True)
print(f"Number of columns after deleting: {adult.shape[1]}")
adult.head()
hrs_per_week = adult[adult['hours-per-week'] == 99]
print("Number of people working for 99 hours per week:", hrs_per_week.shape[0])
# feature selection
print("Number of observation before removing:",adult.shape)
index_age = adult[adult['age'] == 90].index
adult.drop(labels = index_age,axis = 0,inplace =True)
print("Number of observation after removing:",adult.shape)
X=adult.iloc[:,:-1] #independent variables
X
y=adult.iloc[:,-1] # target variable
y
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# split data randomly into 70% training and 30% test
X_train,x_test, y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)

#Instantiate the classifiers
clf_logreg = LogisticRegression()
clf_tree = DecisionTreeClassifier()
clf_knn =  KNeighborsClassifier()
clf_svc = SVC()
clf_forest = RandomForestClassifier()
classifiers = ['LogisticRegression', 'DecisionTree', 'KNN', 'SVC', 'RandomForest']
models = {clf_logreg:'LogisticRegression',
          clf_tree:'DecisionTree',
          clf_knn: 'KNN',
          clf_svc: 'SVC',
          clf_forest: 'RandomForest'}
          
# train function fits the model and returns accuracy score
def train(algo,name,X_train,y_train,X_test,y_test):
    algo.fit(X_train,y_train)
    y_pred = algo.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    print(f"--------------------------------------------{name}---------------------------------------------------")
    print(f"Accuracy Score for {name}: {score*100:.4f}%")
    return y_test,y_pred,score

# acc_res function calculates confusion matrix
def acc_res(y_test,y_pred):
    null_accuracy = y_test.value_counts()[0]/len(y_test)
    print(f"Null Accuracy: {null_accuracy*100:.4f}%")
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test,y_pred)
    print(matrix)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    TN = matrix[0,0]
    FP = matrix[0,1]
    FN = matrix[1,0]
    TP = matrix[1,1]
    accuracy_score=(TN+TP) / float(TP+TN+FP+FN)
    recall_score = (TP)/ float(TP+FN)
    specificity = TN / float(TN+FP)
    FPR = FP / float(FP+TN)
    precision_score = TP / float(TP+FP)
    print(f"Accuracy Score: {accuracy_score*100:.4f}%")
    print(f"Recall Score: {recall_score*100:.4f}%")
    print(f"Specificity Score: {specificity*100:.4f}%")
    print(f"False Positive Rate: {FPR*100:.4f}%")
    print(f"Precision Score: {precision_score*100:.4f}%")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Classification Report")
    print(classification_report(y_test,y_pred))
    
def main(models):
    accuracy_scores = []
    for algo,name in models.items():
        y_test_train,y_pred,acc_score = train(algo,name,X_train,y_train,x_test,y_test)
        acc_res(y_test_train,y_pred)
        accuracy_scores.append(acc_score)
    return accuracy_scores
    
accuracy_scores = main(models)    

pd.DataFrame(accuracy_scores,columns = ['Accuracy Scores'],index = classifiers).sort_values(by = 'Accuracy Scores',ascending = False)