# Import Necessary Libraries



#data analysis libraries 

import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# sns.set_palette("GnBu_d")

sns.set_style('whitegrid')



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import StandardScaler



# Importing Classifier Modules

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier



# Load the library for splitting the data

from sklearn.model_selection import train_test_split

from sklearn import metrics
# Read in and Explore the Data



# Import train and test CSV files

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.head()
# Data Analysis

# Observing summarized information of data like Data types, Missing values etc.

print("\nTrain set summary")

print(train.info())



print("\nTest set summary")

print(test.info())
print("\nTrain set summary")

train.describe()
print('Shape before deleting duplicate values:', train.shape)



# Removing duplicate rows if any

train=train.drop_duplicates()

print('Shape After deleting duplicate values:', train.shape)
# check for any missing values

print("\n Check for any null values in Train set:")

print(pd.isnull(train).sum())



print("\n Check for any null values in Test set:")

print(pd.isnull(test).sum())
# sns.pairplot(train)
# Observe the distribution of target variable

print(train['Survived'].value_counts())

sns.countplot(x="Survived", data=train)

plt.title("Distribution of Survived")

plt.show()
# Finging unique values for each column

# TO understand which column is categorical and which one is Continuous

# Typically if the numer of unique values are < 20 then the variable is likely to be a category 

# otherwise continuous

train.nunique()
# Plotting multiple bar charts for categorical variables

# Since there is no default function which can plot bar charts for multiple columns at once

# we are defining our own function for the same



def PlotBarCharts(inpData, colsToPlot):

    %matplotlib inline

    

    import matplotlib.pyplot as plt

    

    # Generating multiple subplots

    fig, subPlot=plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(18,5))

    fig.suptitle('Bar charts of: '+ str(colsToPlot))



    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):

        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])



# Calling the function

PlotBarCharts(inpData=train, colsToPlot=['Pclass', 'Sex','SibSp','Parch','Embarked'])
# Plotting histograms of multiple columns together

# Observe that Fare column has outliers

train.hist(['Age', 'Fare'], figsize=(16,4))
# Grouping by Categorical variable Survived to find the aggregated values

GroupedData=train.groupby(['Survived'])



# Printing the aggregated values

#GroupedData.size()

#GroupedData.sum()

#GroupedData.count()

GroupedData.mean()
# Creating the graph with Price

# You can observice that many people who survived has paid higher fares!

GroupedData.mean()['Fare'].plot(kind='bar', title='Average Fare for Each Survival Type')

plt.show()
# Box plot for Categorical Variable Survived Vs Continuous Variable Fare

# Observe the outlier in Fare for Survived=1

sns.boxplot(x='Survived', y='Fare', data=train)
sns.boxplot(y='Fare', x='Pclass', data=train)
# Create a function to return index of outliers 

def indicies_of_outliers(x): 

    q1, q3 = np.percentile(x, [25, 75]) 

    iqr = q3 - q1 

    lower_bound = q1 - (iqr * 1.5) 

    upper_bound = q3 + (iqr * 1.5)

    #print(upper_bound)

    

    return np.where((x > upper_bound) | (x < lower_bound)) 


# indicies_of_outliers(train[train['Pclass']==1]['Fare'])[0]



len(indicies_of_outliers(train[train['Pclass']==1]['Fare'])[0])
# Finding those rows where Fare column has outliers

# All the outlier fares are coming from Pclass=1 which makes sense!

train[train['Fare']>187].head()
# checking the balance of outliers in each category

train[train.index.isin(list(indicies_of_outliers(train[train['Pclass']==1]['Fare'])[0]))]['Survived'].value_counts()
# Replacing the outlier records of Fare with value 187

# train['Fare'][train['Fare']>187] = 187
# train.drop(list(indicies_of_outliers(train[train['Pclass']==1]['Fare'])))



train.drop(train.index[indicies_of_outliers(train[train['Pclass']==1]['Fare'])[0]], inplace=True)

train.reset_index(inplace = True , drop = True)

train.shape
# Observing the relationship with Target variable again after Outlier treatment

# You can see that the distribution has improved now

sns.boxplot(y='Fare', x='Pclass', data=train)

plt.show()
# f_oneway() function takes the group data as input and returns F-statistic and P-value

from scipy.stats import f_oneway



# Running the one-way anova test between Fare and Survived

# Assumption(H0) is that Fare and Survived are NOT correlated with each other

Survived_0 = train['Fare'][train['Survived']==0]

Survived_1 = train['Fare'][train['Survived']==1]



# Performing the ANOVA test

AnovaResults = f_oneway(Survived_0, Survived_1)



print('P-Value for Anova is: ', AnovaResults[1])



# We accept the Assumption(H0) only when P-Value > 0.05

# Here the P-Value is almost Zero which means we will REJECT the Assumption(H0)

# This means Fare and Survived ARE correlated with each other
# Cross tablulation between two categorical variables

CrossTabResult = pd.crosstab(index=train['Sex'], columns=train['Survived'])

CrossTabResult
# Visual Inference using Grouped Bar chart

# Notice that Male Suvival rate is very low as compared to Female

sns.countplot( x='Sex', hue="Survived", data=train)

plt.show()
from scipy.stats import chi2_contingency



# Performing Chi-sq test

ChiSqResult = chi2_contingency(CrossTabResult)



# P-Value is the Probability of H0 being True

# If P-Value>0.05 then only we Accept the assumption(H0)

# In this case it is way way lower than 0.05 Hence, we reject H0

# this means the two columns are correlated with each other and Gender of a person affects the Survival

print('The P-Value of the ChiSq Test is:', ChiSqResult[1])
cont_list=["SibSp", "Parch", "Age", "Fare", "Survived"]



sns.heatmap(train[cont_list].corr(), annot=True, fmt =".2f")

plt.show()
# concatenate train and test set

train['TrainTest'] = 1

test['TrainTest'] = 0

data = pd.concat([train,test])

data.reset_index(inplace = True , drop = True)
# Drop useless columns

# Remove those variables from data which have too many missing values (Missing Values > 30%)

# Remove Qualitative variables which cannot be used in Machine Learning

data.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
#check for any missing values

print("\n Check for any null values in combined set")

print(pd.isnull(data).sum())
# Missing values imputation

data['Age'].fillna(data['Age'].median(skipna=True), inplace = True)

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

data['Fare'].fillna(data['Fare'].median(skipna=True), inplace = True)
## Create categorical variable for traveling alone

data['TravelBuds']=data["SibSp"]+data["Parch"]

data['TravelAlone']=np.where(data['TravelBuds']>0, 0, 1)



data.drop('SibSp', axis=1, inplace=True)

data.drop('Parch', axis=1, inplace=True)

data.drop('TravelBuds', axis=1, inplace=True)
# Converting Categorical Features

# Treating all the nominal variables at once using dummy variables

# data_Numeric=pd.get_dummies(data,drop_first=True)

# data_Numeric = pd.get_dummies(data)

# data_Numeric = pd.get_dummies(data, columns=["Pclass"])

data_Numeric = pd.get_dummies(data, columns=["Sex","Embarked","Pclass"],drop_first=True)

data_Numeric.shape
data_Numeric.head()
# Rearrange columns

# data_Numeric.columns.tolist()

data_Numeric = data_Numeric[['PassengerId','Age','Fare','TravelAlone','Sex_male','Embarked_Q',

 'Embarked_S','Pclass_2','Pclass_3','Survived','TrainTest']]
# Standardize the Variables

scaler = StandardScaler()



data_Numeric.iloc[:,1:-2] = scaler.fit_transform(data_Numeric.iloc[:,1:-2])

data_Numeric.head()
# After data preprocess, again split the data into train and test set

train = data_Numeric[data_Numeric['TrainTest'] == 1]

test = data_Numeric[data_Numeric['TrainTest'] == 0]



train = train.drop(['PassengerId','TrainTest'], axis=1)

test = test.drop(['Survived','TrainTest'], axis=1)



print(train.shape, "  ", test.shape)
X = train.drop('Survived', axis=1)

y = train['Survived']
# Split the data into training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)



# Quick sanity check with the shapes of Training and testing datasets

print("X_train - ",X_train.shape)

print("y_train - ",y_train.shape)

print("X_test - ",X_test.shape)

print("y_test - ",y_test.shape)
model_lst = []

accuracy_lst = []
# Logistic Regression

model_lst.append('Logistic Regression')



lr = LogisticRegression(C=1,penalty='l2', solver='liblinear')

lr.fit(X_train, y_train)

y_pred_log_reg = lr.predict(X_test)



# calculate accuracy

acc_log_reg = round( metrics.accuracy_score(y_pred_log_reg , y_test) * 100, 2)

accuracy_lst.append(acc_log_reg)



# calculate auc

Aucs = metrics.roc_auc_score(y_test , y_pred_log_reg)



# calculate precision

PrecisionScore = metrics.precision_score(y_test , y_pred_log_reg)



# calculate recall

RecallScore = metrics.recall_score(y_test , y_pred_log_reg)



# calculate f1 score

F1Score = metrics.f1_score(y_test , y_pred_log_reg)



# draw confusion matrix

cnf_matrix = metrics.confusion_matrix(y_test , y_pred_log_reg)



print("Model Name : Logistic Regression")

print('Accuracy :{0:0.2f} %'.format(acc_log_reg)) 

print('AUC : {0:0.2f}'.format(Aucs))

print('Precision : {0:0.2f}'.format(PrecisionScore))

print('Recall : {0:0.2f}'.format(RecallScore))

print('F1 : {0:0.2f}'.format(F1Score))

print('Confusion Matrix : \n', cnf_matrix)

print("\n")
# Support Vector Classifier

model_lst.append('Support Vector Classifier')



svc = SVC()

svc.fit(X_train, y_train)

y_pred_svc = svc.predict(X_test)



# calculate accuracy

acc_svc = round( metrics.accuracy_score(y_pred_svc , y_test) * 100, 2)

accuracy_lst.append(acc_svc)



print("Model Name : SVC")

print('Accuracy :{0:0.2f} %'.format(acc_svc)) 
# Linear Support Vector Classifier

model_lst.append('Linear Support Vector Classifier')



lsvc = LinearSVC()

lsvc.fit(X_train, y_train)

y_pred_linear_svc = lsvc.predict(X_test)



# calculate accuracy

acc_linear_svc = round( metrics.accuracy_score(y_pred_linear_svc , y_test) * 100, 2)

accuracy_lst.append(acc_linear_svc)



print("Model Name : Linear SVC")

print('Accuracy :{0:0.2f} %'.format(acc_linear_svc)) 
# K-Nearest Neighbors

model_lst.append('K-Nearest Neighbors')

                 

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)



# calculate accuracy

acc_knn = round( metrics.accuracy_score(y_pred_knn , y_test) * 100, 2)

accuracy_lst.append(acc_knn)



print("Model Name : K Neighbors Classifier")

print('Accuracy :{0:0.2f} %'.format(acc_knn)) 
# Decision Tree

model_lst.append('Decision Tree')

                 

dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred_decision_tree = dt.predict(X_test)



# calculate accuracy

acc_decision_tree = round( metrics.accuracy_score(y_pred_decision_tree , y_test) * 100, 2)

accuracy_lst.append(acc_decision_tree)



print("Model Name : Decision Tree Classifier")

print('Accuracy :{0:0.2f} %'.format(acc_decision_tree)) 
# Random Forest

model_lst.append('Random Forest')



rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)

y_pred_random_forest = rf.predict(X_test)



# calculate accuracy

acc_random_forest = round( metrics.accuracy_score(y_pred_random_forest , y_test) * 100, 2)

accuracy_lst.append(acc_random_forest)



print("Model Name : Random Forest Classifier")

print('Accuracy :{0:0.2f} %'.format(acc_random_forest)) 
# Gaussian Naive Bayes

model_lst.append('Gaussian Naive Bayes')

                 

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred_gnb = gnb.predict(X_test)



# calculate accuracy

acc_gnb = round( metrics.accuracy_score(y_pred_gnb , y_test) * 100, 2)

accuracy_lst.append(acc_gnb)



print("Model Name : Gaussian NB")

print('Accuracy :{0:0.2f} %'.format(acc_gnb)) 
# Perceptron

model_lst.append('Perceptron')

                 

prct = Perceptron(max_iter=5, tol=None)

prct.fit(X_train, y_train)

y_pred_perceptron = prct.predict(X_test)



# calculate accuracy

acc_perceptron = round( metrics.accuracy_score(y_pred_perceptron , y_test) * 100, 2)

accuracy_lst.append(acc_perceptron)



print("Model Name : Perceptron")

print('Accuracy :{0:0.2f} %'.format(acc_perceptron)) 
# Stochastic Gradient Descent

model_lst.append('Stochastic Gradient Descent')



sgd = SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, y_train)

y_pred_sgd = sgd.predict(X_test)



# calculate accuracy

acc_sgd = round( metrics.accuracy_score(y_pred_sgd , y_test) * 100, 2)

accuracy_lst.append(acc_sgd)



print("Model Name : SGD Classifier")

print('Accuracy :{0:0.2f} %'.format(acc_sgd)) 
# Gradient Boosting Classifier

model_lst.append('Gradient Boosting Classifier')



gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

y_pred_gbk = gbk.predict(X_test)





# calculate accuracy

acc_gbk = round( metrics.accuracy_score(y_pred_gbk , y_test) * 100, 2)

accuracy_lst.append(acc_gbk)



print("Model Name : GB Classifier")

print('Accuracy :{0:0.2f} %'.format(acc_gbk)) 
# Xtreme Gradient Boosting (XGBoost)

model_lst.append('Xtreme Gradient Boosting')



xgb=XGBClassifier(max_depth=2,learning_rate=0.01,n_estimators=400,objective='binary:logistic',booster='gbtree')

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)



# calculate accuracy

acc_xgb = round( metrics.accuracy_score(y_pred_xgb , y_test) * 100, 2)

accuracy_lst.append(acc_xgb)



print("Model Name : XGBoost Classifier")

print('Accuracy :{0:0.2f} %'.format(acc_xgb)) 
# Performance measures of various classifiers

data = {'Models':model_lst,

       'Accuracy':accuracy_lst}



print("Performance measures of various classifiers: \n")

models = pd.DataFrame(data) 

models.sort_values(['Accuracy'],ascending=False)
sns.barplot(y='Models', x='Accuracy', data=models.sort_values(['Accuracy'],ascending=False))

plt.show()
# Creating Submission File



#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1)).astype(int)





#set the output as a dataframe and convert to csv file named submission.csv

submission = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

submission.head()
submission.to_csv('submission.csv', header=True, index=False)