#Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # import seaborn
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import warnings
warnings.filterwarnings('ignore')
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
titanic_train.head(5)
titanic_train.info()
titanic_test.info()
titanic_train.describe()
titanic_train.describe(include=['O'])
print("Age broken down by P-class")
titanic_train.groupby('Pclass').mean()[['Age']]
titanic_train.loc[titanic_train.Age.isnull(), 'Age'] = titanic_train.groupby('Pclass')['Age'].transform('mean')
titanic_test.loc[titanic_test.Age.isnull(), 'Age'] = titanic_test.groupby('Pclass')['Age'].transform('mean')
titanic_train.iloc[[5, 17]]
titanic_train = titanic_train.drop('Cabin', axis=1)
titanic_test = titanic_test.drop('Cabin', axis=1)
titanic_train['Embarked'].fillna(titanic_train['Embarked'].mode()[0], inplace=True)
titanic_test['Fare'].fillna(titanic_test['Fare'].median(), inplace = True)
print('Training Data Null Values')
print(titanic_train.isnull().sum())
print("-" * 30)
print('Test Data Null Values')
print(titanic_test.isnull().sum())
titanic_train.head()
sns.countplot(x='Survived', data=titanic_train)
sns.boxplot(x = 'Survived', y = 'Fare', data = titanic_train)
titanic_train.groupby('Survived').mean()[['Fare']]
titanic_train.loc[titanic_train['Fare'] > 500, :]
titanic_no_500s = titanic_train.loc[titanic_train['Fare'] < 500, :]
sns.boxplot(x = 'Survived', y = 'Fare', data = titanic_no_500s, palette = 'RdBu_r')
titanic_no_500s.groupby('Survived').mean()[['Fare']]
sns.countplot(x = 'Sex', data = titanic_train, hue = 'Survived')
hist = sns.distplot(titanic_train['Age'], color='b', bins=30, kde=False)
hist.set(xlim=(0, 100), title = "Distribution of Passenger Age's")
titanic_train.Age.describe()
age_box = sns.boxplot(y = 'Age', x = 'Survived',data = titanic_train, palette='coolwarm')
age_box.set(title='Boxplot of Age')
titanic_train.groupby(['Embarked']).count()
sns.countplot(x = 'Embarked', hue = 'Survived', data=titanic_train)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass', data=titanic_train, palette = 'rainbow')
#Make copies of both dataframes.
traindf = titanic_train.copy()
testdf = titanic_test.copy()
#Create list of both data frames to apply similar functions to.
all_data = [traindf, testdf]

#Drop name and ticket columns
for dat in all_data:
    dat.drop(['Name', 'Ticket'], axis=1, inplace=True)
traindf.describe()['Fare']
#Perform operation on both frames
for dat in all_data:
    
    #Create bins to separate fares
    bins = (0, 8, 15, 31, 515)

    #Assign group names to bins
    group_names = ['Fare_Group_1', 'Fare_Group_2', 'Fare_Group_3', 'Fare_Group_4']

    #Bin the Fare column based on bins
    categories = pd.cut(dat.Fare, bins, labels=group_names)
    
    #Assign bins to column
    dat['Fare'] = categories

traindf.describe()['Age']
#Perform operation on both frames
for dat in all_data:
    
    #Create bins to separate fares
    bins = (0, 15, 30, 45, 60, 75, 90)

    #Assign group names to bins
    group_names = ['Child', 'Young Adult', 'Adult', 'Experienced', 'Senior', 'Elderly']

    #Bin the Fare column based on bins
    categories = pd.cut(dat.Age, bins, labels=group_names)
    
    #Assign bins to column
    dat['Age'] = categories
traindf.head()
for dat in all_data:
    dat['Fam_Size'] = dat['SibSp'] + dat['Parch']
traindf = pd.get_dummies(traindf)
traindf.head()
testdf = pd.get_dummies(testdf)
testdf.head()
#Import libraries
from sklearn.metrics import confusion_matrix #confusion matrix
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier
from sklearn.svm import SVC #Support Vector Machine
from sklearn.preprocessing import StandardScaler #For scaling data
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.model_selection import train_test_split #Split data into training and validation sets.
from sklearn.metrics import accuracy_score  #Accuracy Score
#Split data into training and validation set
X = traindf.drop(columns=['PassengerId', 'Survived'], axis=1)
y = traindf['Survived']

#Note they are labeled as test sets but I'm treating them as validation data sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
results = pd.DataFrame(columns=['Validation'], index=['Logistic Regression', 'Support Vector Machine', 'KNN', 'Random Forest'])
def log_reg(X_train, X_test, y_train, y_test):
    #Create logmodel object
    logmodel = LogisticRegression(C=.01)

    #fit logistic regression model
    logmodel.fit(X_train, y_train)

    #Make predictions on validation data
    predictions = logmodel.predict(X_test)
    
    #Print Statistics
    print(accuracy_score(y_test, predictions))
    
    #Return predictions
    return accuracy_score(y_test, predictions)
#Get prediction accuracy for model.
LR_preds = log_reg(X_train, X_test, y_train, y_test)

#Add to dataframe.
results.loc['Logistic Regression', 'Validation'] = LR_preds
def svm(X_train, X_test, y_train, y_test):
    
    #Scale data
    #scaler = StandardScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    
    #Create list of c values to try
    c_vals = list(range(1, 100))
    
    #Accuracy list
    accuracy = [0 for i in range(99)]
    
    #Loop through c_values
    for i, c in enumerate(c_vals):
        #Create support vector machine object
        svc_model = SVC(C=c)
        
        #fit support vector machine model
        svc_model.fit(X_train, y_train)
        
        #Make predictions
        predictions = svc_model.predict(X_test)
        
        #add accuracy score to accuracy list
        accuracy[i] = accuracy_score(y_test, predictions)
    
    print("Best C Value:", c_vals[accuracy.index(max(accuracy))])
    print(accuracy)
    print("Prediction Accuracy: ", max(accuracy))
    
    return max(accuracy)
        
        
#Get support vector machine results
svm_preds = svm(X_train, X_test, y_train, y_test)

#Add to dataframe.
results.loc['Support Vector Machine', 'Validation'] = svm_preds
results.head()

def knn(X_train, X_test, y_train, y_test):
    
    #Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    #Create list of c values to try
    ks = [i + 1 for i in range(20)]
    
    #Accuracy list
    accuracy = [0 for i in range(20)]
    
    #Loop through c_values
    for i, k in enumerate(ks):
        #Create support vector machine object
        knn = KNeighborsClassifier(n_neighbors = k)
        
        #fit support vector machine model
        knn.fit(X_train, y_train)
        
        #Make predictions
        predictions = knn.predict(X_test)
        
        #add accuracy score to accuracy list
        accuracy[i] = accuracy_score(y_test, predictions)
    
    print(ks)
    print(accuracy)
    print("Best k Value:", ks[accuracy.index(max(accuracy))])
    
    print("Prediction Accuracy: ", max(accuracy))
    
    return max(accuracy)
knn_preds = knn(X_train, X_test, y_train, y_test)
results.loc['KNN', 'Validation'] = knn_preds
results.head()
from sklearn.decomposition import PCA
scaler = StandardScaler()
scaler.fit(X)
test_feats = testdf.drop('PassengerId', axis=1)
X = scaler.transform(X)
test_feats = scaler.transform(test_feats)
pca = PCA(n_components = 4)
pca.fit(X)
x_train_pca = pca.transform(X)
x_test_pca = pca.transform(test_feats)
svc_model = SVC(C = 1)
svc_model.fit(x_train_pca, y)
svm_predictions = svc_model.predict(x_test_pca)
output = pd.DataFrame({ 'PassengerId' : testdf['PassengerId'], 'Survived': svm_predictions })
output.to_csv('titanic-predictions-svm-pca.csv', index=False)
output

#svc_model = SVC(C = 1)
#svc_model.fit(X, y)
#svm_predictions = svc_model.predict(test_feats)
#output = pd.DataFrame({ 'PassengerId' : testdf['PassengerId'], 'Survived': svm_predictions })
#output.to_csv('titanic-predictions-svm.csv', index=False)
