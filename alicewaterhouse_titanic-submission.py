#Import relevant modules
import numpy as np # linear algebra
import pandas as pd # data processing, (e.g. pd.read_csv)

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

#Read the data and take a look at the first few lines
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head(10)

#Find the number of missing values in each column of the training data and print if it is non-zero
missing_vals_train = (train.isnull().sum())
print('Missing values in training data:\n')
print(missing_vals_train[missing_vals_train > 0])

#Do the same for test data
print('\nMissing values in test data:\n')
missing_vals_test = (test.isnull().sum())
print(missing_vals_test[missing_vals_test > 0])
#Keep PassengerId for submission later
passengerid = test.PassengerId

#Drop irrelevant columns from both test and training data
Columns_to_drop = ['Ticket','PassengerId']
train.drop(Columns_to_drop, axis=1, inplace=True)
test.drop(Columns_to_drop, axis=1, inplace=True)

#Add 'N' for passengers without cabin data
train.Cabin.fillna("N", inplace=True)
test.Cabin.fillna("N", inplace=True)

#Take the first character of the string to remove the numeric part of the cabin data
train.Cabin = [i[0] for i in train.Cabin]
test.Cabin = [i[0] for i in test.Cabin]

train.head()
#Add a column for name length
train['NameLength'] = [len(i) for i in train.Name]
test['NameLength'] = [len(i) for i in test.Name]

#Add a column for title
train['Title'] = [i.split('.')[0] for i in train.Name]
train['Title'] = [i.split(', ')[1] for i in train.Title]
test['Title'] = [i.split('.')[0] for i in test.Name]
test['Title'] = [i.split(', ')[1] for i in test.Title]

#Drop Name
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

train.head()
print('Title value counts in training data: \n')
print(train.Title.value_counts())
print('\nTitle counts in test data: \n')
print(test.Title.value_counts())
#Simplify titles in training data
train["Title"] = [i.replace('Ms', 'Miss') for i in train.Title]
train["Title"] = [i.replace('Mlle', 'Miss') for i in train.Title]
train["Title"] = [i.replace('Mme', 'Mrs') for i in train.Title]
train["Title"] = [i.replace('Col', 'Military') for i in train.Title]
train["Title"] = [i.replace('Major', 'Military') for i in train.Title]
train["Title"] = [i.replace('Don', 'Military') for i in train.Title]
train["Title"] = [i.replace('Jonkheer', 'Nobility') for i in train.Title]
train["Title"] = [i.replace('Sir', 'Nobility') for i in train.Title]
train["Title"] = [i.replace('Lady', 'Nobility') for i in train.Title]
train["Title"] = [i.replace('Capt', 'Military') for i in train.Title]
train["Title"] = [i.replace('the Countess', 'Nobility') for i in train.Title]

#Simplify titles in test data
test["Title"] = [i.replace('Ms', 'Miss') for i in test.Title]
test["Title"] = [i.replace('Col', 'Military') for i in test.Title]
test["Title"] = [i.replace('Dona', 'Nobility') for i in test.Title]

train.head()
train["Sex"] = train["Sex"].replace({"female":0, "male":1})
test["Sex"] = test["Sex"].replace({"female":0, "male":1})
train.head()
#Change options to make pandas display all the columns
pd.options.display.max_columns = 99

#Print the correlation matrix
print(train.corr())
#Make a copy of the training data
train1 = train.copy(deep=True)
#Add a new column stating that 
train1['Dataset'] = 'Training Data'

#Do the same for the test data. Add 'Unknown' for Survived
test1 = test.copy(deep=True)
test1['Dataset'] = 'Test Data'
test1['Survived'] = 'Unknown'

#Combine the training and test data
combined_data = pd.concat([train1,test1],ignore_index=True)
combined_data.tail()
#Examine the modal value for embarked in both training and test data
print('Embarked value counts in training data: \n')
print(train.Embarked.value_counts())
print('\nEmbarked value counts in test data: \n')
print(test.Embarked.value_counts())

#Extract passengers whose Embarkation data is missing
train[train.Embarked.isnull()]
#Import plotting libraries
import seaborn as sns #Statistical visualisation
import matplotlib.pyplot as plt #Plotting

#Make a boxplot of Fare against embarked for each dataset
plt.figure(figsize=(15,9))
sns.set(style="whitegrid")
ax = sns.boxplot(y="Embarked", x="Fare", hue="Dataset", data=combined_data);
plt.title('Fare Boxplots for each Embarkation Point', fontsize=18)
#Fill in missing Embarked values with 'C'
train.Embarked.fillna("C", inplace=True)
test[test.Fare.isnull()]
#Impute Fare with the mean for the subset described above
missing_fare = combined_data[(combined_data.Pclass == 3) & (combined_data.Embarked == "S")].Fare.mean()
#Replace the test.Fare null values with missing_fare
test.Fare.fillna(missing_fare, inplace=True)
test.iloc[152]
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", y = "Survived", data=train)
plt.title("Fraction of Passengers that Survived by Sex", fontsize = 18)
plt.ylabel("Fraction of Passengers that Survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Pclass", y = "Survived", data=train)
plt.title("Fraction of Passengers that Survived by Pclass", fontsize = 18)
plt.ylabel("Fraction of Passengers that Survived", fontsize = 15)
plt.xlabel("Pclass",fontsize = 15);
plt.figure(figsize=(15,8))
ax = sns.kdeplot(train[(train['Survived'] == 1)].Age , shade=True, label='Survived')
ax = sns.kdeplot(train[(train['Survived'] == 0)].Age , shade=True, label='Did not survive')
plt.title('Estimated probability density of age given survival', fontsize = 18)
plt.xlabel('Age', fontsize = 15)
plt.ylabel('Probability density', fontsize = 15);
plt.figure(figsize=(15,8))
ax = sns.kdeplot(train[(train['Survived'] == 1)].Fare, label='Survived', shade=True)
ax = sns.kdeplot(train[(train['Survived'] == 0)].Fare, label='Did not survive', shade=True)
plt.title('Estimated probability density of fare given survival', fontsize = 18)
plt.xlabel('Fare', fontsize = 15)
plt.ylabel('Probability density', fontsize = 15);
#Use the mean age from the combined data to fill in unknown ages
train.Age.fillna(combined_data.Age.mean(), inplace=True)
test.Age.fillna(combined_data.Age.mean(), inplace=True)
train['Child'] = [1 if i<10 else 0 for i in train.Age]
test['Child'] = [1 if i<10 else 0 for i in test.Age]
train.head(10)
#Get dummies for one-hot-encoding of categorical features
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

#Drop the T Cabin...
train.drop(['Cabin_T'], axis=1, inplace=True)

#Split training data into dependent and independent variables
X = train.drop(['Survived'], axis=1)
y = train["Survived"]
from sklearn.model_selection import cross_val_score
#Import the models and then construct a pipeline for each including imputation
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=1)
from sklearn.svm import SVC
SVC = SVC(random_state=1)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(random_state=1)
from xgboost import XGBClassifier
XGB = XGBClassifier(random_state=1)

#Function that takes a model and returns its cross-validation score
def cv_score(model):
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=10)
    return scores.mean()

#Calculate cross-validation scores using this function
CV_scores = pd.DataFrame({'Cross-validation score':[cv_score(LR),cv_score(SVC),cv_score(RF),cv_score(XGB)]})
CV_scores.index = ['LR','SVC','RF','XGB']
print(CV_scores)
#Import GridSearchCV
from sklearn.model_selection import GridSearchCV

#Define parameter grid on the number of decision trees used and the maximum depth of the trees
parameters = {'n_estimators':[100,120,150], 'max_depth':[5,10,15,20,25,30]}
#Peform gridsearch
RF_grid = GridSearchCV(RF, param_grid=parameters)
RF_grid.fit(X,y)

#Print the best parameters and what they scored
print('Best parameters:'+str(RF_grid.best_params_))
print('Best score:'+str(RF_grid.best_score_))
#Make prediction and create a dataframe for submission
predictions = RF_grid.predict(test)
submission = pd.DataFrame({'PassengerId': passengerid, 'Survived': predictions})
submission.head(10)
submission.to_csv('submission.csv', index=False)