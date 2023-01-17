import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



#To display the rows and columns without getting truncated

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', -1) # this is to view complete text data in the column rather truncated
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import os



path = os.path.join(dirname)

os.chdir(path)



train_data = pd.read_csv('train.csv',encoding = 'cp1252', index_col= False)

test_data = pd.read_csv('test.csv',encoding = 'cp1252', index_col= False)



train_data.head(10)
train_data.info()
train_data.describe(include = 'all')
train_data.shape
# percentage of missing values in each column

round(train_data.isnull().sum()/len(train_data.index), 2)*100
# Imputing Missing values in Age (Numeric Column) Using simpleImputer

from sklearn.impute import SimpleImputer



imputer = SimpleImputer(missing_values= np.nan, strategy='median')



imputer = imputer.fit(train_data[['Age']])

train_data['Age'] = imputer.transform(train_data[['Age']]).ravel()



train_data.info()
# checking for Missing values again

round(train_data.isnull().sum()/len(train_data.index), 2)*100
#convert data type of age to int

train_data['Age']= train_data['Age'].astype(int)

train_data['Age'].dtype
#Fill the categorical variable (Cabin, Embarked Feature) with the most frequently occuring value

#idxmax() function returns index of first occurrence of maximum over

train_data['Cabin'] = train_data['Cabin'].fillna(train_data['Cabin'].value_counts().idxmax())

train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].value_counts().idxmax())
# checking for Missing values again

round(train_data.isnull().sum()/len(train_data.index), 2)*100
# checking whether some rows have more than 50 missing values

len(train_data[train_data.isnull().sum(axis=1) > 50].index)
train_data.columns
#checking for all numeric columns -.columns.tolist() - Excluding the number of siblings, spouse, parents and children

num_col = [col for col in train_data.select_dtypes(include=np.number) if col not in ['PassengerId', 'Survived', 'Age', 'SibSp','Parch']]

print(num_col)
## Check for outliers in Numerical columns- calling the outlier function

fig = plt.subplots(figsize=(9,6))

train_data.boxplot(['Fare'])



plt.show()
#Binning AGE GROUP of applicants

def Age_Group(n):

    if n < 5:

        return 'Infants'

    elif n >=5 and n < 13:

        return 'Children'

    elif n >= 13 and n < 20:

        return 'Teenager'

    elif n >= 20 and n < 26:

        return 'Student'

    elif n >= 26 and n < 41:

        return 'Youth'

    elif n >= 41 and n < 61:

        return 'Mid-Aged'

    else:

        return 'Senior Citizens'

        

train_data['Age_group'] = train_data['Age'].apply(lambda x: Age_Group(x))

#Implementing the changes to test data as well

test_data['Age_group'] = test_data['Age'].apply(lambda x: Age_Group(x))



#plotting the continous variable (AGE) using distplt and Categorical variable (Age_group) using barplot

plt.figure(figsize=(10,4))

    

plt.subplot(1, 2, 1) # This subplot will show the age group distribution

sns.distplot(train_data['Age'])



plt.subplot(1, 2, 2) # This Subplot will show how default rates vary across continous variables 

sns.barplot('Age_group', 'Survived', data=train_data).set(title = 'Distribution of Age-Group survived', ylabel = 'Survived' )

plt.xticks(rotation=90)

age_group = train_data.Age_group.value_counts()

print('\033[1m'+'\033[94m'+"Count per Age Group \n"+'\033[0m', age_group)

#print(age_group)

age_group.plot.bar()
#This is for the Fair paid

def totalfair_Group(n):

    if n < 15:

        return 'Low Fare'

    elif n >=15 and n < 40:

        return 'Avg Fare'

    else:

        return 'High Fare'

        

train_data['Fare_group'] = train_data['Fare'].apply(lambda x: totalfair_Group(x))

test_data['Fare_group'] = test_data['Fare'].apply(lambda x: totalfair_Group(x))



#plotting the continous variable using distplt and Categorical variable (Fare_group) using barplot

plt.figure(figsize=(10,4))

    

plt.subplot(1, 2, 1) # This subplot will show the spread of Fare

sns.distplot(train_data['Fare'])



plt.subplot(1, 2, 2) # This Subplot will show how default rates vary across continous variables 

sns.barplot('Fare_group', 'Survived', data=train_data).set(title = 'Distribution of fare per survived', ylabel = 'Survived' )

plt.xticks(rotation=90)
fare_group = train_data.Fare_group.value_counts()

print('\033[1m'+'\033[94m'+"Count per Age Group \n"+'\033[0m', fare_group)

#print(age_group)

fare_group.plot.bar()
# summarising the values

#print(train_data['Survived'].value_counts())

print("Survival Ratio: \n", train_data.Survived.value_counts()*100/train_data.shape[0])
plt.figure(figsize=(10, 6))

plt.title('Titanic Survival (Survived Vs Deceased) distribution')

sns.set_color_codes("pastel")

sns.countplot(x='Survived', data=train_data)

locs, labels = plt.xticks()

plt.show()
corr = train_data.corr(method ='pearson').abs() # mapping features to their absolute correlation values



#cor_target = corr[corr>=0.8]

plt.figure(figsize=(10,6))

sns.heatmap(corr, linewidths=0.5, vmin=-1, vmax=1, cmap='coolwarm',annot=True)
# Correlation of Survived with other columns

plt.figure(figsize=(10,6))

train_data.corr()['Survived'].sort_values(ascending = False).plot(kind='bar')

plt.show()
#map each Age value to a numerical value

age_mapping = {'Infants': 1, 'Children': 2, 'Teenager': 3, 'Student': 4, 'Youth': 5, 'Mid-Aged': 6, 'Senior Citizens': 7}

train_data['Age_group'] = train_data['Age_group'].map(age_mapping)

test_data['Age_group'] = test_data['Age_group'].map(age_mapping)



train_data.head()



#dropping the Age feature for now, might change

train_data = train_data.drop(['Age'], axis = 1)

test_data = test_data.drop(['Age'], axis = 1)
train_data.columns
#map each Fare value to a numerical value



fare_mapping = {'Low Fare': 0, 'Avg Fare': 1, 'High Fare': 2}

train_data['Fare_group'] = train_data['Fare_group'].map(fare_mapping)

test_data['Fare_group'] = test_data['Fare_group'].map(fare_mapping)



train_data.head()



#dropping the Age feature for now, might change

train_data = train_data.drop(['Fare'], axis = 1)

test_data = test_data.drop(['Fare'], axis = 1)
#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

train_data['Sex'] = train_data['Sex'].map(sex_mapping)

test_data['Sex'] = test_data['Sex'].map(sex_mapping)
#map each Embarked value to a numerical value

Embarked_mapping = {"C": 0, "Q": 1,"S":2}

train_data['Embarked'] = train_data['Embarked'].map(Embarked_mapping)

test_data['Embarked'] = test_data['Embarked'].map(Embarked_mapping)
#Dropping Name, Ticket and Cabin from train and test dataset

train_data.drop(['Name','Cabin','Ticket'], axis=1, inplace = True)

test_data.drop(['Name','Cabin','Ticket'], axis=1, inplace = True)
train_data.head()
train_data.info()
test_data.head()
##Function to plot grahps for various single features/variables

def plot_surv(var):

    plt.figure(figsize=(10,4))

    

    plt.subplot(1, 2, 1)

    x=train_data[var].value_counts()

    sns.barplot(x.index, x.values, order=x.index,alpha=0.8)

    plt.xlabel(var, labelpad=14)

    plt.ylabel("Total count", labelpad=14)

    plt.xticks(rotation=90)

    

    

    plt.subplot(1, 2, 2)

    #sorting the values in descending order

    target_perc = train_data[[var, 'Survived']].groupby([var],as_index=False).mean()

    target_perc.sort_values(by='Survived', ascending=False, inplace=True)

    sns.barplot(x=var, y='Survived',order=target_perc[var], data=target_perc, alpha=0.8)

    plt.xlabel(var, labelpad=14)

    plt.ylabel("Percent of Survived (%)")

    plt.xticks(rotation=90)

    

    fig = plt.figure()

    fig.subplots_adjust(right = 0.9, hspace=0.5, wspace=0.5)

    

    plt.show()
#plotting the gender distribution

plot_surv('Sex')
#Plotting by ticket class

plot_surv('Pclass')
plot_surv('SibSp')
plot_surv('Parch')
#Checking Correlation of dataset



corr = train_data.corr(method ='pearson').abs() # mapping features to their absolute correlation values



plt.figure(figsize=(10,6))

sns.heatmap(corr, linewidths=0.5, vmin=-1, vmax=1, cmap='coolwarm',annot=True)
# Taking copy of the original dataset



train_data_cpy = train_data[:].copy()

train_data_cpy.shape
train_data_cpy.info()
train_data_cpy.head()
#Dropping PassengerId from the dataset

train_data_cpy.drop(['PassengerId'], axis=1, inplace = True)
# X & y dataset for model building, X will obviously not have "Survived" and y will only have "Survived"

X = train_data_cpy.drop(['Survived'], axis=1)

y = train_data_cpy['Survived']



train_data_cpy.drop('Survived', axis=1, inplace=True)
from sklearn.model_selection import train_test_split



#Splitting Train_data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .25 ,random_state = 0)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



lgr = LogisticRegression()

lgr.fit(X_train, y_train)



# make the predictions

y_pred = lgr.predict(X_test)



# convert prediction array into a dataframe

y_pred_df = pd.DataFrame(y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score



print("Confusion Matrix: \n", confusion_matrix(y_test,y_pred))

acc_logistic = round(accuracy_score(y_test,y_pred)*100, 2)

print("Accuracy of the logistic regression model is",acc_logistic)
# Importing decision tree classifier from sklearn library

from sklearn.tree import DecisionTreeClassifier



# max_depth = 5 so that we can plot and read the tree.

dt_default = DecisionTreeClassifier(max_depth=5)

dt_default.fit(X_train, y_train)
# Let's check the evaluation metrics of our default model



# Importing classification report and confusion matrix from sklearn metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Making predictions

y_pred_default = dt_default.predict(X_test)



# Printing classification report

print(classification_report(y_test, y_pred_default))
print(confusion_matrix(y_test,y_pred_default))

acc_decision = round(accuracy_score(y_test,y_pred_default)*100, 2)

print("Accuracy of the Decsion Tree model is",acc_decision)
# Importing random forest classifier from sklearn library

from sklearn.ensemble import RandomForestClassifier



# Running the random forest with default parameters.

rfc = RandomForestClassifier()



# fit

rfc.fit(X_train,y_train)
# Making predictions

y_pred = rfc.predict(X_test)



# Let's check the report of our default model

print(classification_report(y_test,y_pred))
# Printing confusion matrix

print(confusion_matrix(y_test,y_pred))



acc_randomF = round(accuracy_score(y_test,y_pred)*100, 2)

print("Accuracy of the Random Forest model is",acc_randomF)
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)

# Let's check the report of our default model

print(classification_report(y_test,y_pred))
# Printing confusion matrix

print(confusion_matrix(y_test,y_pred))



acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)

print("Accuracy of the Random Forest model is", acc_gaussian)
from sklearn.svm import SVC



svc = SVC()

svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

# Let's check the report of our default model

print(classification_report(y_test,y_pred))
# Printing confusion matrix

print(confusion_matrix(y_test,y_pred))



acc_svm = round(accuracy_score(y_pred, y_test) * 100, 2)

print("Accuracy of the Support Vector Machine model is", acc_svm)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)
y_pred = gbk.predict(X_test)

# Let's check the report of our default model

print(classification_report(y_test,y_pred))
# Printing confusion matrix

print(confusion_matrix(y_test,y_pred))



acc_gradientBoost = round(accuracy_score(y_pred, y_test) * 100, 2)

print("Accuracy of the Gradient Boosting model is", acc_gradientBoost)
model_comp = pd.DataFrame({ 'Model': ['Logistic Regression','Decision Tree', 'Random Forest', 'Gaussian Naive Bayes', 'Support Vector Machines', 'Gradient Boosting'],

                                        'Score': [acc_logistic, acc_decision, acc_randomF, acc_gaussian, acc_svm, acc_gradientBoost]})

model_comp.sort_values(by='Score', ascending=False)
# Predicting Survival using RF

PassengerId = test_data['PassengerId']

test_pred = rfc.predict(test_data.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : PassengerId, 'Survived': test_pred })

print(output)

#output.to_csv('submission.csv', index=False)