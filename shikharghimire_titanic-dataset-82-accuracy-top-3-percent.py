#Importing the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the dataset
train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

train_dataset.isnull().sum() #Age has 177 missing values and Cabin has 687 missing values but we won't bother with
                                #with missing values of cabin because we are going to remove it at the end
                                

train_dataset['name_initials']=0

for i in train_dataset:
    train_dataset['name_initials']=train_dataset.Name.str.extract('([A-Za-z]+)\.')

train_dataset['name_initials'].unique()

#Creating a name initials coloumns
train_dataset['name_initials'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
       'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',
       'Jonkheer'],['Mr','Mrs','Miss','Master','Mr','Other','Mr','Miss','Miss','Mr','Mrs','Mr','Mr','Other','Mr','Other','Other'],inplace=True)


#Determining the age by the initials by calculating the mean from each
train_dataset.head()
train_dataset.groupby('name_initials')['Age'].mean()
#Countess    33.000000
#Master       4.574167
#Miss        21.860000
#Mr          32.739609
#Mrs         36.009174
#Other       45.888889

train_dataset.loc[(train_dataset.Age.isnull())&(train_dataset.name_initials=='Master'),'Age']=5
train_dataset.loc[(train_dataset.Age.isnull())&(train_dataset.name_initials=='Miss'),'Age']=22
train_dataset.loc[(train_dataset.Age.isnull())&(train_dataset.name_initials=='Mr'),'Age']=33
train_dataset.loc[(train_dataset.Age.isnull())&(train_dataset.name_initials=='Mrs'),'Age']=36
train_dataset.loc[(train_dataset.Age.isnull())&(train_dataset.name_initials=='Other'),'Age']=45

#Dropping useless coloumns from training set
train_dataset.drop(['Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True) 
train_dataset.head()

#Changing name_initials category data to numerical values with hotencdoing using labelbinarizer and concating two datasets
from sklearn.preprocessing import LabelBinarizer
name_encoder = LabelBinarizer()
encoded=name_encoder.fit_transform(train_dataset['name_initials'])
final_training = pd.concat([pd.DataFrame(train_dataset),pd.DataFrame(encoded)],axis=1).drop(['name_initials'],axis=1)

#Filling the embarked data with the most repeated categorical values 
final_training = final_training.fillna(final_training['Embarked'].value_counts().index[0]) #Filling all the values with "S". Southampton

##Renaming the coloumn named 0,1,2,3,4
final_training.rename(columns={final_training.columns[8]:'Master',
                               final_training.columns[9]:'Miss',final_training.columns[10]:'Mr',
                               final_training.columns[11]:'Mrs',
                               final_training.columns[12]:'Other'}, inplace=True)

embarked_encoder = LabelBinarizer()
embarked_encoded = embarked_encoder.fit_transform(final_training['Embarked'])
final_training = pd.concat([pd.DataFrame(final_training),pd.DataFrame(embarked_encoded)],axis=1).drop(['Embarked'],axis=1)

#Changing the sex category values to the binary
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
final_training['Sex'] = labelencoder.fit_transform(final_training['Sex']) #1 male and 0 female

#Renaming the coloumn for embarked after it had been label binarised
final_training.rename(columns={final_training.columns[12]:'Cherbourg',
                               final_training.columns[13]:'Queenstown',final_training.columns[14]:'Southampton'},inplace=True)

#Survival rate goes up significantly when 
final_training['total members'] = 0
final_training['total_members'] = final_training['SibSp'] + final_training['Parch'] + 1
ax = sns.factorplot('total_members','Survived',data = final_training,size=5,aspect=2)
plt.show()

#Turns out if the person has more than 2 children and less than or equal to 4 children the chances of survival are relatively high

#Seperating the classes
Firstclass = final_training[final_training.Pclass==1] #For first class
Middleclass = final_training[final_training.Pclass==2] #For second class
ThirdClass = final_training[final_training.Pclass==3] #For third class


#Using fare categories in the data
#Setting it up for the first class
final_training.loc[(final_training.Pclass==1) & (final_training.Fare<=30.95),'Fare_category'] =1
final_training.loc[(final_training.Pclass==1) & (final_training.Fare>30.95) & (final_training.Fare <= 60.287),'Fare_category'] = 2
final_training.loc[(final_training.Pclass==1) & (final_training.Fare>60.287) & (final_training.Fare <= 93.5),'Fare_category'] = 3
final_training.loc[(final_training.Pclass==1) & (final_training.Fare>93.5),'Fare_category'] = 4

#Setting it up for the third class
final_training.loc[(final_training.Pclass==2) & (final_training.Fare<=13),'Fare_category'] =5
final_training.loc[(final_training.Pclass==2) & (final_training.Fare>13) & (final_training.Fare <= 14.35),'Fare_category'] = 6
final_training.loc[(final_training.Pclass==2) & (final_training.Fare>14.35) & (final_training.Fare <= 73.5),'Fare_category'] = 7
final_training.loc[(final_training.Pclass==2) & (final_training.Fare>73.5),'Fare_category'] = 8

#For third class
final_training.loc[(final_training.Pclass==3) & (final_training.Fare<=7.75),'Fare_category'] =9
final_training.loc[(final_training.Pclass==3) & (final_training.Fare>7.75) & (final_training.Fare <= 8.05),'Fare_category'] = 10
final_training.loc[(final_training.Pclass==3) & (final_training.Fare>8.05) & (final_training.Fare <= 15.5),'Fare_category'] = 11
final_training.loc[(final_training.Pclass==3) & (final_training.Fare>15.5),'Fare_category'] = 12

sns.factorplot('Fare_category','Survived',data = final_training,size=5,aspect=2)
plt.show() #0 for not survived and 1 for survived...Turns out Fare category 3 and 4 has the same amount of survival rate

#Removing all the value that causes the dummy variable trap
final_training = final_training.drop('Master',axis=1) #Dropped Master from the initials
final_training = final_training.drop('Cherbourg',axis=1) #Dropped Cherbourg from embarked


#Prepraring the test dataset
test_dataset['name_initials']=0

for i in test_dataset:
    test_dataset['name_initials']=test_dataset.Name.str.extract('([A-Za-z]+)\.')
    
test_dataset['name_initials'].unique()

#Creating a name initials coloumns
test_dataset['name_initials'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Ms', 'Col', 'Rev', 'Dr', 'Dona'],
             ['Mr','Mrs','Miss','Master','Mrs','Mr','Other','Mr','Mrs'],inplace=True)

#Determining the age by the initials by calculating the mean from each
test_dataset.head()
test_dataset.groupby('name_initials')['Age'].mean()

#Master     7.406471
#Miss      21.774844
#Mr        32.306452
#Mrs       38.904762
#Other     35.500000

test_dataset.loc[(test_dataset.Age.isnull())&(test_dataset.name_initials=='Master'),'Age']=7
test_dataset.loc[(test_dataset.Age.isnull())&(test_dataset.name_initials=='Miss'),'Age']=22
test_dataset.loc[(test_dataset.Age.isnull())&(test_dataset.name_initials=='Mr'),'Age']=32
test_dataset.loc[(test_dataset.Age.isnull())&(test_dataset.name_initials=='Mrs'),'Age']=39
test_dataset.loc[(test_dataset.Age.isnull())&(test_dataset.name_initials=='Other'),'Age']=35

#Dropping useless coloumns from training set
test_dataset.drop(['Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True) 
test_dataset.head()

#Changing name_initials category data to numerical values with hotencdoing using labelbinarizer and concating two datasets
from sklearn.preprocessing import LabelBinarizer
name_encoder_one = LabelBinarizer()
encoded_one=name_encoder_one.fit_transform(test_dataset['name_initials'])
final_testing = pd.concat([pd.DataFrame(test_dataset),pd.DataFrame(encoded_one)],axis=1).drop(['name_initials'],axis=1)

#Filling the embarked data with the most repeated categorical values 
final_testing = final_testing.fillna(final_testing['Embarked'].value_counts().index[0]) #Filling all the values with "S". Southampton

##Renaming the coloumn named 0,1,2,3,4
final_testing.rename(columns={final_testing.columns[7]:'Master',
                               final_testing.columns[8]:'Miss',final_testing.columns[9]:'Mr',
                               final_testing.columns[10]:'Mrs',
                               final_testing.columns[11]:'Other'}, inplace=True)


embarked_encoder_one = LabelBinarizer()
embarked_encoded_one = embarked_encoder_one.fit_transform(final_testing['Embarked'])
final_testing = pd.concat([pd.DataFrame(final_testing),pd.DataFrame(embarked_encoded_one)],axis=1).drop(['Embarked'],axis=1)

#Changing the sex category values to the binary
from sklearn.preprocessing import LabelEncoder
labelencoder_one = LabelEncoder()
final_testing['Sex'] = labelencoder_one.fit_transform(final_testing['Sex']) #1 male and 0 female

#Renaming the coloumn for embarked after it had been label binarised
final_testing.rename(columns={final_testing.columns[11]:'Cherbourg',
                               final_testing.columns[12]:'Queenstown',final_testing.columns[13]:'Southampton'},inplace=True)

#Turns out if the person has more than 2 children and less than or equal to 4 children the chances of survival are relatively high
#Seperating the classes
Firstclass_test = final_testing[final_testing.Pclass==1] #For first class
Middleclass_test = final_testing[final_testing.Pclass==2] #For second class
ThirdClass_test = final_testing[final_testing.Pclass==3] #For third class

final_testing['Fare'] = final_testing['Fare'].replace('S',np.nan)
final_testing['Fare'] = final_testing['Fare'].fillna(final_testing['Fare'].mean())


#Using fare categories in the data
#Setting it up for the first class
final_testing.loc[(final_testing.Pclass==1) & (final_testing.Fare<=30.95),'Fare_category'] =1
final_testing.loc[(final_testing.Pclass==1) & (final_testing.Fare>30.95) & (final_testing.Fare <= 60.287),'Fare_category'] = 2
final_testing.loc[(final_testing.Pclass==1) & (final_testing.Fare>60.287) & (final_testing.Fare <= 93.5),'Fare_category'] = 3
final_testing.loc[(final_testing.Pclass==1) & (final_testing.Fare>93.5),'Fare_category'] = 4

#Setting it up for the third class
final_testing.loc[(final_testing.Pclass==2) & (final_testing.Fare<=13),'Fare_category'] =5
final_testing.loc[(final_testing.Pclass==2) & (final_testing.Fare>13) & (final_testing.Fare <= 14.35),'Fare_category'] = 6
final_testing.loc[(final_testing.Pclass==2) & (final_testing.Fare>14.35) & (final_testing.Fare <= 73.5),'Fare_category'] = 7
final_testing.loc[(final_testing.Pclass==2) & (final_testing.Fare>73.5),'Fare_category'] = 8

#For third class
final_testing.loc[(final_testing.Pclass==3) & (final_testing.Fare<=7.75),'Fare_category'] =9
final_testing.loc[(final_testing.Pclass==3) & (final_testing.Fare>7.75) & (final_testing.Fare <= 8.05),'Fare_category'] = 10
final_testing.loc[(final_testing.Pclass==3) & (final_testing.Fare>8.05) & (final_testing.Fare <= 15.5),'Fare_category'] = 11
final_testing.loc[(final_testing.Pclass==3) & (final_testing.Fare>15.5),'Fare_category'] = 12

#Removing all the value that causes the dummy variable trap
final_testing = final_testing.drop('Master',axis=1) #Dropped Master from the initials
final_testing = final_testing.drop('Cherbourg',axis=1) #Dropped Cherbourg from embarked

#Using the model now
#Defining the training data
X_train = final_training.drop(['Survived'],axis=1)
Y_train = final_training.Survived

#Defining the testing data
X_test = final_testing

#Using random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='gini',n_estimators = 44) #Changed 30 to 44 after grid search in n_estimators
classifier.fit(X_train,Y_train)

#Predicting the survival rate of X_test
Y_pred = classifier.predict(X_test)

#Using gridsearch CV to improve the results
from sklearn.model_selection import GridSearchCV
parameters = [{'criterion':['gini','entropy'],'n_estimators':np.arange(1,100)}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv=10)
grid_search = grid_search.fit(X_train,Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

#Using cross validation to find the percentage accuracy ----82%
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X = X_train, y = Y_train, cv = 10, n_jobs=-1)
mean_data = accuracies.mean()