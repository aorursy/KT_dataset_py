#To handle dataframe and array.

import  pandas as pd

import numpy as np

from numpy import array

from sklearn.cross_validation import train_test_split



#To visualize data

import seaborn as sns

import matplotlib.pylab as plt



#To perform Imputation, fill missing values

from fancyimpute import KNN



#To encode Nominal data without natural relationship

from sklearn.preprocessing import LabelEncoder, OneHotEncoder



#To do Machine Learning/Modelling

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

#Import Training and Test Data

train_data = pd.read_csv('../input/train.csv')

test_data  = pd.read_csv('../input/test.csv')



#Check Columns of both data sets and Visualize data

print (train_data.columns.values)

print ('-------------------------------------------------------------')

print (test_data.columns.values)
#let's keep "Survived" data and dropped it off from Training Dataframe

targets = train_data.Survived

train_data.drop('Survived', 1, inplace = True)



#Merge train_data and test_data

Merged_data = train_data.append(test_data)
Merged_data.info()
Merged_data[Merged_data['Fare'].isnull()]
#I filtered data with embarking from "Southampton = S" and "Lower Class = 3" that matched with Mr. Thomas Storey.

Filtered_PClass_Embarked = Merged_data[(Merged_data['Pclass'] == 3) & (Merged_data['Embarked'] == "S")]

sns.distplot(Filtered_PClass_Embarked.Fare.dropna())
#Replace Fare NaN with Median value

Merged_data['Fare'].fillna(Filtered_PClass_Embarked.Fare.median(), inplace=True)
#List all row with NaN in column 'Embarked'

Merged_data[Merged_data['Embarked'].isnull()]
#Plot boxplot Embarked vs Fare with Pclass as legend

ax = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", order = ['C','S','Q'], data = Merged_data)

#Add Fare = 80£ of missing Embarked

ax.hlines(80,-1, 3, linestyle='--', linewidth=1)
#Fill NaN of Embarked with "C"  = Charbourg.

Merged_data['Embarked'].fillna('C', inplace=True)

Merged_data.info()
Merged_data.head(8)
#Fill NaN with "U" to represent Unknown

Merged_data['Cabin'] = Merged_data['Cabin'].fillna("U")



#Take the 1st alphabet that indicates cabin location in Titanic

Merged_data.Cabin = Merged_data.Cabin.str[0]



#Put this plot for data visualizat (Upwards)

ax = sns.boxplot(x="Cabin", y="Fare",  hue="Pclass", order =['A','B','C','D','E','F','G','T','U'] , data = Merged_data) 

Merged_data['Title']  = Merged_data['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

Merged_data.head(5)
CountPerTitle = Merged_data['Title'][:].value_counts()    #Count people per Title

pd.crosstab(Merged_data['Title'], Merged_data['Sex'], dropna = 'FALSE' )    #Breakdown by Gender
def Group_Title():

    Title_Dictionary = []

    Title_Dictionary = {

                        "Capt"             : "Minority",

                        "Col"              : "Minority",

                        "Don"              : "Minority", 

                        "Dona"             : "Minority",

                        "Dr"               : "Minority",

                        "Jonkheer"         : "Minority", 

                        "Lady"             : "Minority",

                        "Major"            : "Minority",

                        "Master"           : "Master",

                        "Miss"             : "Miss", 

                        "Mlle"             : "Minority",    

                        "Mme"              : "Minority", 

                        "Mr"               : "Mr",

                        "Mrs"              : "Mrs",

                        "Ms"               : "Minority",

                        "Rev"              : "Minority", 

                        "Sir"              : "Minority",    

                        "the Countess"     : "Minority", 

    }

    Merged_data['Title']     =  Merged_data.Title.map(Title_Dictionary)

    

Group_Title()

#CountPerTitle = Merged_data['Title'][:].value_counts() 

pd.crosstab(Merged_data['Title'], Merged_data['Sex'], dropna = 'FALSE' )    #Breakdown by Gender

#Find the family size including passenger.

Merged_data['FamilySize']  = Merged_data.Parch  +  Merged_data.SibSp + 1

Merged_data.tail(5)
#from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

Merged_data.loc[:,'Sex'] = label_encoder.fit_transform(Merged_data.loc[:,'Sex'])

Merged_data.loc[:,'Title'] = label_encoder.fit_transform(Merged_data.loc[:,'Title'])

Merged_data.loc[:,'Embarked'] = label_encoder.fit_transform(Merged_data.loc[:,'Embarked'])

Merged_data.head(10)



#Do one hot encoder that breakdown into multiple columns with Cabin Prefix

one_hot_cabin =  pd.get_dummies(Merged_data['Cabin'], prefix = 'Cabin')



#Combined output of one hot encoder to the original dataframe

Merged_data = pd.concat([Merged_data, one_hot_cabin ], axis =1)

Merged_data.head(5)
#Select features with continuous data as input of Age Imputation

Impute_Age = Merged_data[['PassengerId','Sex','Fare', 'Title','FamilySize','Embarked', 'Age']]  



#Do imputation with KNN algorithm

Impute_Age_filled_knn = KNN(k=5).complete(Impute_Age)

Impute_Age_filled_knn = np.rint(Impute_Age_filled_knn)

Impute_Age_filled_knn[0:100,6]

Merged_data['Age'] = Impute_Age_filled_knn[:,6]



#Plot Age Distribution before imputing

figure = plt.figure()

figure.add_subplot(121)

plt.hist(Impute_Age.Age.dropna() ,facecolor='red',alpha=0.75) 

plt.xlabel("Age") 

plt.title("Age_Dist before Imputation ")



#Plot Age Distribution after imputing

figure.add_subplot(122)

plt.hist(Merged_data.Age,facecolor='blue',alpha=0.75) 

plt.xlabel("Age") 

plt.title("Age_Dist After Imputation ")
#Drop unwanted features

Merged_data.drop(['Cabin', 'Name','Ticket'], axis=1, inplace=True)



#Let's check the final look of Dataframe!

Merged_data.head(6)
#Separate training set and data set

X_Train_Final = Merged_data[0:891]

Y_Train_Final = targets    #We separatedly keep from the beginning.





# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X_Train_Final, Y_Train_Final, test_size = 0.2, random_state = 0)



print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Select one model at a time. 

model = RandomForestClassifier(n_estimators =100)  #Score = 82.83%

#model = SVC()   #SVC = Support Vector Machine for Classification , Score = 60.44%

#model = GradientBoostingClassifier()  #Score = 83.58%

#model = KNeighborsClassifier(n_neighbors = 4)    #Score = 66.04#

#model = GaussianNB()      #Score = 77.23%

#model = LogisticRegression() #Score 80.59%



#Fit model to traning data.

model.fit(X_train,y_train )
print (model.score( X_train , y_train ) , model.score( X_test , y_test ))