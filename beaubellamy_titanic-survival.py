# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

trainingData = pd.read_csv('../input/train.csv')
testingData = pd.read_csv('../input/test.csv')

trainingData.head()
#testingData.head()
trainingData.isnull().sum() #checking for total null values
#testingData.isnull().sum()
trainingData['Salutaion']=0

# Extract the salutation
for i in trainingData:
    trainingData['Salutaion']=trainingData.Name.str.extract('([A-Za-z]+)\.') 
    
# Create a salutation column
trainingData['Salutaion'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                                ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

# Determine the Average ages by salutation
trainingData.groupby('Salutaion')['Age'].mean()


# Use the ceiling values for the ages.
trainingData.loc[(trainingData.Age.isnull())&(trainingData.Salutaion=='Master'),'Age']=5
trainingData.loc[(trainingData.Age.isnull())&(trainingData.Salutaion=='Miss'),'Age']=22
trainingData.loc[(trainingData.Age.isnull())&(trainingData.Salutaion=='Mr'),'Age']=33
trainingData.loc[(trainingData.Age.isnull())&(trainingData.Salutaion=='Mrs'),'Age']=36
trainingData.loc[(trainingData.Age.isnull())&(trainingData.Salutaion=='Other'),'Age']=46

trainingData.groupby('Embarked')['PassengerId'].count()
# The most passengers embarked at S (Southampton), so we can assume that 
# the 2 missing values are likley to have embarked at Southampton.
trainingData['Embarked'].fillna('S',inplace=True)
trainingData['Age_band']=0
trainingData.loc[trainingData['Age']<=8,'Age_band']=0
trainingData.loc[(trainingData['Age']>8)&(trainingData['Age']<=16),'Age_band']=1
trainingData.loc[(trainingData['Age']>16)&(trainingData['Age']<=24),'Age_band']=2
trainingData.loc[(trainingData['Age']>24)&(trainingData['Age']<=32),'Age_band']=3
trainingData.loc[(trainingData['Age']>32)&(trainingData['Age']<=40),'Age_band']=4
trainingData.loc[(trainingData['Age']>40)&(trainingData['Age']<=48),'Age_band']=5
trainingData.loc[(trainingData['Age']>48)&(trainingData['Age']<=56),'Age_band']=6
trainingData.loc[(trainingData['Age']>56)&(trainingData['Age']<=64),'Age_band']=7
trainingData.loc[(trainingData['Age']>64)&(trainingData['Age']<=72),'Age_band']=8
trainingData.loc[(trainingData['Age']>72),'Age_band']=9

trainingData.head()
trainingData['Size_Of_Party']=1
trainingData['Size_Of_Party']=trainingData['Parch']+trainingData['SibSp']+1

ax = sns.factorplot('Size_Of_Party','Survived',data=trainingData,size=5,aspect=2)
#ax.set_title('Size of party Vs Survived')
plt.show()
sns.factorplot('Size_Of_Party','Survived',data=trainingData,hue='Sex',col='Pclass')
plt.show()
# Seperate the classes
FirstClass = trainingData[trainingData.Pclass == 1]
MiddleClass = trainingData[trainingData.Pclass == 2]
CattleClass = trainingData[trainingData.Pclass == 3]

# Determine the bin boundaries
#pd.qcut(FirstClass['Fare'],4)
#[(-0.001, 30.924] < (30.924, 60.287] < (60.287, 93.5] < (93.5, 512.329]]

#pd.qcut(MiddleClass['Fare'],4)
#[(-0.001, 13.0] < (13.0, 14.25] < (14.25, 26.0] < (26.0, 73.5]]

#pd.qcut(CattleClass['Fare'],4)
#[(-0.001, 7.75] < (7.75, 8.05] < (8.05, 15.5] < (15.5, 69.55]]
# Create the Fare catogories based on the class 

# First Class
#[(-0.001, 30.924] < (30.924, 60.287] < (60.287, 93.5] < (93.5, 512.329]]
trainingData.loc[(trainingData.Pclass == 1) & (trainingData.Fare <= 30.924),'Fare_cat'] = 1
trainingData.loc[(trainingData.Pclass == 1) & (trainingData.Fare > 30.924) & (trainingData.Fare <= 60.287),'Fare_cat'] = 2
trainingData.loc[(trainingData.Pclass == 1) & (trainingData.Fare > 60.287) & (trainingData.Fare <= 93.5),'Fare_cat'] = 3
trainingData.loc[(trainingData.Pclass == 1) & (trainingData.Fare > 93.5),'Fare_cat'] = 4

# Middle Class
#[(-0.001, 13.0] < (13.0, 14.25] < (14.25, 26.0] < (26.0, 73.5]]
trainingData.loc[(trainingData.Pclass == 2) & (trainingData.Fare <= 13),'Fare_cat'] = 5
trainingData.loc[(trainingData.Pclass == 2) & (trainingData.Fare > 13) & (trainingData.Fare <= 14.35),'Fare_cat'] = 6
trainingData.loc[(trainingData.Pclass == 2) & (trainingData.Fare > 14.35) & (trainingData.Fare <= 73.5),'Fare_cat'] = 7
trainingData.loc[(trainingData.Pclass == 2) & (trainingData.Fare > 73.5),'Fare_cat'] = 8

# Cattle Class
#[(-0.001, 7.75] < (7.75, 8.05] < (8.05, 15.5] < (15.5, 69.55]]
trainingData.loc[(trainingData.Pclass == 3) & (trainingData.Fare <= 7.75),'Fare_cat'] = 9
trainingData.loc[(trainingData.Pclass == 3) & (trainingData.Fare > 7.75) & (trainingData.Fare <= 8.05),'Fare_cat'] = 10
trainingData.loc[(trainingData.Pclass == 3) & (trainingData.Fare > 8.05) & (trainingData.Fare <= 15.5),'Fare_cat'] = 11
trainingData.loc[(trainingData.Pclass == 3) & (trainingData.Fare > 15.5),'Fare_cat'] = 12

trainingData.head()
sns.factorplot('Fare_cat','Survived',data=trainingData,hue='Sex',size=5,aspect=2)
plt.show()
trainingData['Sex'].replace(['male','female'],[0,1],inplace=True)
trainingData['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
trainingData['Salutaion'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)
trainingData.head()
trainingData.drop(['Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True)
#trainingData.drop(['Age','SibSp','Parch'],axis=1,inplace=True)

sns.heatmap(trainingData.corr(),annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':20})
fig=plt.gcf()
fig.set_size_inches(18,15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
# Import the required modules
from sklearn import svm #support vector Machine
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import GridSearchCV

# Split the data into training and test set.
train,test=train_test_split(trainingData,test_size=0.3,random_state=0,stratify=trainingData['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=trainingData[trainingData.columns[1:]]
Y=trainingData['Survived']

trainingData.isnull().sum() 


# Fit a model and test the predictions
model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
model.fit(train_X,train_Y)
prediction=model.predict(test_X)
print('Accuracy for rbf SVM is ',metrics.accuracy_score(prediction,test_Y))


# Set up Hyper parameters to search
#C=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.90,0.95,1]
#gamma=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.90,0.95,1.0]

C=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.90,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}

gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd.fit(X,Y)
print(gd.best_score_)
print(gd.best_estimator_)
# Salutation
# Create the salutation feature for the testing data
testingData['Salutaion']=0

for i in testingData:
    testingData['Salutaion']=testingData.Name.str.extract('([A-Za-z]+)\.') 
    
# Create a salutation column
testingData['Salutaion'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
                                ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Mr'],inplace=True)





# Use the ceiling values for the ages.
#testingData.head()


testingData.loc[(testingData.Age.isnull())&(testingData.Salutaion=='Master'),'Age']=5
testingData.loc[(testingData.Age.isnull())&(testingData.Salutaion=='Miss'),'Age']=22
testingData.loc[(testingData.Age.isnull())&(testingData.Salutaion=='Mr'),'Age']=33
testingData.loc[(testingData.Age.isnull())&(testingData.Salutaion=='Mrs'),'Age']=36
testingData.loc[(testingData.Age.isnull())&(testingData.Salutaion=='Other'),'Age']=46

# Age Band
# Create the age band for the testing data.
testingData['Age_band']=0
testingData.loc[testingData['Age']<=8,'Age_band']=0
testingData.loc[(testingData['Age']>8)&(testingData['Age']<=16),'Age_band']=1
testingData.loc[(testingData['Age']>16)&(testingData['Age']<=24),'Age_band']=2
testingData.loc[(testingData['Age']>24)&(testingData['Age']<=32),'Age_band']=3
testingData.loc[(testingData['Age']>32)&(testingData['Age']<=40),'Age_band']=4
testingData.loc[(testingData['Age']>40)&(testingData['Age']<=48),'Age_band']=5
testingData.loc[(testingData['Age']>48)&(testingData['Age']<=56),'Age_band']=6
testingData.loc[(testingData['Age']>56)&(testingData['Age']<=64),'Age_band']=7
testingData.loc[(testingData['Age']>64)&(testingData['Age']<=72),'Age_band']=8
testingData.loc[(testingData['Age']>72),'Age_band']=9

testingData.head()

# Size of the Party
testingData['Size_Of_Party']=1
testingData['Size_Of_Party']=testingData['Parch']+testingData['SibSp']+1

# Fare catagory for each class
testingData.loc[(testingData.Pclass == 1) & (testingData.Fare <= 30.924),'Fare_cat'] = 1
testingData.loc[(testingData.Pclass == 1) & (testingData.Fare > 30.924) & (testingData.Fare <= 60.287),'Fare_cat'] = 2
testingData.loc[(testingData.Pclass == 1) & (testingData.Fare > 60.287) & (testingData.Fare <= 93.5),'Fare_cat'] = 3
testingData.loc[(testingData.Pclass == 1) & (testingData.Fare > 93.5),'Fare_cat'] = 4

# Middle Class
#[(-0.001, 13.0] < (13.0, 14.25] < (14.25, 26.0] < (26.0, 73.5]]
testingData.loc[(testingData.Pclass == 2) & (testingData.Fare <= 13),'Fare_cat'] = 5
testingData.loc[(testingData.Pclass == 2) & (testingData.Fare > 13) & (testingData.Fare <= 14.35),'Fare_cat'] = 6
testingData.loc[(testingData.Pclass == 2) & (testingData.Fare > 14.35) & (testingData.Fare <= 73.5),'Fare_cat'] = 7
testingData.loc[(testingData.Pclass == 2) & (testingData.Fare > 73.5),'Fare_cat'] = 8

# Cattle Class
#[(-0.001, 7.75] < (7.75, 8.05] < (8.05, 15.5] < (15.5, 69.55]]
testingData.loc[(testingData.Pclass == 3) & (testingData.Fare <= 7.75),'Fare_cat'] = 9
testingData.loc[(testingData.Pclass == 3) & (testingData.Fare > 7.75) & (testingData.Fare <= 8.05),'Fare_cat'] = 10
testingData.loc[(testingData.Pclass == 3) & (testingData.Fare > 8.05) & (testingData.Fare <= 15.5),'Fare_cat'] = 11
testingData.loc[(testingData.Pclass == 3) & (testingData.Fare > 15.5),'Fare_cat'] = 12

testingData['Sex'].replace(['male','female'],[0,1],inplace=True)
testingData['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
testingData['Salutaion'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)



# Drop the unneccessary features from the test set
Passengers = testingData.PassengerId
testingData.drop(['Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True)
# Correct the one Nan value.
testingData.iloc[152,5]= testingData.Fare.mean() # Fare
testingData.iloc[152,10] = 4 # Fare category

# Create the predictions for the test data.
results = gd.predict(testingData)

Passengers = Passengers.tolist()
survived = results.tolist()

type(Passengers)
zip(Passengers,results)

import csv
with open('output.txt', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(Passengers,results))

