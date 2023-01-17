import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')  

#From the below heatmap could see that Cabin has lot of missing/null values and Age has few which can be filled by imputation
sns.set_style('whitegrid')
sns.countplot(x="Survived",data = train)
#From the above data set it can be said that more people died tha survived , lets do it for sex along to draw conclusions from that data 
sns.countplot(x="Survived", hue= 'Sex',data = train)
#From the above graph we could say that male died more than female
sns.countplot(x="Survived", hue= 'Pclass',data = train)

#From the below count plot could see that people who have not survived are from lower (3rd class) and who have survived are from Highest(1st Class)
sns.distplot(train['Age'].dropna() , kde=False , bins=30)
#Seeing the above plot its a bimodal distribution with some pax at 0 to 10 years and then more centric at 20 to 30 years of age 
train.info()
sns.countplot(x='SibSp' ,data=train)
train['Fare'].hist(bins=30)
#Cleaning out Data for performing M Learning 
    
sns.boxplot(x='Pclass',y='Age',data = train )

#Wealthier pax are older that the pax in third class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37 
        elif Pclass == 2:
            return 29 
        else:
            return 24
    else:
        return Age 
    
train['Age'] = train[['Age','Pclass']].apply( lambda x : impute_age(x),axis = 1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')  

#After the imputation we could see that the Age column is not having any null values , let work on cabin column it can be deleted as it has a lot of nullvalues 
train.dropna(inplace = True)
#Dropped all null values 
#Lets see how it look son the heatmap withour null values 
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Its awsome clean Data to start our work on 
train.head()
sex = pd.get_dummies(train['Sex'],drop_first=True)
Embarked = pd.get_dummies(train['Embarked'],drop_first=True)
#Here we are making male /female as 0 and 1 using get dummies method and then removing the first 
#column a sit woube be causing issue in ML terms as MultiColinearity which says that ML algo can predict that 
#if one column is True then the other will be automatically false  to avoid that we are dropping the 
#First coumn 
print(sex.head() )
print()
print()
print(Embarked.head())
#Similarly we need to do for embared
train = pd.concat([train,sex,Embarked],axis =1 )
train.head()
#Now we can drop all the columns which are not necessary for us anymore for performing/applying ML algo 
#on the data 
train.drop(['Sex','Embarked','Name','Ticket','Cabin'],axis= 1, inplace = True)
train.tail()
train.head()
    # we obtained a perfect data set to start out validation , we even dropped passengerID coumn  
#We take X and y to start building our model , X is a datset with all the columns except the one which we need to drop off 
#y is the one which we need to predict 

X = train.drop('Survived',axis=1)
y = train['Survived']
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report

#It is used to get our models accuracy from the F1 square , 
print(classification_report(y_test,predictions))


#Wheckout this video : https://www.youtube.com/watch?v=uZqwmihvTFY for more details on classification report 
# and aslo the meaning of "Sensitivity" and "Specificity"
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
