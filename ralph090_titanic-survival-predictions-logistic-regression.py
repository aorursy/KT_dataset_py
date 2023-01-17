import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.head()
train['Age'].isnull()
#Apart from using this function to fill up the missing values we can use;
#from sklearn.impute import SimpleImputer
#my_imputer = SimpleImputer()
#data_with_imputed_values = my_imputer.fit_transform(original_data)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37 #using pandas mean function or seaborn box plot 

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
Dsex = pd.get_dummies(train['Sex'],drop_first=True) #get dummies method of pandas to convert categorical variable into dummy/indicator variables.
Dembark = pd.get_dummies(train['Embarked'],drop_first=True) #drop_first to drop first column.

train.head(5)
train['Age'].isnull()
train.drop(['PassengerId','Cabin','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.head(1)
train = pd.concat([train,Dsex,Dembark],axis=1)
train.head(1)

X=train.drop(['Survived'],axis=1)
y=train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.50, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
ML = pd.DataFrame(predictions)
ML #The Predictions
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)