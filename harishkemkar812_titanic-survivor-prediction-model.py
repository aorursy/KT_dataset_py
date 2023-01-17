import os

print(os.listdir("../input/"))
import pandas as pd 

import numpy as np



import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix



import seaborn as sns

sns.set(style = "white",color_codes = "True")

sns.set(font_scale = 1.5)





from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report





from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score



from sklearn.metrics import recall_score



from sklearn.metrics import f1_score

## Importing Data Set  



#df_train = pd.read.csv('E:/Harish/DataScience/Machine learning/HKnotebooks/data/mtcars.csv', delimiter = ',',engine='python')



df_train = pd.read_csv('../input/train.csv')

## Applying the same on test data , for creating final predictions file  



df_test = pd.read_csv('../input/test.csv')





df_train.info()

df_test.info()



# Observation from rain data set

# 891 observations

## Most null values in Cabin Column

## some missing values in Age column

## presence of survived column whihc will be missing in test Data 



df_train.head()

df_test.head()
df_train['Sex'].value_counts()

df_train['Survived'].value_counts()
df_train['Embarked'].value_counts()

df_train.isnull().sum()
## We can seecabinhas almost 80% null values , hence it does not makes sense to keep this column , 

##we will drop this column from train Data set

## We will drop other relevant columns as well

df_train.head()
df_train = df_train.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis = 1)

df_test = df_test.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis = 1)

df_train.head()

df_test.head()
## Plotting Age Histogram 



plt.hist(df_train['Age'],bins = 20,color = 'b')

plt.xlabel('Users Age')

plt.ylabel('No of users')

plt.show()
df_train.info()
df1 = df_train.groupby('Pclass')
Age_firstclass = df1.get_group(1)['Age'].dropna()

Age_secondclass = df1.get_group(2)['Age'].dropna()

Age_thirdclass = df1.get_group(3)['Age'].dropna()

count_frstClass = len(df1.get_group(1)['Age'].dropna())

count_secondClass = len(df1.get_group(2)['Age'].dropna())

count_thirdClass = len(df1.get_group(3)['Age'].dropna())
avg_age_first =  Age_firstclass.sum()/count_frstClass

avg_age_second =  Age_secondclass.sum()/count_secondClass

avg_age_third = Age_thirdclass.sum()/count_thirdClass
##Hence we can see average age class wise 

print("Average of first class passenger ",avg_age_first)

print("Average of second class passenger ",avg_age_second)

print("Average of third class passenger ",avg_age_third)



## We will asume below values for average age of three calasses -  38,30,25

## Another way to find mean is below 



df_train.groupby(['Pclass']).mean()



df_test.groupby(['Pclass']).mean()
## We can say the younger the person is , it is more likely to be in first class 



def age_approx(cols):

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

    

    

def age_approx_test_data(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 41

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
df_train['Age'] =  df_train[['Age','Pclass']].apply(age_approx,axis =1)



df_test['Age'] =  df_test[['Age','Pclass']].apply(age_approx_test_data,axis =1)
df_train.isnull().sum()



df_test.isnull().sum()
df_train.dropna(inplace = True)



df_test.dropna(inplace = True)

## Now our training data set looks better , without any null values 

## we can see column sex and embarked are categoricall Data type , we will use get dummies to convert these columnd into categories 

df_train_dummied = pd.get_dummies(df_train,columns = ['Sex'])



df_test_dummied = pd.get_dummies(df_test,columns = ['Sex'])

df_train_dummied.info()



df_test_dummied.info()
df_train_dummied = pd.get_dummies(df_train_dummied,columns = ['Embarked'])



df_test_dummied = pd.get_dummies(df_test_dummied,columns = ['Embarked'])
df_train_dummied.info()



df_test_dummied.info()
## Checking if variables are correlated 

sns.heatmap(df_train_dummied.corr(),cmap = 'bwr')



## Strong correlation between Survived and Sex_female columns 



## creating model  



used_features =  ['Pclass','Age','SibSp','Parch','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']



X = df_train_dummied[used_features].values



Y = df_train_dummied['Survived']



X_Final_test = df_test_dummied[used_features].values

## Splitting Data set into train and test data 

from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size = 0.3,test_size =0.7)

X_train.shape

#(266, 9)

X_test.shape

#(623, 9)

Y_train.shape

# (266,0)

Y_test.shape

#(623,0)





## Now isntantiate and train Classifier 



LogReg = LogisticRegression()
LogReg.fit(X_train,Y_train)
Y_pred = LogReg.predict(X_test)



Y_pred_final = LogReg.predict(X_Final_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test,Y_pred)
accuracy_score(Y_test,Y_pred)
## This means pour model has accuracy score of 79% 
print(classification_report(Y_test,Y_pred))
## Final output is in this data frame 





Y_pred_final = LogReg.predict(X_Final_test)



X_Final_test =  pd.DataFrame(X_Final_test)



print(X_Final_test)

X_Final_test.columns = used_features





Y_pred_final
X_Final_test.head()
Y_pred_final1  =  pd.DataFrame(Y_pred_final)
label_name =  ['Survived']

Y_pred_final1.columns = label_name 
print(Y_pred_final1)


## pd.concat([df,df_target],axis = 1) 



Titanic_predictions = pd.concat([X_Final_test,Y_pred_final1],axis = 1)
Titanic_predictions.head()
## df.to_csv(r'Path where you want to store the exported CSV file\File Name.csv')



Titanic_predictions.to_csv(r'../input/titanic_pred.csv')
Titanic_predictions.info()