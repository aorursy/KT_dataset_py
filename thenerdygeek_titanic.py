# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
dataset= pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
%matplotlib inline
import seaborn
seaborn.set() 

#-------------------Survived/Died by Class -------------------------------------
survived_class = dataset[dataset['Survived']==1]['Pclass'].value_counts()
dead_class = dataset[dataset['Survived']==0]['Pclass'].value_counts()
df_class = pd.DataFrame([survived_class,dead_class], index = ['Survived','Died'])
df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Class")

Class1_survived= df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100
Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100
Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100
print("Percentage of Class 1 that survived:" ,round(Class1_survived),"%")
print("Percentage of Class 2 that survived:" ,round(Class2_survived), "%")
print("Percentage of Class 3 that survived:" ,round(Class3_survived), "%")

# #display table
# from IPython.display import display

#-------------------Survived/Died by SEX------------------------------------
   
Survived = dataset[dataset.Survived == 1]['Sex'].value_counts()
Died = dataset[dataset.Survived == 0]['Sex'].value_counts()
df_sex = pd.DataFrame([Survived , Died], index = ['Survived','Died'])
# df_sex.index = ['Survived','Died']
df_sex.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Sex")


female_survived= df_sex.female[0]/df_sex.female.sum()*100
male_survived = df_sex.male[0]/df_sex.male.sum()*100
print("Percentage of female that survived:" ,round(female_survived), "%")
print("Percentage of male that survived:" ,round(male_survived), "%")

# display table
# from IPython.display import display
display(df_sex)
# dataset
#-------------------- Survived/Died by Embarked ----------------------------

survived_embark = dataset[dataset['Survived']==1]['Embarked'].value_counts()
dead_embark = dataset[dataset['Survived']==0]['Embarked'].value_counts()
df_embark = pd.DataFrame([survived_embark,dead_embark])
df_embark.index = ['Survived','Died']
df_embark.plot(kind='bar',stacked=True, figsize=(5,3))

Embark_S= df_embark.iloc[0,0]/df_embark.iloc[:,0].sum()*100
Embark_C = df_embark.iloc[0,1]/df_embark.iloc[:,1].sum()*100
Embark_Q = df_embark.iloc[0,2]/df_embark.iloc[:,2].sum()*100
print("Percentage of Embark S that survived:", round(Embark_S), "%")
print("Percentage of Embark C that survived:" ,round(Embark_C), "%")
print("Percentage of Embark Q that survived:" ,round(Embark_Q), "%")

# from IPython.display import display
display(df_embark)
X = dataset.drop(['PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'], axis=1)
X_Test = test.drop(['PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'], axis=1)
y = X.Survived                       # vector of labels (dependent variable)
X=X.drop(['Survived'], axis=1)       # remove the dependent variable from the dataframe X
# display(y)
X.head(20)
# ----------------- Encoding categorical data -------------------------

# encode "Sex"
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X.Sex=labelEncoder_X.fit_transform(X.Sex)
X_Test.Sex=labelEncoder_X.fit_transform(X_Test.Sex)

# encode "Embarked"

# number of null values in embarked:
print ('Number of null values in Embarked:', sum(X.Embarked.isnull()))

# fill the two values with one of the options (S, C or Q)
row_index = X.Embarked.isnull()
X.loc[row_index,'Embarked']='S' 
X_Test.loc[X_Test.Embarked.isnull(),'Embarked']='S'

Embarked  = pd.get_dummies(  X.Embarked , prefix='Embarked'  )
Embarked_Test = pd.get_dummies(X_Test.Embarked, prefix='Embarked')
X = X.drop(['Embarked'], axis=1)
X= pd.concat([X, Embarked], axis=1)  
# we should drop one of the columns
X = X.drop(['Embarked_S'], axis=1)

X_Test = pd.concat([X_Test, Embarked_Test], axis=1)
X_Test = X_Test.drop(['Embarked', 'Embarked_S'],axis=1)

print("Number of missing values in AGE: ", sum(X['Age'].isnull()))
print("Number of missing values in AGE in test dataset: ", sum(X_Test['Age'].isnull()))
# -------- Change Name -> Title ----------------------------
got= dataset.Name.str.split(',').str[1]
X.iloc[:,1]=pd.DataFrame(got).Name.str.split('\s+').str[1]

got_test = test.Name.str.split(',').str[1]
X_Test.iloc[:,1]=pd.DataFrame(got_test).Name.str.split('\s+').str[1]

# #------------------ Average Age per title -------------------------------------------------------------
ax = plt.subplot()
ax.set_ylabel('Average age')
X.groupby('Name').mean()['Age'].plot(kind='bar',figsize=(13,8), ax = ax)

title_mean_age=[]
title_mean_age.append(list(set(X.Name)))  #set for unique values of the title, and transform into list
title_mean_age.append(X.groupby('Name').Age.mean())
# title_mean_age

title_mean_age_test=[]
title_mean_age_test.append(list(set(X_Test.Name)))
title_mean_age_test.append(X_Test.groupby('Name').Age.mean())
title_mean_age_test[1][7]= title_mean_age[1][13]

#---------------------------------
# print(len(title_mean_age[0])==len(title_mean_age_test[0]))

# ------------------ Fill the missing Ages ---------------------------
n_training= dataset.shape[0]   #number of rows
n_test = test.shape[0]
n_titles= len(title_mean_age[1])
n_titles_test = len(title_mean_age_test[1])
for i in range(0, n_training):
    if np.isnan(X.Age[i])==True:
        for j in range(0, n_titles):
            if X.Name[i] == title_mean_age[0][j]:
#                 print('Row: ' + str(i) + ' = ' + str(X.Age[i]),end='\t-->\t')
                X.Age[i] = title_mean_age[1][j]
#                 print(X.Age[i],end='\t')
#                 print(X.Name[i])
                break
for i in range(0, n_test):
    if np.isnan(X_Test.Age[i])==True:
        for j in range(0, n_titles_test):
            if X_Test.Name[i] == title_mean_age_test[0][j]:
                X_Test.Age[i] = title_mean_age_test[1][j]
                break
                
#--------------------------------------------------------------------    
print("Number of missing values in AGE After: ", sum(X['Age'].isnull()))
print("Number of missing values in AGE in test dataset After: ", sum(X_Test['Age'].isnull()))
X=X.drop(['Name'], axis=1)

X_Test=X_Test.drop(['Name'], axis=1)
# # got
# # #---------------------------------------------------------------------


for i in range(0, n_training):
    if X.Age[i] > 18:
        X.Age[i]= 0
    else:
        X.Age[i]= 1
for i in range(0, n_test):
    if X_Test.Age[i] >18:
        X_Test.Age[i] = 0
    else:
        X_Test.Age[i] = 1
X_Test.head()
#-----------------------Logistic Regression---------------------------------------------
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty='l2',random_state = 0)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)
print("Logistic Regression:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")

#-----------------------------------K-NN --------------------------------------------------

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)
print("K-NN:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")


#---------------------------------------SVM -------------------------------------------------

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)
print("SVM:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")

#---------------------------------Naive Bayes-------------------------------------------

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)
print("Naive Bayes:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")


#----------------------------Random Forest------------------------------------------

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)
print("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std())
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X,y)

y_test = classifier.predict(X_Test)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('titanic.csv', index=False)
