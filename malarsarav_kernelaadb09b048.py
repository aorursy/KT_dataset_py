## Importing the required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
## Step 1: Importing data from source

dataset = pd.read_csv("../input/Pokemon.csv")
## Analyzing the structure and aspects of data

print(dataset.head(5))

print(dataset.shape)

print(dataset.index)

print(dataset.columns)
## Since the dataset is quite minimal, Let's try the alternative method for handling missing values



## Technique 2 - This can be applied on features which has numerical data like year, values etc. This is an approximation which adds variance to the dataset but can avoid loss of data

## It's a standard technique and for this dataset we have mix of both numerical and categorical data.



## Numerical NaN 

## In the given dataset below are the features with numerical values



## Features :[Total,HP,Attack,Defense,Sp.Atk,Sp.Def,Speed]

## Note: Generation is a numerical value but those values are categorical so we are not considering it.



# print(dataset['Total'].mean())

# print(dataset['Total'].tail())



dataset['Total']= dataset['Total'].fillna(dataset['Total'].mean())







## Similar technique to be adopted for other numerical columns



# print(dataset['HP'].mean())

# print(dataset['HP'].tail())



dataset['HP']= dataset['HP'].replace(np.NaN,dataset['HP'].mean())



# print(dataset['Attack'].mean())

# print(dataset['Attack'].tail())



dataset['Attack'] = dataset['Attack'].replace(np.NaN,dataset['Attack'].mean())

# print(dataset['Defense'].mean())

# print(dataset['Defense'].tail())



dataset['Defense'] = dataset['Defense'].replace(np.NaN,dataset['Defense'].mean())

# print(dataset['Sp. Atk'].mean())

# print(dataset['Sp. Atk'].tail())



dataset['Sp. Atk'] = dataset['Sp. Atk'].replace(np.NaN,dataset['Sp. Atk'].mean())

# print(dataset['Sp. Def'].mean())

# print(dataset['Sp. Def'].tail())



dataset['Sp. Def'] = dataset['Sp. Def'].replace(np.NaN,dataset['Sp. Def'].mean())



# print(dataset['Speed'].mean())

# print(dataset['Speed'].tail())



dataset['Speed'] = dataset['Speed'].replace(np.NaN,dataset['Speed'].mean())



print(dataset['Speed'])

print(dataset.isna().any())

## 2. Label Encoding 

##  LabelEncoder encode labels with a value between 0 and n_classes-1 where n is the number of distinct labels. If a label repeats it assigns the same value to as assigned earlier.



from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



dataset1 = dataset

ds = dataset1[['Type 1','Type 2','Generation','Legendary']]

print(dataset['Total'])

#print(ds)

X = ds.iloc[:,:4].values

print(X)

#print(dataset.tail())



#[:,0]=label_encoder.fit_transform(X[:,0])

#print(X)

X[:,1]=label_encoder.fit_transform(X[:,1].astype(str))

X[:,2]=label_encoder.fit_transform(X[:,2])

X[:,3]=label_encoder.fit_transform(X[:,3])



##print(X[:,1])

 

columns = ['Type 1','Type 2','Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Legendary']



Type1 = pd.DataFrame(X[:,0])

Type2 = pd.DataFrame(X[:,1])

Total = pd.DataFrame(dataset['Total'])

HP = pd.DataFrame(dataset['HP'])

Attack = pd.DataFrame(dataset['Attack'])

Defense = pd.DataFrame(dataset['Defense'])

SpAtk = pd.DataFrame(dataset['Sp. Atk'])

SpDef= pd.DataFrame(dataset['Sp. Def'])

Speed= pd.DataFrame(dataset['Speed'])

Generation= pd.DataFrame(X[:,2])

Legendary= pd.DataFrame(X[:,3])



encoded_dataset = pd.DataFrame()

encoded_dataset = pd.concat([encoded_dataset,Type1,Type2,Total,HP,Attack,Defense,SpAtk,SpDef,Speed,Generation,Legendary],axis =1)

encoded_dataset.columns = columns

print(encoded_dataset.columns)

## The problem here is, since there are different numbers in the same column, 

## the model will misunderstand the data to be in some kind of order, 0 < 1 < 2. But this isn’t the case at all. 

## To overcome this problem, we use One Hot Encoder.
## 4. Creating dummies is another method of handling categorical data and it is somewhat similar to one hot encoding 

## Dummy Variables is one that takes the value 0 or 1 to indicate the absence or presence of some categorical effect that may be expected to shift the outcome.

## Number of columns = number of category values



dummy = pd.get_dummies(encoded_dataset['Type 1'])

print(dummy.columns)

tdataset = dataset[['#', 'Name']]

transformed_dataset = pd.concat([tdataset,encoded_dataset],axis = 1)

transformed_dataset = pd.concat([transformed_dataset,dummy],axis =1)

transformed_dataset = transformed_dataset.drop(['Type 1'],axis = 1)



print(transformed_dataset)







## 5. Sometimes, we use KNN Imputation(for Categorical variables): In this method of imputation, 

## the missing values of an attribute are imputed using the given number of attributes that are most similar to the attribute whose values are missing. 

## The similarity of two attributes is determined using a distance function, but we are going to stop our experiment only with dummies.
# 'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',

#        'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',

#        'Psychic', 'Rock', 'Steel', 'Water'

print(transformed_dataset.columns)



## Eliminating the name columns as we have '#' 

X = transformed_dataset[['#','Total','HP','Attack','Defense','Sp. Atk',

       'Sp. Def', 'Speed', 'Generation','Legendary','Bug', 'Dark', 'Dragon',

       'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass',

       'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel',

       'Water']]

y = transformed_dataset[['Type 2']]

y=y.astype('long')

print(X.isna().any())

print(y.isna().any())



print(np.where(y.values >= np.finfo(np.float64).max))
# Import train_test_split function

from sklearn.model_selection import train_test_split

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier



#Create a Gaussian Classifier

clf=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)



y_pred=clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))





from sklearn.metrics import confusion_matrix



conf_mat = confusion_matrix(y_test, y_pred)

print(conf_mat)



conf_mat=np.matrix(conf_mat)

FP = conf_mat.sum(axis=0) - np.diag(conf_mat)  

FN = conf_mat.sum(axis=1) - np.diag(conf_mat)

TP = np.diag(conf_mat)

TN = conf_mat.sum() - (FP + FN + TP)



# Sensitivity, hit rate, recall, or true positive rate

TPR = TP/(TP+FN)

# Specificity or true negative rate

TNR = TN/(TN+FP) 

# Precision or positive predictive value

PPV = TP/(TP+FP)

# Negative predictive value

NPV = TN/(TN+FN)

# Fall out or false positive rate

FPR = FP/(FP+TN)

# False negative rate

FNR = FN/(TP+FN)

# False discovery rate

FDR = FP/(TP+FP)



# Overall accuracy

ACC = (TP+TN)/(TP+FP+FN+TN)



print('ACC',ACC)
## This exercise of work is for demonstrating pre-processing techniques only, The model can give around 50% accuracy for now.

## We got to apply some more data to make it improve it's accuracy as well hyper tuning of parameters in the algorithm.



## The overall problem that the solution covers is to identify type 1 of the pokemon using other features in the dataset.