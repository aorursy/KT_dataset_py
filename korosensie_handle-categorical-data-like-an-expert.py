#Problem---Handle the ordinal data from the datset

#Solution---We will use One_hot_encoding 



#importing numpy and sklearn libraries

import numpy as np

from sklearn.preprocessing import LabelBinarizer,MultiLabelBinarizer



#creating feature

feature=np.array([["Texas"],

                  ["California"],

                  ["Texas"],

                  ["Delaware"],

                  ["Texas"]])



#Creating One-Hot_Encoder

one_hot =LabelBinarizer()



#Encoding One-Hot feature

one_hot.fit_transform(feature)

#To see the classes of one-hot

one_hot.classes_
#Reversing the One-Hot Encoding

one_hot.inverse_transform(one_hot.transform(feature))
#We can use pandas to do one-hot encoding



import pandas as pd



#Creating dummy variables from feature

pd.get_dummies(feature[:,0])
#To handle situation where each observation has multiple classes



#creating multiclass feature

multiclass_feature = [("Texas", "Texas"),

                      ("California", "Alabama"),

                      ("Texas", "Florida"),

                      ("Delware", "Florida"),

                      ("Texas", "Alabama")]



#Creating One-hot encoder object for multiclass

one_hot_multiclass = MultiLabelBinarizer()

one_hot_multiclass.fit_transform(multiclass_feature)
#seeing classes of one-hot encoding

one_hot_multiclass.classes_
#Problem---Encode the ordinal categorical features(whose order is known)

#Solution---We will use pandas replace



#importing library

import pandas as pd



#Creating features

dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})



#creating map

scale_mapper ={"Low":1,

              "Medium":2,

               "High":3}



#Replacing features with values in scale

dataframe["Score"].replace(scale_mapper)
"""When we encode the categorical feature in numerical value.The common strategy 

is to use the dictionary for mapping the categorical values to the numerical values.

The choice of using numerical value for rach categorical feature depends on our prior knowledge."""



dataframe = pd.DataFrame({"Score": ["Low","Low","Medium","Medium","High","Barely More Than Medium"]})

scale_mapper = {"Low":1,

                "Medium":2,

                "Barely More Than Medium": 3,

                "High":4}

dataframe["Score"].replace(scale_mapper)
scale_mapper = {"Low":1,

                "Medium":2,

                "Barely More Than Medium": 2.1,

                "High":3}

dataframe["Score"].replace(scale_mapper)
#Problem---Convert the dictionary into feature matrix

#Solution---We will use sklearn DictVectorizer



#importing sklearn library

from sklearn.feature_extraction import DictVectorizer



#creating dictionary

data_dict =[{"Red":2,"Blue":4},

           {"Red":4,"Blue":3},

           {"Red":1,"Yellow":2},

           {"Red":2,"Yellow":2}]

#creating dictionary vectorizer

dictvectorizer = DictVectorizer(sparse=False)



#Converting dictionary to feature matrix

features =dictvectorizer.fit_transform(data_dict)
#displaying the feature matrix

features
#Getting names of generated feature

feature_names =dictvectorizer.get_feature_names()



#view feature name

feature_names
#it is not neccessary to use dataframe to display the output



#import library

import pandas as pd



#creating dataframe from features

pd.DataFrame(features,columns=feature_names)
# Create word counts dictionaries for four documents

doc_1_word_count = {"Red": 2, "Blue": 4}

doc_2_word_count = {"Red": 4, "Blue": 3}

doc_3_word_count = {"Red": 1, "Yellow": 2}

doc_4_word_count = {"Red": 2, "Yellow": 2}

# Creating  list

doc_word_counts = [doc_1_word_count,

                   doc_2_word_count,

                   doc_3_word_count,

                   doc_4_word_count]

# Converting  list of word count dictionaries into feature matrix

dictvectorizer.fit_transform(doc_word_counts)
#Problem---Fill the missing class of the categorical feature

#Solution---We will use KNN to predict the missing class



#importing libraries

import numpy as np

from sklearn.neighbors import KNeighborsClassifier



#Creating feature matrix with categorical features

X=np.array([[0, 2.10, 1.45],

            [1, 1.18, 1.33],

            [0, 1.22, 1.27],

            [1, -0.21, -1.19]])



#creating feature matrix with missing values

X_with_nan=np.array([[np.nan, 0.87, 1.31],

                     [np.nan, -0.67, -0.22]])



#training the KNN

clf=KNeighborsClassifier(3,weights='distance')

trained_model =clf.fit(X[:,1:],X[:,0])



#predicting missing values

imputed_values =trained_model.predict(X_with_nan[:,1:])



#join column of the predicted class with other features

X_with_imputed =np.hstack((imputed_values.reshape(-1,1),X_with_nan[:,1:]))



#joining two feature matrices

np.vstack((X_with_imputed,X))
X_with_imputed
imputed_values
from sklearn.impute import SimpleImputer



#Joining the two feature matrices

X_complete=np.vstack((X_with_nan,X))



#Loading imputer object

imputer =SimpleImputer(strategy='most_frequent')



#Fitting and transforming matrix

imputer.fit_transform(X_complete)
#Problem---Balance the imbalaced classes

#Solution---First we will create an imbalanced class and then implement strategy.



#importing libraries

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import load_iris



#loading iris data

iris=load_iris()



#creating feature matrix

features=iris.data



#creating target vector

target = iris.target



#removing first 40 observations

features=features[40:,:]

target=target[40:]



#creatinga binary target vector indicating if class 0

target =np.where((target==0),0,1)



#displaying imbalanced target vector

target
#creating weights

weights={0:0.9,1:0.1}



#creating randomforestClassifier with weights

RandomForestClassifier(class_weight=weights)

#we can pass balanced as it will automatically creates weights inversely proportional to class frequencies

#creating randomForestClassifiers with balanced weights

RandomForestClassifier(class_weight="balanced")
#we downsample the majority class to equl to minority  class

#Indices of each class observations

i_class0 = np.where(target==0)[0]

i_class1 = np.where(target==1)[0]



#Number of observations in each class

n_class0 = len(i_class0)

n_class1 = len(i_class1)



#for every observation of class 0 randomly sample from class 1 without replacement

i_class1_downsampled = np.random.choice(i_class1,size=n_class0,replace=False)



# Join together class 0's target vector with the downsampled class 1's target vector

np.hstack((target[i_class0], target[i_class1_downsampled]))

# Join together class 0's feature matrix with the downsampled class 1's feature matrix

np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5]
#we will use upsampling here

# For every observation in class 1, randomly sample from class 0 with replacement

i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

# Join together class 0's upsampled target vector with class 1's target vector

np.concatenate((target[i_class0_upsampled], target[i_class1]))
# Join together class 0's upsampled feature matrix with class 1's feature matrix

np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5]