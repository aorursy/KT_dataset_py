# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading the training file

training_text_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_text",sep='\|\|', header=None,skiprows=1,names=["ID","Text"])
#checking the training text data 

training_text_data.head()

#reading the training_variants file

training_variants_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_variants")

#checking the training variant data 

training_variants_data.head()
#merging training text data and training variant data

total_data=pd.merge(left=training_text_data,

    right=training_variants_data,

    how='inner',

    on="ID",)

#checking total merged data

total_data.head()
#anaysing if we can see all Gene values present in Text column in our data



print(total_data['Text'][0])

print(total_data['Gene'][0])

#from the Below , we can see that all the Gene column values are present in Text data column,Hence I am ignoring Gene Column
#checking for null values on our data

total_data.isnull().sum()
len(total_data)
#imputing gene row value to null text rows as for all other columns, Gene values are present in Text data

total_data['Text'] = total_data.apply(lambda row: row['Gene'] if pd.isnull(row['Text']) else row['Text'],

    axis=1

)
#we can see  no missing values in our data

total_data.isnull().sum()
#checking the shape of our data

total_data.shape
total_data.info()
#taking class column as dependent variable ie which needs to be find out from all other columns in our data

y=total_data.Class
#taking Text and Variation as independent variables or Predictive columns to train the data

X=total_data[["Text","Variation",]]
X.head()
# splitting into test and train

from sklearn.model_selection  import train_test_split

from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train.head()
# vectorizing the sentences; removing stop words

from sklearn.feature_extraction.text import CountVectorizer



# Definig vectorizing object for Text column

vect_text= CountVectorizer(stop_words ='english')



#Defining vectorizing object for Variation column

vect_variation= CountVectorizer(stop_words ='english')
#vectorizing  for Text column which gives the count of repeated words for each row

vect_text.fit(X_train["Text"])

vect_text.fit(X_test["Text"])
#vectorizing for Variation column  which gives the count of repeated words for each row

vect_variation.fit(X_train["Variation"])

vect_variation.fit(X_test["Variation"])
#transforming count of Variation words in to matrix

variation_tranform_train=vect_variation.transform(X_train["Variation"])

variation_tranform_test=vect_variation.transform(X_test["Variation"])
#transforming count of Text words in to matrix

text_transformed_train= vect_text.transform(X_train["Text"])

text_transformed_test=vect_text.transform(X_test["Text"])
#merging train data of two Matrix horixzontally to train the model

import scipy.sparse as sp

x_train_final = sp.hstack((variation_tranform_train,text_transformed_train))
print(x_train_final.shape)
#merging test data of two Matrix horixzontally to train the model

import scipy.sparse as sp

x_test_final = sp.hstack((variation_tranform_test,text_transformed_test))
print(x_test_final.shape)
# Let's run Linear SVM model using the selected variables

from sklearn import metrics

from sklearn import svm

from sklearn.metrics import classification_report

svc_model=svm.LinearSVC()

svc_model.fit(x_train_final,y_train)



#predicting the Test data using our trained Linear SVM model

y_pred_class = svc_model.predict(x_test_final)



print(classification_report(y_test, y_pred_class))
# Let's run Logistic regression model using the selected variables

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logsk = LogisticRegression()

logsk.fit(x_train_final,y_train)

y_pred_class = logsk.predict(x_test_final)

print(classification_report(y_test, y_pred_class))

#reading stage 2 test data

test_text_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/stage2_test_text.csv",sep='\|\|', header=None,skiprows=1,names=["ID","Text"])

test_variant_data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/stage2_test_variants.csv")
#checking for any missing values in test data

test_text_data.isnull().sum()
test_variant_data.isnull().sum()
#merging two tables of stage 2 test data

test_total_data=pd.merge(left=test_text_data,

    right=test_variant_data,

    how='inner',

    on="ID",)

test_total_data.head()

len(test_total_data)
test_total_data.head()
#generating matrix with the vectorizor object used for training data so that we get same number columns for test data as in Training data

final_variation_tranform_test=vect_variation.transform(test_total_data["Variation"])

final_text_transformed_test=vect_text.transform(test_total_data["Text"])



#concatinating two columns data.

import scipy.sparse as sp

x_train_final_submission = sp.hstack((final_variation_tranform_test,final_text_transformed_test))



#checking the shapes of stage 2 test data and train data

print("shape of train data",x_train_final.shape)

print("shape of stage 2 test data",x_train_final_submission.shape)
#building the Logistic regression model for predicting stage 2 test as it is giving more accuracy compared ot SVM as shown above.

logsk_final = LogisticRegression()

logsk_final.fit(x_train_final,y_train)
#predicting the stage 2 test data

y_pred_test = logsk_final.predict_proba(x_train_final_submission)

print(y_pred_test)
#convert the predicted columns in to data frame

y_pred_test=pd.DataFrame(y_pred_test)  

y_pred_test.head(10)
#renaming the columns

y_pred_test.rename(columns={0:'class1',1:'class2',2:'class3',3:'class4',4:'class5',5:'class6',6:'class7',7:'class8',8:'class9'}, 

                 inplace=True)
y_pred_test.head()
test_total_data.head()
y_pred_test.head()

Submission_File=pd.concat([test_total_data,y_pred_test],axis=1)

Submission_File.head()
#dropiing all other coulumns except predicted class column

Submission_File=Submission_File.drop(columns=["Text","Gene","Variation"])

Submission_File.head()
#converting ID column in to int type

Submission_File["ID"]=Submission_File["ID"].astype(int)
Submission_File.tail()
#converting the data frame in to csv file

Submission_File.to_csv('Submission_File',sep=',',header=True,index=None)
Submission_File.to_csv(r'Submission_File.csv',index=False)
#generating or exporting the file for submission 

from IPython.display import FileLink

FileLink(r'Submission_File.csv')