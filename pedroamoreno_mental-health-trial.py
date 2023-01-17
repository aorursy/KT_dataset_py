# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Lets import common package needed for the notebook

import sys 

import sklearn

import matplotlib as mpl

import matplotlib.pyplot as plt

#Useless cell since The fetch function cannot be called due to w right (try with a local notebook)

#import os

#import tarfile

#from six.moves import urlib



#download_root="../input/"

#survey_path=os.path.join(download_root,"MentalHealthTrial")

#survey_url=download_root+"survey.csv"



#def fetch_mentalHealthSurvey (url=survey_url,path=survey_path):

 #   if not os.path.isdir(survey_path):

#        os.makedirs(survey_path)

 #fetch_mentalHealthSurvey()       
survey_path=os.path.join("../input","survey.csv")



#note acronym mh is equal to mental_health

mh_data=pd.read_csv(survey_path)

mh_data.head()
mh_data.info()
def df_values(df):

    for i in range(0, len(df.columns)):

        print("*****start of feature ", i, "*************************")

        print (df.iloc[:,i].value_counts())

        print ("*****end of feature ", i, "************************** \n")



df_values(mh_data)
#We create a copy to for cleaning purposes

mh_dataclean=mh_data

#Cleaning "self_employed" feature

mh_dataclean["self_employed"].isnull().value_counts()

mh_dataclean[["self_employed"]]=mh_dataclean[["self_employed"]].fillna("Don't know")

#df_values(mh_dataclean)
#Cleaning feature work_interfere by shifting NaN by "Don't know"

mh_dataclean["work_interfere"].isnull().value_counts()

mh_dataclean[["work_interfere"]]=mh_dataclean[["work_interfere"]].fillna("Don't know")
#Cleaning feature states by shifting NaN by "no state referred"

mh_dataclean["state"].isnull().value_counts()

mh_dataclean[["state"]]=mh_dataclean[["state"]].fillna("no state referred")
mh_dataclean
# Drop values with strange Age



mh_dataclean.drop(mh_dataclean[mh_dataclean["Age"]==-1].index, inplace=True)

mh_dataclean.drop(mh_dataclean[mh_dataclean["Age"]==329].index, inplace=True)

mh_dataclean.drop(mh_dataclean[mh_dataclean["Age"]==-1726].index, inplace=True)

mh_dataclean.drop(mh_dataclean[mh_dataclean["Age"]==99999999999].index, inplace=True)

mh_dataclean.drop(mh_dataclean[mh_dataclean["Age"]==5].index, inplace=True)

mh_dataclean.drop(mh_dataclean[mh_dataclean["Age"]==8].index, inplace=True)

mh_dataclean.drop(mh_dataclean[mh_dataclean["Age"]==11].index, inplace=True)

mh_dataclean.loc[mh_dataclean["Age"]==-29, "Age"]=29

mh_dataclean.info()

#*********************************************************

#other alternative, not consider subjects under 16

#mh_dataclean=mh_dataclean[mh_dataclean["Age"]>16 && mh_dataclean["Age"]<100 ]

mh_dataclean=mh_dataclean.reset_index()
mh_dataclean.info()
#drop comments feature

mh_dataclean=mh_dataclean.drop("comments", axis=1)

#drop timestamp feature

mh_dataclean=mh_dataclean.drop("Timestamp", axis=1)

#mh_dataclean.info()
#Cleaning up the male Gender

#mh_dataclean["Gender"].value_counts()

mh_dataclean.loc[mh_dataclean["Gender"]=="male","Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="M", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="m", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Make", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Cis Male", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Man", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="ostensibly male, unsure what that really means", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Male-ish", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Cis Man", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="msle", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Male (CIS)", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Mal", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="maile", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Guy (-ish) ^_^", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Mail", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="male leaning androgynous", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Malr", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="cis male", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="something kinda male?", "Gender"]="Male"

mh_dataclean.loc[mh_dataclean["Gender"]=="Male ", "Gender"]="Male"

mh_dataclean["Gender"].value_counts()

#df_values(mh_dataclean)
#Cleaning up female gender

mh_dataclean.loc[mh_dataclean["Gender"]=="female", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="F", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="f", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="Woman", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="Female (trans)", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="woman", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="cis-female/femme", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="Trans-female", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="Trans woman", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="Femake", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="femail", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="Female (cis)", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="Cis Female", "Gender"]="Female"

mh_dataclean.loc[mh_dataclean["Gender"]=="Female ", "Gender"]="Female"

mh_dataclean["Gender"].value_counts()

#df_values(mh_dataclean)
#Cleaning up other gender option

mh_dataclean.loc[(mh_dataclean["Gender"]!="Male")&(mh_dataclean["Gender"]!="Female"), "Gender"]="Other"

mh_dataclean["Gender"].value_counts()

df_values(mh_dataclean)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder



#First lets encoded categorical features with ordina encoder.

mh_dataclean_encoded=mh_dataclean

#Encoding Country

country_1h_encoder=OrdinalEncoder()

mh_dataclean_encoded.Country=country_1h_encoder.fit_transform(mh_dataclean_encoded[["Country"]])

#mh_dataclean_encoded=pd.DataFrame(country_cat_1hot, columns="Country")

mh_dataclean_encoded

#Encoding state

state_encoder=OrdinalEncoder()

mh_dataclean_encoded.state=state_encoder.fit_transform(mh_dataclean_encoded[["state"]])



#Encoding work_interfere

workinterfere_enconder=OrdinalEncoder(categories=[["Don't know","Never","Rarely","Sometimes","Often"]])

mh_dataclean_encoded["work_interfere"]=workinterfere_enconder.fit_transform(mh_dataclean_encoded[["work_interfere"]])



#Encoding no_employees

noemployees_encoder=OrdinalEncoder(categories= [["1-5","6-25","26-100","100-500","500-1000","More than 1000"]])

mh_dataclean_encoded["no_employees"]=noemployees_encoder.fit_transform(mh_dataclean_encoded[["no_employees"]])



mh_dataclean_encoded.info()



#Encoding gender

#gender_1hot_encoder=OneHotEncoder()

#gender_cat_1hot=gender_1hot_encoder.fit_transform(mh_dataclean_encoded[["Gender"]])

#gender_encoded=pd.DataFrame(gender_cat_1hot.toarray(),columns=["Male_gender","Female_gender", "Other_gender"])

#mh_dataclean_encoded=mh_dataclean_encoded.drop("Gender", axis=1)

#mh_dataclean_encoded["Male_gender","Female_gender", "Other_gender"]= gender_encoded



#Encoding self_employed

#selfemp_1hot_encoder=OneHotEncoder()

#selfemp_cat_1hot=selfemp_1hot_encoder.fit_transform(mh_dataclean_encoded[["self_employed"]])

#selfemp_encoded=pd.DataFrame(selfemp_cat_1hot.toarray(),columns=["No_selfemployed","Yes_selfemployed", "Dontknow_selfemployed"])

#mh_dataclean_encoded=mh_dataclean_encoded.drop("self_employed", axis=1)

3#mh_dataclean_encoded["No_selfemployed","Yes_selfemployed", "Dontknow_selfemployed"]=selfemp_encoded

#mh_dataclean_encoded=pd.merge(mh_dataclean_encoded, selfemp_encoded)
mh_dataclean_encoded_onehot=mh_dataclean_encoded.drop("Age",axis=1)

mh_dataclean_encoded_onehot=mh_dataclean_encoded_onehot.drop("Country",axis=1)

mh_dataclean_encoded_onehot=mh_dataclean_encoded_onehot.drop("state",axis=1)

mh_dataclean_encoded_onehot=mh_dataclean_encoded_onehot.drop("work_interfere",axis=1)

mh_dataclean_encoded_onehot=mh_dataclean_encoded_onehot.drop("no_employees",axis=1)

from sklearn.feature_extraction import DictVectorizer

mh_dict = mh_dataclean_encoded_onehot.to_dict(orient='records') # turn each row as key-value pairs

vec=DictVectorizer(sparse=True, dtype=int)

mh_one_hot=vec.fit_transform(mh_dict)

mh_dataclean_encoded_onehot=pd.DataFrame(mh_one_hot.toarray(), columns=vec.get_feature_names())

mh_dataclean_encoded_onehot
mh_dataclean_encoded_ordinal=mh_dataclean_encoded[["Age","Country","state","work_interfere","no_employees","treatment"]]

mh_dataclean_encoded=pd.concat([mh_dataclean_encoded_ordinal,mh_dataclean_encoded_onehot],axis=1)

mh_dataclean_encoded
#The target class in the data set is the column treatment, thus let apply LabelEncoder for 1= treatment equals of yes; and 0= treatment equals of no



from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

mh_dataclean_encoded["treatment"] = labelencoder.fit_transform(mh_dataclean_encoded["treatment"])

mh_dataclean_encoded
from sklearn.model_selection import train_test_split

train_set,test_set=train_test_split(mh_dataclean_encoded, test_size=0.2, random_state=3)

X_train_set=train_set.drop("treatment", axis=1)

X_test_set=test_set.drop("treatment", axis=1)

y_train_set=train_set["treatment"].copy()

y_test_set=test_set["treatment"].copy()



#Further, it could be convenient to make a stratified split
X_train_set
#Option 1.1: At this time we apply a SGDClassifier without considering cross_val and the scaler

from sklearn.linear_model import SGDClassifier

classifier_withoutscale=SGDClassifier(random_state=3)

classifier_withoutscale.fit(X_train_set, y_train_set)

#Lets predict only one sample

one_sample=X_train_set.iloc[6,:]

print("Predicting 6th sample: ", classifier_withoutscale.predict([one_sample]))

#lets predict the entire train_set

y_predict=classifier_withoutscale.predict(X_train_set)

#see the metrics to compare with ML options taken below

from sklearn.metrics import precision_score, recall_score, confusion_matrix

print("************Metrics results of SGDclassifier without cross_val and scaler")

print("confusion matrix: ", confusion_matrix(y_train_set, y_predict))

print("precision score: ", precision_score(y_train_set, y_predict))

print("recall score: ", recall_score(y_train_set, y_predict))

#Option 1.2: Applying cross validation

from sklearn.model_selection import cross_val_predict

y_train_pred_cv=cross_val_predict(classifier_withoutscale, X_train_set, y_train_set, cv=5)

from sklearn.metrics import precision_score, recall_score, confusion_matrix

print("************Metrics results of SGDclassifier with cross_val and without scaler")

print("confusion matrix: ", confusion_matrix(y_train_set, y_train_pred_cv))

print("precision score: ", precision_score(y_train_set, y_train_pred_cv))

print("recall score: ", recall_score(y_train_set, y_train_pred_cv))
#Option 1.3: Scaling the training set and cross val

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_set_scaled_data = pd.DataFrame(scaler.fit_transform(X_train_set), columns=X_train_set.columns)

classifier_scaled=SGDClassifier(random_state=5)

classifier_scaled.fit(X_train_set_scaled_data, y_train_set)

y_predict_cv_scaled=cross_val_predict(classifier_scaled, X_train_set_scaled_data, y_train_set,cv=5)

print("************Metrics results of SGDclassifier with cross_val and without scaler")

print("confusion matrix: ", confusion_matrix(y_train_set, y_predict_cv_scaled))

print("precision score: ", precision_score(y_train_set, y_predict_cv_scaled))

print("recall score: ", recall_score(y_train_set, y_predict_cv_scaled))
from sklearn.linear_model import LogisticRegression

logreg_classifier=LogisticRegression()

logreg_classifier.fit(X_train_set, y_train_set)

#Lets predict only one sample

one_sample=X_train_set.iloc[6,:]

print("Predicting 6th sample: ", logreg_classifier.predict([one_sample]))#the prediction is the opposite of the treatment value

#lets predict the entire train_set

y_predict_log=logreg_classifier.predict(X_train_set)

#see the metrics to compare with ML options taken below

print("************Metrics results of logclassifier without cross_val and scaler")

print("confusion matrix: ", confusion_matrix(y_train_set, y_predict_log))

print("precision score: ", precision_score(y_train_set, y_predict_log))

print("recall score: ", recall_score(y_train_set, y_predict_log))
#Option 2.2: Applying cross validation

from sklearn.model_selection import cross_val_predict

y_train_pred_log_cv=cross_val_predict(logreg_classifier, X_train_set, y_train_set, cv=5)

print("************ Metrics results of logclassifier with cross_val")

print("confusion matrix: ", confusion_matrix(y_train_set, y_train_pred_log_cv))

print("precision score: ", precision_score(y_train_set, y_train_pred_log_cv))

print("recall score: ", recall_score(y_train_set, y_train_pred_log_cv))
y_test_pred=logreg_classifier.predict(X_test_set)

print("************ Metrics results of logclassifier with cross_val applied to test_set")

print("confusion matrix: ", confusion_matrix(y_test_set, y_test_pred))

print("precision score: ", precision_score(y_test_set, y_test_pred))

print("recall score: ", recall_score(y_test_set, y_test_pred))