# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



df_data = pd.read_csv('/kaggle/input/titanic/train.csv')



df_data.head(20)
#Check missing data from columns



print(df_data['Pclass'].isnull().sum())

print(df_data['Sex'].isnull().sum())

print(df_data['SibSp'].isnull().sum())

print(df_data['Parch'].isnull().sum())

print(df_data['Ticket'].isnull().sum())

print(df_data['Fare'].isnull().sum())

print(df_data['Embarked'].isnull().sum())

print(df_data['Name'].isnull().sum())
df_data.Cabin.fillna("", inplace = True) 

df_data.Age.fillna(np.mean(df_data.Age), inplace = True) 

df_data.Embarked.fillna("N", inplace = True) 



df_data.head(20)

def num_of_cabins(cabins):

    if len(cabins) == 0:

        return 0

    else:

        return len(cabins.split(' '))



def cabin_class(cabins):

    if len(cabins) == 0:

        return 'Z'

    else:

        return cabins[0]



df_data['No_Cabins'] = df_data.Cabin.apply(num_of_cabins)

df_data['Cabin_class'] = df_data.Cabin.apply(cabin_class)

df_data.head(20)

        
#seperate the names, by making everything lower case, and making use the fact that the names are formatted as "lastname, title. firstnames"



def cleanname(name):

    name = name.lower()

    name = name.replace("(", "")

    name = name.replace(")", "")

    return name

   



def lastname(name):

    return name.split(',')[0]



def title(name):

    return name.split(",")[1].split(" ")[1]



def firstname(name):

    return name.split(",")[1].split(" ")[2]

    

df_data['Name'] = df_data.Name.apply(cleanname)

df_data['Last_Name'] = df_data.Name.apply(lastname)

df_data['First_Name'] = df_data.Name.apply(firstname)

df_data['Title'] = df_data.Name.apply(title)



df_data.head(20)
#Let's see the how many unique values are there in the names and the title



print(df_data['Last_Name'].describe())

print(df_data['First_Name'].describe())

print(df_data['Title'].describe())
df_data['Last_Name_Unique'] = df_data['Last_Name'].is_unique

df_data['First_Name_Unique'] = df_data['First_Name'].is_unique

df_data['First_Name_Unique'].describe()
#Extract and group headers by turning everything into lower case, as well as removing punctuations and spaces

#Tickets with no headers are assigned a zero length string header



import re

import string



def ticket_header(ticket):

    ticket_components = ticket.split(" ")

    if  len(ticket_components) > 1:

        header = ticket_components[0].lower()

        header = re.sub('[%s]' % re.escape(string.punctuation), " ", header)

        return header.replace(" ", "")

    else:

        return ""



def ticket_number(ticket):

    ticket_components = ticket.split(" ")

    if  len(ticket_components) > 1:

        try:

            return int(ticket_components[-1])

        except ValueError:

            return -1

    else:

        try:

            return int(ticket_components[0])

        except ValueError:

            return -1



df_data['Ticket_Header'] = df_data.Ticket.apply(ticket_header)

df_data['Ticket_number'] = pd.to_numeric(df_data.Ticket.apply(ticket_number))



#Store all unique header - for use later

unique_headers = df_data.Ticket_Header.unique()



print(unique_headers)
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split





Y = df_data['Survived']

X = df_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'No_Cabins', 'Cabin_class', 'Title', 'Ticket_Header', 'Ticket_number']]





title_le = LabelEncoder()

sex_le = LabelEncoder()

cabin_class_le = LabelEncoder()

ticket_le = LabelEncoder()

Embarked_le = LabelEncoder()

sc = StandardScaler()





title_le.fit(X['Title'])

sex_le.fit(X['Sex'])

cabin_class_le.fit(X['Cabin_class'])

ticket_le.fit(X['Ticket_Header'])

Embarked_le.fit(X['Embarked'])

X['Title'] = title_le.transform(X['Title'])

X['Sex'] = sex_le.transform(X['Sex'])

X['Cabin_class'] = cabin_class_le.transform(X['Cabin_class'])

X['Ticket_Header'] = ticket_le.transform(X['Ticket_Header'])

X['Embarked'] = Embarked_le.transform(X['Embarked'])

sc.fit(X)

X = sc.transform(X)



X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.25, random_state=1079)





import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from sklearn.metrics import accuracy_score



#Model 1: XGB model (This model gives a leaderboard score of 0.76315)

xgb_model = XGBClassifier(learning_rate = 0.1, n_estimators = 500, max_depth = 5, min_child_weight = 1, gamma = 0, subsample = 0.8, colsample_bytree = 0.8, objective = 'binary:logistic', nthread = 4, scale_pos_weight = 1, seed = 323, verbosity = 0)



#Wrap parameters

xgb_param = xgb_model.get_xgb_params()



#Wrap data into XGB DMatrix formate

xgtrain = xgb.DMatrix(X_train, label = y_train)



#XGB CV module to obtain optimal n_estimator (i.e. number of boosting rounds)

cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = xgb_model.get_params()['n_estimators'], nfold = 5, metrics = 'error', early_stopping_rounds = 50)



#set XGB Model with the determined n_estimator

xgb_model.set_params(n_estimators = cvresult.shape[0])



xgb_model.fit(X_train, y_train, eval_metric = 'error')



y_pred = xgb_model.predict(X_train)

y_pred_v = xgb_model.predict(X_valid)

print("XGB Model Accuracy, training data:", accuracy_score(y_train, y_pred))

print("XGB Model Accuracy, validation data:", accuracy_score(y_valid, y_pred_v))

#Random Forest Model - n_estimator tuned using validation data

from sklearn.ensemble import RandomForestClassifier



X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.25, random_state=2732)



clf_rf = RandomForestClassifier(n_estimators = 30)

clf_rf.fit(X_train, y_train)

y_pred_rf = clf_rf.predict(X_train)

y_pred_rf_v = clf_rf.predict(X_valid)

print("Random Forest Accuracy - Training data:", accuracy_score(y_train, y_pred_rf))

print("Random Forest Accuracy - Validation data:", accuracy_score(y_valid, y_pred_rf_v))
#Logistic Regression Model

from sklearn.linear_model import LogisticRegression



X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.25, random_state=1024)



clf_LR = LogisticRegression()

clf_LR.fit(X_train, y_train)

y_pred_LR = clf_LR.predict(X_train)

y_pred_LR_v = clf_LR.predict(X_valid)

print("Logistic Regression Accuracy - Training data:", accuracy_score(y_train, y_pred_LR))

print("Logistic Regression Accuracy - Validation data:", accuracy_score(y_valid, y_pred_LR_v))
#State Vector Machine Classifier Model



from sklearn.svm import SVC



X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.25, random_state=27)



clf_SVC = SVC(probability = True)

clf_SVC.fit(X_train, y_train)

y_pred_SVC = clf_SVC.predict(X_train)

y_pred_SVC_v = clf_SVC.predict(X_valid)

print("State Vector Machine Accuracy - Training data:", accuracy_score(y_train, y_pred_SVC))

print("State Vector Machine Accuracy - Validation data:", accuracy_score(y_valid, y_pred_SVC_v))
#K nearest Neighbour, n_neighbors tuned using validation data



from sklearn.neighbors import KNeighborsClassifier



X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.25, random_state=323)



clf_knn = KNeighborsClassifier(n_neighbors=12)

clf_knn.fit(X_train, y_train)

y_pred_knn = clf_knn.predict(X_train)

y_pred_knn_v = clf_knn.predict(X_valid)

print("KNN Accuracy - Training data:", accuracy_score(y_train, y_pred_knn))

print("KNN Accuracy - Validation data:", accuracy_score(y_valid, y_pred_knn_v))





#Ensemble model by voting - average probability from each model (for positive class) is averaged, 

#and a final prediction of positive is given if the probability average is greater than 0.6



X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.25, random_state=17430)



y_pred = ((xgb_model.predict_proba(X_valid)+ clf_rf.predict_proba(X_valid)+ clf_SVC.predict_proba(X_valid) + clf_knn.predict_proba(X_valid) + clf_LR.predict_proba(X_valid))[:,1])/5 > 0.6

print("Ensemble model accuracy: validation data:", accuracy_score(y_valid, y_pred))

def replace_ticket_header(text, unique_headers):

    if text not in unique_headers:

        return ""

    else: 

        return text





df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



df_test.Cabin.fillna("", inplace = True) 

df_test.Age.fillna(np.mean(df_data.Age), inplace = True) 

df_test.Embarked.fillna("N", inplace = True) 

df_test['No_Cabins'] = df_test.Cabin.apply(num_of_cabins)

df_test['Cabin_class'] = df_test.Cabin.apply(cabin_class)

df_test['Name'] = df_test.Name.apply(cleanname)

df_test['Last_Name'] = df_test.Name.apply(lastname)

df_test['First_Name'] = df_test.Name.apply(firstname)

df_test['Title'] = df_test.Name.apply(title)

df_test['Ticket_Header'] = df_test.Ticket.apply(ticket_header)

df_test['Ticket_number'] = pd.to_numeric(df_test.Ticket.apply(ticket_number))





df_test['Title'] = df_test['Title'].apply(lambda x: re.sub("dona.", "ms.", x))

df_test['Ticket_Header'] = df_test['Ticket_Header'].apply(lambda x: replace_ticket_header(x, unique_headers))



df_test['Title'] = title_le.transform(df_test['Title'])

df_test['Sex'] = sex_le.transform(df_test['Sex'])

df_test['Cabin_class'] = cabin_class_le.transform(df_test['Cabin_class'])

df_test['Ticket_Header'] = ticket_le.transform(df_test['Ticket_Header'])

df_test['Embarked'] = Embarked_le.transform(df_test['Embarked'])



df_test.Fare.fillna(np.mean(df_data.Fare), inplace = True) 





X_test = df_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'No_Cabins', 'Cabin_class', 'Title', 'Ticket_Header', 'Ticket_number']]

df_sub = df_test[['PassengerId']]





X_test = sc.transform(X_test)











y_pred = ((xgb_model.predict_proba(X_test)+ clf_rf.predict_proba(X_test)+ clf_SVC.predict_proba(X_test) + clf_knn.predict_proba(X_test) + clf_LR.predict_proba(X_test))[:,1])/5 > 0.6

df_sub['Survived'] = y_pred*1







df_sub.to_csv("submission.csv", index = False)
