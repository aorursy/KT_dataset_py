#import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



plt.style.use("fivethirtyeight")
df = pd.read_csv("../input/income-classification/income_evaluation.csv")

df.head()



#fnlwgt referst to final weight (this is the number of people the census believes the entry represents)
#check out the data set

df.shape
df.info()
#Fixing the columns



for x in df.columns:

    x_new = x.strip()

    df=df.rename(columns={x:x_new})



df.columns
data = df.drop(["fnlwgt","capital-gain","capital-loss","native-country"],axis=1)

data
#Remove Whitespace from dataframe



for column in data[["workclass","education","marital-status","occupation","race","sex"]]:

    data[column] = data[column].str.strip()
data.head(6)
from sklearn import preprocessing
#categorical values of workclass

print("Unique values of workclass: ", data["workclass"].unique())

print("Unique values of education: ", data["education"].unique())

print("Unique values of marital status: ", data["marital-status"].unique())

print("Unique values of occupation: ", data["occupation"].unique())

print("Unique values of relationship: ", data["relationship"].unique())

print("Unique values of race: ", data["race"].unique())

print("Unique values of sex: ", data["sex"].unique())
#workclass labels

lb_workclass = preprocessing.LabelEncoder()

lb_workclass.fit(["Private","Self-emp-not-inc","Local-gov","?",

                  "State-gov","Self-emp-inc",

                 "Federal-gov","Without-pay","Never-worked"])

data.iloc[:,1] = lb_workclass.transform(data.iloc[:,1])



#education labels

lb_educ = preprocessing.LabelEncoder()

lb_educ.fit(["HS-grad","Some-college","Bachelors","Masters",

             "Assoc-voc","11th","Assoc-acdm","10th","7th-8th","Prof-school",

             "9th","12th","Doctorate","5th-6th","1st-4th","Preschool"])

data.iloc[:,2] = lb_educ.transform(data.iloc[:,2])



#marriage labels

lb_marry = preprocessing.LabelEncoder()

lb_marry.fit(["Married-civ-spouse","Never-married","Divorced","Separated",

              "Widowed","Married-spouse-absent","Married-AF-spouse"])

data.iloc[:,4] = lb_marry.transform(data.iloc[:,4])



#occupation labels

lb_occ = preprocessing.LabelEncoder()

lb_occ.fit(['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',

       'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',

       'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',

       'Tech-support', '?', 'Protective-serv', 'Armed-Forces',

       'Priv-house-serv'])

data.iloc[:,5] = lb_occ.transform(data.iloc[:,5])



#relationship labels

lb_rel = preprocessing.LabelEncoder()

lb_rel.fit([' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',

       ' Other-relative'])

data.iloc[:,6] = lb_rel.transform(data.iloc[:,6])



#race labels

lb_race = preprocessing.LabelEncoder()

lb_race.fit(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',

       'Other'])

data.iloc[:,7] = lb_race.transform(data.iloc[:,7])



#gender labels

lb_sex = preprocessing.LabelEncoder()

lb_sex.fit(['Male', 'Female'])

data.iloc[:,8] = lb_sex.transform(data.iloc[:,8])
data.head(8)
X=data.iloc[:,:-1]

y=data[["income"]]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)



print("Size of train set:", X_train.shape, y_train.shape)

print("Size of test set: ", X_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier



incomeGuess = DecisionTreeClassifier(criterion="entropy",max_depth=4)

incomeGuess
incomeGuess.fit(X_train,y_train)
predict_income = incomeGuess.predict(X_test)
## Evaluation

from sklearn import metrics

print("Accuracy of decision tree model regarding to income prediction: ", metrics.accuracy_score(y_test,predict_income))
from sklearn.neighbors import KNeighborsClassifier



for i in range(1,11):

    #Train the model

    neigh=KNeighborsClassifier(n_neighbors=i).fit(X_train,np.ravel(y_train))

    y_pred=neigh.predict(X_test)

    acc=metrics.accuracy_score(y_test,y_pred)

    print(i,acc)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

log_r = LogisticRegression(C=0.01,solver="liblinear").fit(X_train,np.ravel(y_train))
y_hat = log_r.predict(X_test)

y_hat
#Evaluation

import math

con_mat = confusion_matrix(y_test,y_hat)

total_accuracy = (con_mat[0, 0] + con_mat[1, 1]) / float(np.sum(con_mat))

class1_accuracy = (con_mat[0, 0] / float(np.sum(con_mat[0, :])))

class2_accuracy = (con_mat[1, 1] / float(np.sum(con_mat[1, :])))

print(con_mat)

print('Total accuracy of income model: %.2f' % total_accuracy)

print('Accuracy "Income more than 50K": %.2f' % class1_accuracy)

print('Accuracy "Income less than 50K": %.2f' % class2_accuracy)

print('Geometric mean accuracy: %.5f' % math.sqrt((class1_accuracy * class2_accuracy)))