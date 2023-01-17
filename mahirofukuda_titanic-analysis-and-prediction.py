import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split as t_t_s

data_train = pd.read_csv("../input/titanic/train.csv")
data_test = pd.read_csv("../input/titanic/test.csv")
#summary
print(data_train.head(5)) #train.csv
print(data_test.head(5)) #test.csv
#category(survived) plot age vs fare
#classify survived data and dead data
grouped_data = data_train.groupby("Survived")
data_train_survived = grouped_data.get_group(1)
data_train_dead = grouped_data.get_group(0)

plt.scatter(data_train_survived["Age"],data_train_survived["Fare"],c="r",label="survived")
plt.scatter(data_train_dead["Age"],data_train_dead["Fare"],c="b",label="dead")
plt.title("Age vs Fare", fontsize=15)
plt.xlabel("Age",fontsize=10)
plt.ylabel("Fare",fontsize=10)
plt.legend()
#flags
male_s_bool = ((data_train["Sex"] == "male") & (data_train["Survived"]==1)) #male & survived
male_d_bool = ((data_train["Sex"] == "male") & (data_train["Survived"]==0)) #male & dead
female_s_bool = ((data_train["Sex"] == "female") & (data_train["Survived"]==1)) #female & survived
female_d_bool = ((data_train["Sex"] == "female") & (data_train["Survived"]==0)) #female & dead

#count the data
count_s_male = male_s_bool.values.sum()
count_d_male = male_d_bool.values.sum()
count_s_female = female_s_bool.values.sum()
count_d_female = female_d_bool.values.sum()

#gather data
entry = ["male-survived", "female-survived","male-dead","female-dead"] #name
count = [count_s_male,count_s_female,count_d_male,count_d_female] #count
color_list = ["green","orange","green","orange"] #color

#Barplot
plt.bar(entry,count,color=color_list)
#revise data( Sex → SexID )
data_train.loc[data_train['Sex'] == "male", 'Sex'] = 0
data_train.loc[data_train['Sex'] == "female", 'Sex'] = 1

#name list
x_column_list=["Sex","Fare"]
y_column_list=["Survived"]

logit=LogisticRegression() 
#split data for trainning and test
X_train, X_test, y_train, y_test = t_t_s(data_train[x_column_list], data_train[y_column_list], test_size=0.2)

logit.fit(X_train,y_train) #learning
y_pred = logit.predict(X_test) #prediction

#calculate accuracy
print(accuracy_score(y_test, y_pred))

#revise data( Age → GrowingID )
data_train.loc[data_train['Age']>=61,'GrowingID'] = 4 #old 
data_train.loc[(data_train['Age']>=26)&(data_train['Age']<61),"GrowingID"] = 3
data_train.loc[(data_train['Age']>=16)&(data_train['Age']<26),'GrowingID'] = 2 
data_train.loc[(data_train['Age']>= 1)&(data_train['Age']<16),"GrowingID"]=1

data_train.head(5) # summary
#flags
old_bool = (data_train["GrowingID"] == 4)
adult_bool = (data_train["GrowingID"] == 3)
young_bool = (data_train["GrowingID"] == 2)
child_bool = (data_train["GrowingID"] == 1)

#count the data
count_old = old_bool.values.sum()
count_adult = adult_bool.values.sum()
count_young = young_bool.values.sum()
count_child = child_bool.values.sum()

count_grow = [count_old,count_adult,count_young,count_child]
label_grow = ["Old","Adult","Young","Child"]

#pie plot
plt.pie(count_grow, labels=label_grow)
#replace NaN into 3(means adult)
data_train["GrowingID"] = data_train["GrowingID"].fillna(3)

data_train.head(20)
#name list
x_column_list=["Sex","Fare","GrowingID"]
y_column_list=["Survived"]

logit=LogisticRegression() 
#split data for trainning and test
X_train, X_test, y_train, y_test = t_t_s(data_train[x_column_list], data_train[y_column_list], test_size=0.2)

logit.fit(X_train,y_train) #learning
y_pred = logit.predict(X_test) #prediction

#calculate accuracy
print(accuracy_score(y_test, y_pred))
logit2=LogisticRegression() 

#make model
X_train, y_train = (data_train[x_column_list], data_train[y_column_list])

logit2.fit(X_train,y_train) #learning

#prepare test data
#revise data( Sex → SexID )
data_test.loc[data_test['Sex'] == "male", 'Sex'] = 0
data_test.loc[data_test['Sex'] == "female", 'Sex'] = 1

#revise data( Age → GrowingID )
data_test.loc[data_test['Age']>=61,'GrowingID'] = 4 #old 
data_test.loc[(data_test['Age']>=26)&(data_test['Age']<61),"GrowingID"] = 3
data_test.loc[(data_test['Age']>=16)&(data_test['Age']<26),'GrowingID'] = 2 
data_test.loc[(data_test['Age']>= 1)&(data_test['Age']<16),"GrowingID"]=1

#replace NaN into 3(means adult)
data_test["GrowingID"] = data_test["GrowingID"].fillna(3)

#replace NaN into 10
data_test["Fare"] = data_test["Fare"].fillna(10)

print(data_test.info())

X_test = data_test[x_column_list]

#Prediction
y_pred = logit2.predict(X_test) 
data_test = data_test.assign(Y_pred=y_pred)
print(y_pred)

#write into submission file
data_submission = data_test[["PassengerId","Y_pred"]]
data_submission.to_csv("submission2.csv")
#flags
pclass1_s_bool = ((data_train["Pclass"] == 1) & (data_train["Survived"]==1)) 
pclass1_d_bool = ((data_train["Pclass"] == 1) & (data_train["Survived"]==0))
pclass2_s_bool = ((data_train["Pclass"] == 2) & (data_train["Survived"]==1)) 
pclass2_d_bool = ((data_train["Pclass"] == 2) & (data_train["Survived"]==0))
pclass3_s_bool = ((data_train["Pclass"] == 3) & (data_train["Survived"]==1)) 
pclass3_d_bool = ((data_train["Pclass"] == 3) & (data_train["Survived"]==0))

#count the data
count_pclass1_s = pclass1_s_bool.values.sum()
count_pclass1_d = pclass1_d_bool.values.sum()
count_pclass2_s = pclass2_s_bool.values.sum()
count_pclass2_d = pclass2_d_bool.values.sum()
count_pclass3_s = pclass3_s_bool.values.sum()
count_pclass3_d = pclass3_d_bool.values.sum()

#gather data
entry = ["PClass1 S", "PClass1 D","PClass2 S", "PClass2 D","PClass3 S", "PClass3 D"] #name
count = [count_pclass1_s,count_pclass1_d,
         count_pclass2_s,count_pclass2_d,
         count_pclass3_s,count_pclass3_d,] #count
c_list = ["green","orange","green","orange","green","orange"] #color

#Barplot
plt.bar(entry,count,color=c_list)
#draw boxplot: Parch by Survived
sns.boxplot(x=data_train["Survived"],y=data_train["Parch"])
plt.title("Parch by Survived")
plt.show()
from sklearn.preprocessing import StandardScaler

#scale data
scaler = StandardScaler()
data_train_std = scaler.fit_transform(data_train[["Sex","Fare","Pclass","GrowingID","Parch"]])

#Logistic Regression
y_column_list=["Survived"]

logit3=LogisticRegression() 
#split data for trainning and test
X_train, X_test, y_train, y_test = t_t_s(data_train_std, data_train[y_column_list], test_size=0.2)

logit3.fit(X_train,y_train) #learning
y_pred = logit3.predict(X_test) #prediction

#calculate accuracy
print(accuracy_score(y_test, y_pred))
#make model
X_train, y_train = (data_train_std, data_train[y_column_list])
logit4=LogisticRegression()
logit4.fit(X_train,y_train) #learning

#Scale data_test
data_test_std = scaler.fit_transform(data_test[["Sex","Fare","Pclass","GrowingID","Parch"]])

X_test = data_test_std

#Prediction
y_pred = logit4.predict(X_test) 
data_test = data_test.assign(Survived=y_pred)
print(y_pred)

#write into submission file
data_submission = data_test[["PassengerId","Survived"]]
data_submission.to_csv("submission4.csv")