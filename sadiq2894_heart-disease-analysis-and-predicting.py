import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

heart=pd.read_csv('../input/heart.csv')
# To see the first top five rows in our uploaded data #

heart.head()
# To see the Last five rows in our uploaded data #

heart.tail()
# Now we will look at summary statistics of our data #

# summary statistics are used to summarize a set of observations, in order to communicate the largest amount of information as simply as possible #



heart.describe()
#   To get a concise summary of the dataframe #

heart.info()
# We will list all the columns in our loaded dataset #

heart.columns
# Now we will see how many rows and columns are present in our loaded dataset #

heart.shape

# To find the how many missing values in our data #

heart.isnull().sum()
heart.target.value_counts()
sns.countplot(x="target",data=heart)
heart.age.value_counts()[:15]
sns.barplot(x=heart.age.value_counts()[:15].index,y=heart.age.value_counts()[:15].values)

plt.xlabel('Age')

plt.ylabel('Age Count')

plt.title('Age Analysis System')

plt.show()
heart.sex.value_counts()
sns.countplot(x='sex', data=heart)

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
male_disease=heart[(heart.sex==1) & (heart.target==1)]          ## Here we have sex=1(male) and target =1(have disease)

male_NO_disease=heart[(heart.sex==1) & (heart.target==0)]       ## Here we have sex=1(male) and target =0(have no disease )

print(len(male_disease),"male_disease")

print(len(male_NO_disease),"male_NO_disease")
a=len(male_disease)

b=len(male_NO_disease)

sns.barplot(x=['male_disease ','male_NO_disease'],y=[a,b])

plt.xlabel('Male and Target')

plt.ylabel('Count')

plt.title('State of the Gender')

plt.show()

female_disease=heart[(heart.sex==0) & (heart.target==1)]          ## Here we have sex=0(female) and target =1(have disease)

female_NO_disease=heart[(heart.sex==0) & (heart.target==0)]       ## Here we have sex=0(female) and target =0(have no disease )

print(len(female_disease),"female_disease")

print(len(female_NO_disease),"female_NO_disease")
c=len(female_disease)

d=len(female_NO_disease)

sns.barplot(x=['female_disease ','female_NO_disease'],y=[c,d])

plt.xlabel('Female and Target')

plt.ylabel('Count')

plt.title('State of the Gender')

plt.show()
heart["cp"].value_counts()
sns.countplot(x='cp', data=heart)

plt.xlabel(" Chest type")

plt.ylabel("Count")

plt.title("Chest type Vs count plot")

plt.show()
print(len(heart[(heart.cp==0)&(heart.target==0)]),"=cp_zero_target_zero")

print(len(heart[(heart.cp==0)&(heart.target==1)]),"=cp_zero_target_one")

print(len(heart[(heart.cp==1)&(heart.target==0)]),"=cp_one_target_zero")

print(len(heart[(heart.cp==1)&(heart.target==1)]),"=cp_one_target_one")
target_0=len(heart[(heart.cp==0)&(heart.target==0)])

target_1=len(heart[(heart.cp==0)&(heart.target==1)])

plt.subplot(1,2,1)

sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])

plt.ylabel("Count")

plt.title("Chest_type_0 Vs count plot")





target_0=len(heart[(heart.cp==1)&(heart.target==0)])

target_1=len(heart[(heart.cp==1)&(heart.target==1)])

plt.subplot(1,2, 2)

sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])

plt.ylabel("Count")

plt.title("Chest_type_1 Vs count plot")



print(len(heart[(heart.cp==2)&(heart.target==0)]),"=cp_two_target_zero")

print(len(heart[(heart.cp==2)&(heart.target==1)]),"=cp_two_target_one")

print(len(heart[(heart.cp==3)&(heart.target==0)]),"=cp_three_target_zero")

print(len(heart[(heart.cp==3)&(heart.target==1)]),"=cp_three_target_one")
target_0=len(heart[(heart.cp==2)&(heart.target==0)])

target_1=len(heart[(heart.cp==2)&(heart.target==1)])

plt.subplot(1,2,1)

sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])

plt.ylabel("Count")

plt.title("Chest_type_2 Vs count plot")





target_0=len(heart[(heart.cp==3)&(heart.target==0)])

target_1=len(heart[(heart.cp==3)&(heart.target==1)])

plt.subplot(1,2, 2)

sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])

plt.ylabel("Count")

plt.title("Chest_type_3 Vs count plot")

plot = heart[heart.target == 1].trestbps.value_counts().sort_index().plot(kind = "bar", figsize=(15,4), fontsize = 15)

plot.set_title("Resting blood pressure", fontsize = 20)
heart.chol.value_counts()[:20]
sns.barplot(x=heart.chol.value_counts()[:20].index,y=heart.chol.value_counts()[:20].values)

plt.xlabel('chol')

plt.ylabel('Count')

plt.title('chol Counts')

plt.xticks(rotation=45)

plt.show()
age_unique=sorted(heart.age.unique())

age_chol_values=heart.groupby('age')['chol'].count().values

mean_chol=[]

for i,age in enumerate(age_unique):

    mean_chol.append(sum(heart[heart['age']==age].chol)/age_chol_values[i])

    
plt.figure(figsize=(10,5))

sns.pointplot(x=age_unique,y=mean_chol,color='red',alpha=0.8)

plt.xlabel('age',fontsize = 15,color='blue')

plt.xticks(rotation=45)

plt.ylabel('chol',fontsize = 15,color='blue')

plt.title('age vs chol',fontsize = 15,color='blue')

plt.grid()

plt.show()
print(len(heart[(heart.fbs==1)&(heart.target==0)]),"=fbs_one_target_zero")

print(len(heart[(heart.fbs==1)&(heart.target==1)]),"=fbs_one_target_one")
target_0=len(heart[(heart.fbs==1)&(heart.target==0)])

target_1=len(heart[(heart.fbs==1)&(heart.target==1)])

plt.subplot(1,2,1)

sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])

plt.ylabel("Count")

plt.title("fbs_type_1 Vs count plot")

print(len(heart[(heart.restecg==1)&(heart.target==0)]),"=restecg_one_target_zero")

print(len(heart[(heart.restecg==1)&(heart.target==1)]),"=restecg_one_target_one")
plot = heart[heart.target == 1].thalach.value_counts().sort_index().plot(kind = "bar", figsize=(15,4), fontsize = 10)

plot.set_title("thalach", fontsize = 15)
heart.thal.value_counts()
print(len(heart[(heart.thal==3)&(heart.target==0)]),"=thal_three_target_zero")

print(len(heart[(heart.thal==3)&(heart.target==1)]),"=thal_three_target_one")
target_0=len(heart[(heart.thal==3)&(heart.target==0)])

target_1=len(heart[(heart.thal==3)&(heart.target==1)])

plt.subplot(1,2,1)

sns.barplot(x=["target_0","target_1"],y=[target_0,target_1])

plt.ylabel("Count")

plt.title("thal_type_3 Vs count plot")
print(len(heart[(heart.thal==6)&(heart.target==0)]),"=thal_7_target_zero")   # Here thal for (6 = fixed defect) has no heart disease

print(len(heart[(heart.thal==6)&(heart.target==1)]),"=thal_7_target_one")
print(len(heart[(heart.thal==7)&(heart.target==0)]),"=thal_7_target_zero")  # Here thal for (7 = reversable defect) has no heart disease

print(len(heart[(heart.thal==7)&(heart.target==1)]),"=thal_7_target_one")
cp = pd.get_dummies(heart['cp'], prefix = "cp", drop_first=True)

thal = pd.get_dummies(heart['thal'], prefix = "thal" , drop_first=True)

slope = pd.get_dummies(heart['slope'], prefix = "slope", drop_first=True)



#Removing the first level.
data = pd.concat([heart, cp, thal, slope], axis=1)

data.head()
data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)

data.head()
x = data.drop(['target'], axis=1)

y = data.target
print(x.shape)
x.corr()
x = (x - x.min())/(x.max()-x.min())

x.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
from sklearn.linear_model import LogisticRegression

logi = LogisticRegression()

logi.fit(x_train, y_train)

logi.score(x_test, y_test)
from sklearn.model_selection import GridSearchCV

 ## Setting parameters for GridSearchCV

params = {'penalty':['l1','l2'],

         'C':[0.01,0.1,1,10,100],

         'class_weight':['balanced',None]}

logi_model = GridSearchCV(logi,param_grid=params,cv=10)
logi_model.fit(x_train,y_train)

logi_model.best_params_
logi = LogisticRegression(C=1, penalty='l2')

logi.fit(x_train, y_train)

logi.score(x_test, y_test)
from sklearn.metrics import confusion_matrix

cm_lg = confusion_matrix(y_test, logi.predict(x_test))

sns.heatmap(cm_lg, annot=True)

plt.plot()
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()

dtree.fit(x_train,y_train)                                # HERE WE ARE FITTING THE VALUES OF BOTH x_train,y_train
predict=dtree.predict(x_test)                               # HERE WE ARE PREDICTING y_test values.

predict
#NOW WE WILL SEE CONFUSION MATRIX FOR DECISION TREE



from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predict))





from sklearn.metrics import confusion_matrix

cm_tree = confusion_matrix(y_test,predict )

sns.heatmap(cm_tree, annot=True)

plt.plot()
from sklearn.metrics import accuracy_score

print("Accuracy is:",accuracy_score(y_test,predict)*100)    #HERE WE ARE GETTING OUR ACCURACY OF OUR MODEL
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(x_train,y_train)
rfc_predict=rfc.predict(x_test)                                # HERE WE ARE PREDICTING y_test values.

rfc_predict
#NOW WE WILL SEE CONFUSION MATRIX FOR RANDOM FOREST



from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_test,rfc_predict )

sns.heatmap(cm_rf, annot=True)

plt.plot()





print(classification_report(y_test,rfc_predict))
from sklearn.metrics import accuracy_score

print("Accuracy is:",accuracy_score(y_test,rfc_predict)*100)    #HERE WE ARE GETTING OUR ACCURACY OF OUR MODEL
from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

x_train=sc_x.fit_transform(x_train)

x_test=sc_x.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier

Classifier=KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)

Classifier.fit(x_train,y_train)
y_predict=Classifier.predict(x_test)                                # HERE WE ARE PREDICTING y_test values.                

y_predict
#NOW WE WILL SEE CONFUSION MATRIX FOR K NEAREST NEIGHBOR



from sklearn.metrics import confusion_matrix

cm_knn = confusion_matrix(y_test,y_predict )

sns.heatmap(cm_knn, annot=True)

plt.plot()
from sklearn.metrics import accuracy_score

print("Accuracy is:",accuracy_score(y_test,y_predict)*100)    #HERE WE ARE GETTING OUR ACCURACY OF OUR MODEL
plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lg,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,2)

plt.title("Decision Tree Confusion Matrix")

sns.heatmap(cm_tree,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,3)

plt.title("Random forest")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,4)

plt.title("K Nearest Neighbor Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False)
