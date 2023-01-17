import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from sklearn.preprocessing import StandardScaler



# Import tools

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



#machine learning models

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC



%matplotlib inline
# We are reading our data

df = pd.read_csv("../input/processed-cleveland-data/heart_disease.csv")
df.head()
df.info()
df.isna().sum()
df.ca.value_counts()
df.thal.value_counts()
df.shape
pd.set_option("display.float", "{:.2f}".format)

df.describe()
df.target.value_counts()
df['ca'].replace('?', np.nan, inplace= True)

df['thal'].replace('?', np.nan, inplace= True)

df = df.dropna()

df.shape
sns.countplot(x="target", data=df, palette="bwr")

plt.show()
countNoDisease = len(df[df.target == 0])

countHaveDisease = len(df[df.target == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
sns.countplot(x='sex', data=df, palette="mako_r")

plt.xlabel("Sex (0 = female, 1= male)")

plt.show()
countFemale = len(df[df.sex == 0])

countMale = len(df[df.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))
pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")

plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
plt.figure(figsize=(18,10))

sns.heatmap(df.corr(), annot=True,cmap='coolwarm',linewidths=.1)

plt.show()
X= df.drop('target',axis=1)

y=df['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=45)
# Pre-processing scaling the values

scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train)

X_train = pd.DataFrame(X_train_scaled)



X_test_scaled = scaler.transform(X_test)

X_test = pd.DataFrame(X_test_scaled)
#Implementing GridSearchCv to select best parameters and applying k-NN Algorithm

knn =KNeighborsClassifier()

params = {'n_neighbors':list(range(1,20)),

    'p':[1, 2, 3, 4,5],

    'leaf_size':list(range(1,20)),

    'weights':['uniform', 'distance']

         }
model = GridSearchCV(knn,params,cv=3, n_jobs=-1)

model.fit(X_train,y_train)

model.best_params_ 
predict = model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

print('Accuracy Score: ',accuracy_score(y_test,predict))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,predict),5)*100,'%')
test_score = accuracy_score(y_test, model.predict(X_test)) * 100

train_score = accuracy_score(y_train, model.predict(X_train)) * 100



results_df = pd.DataFrame(data=[["K-Nearest Neighor Algorithm", train_score, test_score]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df
log = LogisticRegression()

# Setting parameters for GridSearchCV

params = {'penalty':['l2'],

         'C':[0.01,0.1,1,5],

         'class_weight':['balanced',None]}

log_model = GridSearchCV(log,param_grid=params,cv=10)
log_model.fit(X_train,y_train)

# Printing best parameters choosen through GridSearchCV

log_model.best_params_
predict = log_model.predict(X_test)
from sklearn.metrics import accuracy_score

print('Accuracy Score: ',accuracy_score(y_test,predict))

print('Using Logistic Regression we get an accuracy score of: ',

      round(accuracy_score(y_test,predict),5)*100,'%')
test_score = accuracy_score(y_test, log_model.predict(X_test)) * 100

train_score = accuracy_score(y_train, log_model.predict(X_train)) * 100



results_df_2 = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 

                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
svm_model = SVC(kernel='rbf', gamma=0.1, C=1.0)

svm_model.fit(X_train, y_train)
predict = svm_model.predict(X_test)
from sklearn.metrics import accuracy_score

print('Accuracy Score: ',accuracy_score(y_test,predict))

print('Using Decision Tree we get an accuracy score of: ',

      round(accuracy_score(y_test,predict),5)*100,'%')