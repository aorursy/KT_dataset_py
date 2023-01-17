import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/Heart Disease UCI/heart-Copy1.csv")



df.head()
dfml = df.copy()
dfml.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
dfml.head()
dfml['sex'][dfml['sex'] == 0] = 'female'

dfml['sex'][dfml['sex'] == 1] = 'male'



dfml['chest_pain_type'][dfml['chest_pain_type'] == 1] = 'typical angina'

dfml['chest_pain_type'][dfml['chest_pain_type'] == 2] = 'atypical angina'

dfml['chest_pain_type'][dfml['chest_pain_type'] == 3] = 'non-anginal pain'

dfml['chest_pain_type'][dfml['chest_pain_type'] == 4] = 'asymptomatic'



dfml['fasting_blood_sugar'][dfml['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

dfml['fasting_blood_sugar'][dfml['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



dfml['rest_ecg'][dfml['rest_ecg'] == 0] = 'normal'

dfml['rest_ecg'][dfml['rest_ecg'] == 1] = 'ST-T wave abnormality'

dfml['rest_ecg'][dfml['rest_ecg'] == 2] = 'left ventricular hypertrophy'



dfml['exercise_induced_angina'][dfml['exercise_induced_angina'] == 0] = 'no'

dfml['exercise_induced_angina'][dfml['exercise_induced_angina'] == 1] = 'yes'



dfml['st_slope'][dfml['st_slope'] == 1] = 'upsloping'

dfml['st_slope'][dfml['st_slope'] == 2] = 'flat'

dfml['st_slope'][dfml['st_slope'] == 3] = 'downsloping'



dfml['thalassemia'][dfml['thalassemia'] == 1] = 'normal'

dfml['thalassemia'][dfml['thalassemia'] == 2] = 'fixed defect'

dfml['thalassemia'][dfml['thalassemia'] == 3] = 'reversable defect'
dfml.head()
dfml['sex'] = dfml['sex'].astype('object')

dfml['chest_pain_type'] = dfml['chest_pain_type'].astype('object')

dfml['fasting_blood_sugar'] = dfml['fasting_blood_sugar'].astype('object')

dfml['rest_ecg'] = dfml['rest_ecg'].astype('object')

dfml['exercise_induced_angina'] = dfml['exercise_induced_angina'].astype('object')

dfml['st_slope'] = dfml['st_slope'].astype('object')

dfml['thalassemia'] = dfml['thalassemia'].astype('object')
dfml.dtypes
dfml.target.value_counts()
countNoDisease = len(dfml[dfml.target == 0])

countHaveDisease = len(dfml[dfml.target == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(dfml.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(dfml.target))*100)))
countFemale = len(dfml[dfml.sex == 'female'])

countMale = len(dfml[dfml.sex == 'male'])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(dfml.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(dfml.sex))*100)))
dfml.groupby('target').mean()
pd.crosstab(dfml.age,dfml.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
pd.crosstab(dfml.sex, dfml.target).plot(kind="bar",figsize=(10,6))

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
f,ax=plt.subplots(1,2,figsize=(16,7))

dfml['target'][df['sex']== 'male'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)

dfml['target'][df['sex']== 'female'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[0].set_title('Male')

ax[1].set_title('Female')

plt.legend(["Haven't Disease", "Have Disease"])
pd.crosstab(dfml.st_slope,dfml.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency for Slope')

plt.xlabel('The Slope of The Peak Exercise ST Segment ')

plt.legend(["Haven't Disease", "Have Disease"])

plt.xticks(rotation = 0)

plt.ylabel('Frequency')

plt.show()
pd.crosstab(dfml.fasting_blood_sugar,dfml.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])

plt.title('Heart Disease Frequency According To Fasting Blood Sugar')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
pd.crosstab(dfml.chest_pain_type,dfml.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency According To Chest Pain Type')

plt.xlabel('Chest Pain Type')

plt.xticks(rotation = 0)

plt.ylabel('Frequency of Disease or Not')

plt.show()
dfml.head(2)
dfml = pd.get_dummies(dfml, drop_first=True)
dfml.head()
y = dfml.target.values

x_data = dfml.drop(['target'], axis = 1)
# Normalize

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.metrics import accuracy_score



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
print("DataTrain Shape: {}".format(dfml.shape))

print("Train Shape: {}".format(x_train.shape))

print("Test Shape: {}".format(x_test.shape))
from sklearn.neighbors import KNeighborsClassifier



Model = KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)



print('accuracy is', Model.score(x_test,y_test))
scoreList = []

for i in range(1,20):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(x_train, y_train)

    scoreList.append(knn.score(x_test, y_test))

    

plt.plot(range(1,20), scoreList)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()



KNN = max(scoreList)

print("Maximum KNN Score is " , KNN )

from sklearn.naive_bayes import GaussianNB



Model = GaussianNB().fit(x_train, y_train)



print('accuracy is', Model.score(x_test, y_test))



NBB = Model.score(x_test, y_test)
from sklearn.svm import SVC



Model = SVC().fit(x_train, y_train)



print('accuracy is', Model.score(x_test, y_test))



SVMm =  Model.score(x_test, y_test)
from sklearn.linear_model import LogisticRegression



Model = LogisticRegression().fit(x_train, y_train)



print('accuracy is',Model.score(x_test, y_test))



LR = Model.score(x_test, y_test)
from sklearn.tree import DecisionTreeClassifier



Model = DecisionTreeClassifier().fit(x_train, y_train)



print('accuracy is', Model.score(x_test, y_test))



DT = Model.score(x_test, y_test)
from sklearn.ensemble import RandomForestClassifier



Model=RandomForestClassifier(max_depth=2 , n_estimators = 2000).fit(x_train, y_train)



print('accuracy is ', Model.score(x_test, y_test))



RT = Model.score(x_test, y_test)
models = pd.DataFrame({

    'Model': ['K-Nearest Neighbours', 'Naive Bayes', 'Support Vector Machines', 'Decision Tree', 'LogisticRegression', 'Random Forest'],

    'Score': [ KNN, NBB, SVMm, DT, LR, RT]})

models.sort_values(by='Score', ascending=False)
plt.subplots(figsize =(10, 8))



sns.barplot(x='Score', y = 'Model', data = models, palette="Set2")



plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
submit_gbc = GaussianNB().fit(x, y)



rr = submit_gbc.predict(x)
rr
x.index
submission = pd.DataFrame({

        "Target": rr

    })

submission.to_csv('heart-disease.csv', index=False)



submission.head(10)