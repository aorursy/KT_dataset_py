import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pandas_profiling



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
df=pd.read_csv("../input/heart-disease-uci/heart.csv")
df.head()
df.describe()
df.info()
# target

print(df.target.value_counts())

plt.figure(figsize=(10,8))

sns.countplot(x=df['target'])
# sex

print(df.sex.value_counts())
Male = len(df[df.sex == 1])

Female = len(df[df.sex == 0])



print("Percentage of Male Patients: {:.2f}%".format((Male / (len(df.sex))*100)))

print("Percentage of Female Patients: {:.2f}%".format((Female / (len(df.sex))*100)))
plt.figure(figsize=(8,6))

sns.countplot(x=df['sex'])
# cp: chest_pain

print(df.cp.value_counts())

plt.figure(figsize=(8,6))

sns.countplot(x=df['cp'])
profile = pandas_profiling.ProfileReport(df)

profile
# fasting blood sugar>120 mg/dl(true=1,false=0)

print(df.fbs.value_counts())
pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,6),color=['#1CA53B','orange' ])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
# let's change the names of the  columns for better understanding



df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']



df.columns
df['sex'][df['sex'] == 0] = 'female'

df['sex'][df['sex'] == 1] = 'male'



df['chest_pain_type'][df['chest_pain_type'] == 1] = 'typical angina'

df['chest_pain_type'][df['chest_pain_type'] == 2] = 'atypical angina'

df['chest_pain_type'][df['chest_pain_type'] == 3] = 'non-anginal pain'

df['chest_pain_type'][df['chest_pain_type'] == 4] = 'asymptomatic'



df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



df['rest_ecg'][df['rest_ecg'] == 0] = 'normal'

df['rest_ecg'][df['rest_ecg'] == 1] = 'ST-T wave abnormality'

df['rest_ecg'][df['rest_ecg'] == 2] = 'left ventricular hypertrophy'



df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'



df['st_slope'][df['st_slope'] == 1] = 'upsloping'

df['st_slope'][df['st_slope'] == 2] = 'flat'

df['st_slope'][df['st_slope'] == 3] = 'downsloping'



df['thalassemia'][df['thalassemia'] == 1] = 'normal'

df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'

df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'
df['sex'] = df['sex'].astype('object')

df['chest_pain_type'] = df['chest_pain_type'].astype('object')

df['fasting_blood_sugar'] = df['fasting_blood_sugar'].astype('object')

df['rest_ecg'] = df['rest_ecg'].astype('object')

df['exercise_induced_angina'] = df['exercise_induced_angina'].astype('object')

df['st_slope'] = df['st_slope'].astype('object')

df['thalassemia'] = df['thalassemia'].astype('object')
df.head()
y = df.target



df = df.drop('target', axis=1)

df = pd.get_dummies(df, drop_first=True)

x=df

# display data after encoding

x.head()
# splitting the sets into training and test sets



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



# getting the shapes

print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_test :", y_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(x_train)

x_train = pd.DataFrame(X_train_scaled)



X_test_scaled = scaler.transform(x_test)

x_test = pd.DataFrame(X_test_scaled)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



knn =KNeighborsClassifier()

params = {'n_neighbors':list(range(1,20)),

    'p':[1, 2, 3, 4,5,6,7,8,9,10],

    'leaf_size':list(range(1,20)),

    'weights':['uniform', 'distance']

         }


model = GridSearchCV(knn,params,cv=3, n_jobs=-1)
model.fit(x_train,y_train)

model.best_params_           #print's parameters best values
predict = model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix

print('Accuracy Score: ',accuracy_score(y_test,predict))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,predict),5)*100,'%')