import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
data = pd.read_csv('../input/heart.csv')
data.isnull().any().any()

len(data.columns)
for i in data.columns:

    data[i]=data[i].replace("?",np.nan)

    data[i]=data[i].astype(float)
data.columns
#It's a clean, easy to understand set of data. However, the meaning of some of the column headers are not obvious. Here's what they mean,



#age: The person's age in years

#sex: The person's sex (1 = male, 0 = female)

#cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)

#trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)

#chol: The person's cholesterol measurement in mg/dl

#fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)

#restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)

#thalach: The person's maximum heart rate achieved

#exang: Exercise induced angina (1 = yes; 0 = no)

#oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)

#slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

#ca: The number of major vessels (0-3)

#thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

#target: Heart disease (0 = no, 1 = yes)
data = data.rename(columns={'cp':'Chest Pain' , 'trestbps':'BP','chol':'cholestoral','fbs':'fasting blood sugar','restecg':'Resting ECG','thalach':'Max Hear Rate','exang':'exercise induced angina','thal':'Thalassemia','num       ':'object'})

data
data.count()
for i in data.columns:

    if data[i].count() <  200:

        del data[i]

        
data.count()
data.dtypes
data
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(data)
data = pd.DataFrame(imp_mean.transform(data), columns=data.columns)
data = data.astype(int)
data
plt.figure(figsize=(20,10))

for i in data.columns:

    plt.hist(data[i] , label=i )

    plt.legend(loc='upper right') 
plt.figure(figsize=(20,10))

plt.hist(data["Chest Pain"], label=["1.TypicalAngina \n2.AtypicalAngina  \n3.Non-AnginalPain  \n4.asymptomatic"])

plt.legend(loc='upper left')
CPmale = data[["Chest Pain"]].where(data["sex"]==1).dropna()

CPfemale = data[["Chest Pain"]].where(data["sex"]==0).dropna()

plt.hist([CPmale["Chest Pain"],CPfemale["Chest Pain"]])

plt.hist(data["exercise induced angina"])
MaleHeartRate = data[["Max Hear Rate"]].where(data["sex"]==1)

FemaleHeartRate = data[["Max Hear Rate"]].where(data["sex"]==0)

plt.hist([MaleHeartRate["Max Hear Rate"],FemaleHeartRate["Max Hear Rate"]])
MaleFrame =( data.where(data.sex==1).dropna())
MaleFrame
MaleAgeAngina = MaleFrame[["age"]].where(data["exercise induced angina"]==1)

plt.hist(MaleAgeAngina["age"])
plt.hist(data.where(data.BP > 120)["sex"])
patient = (data.where(data["target"] == 1)).dropna()

MalePatient = (patient.where(patient["sex"]==1)).dropna()

FemalePatient = (patient.where(patient["sex"]==0)).dropna()
plt.hist(data.where((data["fasting blood sugar"] == 1) & (data["target"] == 1) )["sex"])
MaleBP_Patient = MalePatient.where(MalePatient.BP >120).dropna()

MaleBP_Patient.BP.count()

print("All men patient having low BP :", (MalePatient.BP.count() - MaleBP_Patient.BP.count()), " and all men patient having high BP :",MaleBP_Patient.BP.count())
MaleCholestoral_Patient = MalePatient.where(MalePatient.cholestoral >200).dropna()

MaleBP_Patient.cholestoral.count()

print("All men patient having low Cholestoral :", (MalePatient.cholestoral.count() - MaleCholestoral_Patient.cholestoral.count()), " and all men patient having high Cholestoral :",MaleCholestoral_Patient.cholestoral.count())
plt.hist(MalePatient['Chest Pain'] , label =" 1.Typical angina \n 2.Atypical angina \n 3.Non-anginal pain \n 4.Asymptomatic ")

plt.legend()
MaleTAngina_Patient = MalePatient.where(MalePatient["Chest Pain"] == 1).dropna()

print("All men patient having no Typical Angina :", (MalePatient.BP.count() - MaleTAngina_Patient["Chest Pain"].count()), " and all men patient having Typical Angina :",MaleTAngina_Patient["Chest Pain"].count())
MaleATAngina_Patient = MalePatient.where(MalePatient["Chest Pain"] == 2).dropna()

print("All men patient having no ATypical Angina :", (MalePatient.BP.count() - MaleATAngina_Patient["Chest Pain"].count()), " and all men patient having ATypical Angina :",MaleATAngina_Patient["Chest Pain"].count())
MaleNAngina_Patient = MalePatient.where(MalePatient["Chest Pain"] == 3).dropna()

print("All men patient having no Non-Angina Pain :", (MalePatient.BP.count() - MaleNAngina_Patient["Chest Pain"].count()), " and all men patient having Non-Angina Pain :",MaleNAngina_Patient["Chest Pain"].count())
MaleAsymptomaticPatient = MalePatient.where(MalePatient["Chest Pain"] == 4).dropna()

print("All men patient having no Asymptomatic Pain :", (MalePatient.BP.count() - MaleAsymptomaticPatient["Chest Pain"].count()), " and all men patient having Asymptomatic Pain :",MaleAsymptomaticPatient["Chest Pain"].count())
MaleFastingBloodSugar = MalePatient.where(MalePatient["fasting blood sugar"] == 1).dropna()

print("All men patient having no fasting blood sugar :", (MalePatient["fasting blood sugar"].count() - MaleFastingBloodSugar["fasting blood sugar"].count()), " and all men patient having fasting blood sugar:",MaleFastingBloodSugar["fasting blood sugar"].count())
FemaleBP_Patient = FemalePatient.where(FemalePatient.BP >120).dropna()

print("All Women patient having low BP :",( (FemalePatient.BP.count() - FemaleBP_Patient.BP.count())), " and all women patient having high BP :",FemaleBP_Patient.BP.count())
FemaleBP_Patient = FemalePatient.where(FemalePatient.cholestoral >200).dropna()

print("All Women patient having low Cholestoral :",( (FemalePatient.cholestoral.count() - FemaleBP_Patient.cholestoral.count())), " and all women patient having high Cholestoral :",FemaleBP_Patient.cholestoral.count())
FemaleTAngina_Patient = FemalePatient.where(FemalePatient["Chest Pain"] == 1).dropna()

print("All Women patient having no Typical Angina :", (FemalePatient["Chest Pain"].count() - FemaleTAngina_Patient["Chest Pain"].count()), " and all Women patient having Typical Angina :",FemaleTAngina_Patient["Chest Pain"].count())
FemaleATAngina_Patient = FemalePatient.where(FemalePatient["Chest Pain"] == 2).dropna()

print("All Women patient having no ATypical Angina :", (FemalePatient["Chest Pain"].count() - FemaleATAngina_Patient["Chest Pain"].count()), " and all Women patient having ATypical Angina :",FemaleATAngina_Patient["Chest Pain"].count())
FemaleATAngina_Patient = FemalePatient.where(FemalePatient["Chest Pain"] == 3).dropna()

print("All Women patient having Non-Angina Pain :", (FemalePatient["Chest Pain"].count() - FemaleATAngina_Patient["Chest Pain"].count()), " and all Women patient having ATypical Angina :",FemaleATAngina_Patient["Chest Pain"].count())
FemaleAsymptomaticPatient = FemalePatient.where(MalePatient["Chest Pain"] == 4).dropna()

print("All men patient having no Asymptomatic Pain :", (MalePatient.BP.count() - MaleAsymptomaticPatient["Chest Pain"].count()), " and all men patient having Asymptomatic Pain :",MaleAsymptomaticPatient["Chest Pain"].count())
plt.hist(FemalePatient['Chest Pain'] , label =" 1.Typical angina \n 2.Atypical angina \n 3.Non-anginal pain \n 4.Asymptomatic ")

plt.legend()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import svm, datasets

from sklearn.model_selection import KFold
X = data.drop('target', axis=1)

X.head()

X.index
y = data.target

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

len(X_train),len(y_train)
clf = GaussianNB().fit(X_train,y_train)
predicted = clf.predict(X_test)
acc=sum(predicted==y_test)/len(y_test)*100

print(acc)
neigh = KNeighborsClassifier(n_neighbors=13 )

neigh.fit(X_train, y_train)

KNNpredicted = neigh.predict(X_test)

acc=sum(KNNpredicted==y_test)/len(y_test)*100

print(acc)
svc = svm.SVC(kernel='linear').fit(X_train, y_train)

pred_target=svc.predict(X_test)

acc=sum(pred_target==y_test)/len(y_test)*100

print(acc)

predicted = {}

key =0

predicted[1]=[]

predicted[2]=[]

predicted[3]=[]

kfold = KFold(3,True,1)

for train, test in kfold.split(X):

    X_train, X_test = X.iloc[train], X.iloc[test]

    y_train, y_test = y[train], y[test]

    for k in range(1,20,2):    

        neigh = KNeighborsClassifier(n_neighbors=k )

        neigh.fit(X_train, y_train)

        KNNpredicted = neigh.predict(X_test)

        acc=sum(KNNpredicted==y_test)/len(y_test)*100

        key = key+1 if k==1 else key

        predicted[key].append(  acc )
X = list(range(1,20,2))

plt.figure(figsize=(20,5))

plt.plot(X,predicted[1],label ="Fold 1")

plt.plot(X,predicted[2],label ="Fold 2")

plt.plot(X,predicted[3],label ="Fold 3")

plt.legend(loc="upper left")