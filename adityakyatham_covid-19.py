import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Comment this if the data visualisations doesn't work on your side

%matplotlib inline



plt.style.use('bmh')
df= pd.read_csv('../input/COVID_19.csv', index_col=None)

df.head()

df.info()

list(set(df.dtypes.tolist()))

from collections import Counter

Counter(df["Label"]) 
df.columns
plt.figure(figsize = (10,10))

boxplot = df.boxplot()

from sklearn.preprocessing import LabelEncoder

encodings = dict()

for c in df.columns:

    #print df[c].dtype

    if df[c].dtype == "object":

        encodings[c] = LabelEncoder() #to give numerical label to char type labels.

        encodings[c]

        df[c] = encodings[c].fit_transform(df[c])

print(encodings)
df.head()
X = df.iloc[:,0:9]

Y = df.iloc[:,9]

Y.head()
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,Y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

plt.figure(figsize=(10,10))

feat_importances.nlargest(9).plot(kind='barh')

plt.show()

from sklearn.preprocessing import StandardScaler #normalization

std = StandardScaler()

X = std.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)#random for splitting same data when run again.

# Fitting Random Forest Regression to the dataset 

# import the regressor 

from sklearn.ensemble import RandomForestClassifier 



# create regressor object 

clf = RandomForestClassifier(n_estimators=50,criterion='gini',  

random_state=0)



# fit the regressor with x and y data 

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test) 

print(Counter(y_pred))

print(Counter(y_test))

# Python script for confusion matrix creation. 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

 

results = confusion_matrix(y_test, y_pred) 

print('Confusion Matrix :')

print(results) 

print('Accuracy Score :',accuracy_score(y_test, y_pred))

print('Report : ')

print(classification_report(y_test, y_pred))

#0 : Negative (No coronavirus)

#1: Positive
'''age=int(input("Enter your Age : "))

gender=input("Enter your Gender : ")

Region1=input("Enter your City : ")

Region2=input("Enter your District : ")

detected_state=input("Enter your State : ")

nationality=input("Enter your Nationality : ")

Travel_hist=input("Enter last City/Nation travelled : ")

Disease_hist=input("If you have BP/Diabetes, mention it : ")

Symptom=input("Mention present symptom of illness : ")

'''

age=45

gender='Male'

Region1='Solapur'

Region2='Solapur'

detected_state='Maharashtra'

nationality='India'

Travel_hist='Italy'

Disease_hist='Null'

Symptom='Null'



data=[[age,gender,Region1,Region2,detected_state,nationality,Travel_hist,Disease_hist,Symptom]]

dfX = pd.DataFrame(data, columns = ['age','gender','Region1 ','Region2','detected_state','nationality','Travel_hist','Disease_hist','Symptom'])

print(dfX)

for c in dfX.columns:

    #print df[c].dtype

    if dfX[c].dtype == "object":

        dfX[c] = encodings[c].transform(dfX[c])

X_test1 = std.transform(dfX)

y_pred1 = clf.predict(X_test1) 

ans = encodings['Label'].inverse_transform(y_pred1)

for dt in ans:

  if dt=='Positive':

    print("Result : High chances of COVID-19")

  else:

    print("Result : You are not suffering from COVID-19")