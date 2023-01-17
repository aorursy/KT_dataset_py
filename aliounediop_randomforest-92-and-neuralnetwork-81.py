import seaborn as sns
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.models import Sequential
from keras.optimizers import SGD,Adam,Adagrad
from keras.layers import InputLayer, Dense 
%matplotlib inline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,KFold
from sklearn.preprocessing import StandardScaler

pwd
df=pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
#Plotting the variables using histogram
df.hist(figsize=(12,10))
fig,ax=plt.subplots(figsize=(15,15))

plt.hist(df['Pregnancies'],color='skyblue',bins=30,alpha=0.3)
plt.hist(df['Glucose'],color='black',bins=30,alpha=0.3)
plt.hist(df['BloodPressure'],color='lime',bins=30,alpha=0.3)
plt.hist(df['BMI'],color='red',bins=30,alpha=0.3)
plt.hist(df['DiabetesPedigreeFunction'],color='navy',bins=30,alpha=0.1)
plt.hist(df['Age'],color='goldenrod',bins=30,alpha=0.3)

plt.title('Histogram of Variables')
plt.legend(df[['Pregnancies','Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age']])
#Replacing the zero-values for Blood Pressure
df1 = df.loc[df['Outcome'] == 1]
df2 = df.loc[df['Outcome'] == 0]
df1 = df1.replace({'BloodPressure':0}, np.median(df1['BloodPressure']))
df2 = df2.replace({'BloodPressure':0}, np.median(df2['BloodPressure']))
dataframe = [df1, df2]
df = pd.concat(dataframe)
#Replacing the zero-values for BMI
df1 = df.loc[df['Outcome'] == 1]
df2 = df.loc[df['Outcome'] == 0]
df1 = df1.replace({'BMI':0}, np.mean(df1['BMI']))
df2 = df2.replace({'BMI':0}, np.mean(df2['BMI']))
dataframe = [df1, df2]
df = pd.concat(dataframe)
#Replacing the zero-values for Glucose
df1 = df.loc[df['Outcome'] == 1]
df2 = df.loc[df['Outcome'] == 0]
df1 = df1.replace({'Glucose':0}, np.median(df1['Glucose']))
df2 = df2.replace({'Glucose':0}, np.median(df2['Glucose']))
dataframe = [df1, df2]
df = pd.concat(dataframe)
#Replacing the zero-values for Insulin
df1 = df.loc[df['Outcome'] == 1]
df2 = df.loc[df['Outcome'] == 0]
df1 = df1.replace({'Insulin':0}, np.mean(df1['Insulin']))
df2 = df2.replace({'Insulin':0}, np.mean(df2['Insulin']))
dataframe = [df1, df2]
df = pd.concat(dataframe)
#Replacing the zero-values for SkinThickness
df1 = df.loc[df['Outcome'] == 1]
df2 = df.loc[df['Outcome'] == 0]
df1 = df1.replace({'SkinThickness':0}, np.median(df1['SkinThickness']))
df2 = df2.replace({'SkinThickness':0}, np.median(df2['SkinThickness']))
dataframe = [df1, df2]
df = pd.concat(dataframe)
df.describe()
#Checking for missing values
df.isna().sum()
#Checking the correlation between variables
sns.pairplot(df,hue='Outcome')
df1=df.corr('pearson')
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df1,center=0,square=True,annot=True,cmap="YlGnBu")
df.describe()
out=[]
def ZRscore_outlier(df):
    med = np.median(df)
    ma = stats.median_absolute_deviation(df)
    for i in df: 
        z = (0.6745*(i-med))/ (np.median(ma))
        if np.abs(z) > 3: 
            out.append(i)
    print("Outliers:",out)

from sklearn.preprocessing import StandardScaler
#Splitting data into data and target
X=df.drop('Outcome',axis=1)
y=df['Outcome']
X.head()
#Setting a benchmark
y.value_counts()/len(y)
#Converting to category
y=y.astype('category')
#Scaling the X data
columns = X.columns
scale=StandardScaler()
X=scale.fit_transform(X)
X
X=pd.DataFrame(X, columns = columns)
#Splitting the data between Train and Test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)
X.shape
X.describe
from keras.layers import Dropout
#Model Creation
model=Sequential()
model.add(Dense(64,input_shape=(8,),activation='softmax'))
model.add(Dense(32,activation='softmax'))
model.add(Dense(16,activation='softmax'))
model.add(Dense(8,activation='softmax'))
model.add(Dense(4,activation='sigmoid'))
#model.add(Dense(2,activation='sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(Adagrad(lr=0.05),'binary_crossentropy',metrics=['accuracy'])

history=model.fit(X_train, y_train,validation_data=(X_test, y_test),epochs = 100)


print(model.evaluate(X_train,y_train))
print(model.evaluate(X_test,y_test))
history=pd.DataFrame(history.history)
history.head()
y_pred=model.predict_classes(X_test)
score=confusion_matrix(y_test,y_pred)
print('Confusion_matrix \n ', score)
print('acurracy_score:' ,accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

#Using KerasClassifier to run our  model
def model_creation(activation='sigmoid',epochs=15):
    model=Sequential()
    model.add(Dense(256,input_shape=(8,),activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(2,activation='sigmoid'))
#model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(Adagrad(lr=0.018),'binary_crossentropy',metrics=['accuracy'])
    return model
model=KerasClassifier(build_fn=model_creation,epochs=200)

model.fit(X_train,y_train)
pred=model.predict(X_test)
print(classification_report(y_test,pred))

#Cross vaidation on the model
cv=KFold(n_splits=5,shuffle=True)
scores=cross_val_score(model,X,y,cv=cv,verbose=0)
scores.mean()
#Comparing Deep Neural Network with others Classifiers
from sklearn import linear_model as lm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.linear_model import RidgeClassifier as rgc
from sklearn.ensemble import RandomForestClassifier as rfc, AdaBoostClassifier as ada

classifiers = [
    knnc(),
    dtc(),
    SVC(kernel='sigmoid'),
    SVC(kernel='sigmoid'),
    rgc(),
    rfc(),
    ada(),
      ]
classifier_names = [
    'KNN',
    'Decision Tree',
    'SVC',
    'SVC with sigmoid kernel',
    'Gaussian Naive Bayes',
    'RidgeClassifier',
    'RandomForrest',
    'Adaboost'    
    ]
for clf, clf_name in zip(classifiers, classifier_names):
    cv_scores = cross_val_score(clf, X,y, cv=5)
    
    print(clf_name, ' mean accuracy: ', round(cv_scores.mean()*100, 3), '% std: ', round(cv_scores.var()*100, 3),'%')
clf=rfc(n_estimators=100)
clf.fit(X_train,y_train)
rfc_pred=clf.predict(X_test)
print(classification_report(y_test,rfc_pred))
