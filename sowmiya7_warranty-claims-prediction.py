#IMPORTING PACKAGES AND MODULES 



import pandas as pd



import numpy as np



import matplotlib.pyplot as plt



import seaborn as sns



from imblearn.over_sampling import ADASYN



from collections import Counter



from sklearn.model_selection import train_test_split



from sklearn.ensemble import RandomForestClassifier



from sklearn import metrics

#LOADING DATASET



data=pd.read_csv("..//input/warranty-claims//train.csv")

data.head()
#Dropping first column



data1=data.drop(['Unnamed: 0'],axis=1)



print(data1.describe())
#Replace claim with Claim



data1['Purpose'].replace('claim','Claim',inplace=True)



data1['Purpose'].value_counts()



data1['Purpose'].value_counts().plot(kind='bar')
#Replace up with uttar pradesh



data1['State'].replace('UP','Uttar Pradesh',inplace=True)



data1['State'].value_counts()



data1['State'].value_counts().plot(kind='bar')

#Finding the duplicate values



data1.describe()



data1.Claim_Value.duplicated()



data1.Claim_Value.duplicated().sum()



data1.duplicated()



data1.duplicated().sum()



data1.loc[data1.duplicated(),:]



data1.loc[data1.duplicated(keep='first'),:]



data1.loc[data1.duplicated(keep='last'),:]



data1.loc[data1.duplicated(keep=False),:]



data1=data1.drop_duplicates()



data1
#NA Values



data1.isnull()



data1.isnull().sum()



data1.Claim_Value.describe()
#Median imputation



data1.fillna(data1.Claim_Value.median(),inplace=True)



data1.Claim_Value.describe()



data1.isnull().sum()
#Applying ADASYN Technique for imbalanced dataset



from imblearn.over_sampling import ADASYN



from collections import Counter



data2=pd.get_dummies(data1)



X=data2.iloc[:,:-1].values



Y=data2.iloc[:,10:11].values



print('Shape of Future Matrix:',X.shape)



print('Shape of Target Vector:',Y.shape)



ada=ADASYN(sampling_strategy='minority',random_state=42,n_neighbors=5)



X_res, y_res = ada.fit_resample(X,Y)



print('Oversampled Target Variable Distribution:',Counter(y_res))



data3=np.column_stack((X_res,y_res))



df=pd.DataFrame(data3)

#Model building- Random forest



X=df.iloc[:,:-1]



Y=df.iloc[:,80:81]



from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30)



from sklearn.neighbors import KNeighborsClassifier as KNC



neigh = KNC(n_neighbors= 1)



neigh.fit(X_train,Y_train)



y_pred=neigh.predict(X_test)



from sklearn.metrics import classification_report,confusion_matrix



from sklearn import metrics



print(confusion_matrix(Y_test,y_pred))



print("Accuracy:",metrics.accuracy_score(Y_test,y_pred))



print(classification_report(Y_test,y_pred))



#k=5



X=df.iloc[:,:-1]



Y =df.iloc[:,80:81]



from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30)



from sklearn.neighbors import KNeighborsClassifier as KNC



neigh = KNC(n_neighbors= 5)



neigh.fit(X_train,Y_train)



y_pred=neigh.predict(X_test)



from sklearn.metrics import classification_report,confusion_matrix



from sklearn import metrics



print(confusion_matrix(Y_test,y_pred))



print("Accuracy:",metrics.accuracy_score(Y_test,y_pred))



print(classification_report(Y_test,y_pred))



#Random forest



X=df.iloc[:,:-1]



Y=df.iloc[:,80:81]



from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.30)



from sklearn.ensemble import RandomForestClassifier



from sklearn import metrics



rf=RandomForestClassifier()



rf.fit(X_train,Y_train)



predictions=rf.predict(X_test)



cm= metrics.confusion_matrix(Y_test,predictions)



print(cm)



print("Accuracy:",metrics.accuracy_score(Y_test,predictions))



print(classification_report(Y_test,predictions))


