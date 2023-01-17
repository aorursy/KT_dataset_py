import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

import warnings

warnings.filterwarnings("ignore")



from pylab import rcParams





%matplotlib inline
data = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.columns
data.shape
data.head(5)
data.drop(['customerID'], axis=1, inplace=True)
data['Churn'].value_counts(sort = False)
data['Churn'].value_counts(sort = False)
# Data to plot

labels =data['Churn'].value_counts(sort = True).index

sizes = data['Churn'].value_counts(sort = True)





colors = ["whitesmoke","red"]

explode = (0.1,0)  # explode 1st slice

 

rcParams['figure.figsize'] = 8,8

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=270,)



plt.title('Percent of churn in customer')

plt.show()
data['Churn'] = data['Churn'].map(lambda s :1  if s =='Yes' else 0)
data.info()
#missing data

total = data.isnull().sum().sort_values(ascending=False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(6)
data.head(5)
data['gender'].head()
g = sns.factorplot(y="Churn",x="gender",data=data,kind="bar" ,palette = "Pastel1")

data = pd.get_dummies(data=data, columns=['gender'])
data['SeniorCitizen'].value_counts()
data['Partner'].value_counts()
data['Partner'] = data['Partner'].map(lambda s :1  if s =='Yes' else 0)

data['Partner'].value_counts()
data['Dependents'] = data['Dependents'].map(lambda s :1  if s =='Yes' else 0)

data['PhoneService'] = data['PhoneService'].map(lambda s :1  if s =='Yes' else 0)

data['PaperlessBilling'] = data['PaperlessBilling'].map(lambda s :1  if s =='Yes' else 0)

data['tenure'].head()
# tenure distibution 

g = sns.kdeplot(data.tenure[(data["Churn"] == 0) ], color="Red", shade = True)

g = sns.kdeplot(data.tenure[(data["Churn"] == 1) ], ax =g, color="Blue", shade= True)

g.set_xlabel("tenure")

g.set_ylabel("Frequency")

plt.title('Distribution of tenure comparing with churn feature')

g = g.legend(["Not Churn","Churn"])
data['MultipleLines'].value_counts()
data['MultipleLines'].replace('No phone service','No', inplace=True)

data['MultipleLines'] = data['MultipleLines'].map(lambda s :1  if s =='Yes' else 0)

data['MultipleLines'].value_counts()
data['InternetService'].value_counts()
data['Has_InternetService'] = data['InternetService'].map(lambda s :0  if s =='No' else 1)

data['Fiber_optic'] = data['InternetService'].map(lambda s :1  if s =='Fiber optic' else 0)

data['DSL'] = data['InternetService'].map(lambda s :1  if s =='DSL' else 0)

print(data['Has_InternetService'].value_counts())

print(data['Fiber_optic'].value_counts())

print(data['DSL'].value_counts())

data.drop(['InternetService'], axis=1, inplace=True)
data['OnlineSecurity'] = data['OnlineSecurity'].map(lambda s :1  if s =='Yes' else 0)

data['OnlineBackup'] = data['OnlineBackup'].map(lambda s :1  if s =='Yes' else 0)

data['DeviceProtection'] = data['DeviceProtection'].map(lambda s :1  if s =='Yes' else 0)

data['TechSupport'] = data['TechSupport'].map(lambda s :1  if s =='Yes' else 0)

data['StreamingTV'] = data['StreamingTV'].map(lambda s :1  if s =='Yes' else 0)

data['StreamingMovies'] = data['StreamingMovies'].map(lambda s :1  if s =='Yes' else 0)
data['PaymentMethod'].value_counts()
data = pd.get_dummies(data=data, columns=['PaymentMethod'])
data[['PaymentMethod_Electronic check',

      'PaymentMethod_Mailed check',

      'PaymentMethod_Bank transfer (automatic)',

      'PaymentMethod_Credit card (automatic)']].head()
data['Contract'].value_counts()
data = pd.get_dummies(data=data, columns=['Contract'])
data['MonthlyCharges'].head()
g = sns.factorplot(x="Churn", y = "MonthlyCharges",data = data, kind="box", palette = "Pastel1")
data['TotalCharges'].head()
## because 11 rows contain " " , it means 11 missing data in our dataset

len(data[data['TotalCharges'] == " "])
## Drop missing data

data = data[data['TotalCharges'] != " "]
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])

## At first time I use this command but it error because some value contain " "

## That why I know " " hide in our dataset 
g = sns.factorplot(y="TotalCharges",x="Churn",data=data,kind="boxen", palette = "Pastel2")
data.info()
data["Churn"] = data["Churn"].astype(int)



Y_train = data["Churn"]

X_train = data.drop(labels = ["Churn"],axis = 1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import  cross_val_score,GridSearchCV



Rfclf = RandomForestClassifier(random_state=15)

Rfclf.fit(X_train, Y_train)
# 10 Folds Cross Validation 

clf_score = cross_val_score(Rfclf, X_train, Y_train, cv=10)

print(clf_score)

clf_score.mean()
%%time

param_grid  = { 

                'n_estimators' : [500,1200],

               # 'min_samples_split': [2,5,10,15,100],

               # 'min_samples_leaf': [1,2,5,10],

                'max_depth': range(1,5,2),

                'max_features' : ('log2', 'sqrt'),

                'class_weight':[{1: w} for w in [1,1.5]]

              }



GridRF = GridSearchCV(RandomForestClassifier(random_state=15), param_grid)



GridRF.fit(X_train, Y_train)

#RF_preds = GridRF.predict_proba(X_test)[:, 1]

#RF_performance = roc_auc_score(Y_test, RF_preds)



print(

    #'DecisionTree: Area under the ROC curve = {}'.format(RF_performance)

     "\nBest parameters \n" + str(GridRF.best_params_))
rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(X_train, Y_train)
# 10 Folds Cross Validation 

clf_score = cross_val_score(rf, X_train, Y_train, cv=10)

print(clf_score)

clf_score.mean()

    
Rfclf_fea = pd.DataFrame(rf.feature_importances_)

Rfclf_fea["Feature"] = list(X_train) 

Rfclf_fea.sort_values(by=0, ascending=False).head()
g = sns.barplot(0,"Feature",data = Rfclf_fea.sort_values(by=0, ascending=False)[0:5], palette="Pastel1",orient = "h")

g.set_xlabel("Weight")

g = g.set_title("Random Forest")
# Confusion Matrix

from sklearn.metrics import confusion_matrix



y_pred = rf.predict(X_train)



print(confusion_matrix(Y_train, y_pred))
from sklearn.metrics import classification_report



print(classification_report( Y_train, y_pred))
data
sns.relplot(data=data,x="tenure",y="TotalCharges",hue="Churn")
data.corr()["Churn"]
small=data[["Contract_Month-to-month", "Fiber_optic", "tenure", "SeniorCitizen"]]                                                              
rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(small, Y_train)

clf_score = cross_val_score(rf, small, Y_train, cv=10)

print(clf_score)

clf_score.mean()
small=data[["Contract_Month-to-month", "Fiber_optic", "tenure", "SeniorCitizen","Has_InternetService"]]

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(small, Y_train)

clf_score = cross_val_score(rf, small, Y_train, cv=10)

print(clf_score)

clf_score.mean()
data["percantageCharge"]=-data["TotalCharges"]/data["MonthlyCharges"]
data[["Churn","percantageCharge"]].corr()
small=data[["Contract_Month-to-month", "Fiber_optic", "tenure", "SeniorCitizen","Has_InternetService","percantageCharge"]]

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(small, Y_train)

clf_score = cross_val_score(rf, small, Y_train, cv=10)

print(clf_score)

clf_score.mean()
data["partdepends"]=data["Partner"]*data["Dependents"]

data[["Churn","partdepends"]].corr()
data["tvmovies"]=data["StreamingTV"]+data["StreamingMovies"]

data[["Churn","tvmovies"]].corr()
data["backup_security"]=-data["OnlineSecurity"]*data["OnlineBackup"]

data[["Churn","backup_security"]].corr()
small=data[["Contract_Month-to-month", "Fiber_optic", "tenure", "SeniorCitizen","Has_InternetService","percantageCharge","PaperlessBilling"]]

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(small, Y_train)

clf_score = cross_val_score(rf, small, Y_train, cv=10)

print(clf_score)

clf_score.mean()
small=data[["Contract_Month-to-month", "Fiber_optic", "tenure", "SeniorCitizen","Has_InternetService","percantageCharge","PaperlessBilling","OnlineSecurity"]]

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(small, Y_train)

clf_score = cross_val_score(rf, small, Y_train, cv=10)

print(clf_score)

clf_score.mean()
small=data[["Contract_Month-to-month", "Fiber_optic", "tenure", "SeniorCitizen","Has_InternetService","percantageCharge","PaperlessBilling","OnlineSecurity"]]

from sklearn.feature_selection import SelectKBest,f_classif

kbest= SelectKBest(f_classif, k=4)

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(small, Y_train)

clf_score = cross_val_score(rf, kbest.fit_transform(small,Y_train), Y_train, cv=10)

print(clf_score)

clf_score.mean()
small=data[["Contract_Month-to-month", "Fiber_optic", "tenure", "SeniorCitizen","Has_InternetService","percantageCharge","PaperlessBilling","OnlineSecurity"]]

from sklearn.feature_selection import SelectKBest,f_classif

kbest= SelectKBest(f_classif, k=6)

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(small, Y_train)

clf_score = cross_val_score(rf, kbest.fit_transform(small,Y_train), Y_train, cv=10)

print(clf_score)

clf_score.mean()
from sklearn.feature_selection import SelectKBest,f_classif

kbest= SelectKBest(f_classif, k=6)

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(small, Y_train)

clf_score = cross_val_score(rf, kbest.fit_transform(data.drop(labels = ["Churn"],axis = 1),Y_train), Y_train, cv=10)

print(clf_score)

clf_score.mean()
from sklearn.feature_selection import SelectKBest,f_classif

kbest= SelectKBest(f_classif, k=8)

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(small, Y_train)

clf_score = cross_val_score(rf, kbest.fit_transform(data.drop(labels = ["Churn"],axis = 1),Y_train), Y_train, cv=10)

print(clf_score)

clf_score.mean()
from sklearn.feature_selection import SelectKBest,f_classif

kbest= SelectKBest(f_classif, k=2)

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

clf_score = cross_val_score(rf, kbest.fit_transform(data.drop(labels = ["Churn"],axis = 1),Y_train), Y_train, cv=10)

print(clf_score)

clf_score.mean()
rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

from sklearn.model_selection import train_test_split



X2_train, X2_test, y_train, y_test = train_test_split(data.drop(labels = ["Churn"],axis = 1), Y_train, test_size=0.25, random_state=42)



kbest= SelectKBest(f_classif, k=8)

rf = RandomForestClassifier(random_state=15,**GridRF.best_params_)

rf.fit(X2_train,y_train)

ypred=rf.predict(X2_test)

import sklearn.metrics as metrik

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))