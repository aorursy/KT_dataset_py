#load all helpful packages

import numpy as np # for linear algebra

import pandas as pd # to import csv and for data manipulation

import matplotlib.pyplot as plt # to plot graph

from sklearn.preprocessing import LabelEncoder # to encode categorical data in values 

from sklearn.ensemble import RandomForestClassifier # Random forest classifier

from sklearn.model_selection import train_test_split # to split the data

from sklearn.model_selection import GridSearchCV # Grid Search

from sklearn.feature_selection import RFE # Recursive Feature Elimination

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
#load dataset

data=pd.read_csv("../input/LasVegas.csv")
data.head(1)
data.columns
#checking for missing values

data.info()
data.describe(include=np.number)
data.describe(exclude=np.number)
# handling categorical data with only 2 unique values

# create new columns with get_dummies()

swimming_pool=pd.get_dummies(data['Swimming Pool'],drop_first=True)

exercise_room=pd.get_dummies(data['Exercise Room'],drop_first=True)

basketball_court=pd.get_dummies(data['Basketball Court'],drop_first=True)

yoga_classes=pd.get_dummies(data['Yoga Classes'],drop_first=True)

club=pd.get_dummies(data['Club'],drop_first=True)

free_wifi=pd.get_dummies(data['Free Wifi'],drop_first=True)
# renaming the new columns 

swimming_pool.rename(columns={'YES':'Swimming Pool'},inplace=True)

exercise_room.rename(columns={'YES':'Exercise Room'},inplace=True)

basketball_court.rename(columns={'YES':'Basketball Court'},inplace=True)

yoga_classes.rename(columns={'YES':'Yoga Classes'},inplace=True)

club.rename(columns={'YES':'Club'},inplace=True)

free_wifi.rename(columns={'YES':'Free Wifi'},inplace=True)
# Replace old columns with the new ones

data.drop(columns=['Swimming Pool','Exercise Room', 'Basketball Court', 'Yoga Classes', 'Club','Free Wifi'],axis=1,inplace=True)

data=pd.concat([data,swimming_pool,exercise_room,basketball_court,yoga_classes,club,free_wifi],axis=1)
# List of all features with categorical values

categorical=data.describe(exclude=np.number).columns

categorical=np.delete(categorical,4)

for i in categorical:

    print(data[i].unique())
# Encode categorical features with LabelEncoder

# Encode labels with value between 0 and n_classes-1

le=LabelEncoder()

for i in categorical:

    data[i]=le.fit_transform(data[i])
# handling multiple values for feature Hotel Stars

hotel_stars=data['Hotel stars']

stars=[sum(map(int,x.split(',')))/len(x.split(',')) for x in hotel_stars] 

data['Hotel stars']=pd.Series(stars)

data['Score']=data['Score'].astype('float64')
# shape of dataset

data.shape
# Features and labels

X=data.drop(['Score',],axis=1)

y=data['Score']
# splitting data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# count of instances of each class

y_train=pd.DataFrame(y_train,columns=['Score'])

train=pd.concat([X_train,y_train],axis=1)

train.Score.value_counts()
# splitting instances of different classes in a list of dataframes

data_class=[]

for i in range(len(data.Score.value_counts())):

    data_class.append(data[data['Score']==i+1])
# OverSampling training data 

train=pd.DataFrame()

no_of_rows=data_class[-1].shape[0]

for i in range(len(data_class)):

    no_of_rows=no_of_rows

    sampled_data=data_class[i].sample(no_of_rows,replace=True if no_of_rows>data_class[i].shape[0] else False)

    train=pd.concat([train,sampled_data],axis=0)
# Features and Labels 

X_train=train.drop(['Score'],axis=1)

y_train=train['Score']
# Training Random Forest Classifier

rfc=RandomForestClassifier(40)

rfc.fit(X_train,y_train)

y_pred=rfc.predict(X_test)

print(rfc.score(X_test,y_test))
# Classification report, Confusion matrix and Accuracy score

print("\nClassification Report\n")

print(classification_report(y_test,y_pred))

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test,y_pred))

print("\nAccuracy score\n")

print(accuracy_score(y_test,y_pred))
# grid search to find how many features to select as most relevant to the model

param_grid={'n_features_to_select':np.arange(1,20)}

grid=GridSearchCV(RFE(estimator=RandomForestClassifier(random_state=0)),param_grid)

grid.fit(X_train,y_train)
grid.best_params_
# recursive feature elimination for Random Forest Classifier

rfe = RFE(estimator = RandomForestClassifier(),n_features_to_select=grid.best_params_['n_features_to_select'])

rfe.fit(X_train, y_train)

feature_list = pd.DataFrame({'col':list(X_train.columns.values),'sel':list(rfe.support_ *1)})

print("Most contributing features in Score")

print(feature_list[feature_list.sel==1].col.values)
# Subset train data based on selected features

X_sel = pd.DataFrame(X_train, columns=(feature_list[feature_list.sel==1].col.values))

X_sel_t = pd.DataFrame(X_test, columns=(feature_list[feature_list.sel==1].col.values))
# improved score

rfc=RandomForestClassifier(40,random_state=0)

rfc.fit(X_sel,y_train)

print(rfc.score(X_sel_t,y_test))