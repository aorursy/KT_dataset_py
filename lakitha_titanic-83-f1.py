

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport



from sklearn import preprocessing

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA



from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import os



import pickle



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data=pd.read_csv('/kaggle/input/titanic/train.csv')

print(train_data.head())

gender_submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(gender_submission.head())





#Uncomment below lined to generate full profile report of the dataframe

# profile = ProfileReport(train_data, title="Pandas Profiling Report")

# profile.to_widgets()


print(train_data.corr())

print(train_data.describe())





#Preprocessing Step



#select needed columns from the dataframe

train_data=train_data[['Pclass','Sex','SibSp','Name','Parch','Embarked','Age','Fare','Survived']]



#Impute Null values

train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean())

train_data['Fare']=train_data['Fare'].fillna(train_data['Fare'].mean())



#New column name Family created using parch and sibsp values (Feature engineering)

train_data['Family']=0



for x in range(len(train_data)):

    Parch=int(train_data['Parch'].iloc[x])

    SibSp=int(train_data['SibSp'].iloc[x])

    family_size=Parch+SibSp

  

    if family_size ==1:

        train_data['Family'].iloc[x]=0

    elif family_size < 5 and family_size > 1:

        train_data['Family'].iloc[x]=1

    elif family_size >=5 :

        train_data['Family'].iloc[x]=2

        

#Feature engineering the column name and adding it to the dataframe as title

dataset=pd.DataFrame()     



dataset['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

dataset['Title'] = dataset['Title'].replace('Master', 'Mr')



dataset['Title'] = dataset['Title'].replace('Miss', 0)

dataset['Title'] = dataset['Title'].replace('Mr', 1)

dataset['Title'] = dataset['Title'].replace('Mrs', 2)

dataset['Title'] = dataset['Title'].replace('Rare', 3)

    

    

train_data['Title']=dataset['Title'].tolist()



#Binning age and fare columns

bins = [0,10,20,30,40,50,60,70,80,90,100]

labels = [0,1,2,3,4,5,6,7,8,9]

train_data['Age_Binned'] = pd.cut(train_data['Age'], bins=bins,labels=labels)

bins = [0,32,64,128,256,512,1025]

labels = [0,1,2,3,4,5]

train_data['Fare_Binned'] = pd.cut(train_data['Fare'], bins=bins,labels=labels)



#encoding all the catergorical data

enc_embarked = preprocessing.LabelEncoder()

train_data['Embarked']=enc_embarked.fit_transform(train_data['Embarked'].tolist())



enc_sex= preprocessing.LabelEncoder()

train_data['Sex']=enc_sex.fit_transform(train_data['Sex'].tolist())



#pickel saving models

filename = 'enc_embarked.sav'

pickle.dump(enc_embarked, open(filename, 'wb'))



filename = 'enc_sex.sav'

pickle.dump(enc_sex, open(filename, 'wb'))





train_data=train_data[['Pclass','Sex','SibSp','Parch','Embarked','Age_Binned','Family','Title','Survived']]

print(train_data)



np_input_data=train_data.iloc[:,0:len(train_data.columns)-1].values





#Scaling the data 

scaler =  MinMaxScaler()

np_input_data_scaled=scaler.fit_transform(np_input_data)



filename = 'scaling.sav'

pickle.dump(scaler, open(filename, 'wb'))



poly = PolynomialFeatures(3)

np_input_data_scaled=poly.fit_transform(np_input_data_scaled)



filename = 'polynomial.sav'

pickle.dump(poly, open(filename, 'wb'))



pca = PCA(n_components=5)

np_input_data_scaled=pca.fit_transform(np_input_data_scaled)



filename = 'pca.sav'

pickle.dump(pca, open(filename, 'wb'))



np_output_data=train_data.iloc[:,len(train_data.columns)-1].values



#Train split data

X_train, X_test, y_train, y_test = train_test_split(np_input_data_scaled, np_output_data, test_size=0.1, random_state=0)



print(X_train)

print(y_train)







clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)

#Predict the response for test dataset

y_pred = clf.predict(X_test)

print('DT')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Accuracy F1:",metrics.f1_score(y_test, y_pred, average='macro'))





filename = 'dt.sav'

pickle.dump(clf, open(filename, 'wb'))



#Create a Gaussian Classifier

clf= RandomForestClassifier(criterion='gini',

                                           n_estimators=1750,

                                           max_depth=7,

                                           min_samples_split=6,

                                           min_samples_leaf=6,

                                           max_features='auto',

                                           oob_score=True,

                                           random_state=42,

                                           n_jobs=-1,

                                           verbose=1)

#Train the model using the training sets y_pred=clf.predict(X_test)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print('Random Forest')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Accuracy F1:",metrics.f1_score(y_test, y_pred, average='macro'))



filename = 'random_forest.sav'

pickle.dump(clf, open(filename, 'wb'))



model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1.0, gamma=1.5, gpu_id=-1,

              importance_type='gain', interaction_constraints='',

              learning_rate=0.02, max_delta_step=0, max_depth=6,

              min_child_weight=1, monotone_constraints='()',

              n_estimators=400, n_jobs=1, nthread=1, num_parallel_tree=1,

              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

              subsample=1.0, tree_method='exact', validate_parameters=1,

              verbosity=None)





model.fit(X_train, y_train)

y_pred=model.predict(X_test)

print('XGboost')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Accuracy F1:",metrics.f1_score(y_test, y_pred, average='macro'))

filename = 'xgb.sav'

pickle.dump(model, open(filename, 'wb'))



from sklearn.ensemble import GradientBoostingClassifier



clf = GradientBoostingClassifier( loss='deviance',n_estimators=100,learning_rate=0.1)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print('GradientBoosting')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Accuracy F1:",metrics.f1_score(y_test, y_pred, average='macro'))





filename = 'gradien_boost.sav'

pickle.dump(clf, open(filename, 'wb'))


#Below code is used to test the model generated above to obtain the required submission file.



test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

gender_sub=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(test_data.head())





test_data=pd.merge(test_data, gender_sub, on=['PassengerId'])



test_data=test_data[['Pclass','Sex','SibSp','Name','Parch','Embarked','Age','Fare','Survived']]

print(train_data)



test_data['Age']=test_data['Age'].fillna(test_data['Age'].mean())

test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())

# train_data=train_data.drop_duplicates()



test_data['Family']=0





for x in range(len(test_data)):

    Parch=int(test_data['Parch'].iloc[x])

    SibSp=int(test_data['SibSp'].iloc[x])

    family_size=Parch+SibSp

  

    if family_size ==1:

        test_data['Family'].iloc[x]=0

    elif family_size < 5 and family_size > 1:

        test_data['Family'].iloc[x]=1

    elif family_size >=5 :

        test_data['Family'].iloc[x]=2

    

dataset=pd.DataFrame()     



dataset['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

dataset['Title'] = dataset['Title'].replace('Master', 'Mr')



dataset['Title'] = dataset['Title'].replace('Miss', 0)

dataset['Title'] = dataset['Title'].replace('Mr', 1)

dataset['Title'] = dataset['Title'].replace('Mrs', 2)

dataset['Title'] = dataset['Title'].replace('Rare', 3)

    

    

test_data['Title']=dataset['Title'].tolist()









bins = [0,10,20,30,40,50,60,70,80,90,100]

labels = [0,1,2,3,4,5,6,7,8,9]

test_data['Age_Binned'] = pd.cut(test_data['Age'], bins=bins,labels=labels)

bins = [0,32,64,128,256,512,1025]

labels = [0,1,2,3,4,5]

test_data['Fare_Binned'] = pd.cut(test_data['Fare'], bins=bins,labels=labels)







enc_sex = pickle.load(open('./enc_sex.sav', 'rb'))

test_data['Sex'] = enc_sex.transform(test_data['Sex'].tolist())



enc_embarked = pickle.load(open('./enc_embarked.sav', 'rb'))

test_data['Embarked'] = enc_embarked.transform(test_data['Embarked'].tolist())



test_data=test_data[['Pclass','Sex','SibSp','Parch','Embarked','Age_Binned','Family','Title','Survived']]

print(test_data)



np_input_data=test_data.iloc[:,0:len(train_data.columns)-1].values







scaler = pickle.load(open('./scaling.sav', 'rb'))

np_input_data_scaled = scaler.transform(np_input_data)







poly = pickle.load(open('./polynomial.sav', 'rb'))

np_input_data_scaled = poly.transform(np_input_data_scaled)





pca = pickle.load(open('./pca.sav', 'rb'))

np_input_data_scaled = pca.transform(np_input_data_scaled)



np_output_data=train_data.iloc[:,len(train_data.columns)-1].values



loaded_model = pickle.load(open('./gradien_boost.sav', 'rb'))

result=loaded_model.predict(np_input_data_scaled)

print(len(result))

print(len(np_output_data))

print(result)

print(np_output_data)

# print("Accuracy:",metrics.accuracy_score(result, np_output_data))

# print("Accuracy F1:",metrics.f1_score(result, np_output_data, average='macro'))





test_data=pd.read_csv('/kaggle/input/titanic/test.csv')

print(len(test_data['PassengerId'].tolist()))

result_df=pd.DataFrame()

result_df['PassengerId']=test_data['PassengerId'].tolist()

result_df['Survived']=result

print(result_df)



result_df.to_csv('submission.csv',index=False)