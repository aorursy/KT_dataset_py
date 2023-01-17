# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as se

import keras

import pickle 

from keras.layers import Dense

from keras.models import Sequential

from keras.layers import Dropout

from keras.models import save_model

from IPython.display import Image

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,roc_auc_score,f1_score

from xgboost import XGBClassifier

encoder = LabelEncoder()

ohe = OneHotEncoder()

sc = StandardScaler()

import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
df  = pd.read_csv('../input/heart.csv')
df.head()
df.shape
df.describe()
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
df['sex'][df['sex']==1] = 'male'

df['sex'][df['sex']==0] = 'female'
df['chest_pain_type'][df['chest_pain_type'] == 1] = 'typical angina'

df['chest_pain_type'][df['chest_pain_type'] == 2] = 'atypical angina'

df['chest_pain_type'][df['chest_pain_type'] == 3] = 'non-anginal pain'

df['chest_pain_type'][df['chest_pain_type'] == 4] = 'asymptomati'

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
df
se.countplot(data=df,x = 'sex')
impaact_on_gender = pd.crosstab(df['target'],df['sex'])

impaact_on_gender

impaact_on_gender.plot(kind = 'bar')
df[df.duplicated()  == True]
df.drop_duplicates(inplace=True)
male_gender = len(df[df.sex == 1])

female_gender = len(df[df.sex == 0])



print("In this dataset there exists {0} male subjects and {1} female subjects which computes to {2}% for males and {3}% for females.".format(male_gender, female_gender, round((male_gender/len(df.sex)), 2)*100, round((female_gender/len(df.sex)), 2)*100))
impaact_on_gender
male =len(df[df['sex'] == 'male'])

female = len(df[df['sex']== 'female'])



plt.figure(figsize=(8,6))



# Data to plot

labels = 'Male','Female'

sizes = [male,female]

colors = ['skyblue', 'yellowgreen']

explode = (0, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=90)

 

plt.axis('equal')

se.pairplot(df)
df['chest_pain_type'].value_counts()
plt.figure(figsize=(8,6))



# Data to plot

labels = 'Chest Pain Type:0','Chest Pain Type:1','Chest Pain Type:2','Chest Pain Type:3'

sizes = [len(df[df['chest_pain_type'] == 0]),len(df[df['chest_pain_type'] == 'atypical angina']),

         len(df[df['chest_pain_type'] == 'non-anginal pain']),

         len(df[df['chest_pain_type'] == 'non-anginal pain'])]

colors = ['red', 'green','orange','gold']

explode = (0, 0,0,0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=180)

 

plt.axis('equal')

plt.show()
df.head()
plt.figure(figsize=(8,6))



# Data to plot

labels = 'greater than 120mg/ml','greater than 120mg/ml'

sizes = [len(df[df['fasting_blood_sugar'] == 'greater than 120mg/ml']),len(df[df['fasting_blood_sugar'] == 'lower than 120mg/ml'])]

colors = ['red', 'green','orange','gold']

explode = (0,0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,shadow=True, startangle=180)

 

plt.axis('equal')

plt.show()
plt.figure(figsize=(8,6))

se.countplot(x = df['rest_ecg'],data = df)
plt.figure(figsize=(14,8))

se.heatmap(df.corr(), annot = True, cmap='summer',linewidths=.1)

plt.show()
df.info()
df.hist(figsize=(30,30))

plt.show()
se.pairplot(df, hue="target",diag_kind='kde',kind='scatter')
numerical = df.select_dtypes(exclude=['object'])

categorical = df.select_dtypes(include=['object'])

numerical.columns
categorical.columns
se.scatterplot(x = df['max_heart_rate_achieved'],y = df['resting_blood_pressure'],data= numerical,hue = df['age'])
se.scatterplot(x = df['st_depression'],y = df['resting_blood_pressure'],data= numerical,hue = df['age'])
numerical.plot(kind = 'hist')
categorical.columns
df.head(100)
df['chest_pain_type'].value_counts()

df['chest_pain_type'].replace(0,'atypical angina',inplace=True)
df['st_slope'].value_counts()

df['st_slope'].replace(0,'flat',inplace=True)
df['thalassemia'].value_counts()

df['thalassemia'].replace(0,'fixed defect',inplace=True)

df['sex'] = encoder.fit_transform(df['sex'])

df['chest_pain_type'] = encoder.fit_transform(df['chest_pain_type'])

df['fasting_blood_sugar'] = encoder.fit_transform(df['fasting_blood_sugar'])

df['rest_ecg'] = encoder.fit_transform(df['chest_pain_type'])

df['st_slope'] = encoder.fit_transform(df['st_slope'])

df['thalassemia'] = encoder.fit_transform(df['thalassemia'])

X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values
onehotencoder = OneHotEncoder()

X = onehotencoder.fit_transform(X).toarray()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.33)
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
models = []

models.append(('Log Classifier', LogisticRegression()))

models.append(('Support Vector Classifier', SVC()))

models.append(('DT Classifier', DecisionTreeClassifier(random_state=0,max_depth = 10)))

models.append(('Random Forest', RandomForestClassifier(random_state=0,n_estimators=100,min_samples_split=2,min_impurity_decrease=0.1)))

models.append(('Extra Tree', ExtraTreeClassifier(random_state=0,min_samples_leaf=1,max_depth=10)))

models.append(('GB Classifier Accuracy',GradientBoostingClassifier(random_state=0, n_estimators=500, learning_rate=1.0)))

models.append(('XGB Classifier Accuracy',XGBClassifier()))





#evaluate each model in turn

results = []

names = []

for name, model in models:

    model = model.fit(X_train,y_train.ravel())

    predict = model.predict(X_test)

    score = accuracy_score(y_test.ravel(), predict)

    names.append(name)

    msg = name,score

    print(msg)

    

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

accuracies = {'Linear Classifier':76,'Support Vector Classifier':76,'Random Forest':62,'Extra Tree':62,'GB Classifier Accuracy':77,"XGB Classifier Accuracy":76}



se.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

se.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

plt.show()
input_dim = X_train.shape[1]

regressor = Sequential()

regressor.add(Dense(13,activation='relu',input_dim = input_dim,kernel_initializer = 'uniform'))

regressor.add(Dense(13,activation='relu'))

regressor.add(Dense(13,activation='relu'))

regressor.add(Dense(13,activation='relu'))

regressor.add(Dropout(0.5))

regressor.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))

regressor.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



history = regressor.fit(X_train,y_train,batch_size=64,epochs=700,validation_data=(X_test,y_test),verbose=2)

predictions_ann=regressor.predict(X_test)

print(history.history.keys())

regressor.evaluate(x = X_test,y = y_test)
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
from sklearn.model_selection import RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score,roc_curve,auc

from sklearn.model_selection import cross_val_score

knn =KNeighborsClassifier()

lr_model = LogisticRegression()

svc_model = SVC(probability=True)

dt_model = DecisionTreeClassifier()

rf_model = RandomForestClassifier()

gb_model = GradientBoostingClassifier()

params_lg  = {'penalty':['l1','l2'],

         'C':[0.01,0.1,1,10,100],

         'class_weight':['balanced',None]}

params_knn = {'n_neighbors':[i for i in range(1,30,2)]}

params_svc = {'kernel':['linear','poly','rbf'],'gamma':[0.1, 1, 10, 100]}

params_dt  = {'max_features': ['auto', 'sqrt', 'log2'],

          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 

          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}

param_rf = {'n_estimators':[1, 2, 4, 8, 16, 32, 64, 100, 200],'max_depth':[int(x) for x in np.linspace(1, 32, 32, endpoint=True)]}

param_gb = {'learning_rate':[1, 0.5, 0.25, 0.1, 0.05, 0.01],'n_estimators':[1, 2, 4, 8, 16, 32, 64, 100, 200],'max_depth':[int(x) for x in np.linspace(1, 32, 32, endpoint=True)]}
knn_model = RandomizedSearchCV(estimator=knn,param_distributions=params_knn)

knn_model.fit(X_train,y_train)

knn_model.best_params_
print('Accuracy Score: ',accuracy_score(y_test,knn_model.predict(X_test)))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,knn_model.predict(X_test)),5)*100,'%')
predict_probabilities = knn_model.predict_proba(X_test)[:,1]

#Create true and false positive rates

false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,predict_probabilities)

#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_knn,true_positive_rate_knn)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()

print("AUC SCORE FOR KNN:",roc_auc_score(y_test,predict_probabilities))

lr_model = RandomizedSearchCV(estimator=lr_model,param_distributions=params_lg)

lr_model.fit(X_train,y_train)

print(lr_model.best_params_)

print('Accuracy Score: ',accuracy_score(y_test,lr_model.predict(X_test)))

print('Using Log Regression we get an accuracy score of: ',

      round(accuracy_score(y_test,lr_model.predict(X_test)),5)*100,'%')
predict_probabilities = lr_model.predict_proba(X_test)[:,1]

#Create true and false positive rates

false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,predict_probabilities)

#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_knn,true_positive_rate_knn)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()

print("AUC SCORE FOR LR:",roc_auc_score(y_test,predict_probabilities))

svc_model = RandomizedSearchCV(estimator=svc_model,param_distributions=params_svc)

svc_model.fit(X_train,y_train)

print(svc_model.best_params_)

print('Accuracy Score: ',accuracy_score(y_test,svc_model.predict(X_test)))

print('Using SVC we get an accuracy score of: ',

      round(accuracy_score(y_test,svc_model.predict(X_test)),5)*100,'%')
predict_probabilities = svc_model.predict_proba(X_test)[:,1]

#Create true and false positive rates

false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,predict_probabilities)

#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_knn,true_positive_rate_knn)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()

print("AUC SCORE FOR SVC:",roc_auc_score(y_test,predict_probabilities))

dt_model = RandomizedSearchCV(estimator=dt_model,param_distributions=params_dt)

dt_model.fit(X_train,y_train)

print(dt_model.best_params_)

print('Accuracy Score: ',accuracy_score(y_test,dt_model.predict(X_test)))

print('Using Decision tree we get an accuracy score of: ',

      round(accuracy_score(y_test,dt_model.predict(X_test)),5)*100,'%')
predict_probabilities = dt_model.predict_proba(X_test)[:,1]

#Create true and false positive rates

false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,predict_probabilities)

#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_knn,true_positive_rate_knn)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()

print("AUC SCORE FOR DecisionTree:",roc_auc_score(y_test,predict_probabilities))

rf_model = RandomizedSearchCV(estimator=rf_model,param_distributions=param_rf)

rf_model.fit(X_train,y_train)

print(rf_model.best_params_)

print('Accuracy Score: ',accuracy_score(y_test,rf_model.predict(X_test)))

print('Using RF we get an accuracy score of: ',

      round(accuracy_score(y_test,rf_model.predict(X_test)),5)*100,'%')
predict_probabilities = rf_model.predict_proba(X_test)[:,1]

#Create true and false positive rates

false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,predict_probabilities)

#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_knn,true_positive_rate_knn)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()

print("AUC SCORE FOR RandomForest:",roc_auc_score(y_test,predict_probabilities))

gb_model = RandomizedSearchCV(estimator=gb_model,param_distributions=param_gb)

gb_model.fit(X_train,y_train)

print(gb_model.best_params_)

print('Accuracy Score: ',accuracy_score(y_test,gb_model.predict(X_test)))

print('Using RF we get an accuracy score of: ',

      round(accuracy_score(y_test,gb_model.predict(X_test)),5)*100,'%')
predict_probabilities = gb_model.predict_proba(X_test)[:,1]

#Create true and false positive rates

false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,predict_probabilities)

#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_knn,true_positive_rate_knn)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()

print("AUC SCORE FOR Gradient boosting:",roc_auc_score(y_test,predict_probabilities))

# Evaluate the model using 10-fold cross-validation

clf=rf_model



scores = cross_val_score(clf,X_train,y_train, scoring='accuracy', cv=10)

print (scores)

print (scores.mean())



SEED=42

mean_auc = 0.0

n = 10  # repeat the CV procedure 10 times to get more precise results

for i in range(n):

    # for each iteration, randomly hold out 20% of the data as CV set

    X_train, X_cv, y_train, y_cv = train_test_split(

    X_train,y_train, test_size=.20, random_state=i*SEED)



    # train model and make predictions

    clf.fit(X_train, y_train) 

    preds = clf.predict_proba(X_cv)[:, 1]



    # compute AUC metric for this CV fold

    fpr, tpr, thresholds = roc_curve(y_cv, preds)

    roc_auc = auc(fpr, tpr)

    print ("AUC (fold %d/%d): %f" % (i + 1, n, roc_auc))

    mean_auc += roc_auc



print ("Mean AUC: %f" % (mean_auc/n)) 
# Save the trained model as a pickle string. 

model_pickle_path = 'rf_model.pkl'

ohe_model_path = 'ohe_model.pkl'

scaling_model_path = 'scaled_model.pkl'

 

# Create an variable to pickle and open it in write mode

model_pickle = open(model_pickle_path, 'wb')

ohe_pickle = open(ohe_model_path, 'wb')

scaled_pickle = open(scaling_model_path, 'wb')



saved_model = pickle.dump(rf_model, model_pickle)

saved_ohe = pickle.dump(onehotencoder, ohe_pickle)

saved_sc = pickle.dump(sc, scaled_pickle)



model_pickle.close()

ohe_pickle.close()

scaled_pickle.close()

# Load the pickled model 

rf_from_pickle = open(model_pickle_path, 'rb')

ohe_from_pickle = open(ohe_model_path,'rb')

sc_from_pickle = open(scaling_model_path,'rb')

#decision_tree_model = pickle.load(decision_tree_model_pkl)

rf_from_pickle = pickle.load(rf_from_pickle)

ohe_from_pickle = pickle.load(ohe_from_pickle)

sc_from_pickle = pickle.load(sc_from_pickle)



print(rf_from_pickle)

print(ohe_from_pickle)

print(sc_from_pickle)