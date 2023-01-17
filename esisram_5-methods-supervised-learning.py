#import libraries are needed

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,BaggingClassifier,AdaBoostClassifier

from matplotlib.pyplot import show

%matplotlib inline
#create function to calling data in file train.csv

def importdata(data):

    #Memanggil data train.csv

    data=pd.read_csv(data)

    #Shape data (row,col)

    print('sum of observation,columns = ',data.shape)

    #Check , is it any duplicated data?

    print('Duplicated data = ',data.duplicated().sum())

    #if any duplicated data, drop it

    data=data.drop_duplicates()

    #check shape of data

    print('sum of (observation,columns) after remove duplicate = ', data.shape)

    #give it back the data

    return data

#Calling funtions of import data

data_train=importdata('../input/train.csv')
#Separates data_train into x and y

def sep_xy(data,output):

    #data_output merupakan nilai dari variabel output

    data_output=data[output]

    #data_input merupakan nilai dari variabel kecuali variabel output

    data_input=data.drop(output,axis=1)

    #kembalikan nilai data_input,data_output

    return data_input, data_output

x,y=sep_xy(data_train,'Survived')
#Separates x and y into training and testing with size 80/20 using train_test_split library

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=105)
#Info each attribute

x_train.info()
#---------------------------------------

#Noted:

#Numerical data:

#SibSp          668 non-null int64

#Parch          668 non-null int64

#Age            529 non-null float64

#Fare           668 non-null float64

#categorical data:

#Pclass         668 non-null int64

#Sex            668 non-null object

#Embarked       666 non-null object

#Drop data:

#Name           668 non-null object

#Ticket         668 non-null object

#Cabin          146 non-null object

#PassengerId    668 non-null int64

#---------------------------------------
#create functions Separates x_train into numerical and categorical

def numerical_categorical(data,categoric_type,drop_data):

    #create Numerical

    data_numerical=data._get_numeric_data()

    data_numerical=data_numerical.drop(categoric_type,axis=1)

    #create Categorical

    data_categorical=data.drop(data_numerical.columns, axis=1)

    data_categorical=data_categorical.drop(drop_data,axis=1)

    return data_numerical, data_categorical

x_train_numerical, x_train_categorical=numerical_categorical(x_train,['Pclass','PassengerId'],['Name','Ticket','Cabin','PassengerId'])
#info x_train_numerical

x_train_numerical.info()
#Statistika Deskriptif for numerical attribute

x_train_numerical.describe()
#check sum of missing value in x_train_numerical

print('Sum of Missing Values = ', np.count_nonzero(x_train_numerical.isnull()))

print('Sum of Non Missing Values = ', np.count_nonzero(x_train_numerical.notnull()))

#Persen data missing

print('Persen Missing = ', 

      np.count_nonzero(x_train_numerical.isnull())/(np.count_nonzero(x_train_numerical.isnull()) + np.count_nonzero(x_train_numerical.notnull())))                            
#check how much each attribute in x_train_numerical has null value 

x_train_numerical.isnull().sum()
#How many persen missing value in attribute Age?

print('Persen Missing Of Age = ',

      x_train_numerical['Age'].isnull().sum()/(x_train_numerical['Age'].isnull().sum() + x_train_numerical['Age'].notnull().sum()))
#create function to imputer missing value in x_train_numerical

def imput_numerical(data,missing_value,method):

    #create imputer

    imput_numeric=Imputer(missing_values=missing_value, strategy=method)

    #fit imputer into data

    imput_numeric.fit(data)

    #transfor data with imputer for change missing value into kind of imputer, ex:median

    data_imput_numerical=pd.DataFrame(imput_numeric.transform(data))

    #return column name

    data_imput_numerical.columns=data.columns

    #return index 

    data_imput_numerical.index=data.index

    

    return data_imput_numerical, imput_numeric

x_train_imput_numerical,imput_numerical=imput_numerical(x_train_numerical,'NaN','median')
#info x_train_categorical

x_train_categorical.info()
#check sum of missing value in x_train_categorical

print('Sum of Missing Values = ', np.count_nonzero(x_train_categorical.isnull()))

print('Sum of Non Missing Values = ', np.count_nonzero(x_train_categorical.notnull()))

#Persen data missing

print('Persen Missing = ', 

      np.count_nonzero(x_train_categorical.isnull())/(np.count_nonzero(x_train_categorical.isnull()) + np.count_nonzero(x_train_categorical.notnull())))                            
#check how much each attribute in x_train_categorical has null value 

x_train_categorical.isnull().sum()
#How many persen missing value in attribute Embarked?

print('Persen Missing Of Embarked = ',

      x_train_categorical['Embarked'].isnull().sum()/(x_train_categorical['Embarked'].isnull().sum() + x_train_categorical['Embarked'].notnull().sum()))
#check kind of kategorik whom has much proportion in each attribute

pclass_count=x_train_categorical['Pclass'].value_counts(True)

sex_count=x_train_categorical['Sex'].value_counts(True)

embarked_count=x_train_categorical['Embarked'].value_counts(True)

print(pclass_count,'\n')

print(sex_count,'\n')

print(embarked_count)
# let create visualiztion on x_train_categorical

#How much size to draw shape

plt.figure(figsize=(5,5))

#create barplot with Embarked data

sns.barplot(embarked_count.index, embarked_count.values)

#giving name on ylabel

plt.ylabel('count', fontsize=12)

#giving name on xlabel

plt.xlabel('Embarked', fontsize=12)

#giving strecht on barplot

plt.grid()

#show the result

plt.show()
# let create visualiztion on x_train_categorical

#How much size to draw shape

plt.figure(figsize=(5,5))

#create barplot with Embarked data

sns.barplot(pclass_count.index, pclass_count.values)

#giving name on ylabel

plt.ylabel('count', fontsize=12)

#giving name on xlabel

plt.xlabel('Pclass', fontsize=12)

#giving strecht on barplot

plt.grid()

#show the result

plt.show()
# let create visualiztion on x_train_categorical

#How much size to draw shape

plt.figure(figsize=(5,5))

#create barplot with Embarked data

sns.barplot(sex_count.index, sex_count.values)

#giving name on ylabel

plt.ylabel('count', fontsize=12)

#giving name on xlabel

plt.xlabel('Sex', fontsize=12)

#giving strecht on barplot

plt.grid()

#show the result

plt.show()
plt.figure(figsize=(8,8))

plt.grid()

sns.countplot(x="Pclass", hue="Sex", data=x_train_categorical)
total = float(len(x_train_categorical))  

ax = sns.countplot(x="Pclass", hue="Sex", data=x_train_categorical) 

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format(height/total),

            ha="center") 

show()
plt.figure(figsize=(8,8))

plt.grid()

sns.countplot(x="Pclass", hue="Embarked", data=x_train_categorical)
#create function to imputer missing value in x_train_categorical

def imput_categorical(data,value):

    #fill missing value S in Embarked 

    data_categorical_imput=data.fillna(value)

    #Change data type into string beacuse the next step to create dummy must string

    data_categorical_imput=data_categorical_imput.astype('str')

    #create dummy variabel

    data_categorical_imput=pd.get_dummies(data_categorical_imput,drop_first=True)

    return data_categorical_imput, data_categorical_imput.columns

x_train_imput_categorical, dummy_columns=imput_categorical(x_train_categorical,'S')
#Combine x_train_imput_numerical and x_train_imput_categorical

x_train_preprocessed=pd.concat([x_train_imput_numerical,x_train_imput_categorical],axis=1)

x_train_preprocessed.head()
#info from 2 combine x_train imput numerical and categorical

x_train_preprocessed.info()
def male_female_child(data):

    if (data['Age'] > 16 and data['Sex_male']==1 ):

        return 'male' 

    elif (data['Age'] > 16 and data['Sex_male']==0 ):

        return 'female'

    elif (data['Age'] <= 16 and data['Sex_male']==1 ):

        return 'child'

    elif (data['Age'] <= 16 and data['Sex_male']==0 ):

        return 'child'

    else:

        return data['Sex_male']

x_train_preprocessed['Person'] = x_train_preprocessed.apply(male_female_child, axis=1)

x_train_preprocessed.head()
xy_train=pd.concat([x_train_preprocessed,y_train],axis=1)

xy_train.head()
plt.figure(figsize=(8,8))

plt.grid()

sns.countplot(x="Survived", hue="Person", data=xy_train)
xy_train['Person'].value_counts(True)
xy_train.head()
# Buatalah kolom Alone dengan menambahkan titanic_df.Parch + titanic_df.SibSp

# Kemudian tampilkan series sesuai di bawah ini

Alone=data_train.Parch+data_train.SibSp

Alone=pd.DataFrame(Alone)

Alone.columns=['Alone']
xy_train=xy_train.join(Alone)
xy_train.head()
def alone(data):

    if (data['Alone'] == 1):

        return 'With Family' 

    else:

        return 'Alone'

xy_train['Alone'] = xy_train.apply(alone, axis=1)
plt.figure()

sns.barplot(xy_train['Alone'].value_counts().index,xy_train['Alone'].value_counts().values)

plt.ylabel('count', fontsize=12)

plt.xlabel('Alone', fontsize=12)

plt.show()
plt.figure(figsize=(8,8))

sns.countplot(x="Survived", hue="Alone", data=xy_train)

plt.xlabel('Survivor')

#xy_train-->combine x_train_preprocessed and y_train

x_train_preprocessed=x_train_preprocessed.drop(['Person'],axis=1)
#Create function to Standardize x_train_preprocessed

def standardize(data):

    scaler=StandardScaler()

    scaler.fit(data)

    data_standardize=pd.DataFrame(scaler.transform(data))

    data_standardize.columns=data.columns

    data_standardize.index=data.index

    return data_standardize, scaler

x_train_norm, standardize=standardize(x_train_preprocessed)
#Function logistics regression

def logreg_fit(x_train, y_train):

    logreg = LogisticRegression()

    hyperparam = {'C': [1000, 333.33, 100, 33.33, 10, 3.33, 10, 3.33, 1, 0.33, 0.1, 0.033, 0.01, 0.0033, 

                        0.001, 0.00033, 0.0001]}

    #n_jobs=2 artinya tergantung spech leptop

    random_logreg = RandomizedSearchCV(logreg, param_distributions = hyperparam, cv = 10,

                                    n_iter = 10, n_jobs=-1, random_state = 125)

    random_logreg.fit(x_train, y_train)

    print ("Best Accuracy", random_logreg.best_score_)

    print ("Best Param", random_logreg.best_params_)

    return random_logreg
best_logreg = logreg_fit(x_train_norm, y_train) 
best_logreg
logreg = LogisticRegression(C=best_logreg.best_params_.get('C'))

logreg.fit(x_train_norm, y_train)
#Function Decision Tree

def dectree_fit(x_train, y_train, scoring = 'accuracy'):

    dectree = DecisionTreeClassifier(random_state=125)

    #min_samples_split untuk 

    #max_features untuk

    #max_depth untuk

    hyperparam = {'min_samples_split': [3, 5, 7, 9, 13, 17, 21, 27, 33, 41, 50, 60, 80, 100],

                  'max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75],

                  'max_depth': [2, 4, 6, 8]                 }



    random_dectree = RandomizedSearchCV(dectree, 

                                        param_distributions= hyperparam, 

                                        cv = 20, n_iter = 20, 

                                        scoring = scoring, n_jobs=10, random_state = 125)

    

    random_dectree.fit(x_train, y_train)

    

    print ("Best Accuracy", random_dectree.best_score_)

    print ("Best Param", random_dectree.best_params_)

    

    return random_dectree
best_dectree = dectree_fit(x_train_norm, y_train)
best_dectree.score(x_train_norm,y_train)
dectree = DecisionTreeClassifier(random_state=125,

                                  min_samples_split= best_dectree.best_params_.get('min_samples_split'),

                                  max_features = best_dectree.best_params_.get('max_features'),

                                  max_depth = best_dectree.best_params_.get('max_depth'))

dectree.fit(x_train_norm, y_train)
##Function Bagging Boostrap

def bagging_fit(x_train, y_train, scoring = 'accuracy'):

    dectree = DecisionTreeClassifier(random_state=125)

    

    bagging = BaggingClassifier(base_estimator = dectree, 

                                random_state=125)

    

    #base_estimator_min_samples_split untuk

    #base_estimator_max_depth untuk

    #n_estimators untuk

    hyperparam = {'base_estimator__min_samples_split': [3, 5, 7, 9, 13, 17, 21, 27, 33, 41, 50, 60, 80, 100],

                  'base_estimator__max_depth': [2, 4, 6, 8],

                  'n_estimators': [100, 200, 300, 500, 750, 1000]}

    # 'base_estimator__' sebelum 'min_samples_leaf' menandakan hyperparameter yang dicari ada di dalam base estimatornya

    # dalam hal ini berarti decTree

    # (min_samples_leaf ada di dalam decTree)

    

    #scoring ada berapa macam selain accuracy--->

    random_bagging = RandomizedSearchCV(bagging, 

                                        param_distributions = hyperparam, 

                                        cv = 20, 

                                        n_iter = 20, 

                                        scoring = scoring,

                                        n_jobs = 5, 

                                        random_state = 125)

    

    random_bagging.fit(x_train, y_train)

    

    print("Best Accuracy", random_bagging.best_score_)

    print("Best Param", random_bagging.best_params_)

    return random_bagging
best_bagging = bagging_fit(x_train_norm, y_train)
dectree_bagging = DecisionTreeClassifier(min_samples_split = best_bagging.best_params_.get('base_estimator__min_samples_split'),

                                         max_depth = best_bagging.best_params_.get('base_estimator__max_depth'),

                                         random_state=125)

bagging = BaggingClassifier(base_estimator = dectree_bagging, 

                            n_estimators = best_bagging.best_params_.get('n_estimators'),

                            random_state=125, n_jobs=2)

bagging.fit(x_train_norm, y_train)
#Function Random Forest

def rf_fit(x_train, y_train, scoring = 'accuracy'):

    random_forest = RandomForestClassifier(random_state=125)



    hyperparam = {'min_samples_split': [3, 5, 7, 9, 13, 17, 21, 27, 33, 41, 50, 60, 80, 100],

                  'max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75], 

                  'n_estimators': [100, 200, 300, 500, 750, 1000]}

    

    random_rf = RandomizedSearchCV(random_forest, 

                                             param_distributions = hyperparam,

                                             cv = 20, 

                                             n_iter = 20, 

                                             scoring = scoring, 

                                             n_jobs=8, 

                                             random_state = 125)

    

    random_rf.fit(x_train, y_train)

    

    print("Best Accuracy", random_rf.best_score_)

    print("Best Param", random_rf.best_params_)

    return random_rf
best_rf = rf_fit(x_train_norm, y_train)
random_forest = RandomForestClassifier(random_state=125, n_jobs = 2,

                                   min_samples_split = best_rf.best_params_.get('min_samples_split'),

                                   max_features = best_rf.best_params_.get('max_features'),

                                   n_estimators = best_rf.best_params_.get('n_estimators'))

random_forest.fit(x_train_norm, y_train)
#Functions adaboost

def adaboost_fit(x_train, y_train, scoring = 'accuracy'):

    dectree = DecisionTreeClassifier(random_state=125)

    

    adaboost = AdaBoostClassifier(base_estimator = dectree, 

                                  random_state=125)

    

    hyperparam = {'base_estimator__min_samples_split': [3, 5, 7, 9, 

                                                       13, 17, 21, 27, 

                                                       33, 41, 50, 60, 

                                                       80, 100],

                  'base_estimator__max_features': ['sqrt', 'log2', 

                                                   0.25, 0.5, 0.75],

                  'learning_rate': [0.01, 0.015, 0.02, 

                                    0.05, 0.08, 0.1],

                  'n_estimators': [100, 200, 300, 

                                   500, 750, 1000]}

    

    random_adaboost = RandomizedSearchCV(adaboost, 

                                         param_distributions = hyperparam, 

                                         cv = 20, 

                                         n_iter = 20, 

                                         scoring = scoring, 

                                         n_jobs=10, 

                                         random_state = 125)

    

    random_adaboost.fit(x_train, y_train)

    

    print ("Best Accuracy", random_adaboost.best_score_)

    print ("Best Param", random_adaboost.best_params_)

    return random_adaboost
best_adaboost = adaboost_fit(x_train_norm, y_train)
dectree_boost = DecisionTreeClassifier(min_samples_split = best_adaboost.best_params_.get('base_estimator__min_samples_split'),

                                      max_features = best_adaboost.best_params_.get('base_estimator__max_features'),

                                      random_state=125)

adaboost = AdaBoostClassifier(base_estimator = dectree_boost, 

                             n_estimators = best_adaboost.best_params_.get('n_estimators'),

                             learning_rate = best_adaboost.best_params_.get('learning_rate'),

                             random_state=125)

adaboost.fit(x_train_norm, y_train)
#Functions Gradient Boosting

def gradientboost_fit(x_train, y_train, scoring = 'accuracy'):

    gradient_boost = GradientBoostingClassifier(random_state=125)

    

    hyperparam = {'min_samples_split': [3, 5, 7, 9, 13, 17, 21, 27, 33, 41, 50, 60, 80, 100],

                  'max_features': ['sqrt', 'log2', 0.25, 0.5, 0.75], 

                  'n_estimators': [100, 200, 300, 500, 750, 1000],

                  'learning_rate': [0.01, 0.015, 0.02, 0.05, 0.08, 0.1] }

    random_gradientboost = RandomizedSearchCV(gradient_boost, param_distributions = hyperparam, cv = 20,

                                          n_iter = 20, scoring = scoring, n_jobs=10, 

                                          random_state = 125, verbose = True)

    random_gradientboost.fit(x_train, y_train)

    

    print ("Best Accuracy", random_gradientboost.best_score_)

    print ("Best Param", random_gradientboost.best_params_)

    return random_gradientboost
best_gradientboost = gradientboost_fit(x_train_norm, y_train)
gradient_boost = GradientBoostingClassifier(n_estimators=best_gradientboost.best_params_.get('n_estimators'),

                                       min_samples_leaf = best_gradientboost.best_params_.get('min_samples_split'),

                                       max_features = best_gradientboost.best_params_.get('max_features'),

                                       learning_rate = best_gradientboost.best_params_.get('learning_rate'), random_state=123)

gradient_boost.fit(x_train_norm, y_train)
#All functions Test

def testNumeric(data, imputer):

    numerical_data_imputed = pd.DataFrame(imputer.transform(data))

    numerical_data_imputed.columns = data.columns

    numerical_data_imputed.index = data.index

    return  numerical_data_imputed

def testCategorical(data, value, dummy_column):

    categorical_data_imputed = data.fillna(value)

    categorical_data_imputed = categorical_data_imputed.astype('str')

    dummy = pd.get_dummies(categorical_data_imputed)

    dummy = dummy.reindex(columns = dummy_column, fill_value = 0)

    return dummy

def testStandardize(data, standardize):

    data_columns = data.columns  # agar nama column tidak hilang

    data_index = data.index # agar index tidak hilang

    data_standardized = pd.DataFrame(standardize.transform(data))

    data_standardized.columns = data_columns

    data_standardized.index = data_index

    return data_standardized

def newData(data, imput_numerical, value_categorical, dummy_columns, standardizer):

    #Numerical Categorical Split

    data_numerical, data_categorical = numerical_categorical(data, ["Pclass","PassengerId"],["Name","Ticket","Cabin","PassengerId"])

    # Numerical Imputation

    data_numerical_imputed = testNumeric(data_numerical, imput_numerical)

    # Categorical Imputation

    data_categorical_imputed = testCategorical(data_categorical, value_categorical, dummy_columns)

    # Join

    data_preprocessed = pd.concat([data_numerical_imputed, data_categorical_imputed], axis = 1)

    # Normalization

    data_norm = testStandardize(data_preprocessed, standardizer)

    return data_norm

def evalModel(x, y, clf):

    print ("Accuracy  : %.5f" % accuracy_score(y, clf.predict(x)))

    print ("Recall    : %.5f" % recall_score(y, clf.predict(x)))

    print ("Precision : %.5f" % precision_score(y, clf.predict(x)))

    print ("F1 score  : %.5f" % f1_score(y, clf.predict(x)))
x_test_norm = newData(x_test, imput_numerical, "S", dummy_columns, standardize)
print(logreg.score(x_test_norm, y_test))

print(dectree.score(x_test_norm, y_test))

print(bagging.score(x_test_norm, y_test))

print(random_forest.score(x_test_norm, y_test))

print(adaboost.score(x_test_norm, y_test))

print(gradient_boost.score(x_test_norm, y_test))
evalModel(x_test_norm, y_test, logreg)
evalModel(x_test_norm, y_test, dectree)
evalModel(x_test_norm, y_test, bagging)
evalModel(x_test_norm, y_test, random_forest)
evalModel(x_test_norm, y_test, adaboost)
evalModel(x_test_norm, y_test, gradient_boost)
data=pd.read_csv("../input/test.csv")

data1=pd.read_csv("../input/test.csv")

data2=pd.read_csv("../input/test.csv")

data3=pd.read_csv("../input/test.csv")

data4=pd.read_csv("../input/test.csv")

data5=pd.read_csv("../input/test.csv")
x_new_norm = newData(data, imput_numerical, "S", dummy_columns, standardize)

x_new_norm.head()
y_predict_logreg=logreg.predict(x_new_norm)

y_predict_logreg
y_predict_dectree=dectree.predict(x_new_norm)

y_predict_dectree
y_predict_bagging=bagging.predict(x_new_norm)

y_predict_bagging
y_predict_random_forest=random_forest.predict(x_new_norm)

y_predict_random_forest
y_predict_adaboost=adaboost.predict(x_new_norm)

y_predict_adaboost
y_predict_gradient_boost=gradient_boost.predict(x_new_norm)

y_predict_gradient_boost
data["Survived"]=y_predict_logreg

data1["Survived"]=y_predict_dectree

data2["Survived"]=y_predict_bagging

data3["Survived"]=y_predict_random_forest

data4["Survived"]=y_predict_adaboost

data5["Survived"]=y_predict_gradient_boost
#save output to excel

df_output = data.drop(['Pclass','Name','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'], axis=1)

df_output1 = data1.drop(['Pclass','Name','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'], axis=1)

df_output2 = data2.drop(['Pclass','Name','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'], axis=1)

df_output3 = data3.drop(['Pclass','Name','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'], axis=1)

df_output4 = data4.drop(['Pclass','Name','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'], axis=1)

df_output5 = data5.drop(['Pclass','Name','Sex','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age'], axis=1)

writerxls = pd.ExcelWriter('../input/Submission_logreg.xls')

writerxls1 = pd.ExcelWriter('../input/Submission_dectree.xls')

writerxls2 = pd.ExcelWriter('../input/Submission_bagging.xls')

writerxls3 = pd.ExcelWriter('../input/Submission_random_forest.xls')

writerxls4 = pd.ExcelWriter('../input/Submission_adaboost.xls')

writerxls5 = pd.ExcelWriter('../input/Submission_gradient_boost.xls')

df_output.to_excel(writerxls,'Sheet1')

df_output1.to_excel(writerxls1,'Sheet1')

df_output2.to_excel(writerxls2,'Sheet1')

df_output3.to_excel(writerxls3,'Sheet1')

df_output4.to_excel(writerxls4,'Sheet1')

df_output5.to_excel(writerxls5,'Sheet1')

writerxls.save()

writerxls1.save()

writerxls2.save()

writerxls3.save()

writerxls4.save()

writerxls5.save()