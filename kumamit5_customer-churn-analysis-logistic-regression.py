!pip install feature-engine


#Import the libraries

import  numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split,cross_val_score

from feature_engine.categorical_encoders import  OrdinalCategoricalEncoder

from feature_engine.missing_data_imputers import RandomSampleImputer

from feature_engine.variable_transformers import PowerTransformer

from scipy.stats import probplot

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import  Pipeline

from sklearn.metrics import classification_report



#Read the data

data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



#Getting the shape of the data

print(data.shape)



#Check the data head

print(data.head())



#Checking the percentage of missing values

print(data.isnull().mean()*100)

#Make the Churn variable as binary variable

data['Churn'] = np.where(data['Churn'] == 'No', 0, 1)



#Method to Segregate the categorical and numerical columns

def dtype_seg(data):

    cat_col = []

    num_col = []

    for col in data.columns:

        if(data[col].dtype == 'object'):

            cat_col.append(col)

        else:

            num_col.append(col)

    return  cat_col, num_col



categorical, numerical = dtype_seg(data)

print(f'the categorical columns : {categorical}')

print(f'the numerical columns : {numerical}')



#Here we have to cast the data type of TotalCharges as float64

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')



#Methods to join multiple words of the observation

def join_words(data, column):

    for i in range(len(data[column])):

        x = data[column][i].split()

        x = '_'.join(x)

        data[column][i] = x

    return data[column]



#Apply join words on MultipleLines column

data['MultipleLines'] = join_words(data, 'MultipleLines')



#Apply join words on InternetService column

data['InternetService'] = join_words(data, 'InternetService')



#Apply join words on OnlineSecurity column

data['OnlineSecurity'] = join_words(data, 'OnlineSecurity')



#Apply join words on OnlineBackup column

data['OnlineBackup'] = join_words(data, 'OnlineBackup')



#Apply join words on DeviceProtection column

data['DeviceProtection'] = join_words(data, 'DeviceProtection')



#Apply join words on TechSupport column

data['TechSupport'] = join_words(data, 'TechSupport')



#Apply join words on StreamingTV column

data['StreamingTV'] = join_words(data, 'StreamingTV')



#Apply join words on StreamingMovies column

data['StreamingMovies'] = join_words(data, 'StreamingMovies')



#Apply join words on Contract column

data['Contract'] = join_words(data, 'Contract')



#Apply join words on PaymentMethod column

data['PaymentMethod'] = join_words(data, 'PaymentMethod')





#Segregate the data to dependent and Independent

X = data[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',

          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies',

          'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges','SeniorCitizen', 'tenure', 'MonthlyCharges']]

y = data['Churn']



#Split the data to training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)



#Getting the shape

print(X_train.shape, X_test.shape)





#Method to visualise the distribution of the data

def diagonstic_plot(data,column):

    plt.figure(figsize=(10,8))

    plt.subplot(1,2,1)

    sns.distplot(data[column].dropna())



    plt.subplot(1,2,2)

    probplot(data[column], dist='norm', plot=plt)

    plt.show()



#Check the TotalCharges data description

print(X_train['TotalCharges'].describe())



#Impute the missing values for the Totalcharges

imputer = RandomSampleImputer(random_state=['TotalCharges'], seed='observation', seeding_method='add')

imputer.fit(X_train)



#Transform the data

X_train = imputer.transform(X_train)

X_test = imputer.transform(X_test)



#Transform the numerical variable for more Gaussian Approximation

vt = PowerTransformer(exp=0.3)

vt.fit(X_train)



#Transform the variable

X_train = vt.transform(X_train)

X_test = vt.transform(X_test)



#Method to get the number of unique values in each categorical feature

def get_nunique(data,categorical_col_list):

    my_dict = {}

    for column in categorical_col_list:

        if(column!='customerID'):

            my_dict[column] = data[column].nunique()

    return my_dict

#Print the number of unique categories associated with each categorical features

print(get_nunique(X_train,categorical))



#Encode the categorcal variable

encoder = OrdinalCategoricalEncoder(encoding_method='ordered')

encoder.fit(X_train, y_train)



#Transform the variable

X_train = encoder.transform(X_train)

X_test = encoder.transform(X_test)





#Choosing the appropriate Hyperparamters for the model

#Stage 1 choose optimal Penalty parameter

penalty_dict = {}

for penalty in ['l1','l2']:

    pipeline_p = Pipeline([

        ('scaler', StandardScaler()),

        ('log', LogisticRegression(penalty=penalty))

    ])

    score = cross_val_score(estimator=pipeline_p,X=X_train,y=y_train,scoring='f1',cv=10)

    penalty_dict[penalty] = score.mean()

print(f'Penalty : {penalty_dict}')



#Stage 2 choose dual  parameter

dual_dict = {}

for dual in [False, True]:

    pipeline_d = Pipeline([

        ('scaler', StandardScaler()),

        ('log', LogisticRegression(penalty='l2', dual=dual))

    ])

    score = cross_val_score(estimator=pipeline_d,X=X_train,y=y_train,scoring='f1',cv=10)

    dual_dict[dual] = score.mean()

print(f'Dual : {dual_dict}')



#Stage 3 choose optimal C value

C_dict = {}

for C in [0.5,1.0,2.0,3.0,4.0]:

    pipeline_c = Pipeline([

        ('scaler', StandardScaler()),

        ('log', LogisticRegression(penalty='l2', dual=False, C=C))

    ])

    score = cross_val_score(estimator=pipeline_c,X=X_train,y=y_train,scoring='f1',cv=10)

    C_dict[C] = score.mean()

print(f'C : {C_dict}')





#Stage 4 choose  fit_intercept value

intercept_dict = {}

for intercept in [False,True]:

    pipeline_i = Pipeline([

        ('scaler', StandardScaler()),

        ('log', LogisticRegression(penalty='l2', dual=False, C=1.0, fit_intercept=intercept))

    ])

    score = cross_val_score(estimator=pipeline_i,X=X_train,y=y_train,scoring='f1',cv=10)

    intercept_dict[intercept] = score.mean()

print(f'Intercept : {intercept_dict}')



#Stage 5 choose  solver value

solver_dict = {}

for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:

    pipeline_s = Pipeline([

        ('scaler', StandardScaler()),

        ('log', LogisticRegression(penalty='l2', dual=False, C=1.0, fit_intercept=False, solver=solver))

    ])

    score = cross_val_score(estimator=pipeline_s,X=X_train,y=y_train,scoring='f1',cv=10)

    solver_dict[solver] = score.mean()

print(f'Solver : {solver_dict}')



#Stage 6 choose  multi_class value

multiclass_dict = {}

for multiclass in ['auto', 'ovr', 'multinomial']:

    pipeline_m = Pipeline([

        ('scaler', StandardScaler()),

        ('log', LogisticRegression(penalty='l2', dual=False, C=1.0, fit_intercept=False, solver='saga', multi_class=multiclass))

    ])

    score = cross_val_score(estimator=pipeline_m,X=X_train,y=y_train,scoring='f1',cv=10)

    multiclass_dict[multiclass] = score.mean()

print(f'Multi-Class : {multiclass_dict}')





#Scale the data

scaler = StandardScaler()

scaler.fit(X_train)



#Transform the data

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



#Applying the model

classifier = LogisticRegression(penalty='l2', dual=False, C=1.0, fit_intercept=False, solver='saga', multi_class='ovr')

classifier.fit(X_train,y_train)



#Predict the test set result

y_pred = classifier.predict(X_test)



#Print classification Report for training set

print('Training Classification Report')

print(classification_report(y_train,classifier.predict(X_train)))



#Print classification Report for test set

print('Training Classification Report')

print(classification_report(y_test,y_pred))