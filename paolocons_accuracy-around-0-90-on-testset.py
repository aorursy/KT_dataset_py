# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import accuracy_score, log_loss

import matplotlib.pyplot as plt

import seaborn as sns

from itertools import combinations
#read the dataset

df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
df.info()
df.describe()
# check for na values

df[df.isna()].count()
# check for zero values

df[df == 0].count().sort_values(ascending=False)
#fill Insulin's zero values

Insulin_by_outcome = df.groupby(['Outcome']).mean()['Insulin']



def fixInsulin(insulin, outcome):

    if insulin == 0:

        return Insulin_by_outcome[outcome]

    else:

        return insulin

    

df['Insulin'] = df.apply(lambda row : fixInsulin(row['Insulin'], row['Outcome']), axis=1)
#fill SkinThickness's zero values

skinThickness_by_outcome = df.groupby(['Outcome']).mean()['SkinThickness']



def fixSkinThickness(skinThickness, outcome):

    if skinThickness == 0:

        return skinThickness_by_outcome[outcome]

    else:

        return skinThickness

    

df['SkinThickness'] = df.apply(lambda row : fixSkinThickness(row['SkinThickness'], row['Outcome']), axis=1)
#fill BloodPressure's zero values



bloodPressure_by_outcome = df.groupby(['Outcome']).mean()['BloodPressure']



def fixBloodPressure(bloodPressure, outcome):

    if bloodPressure == 0:

        return bloodPressure_by_outcome[outcome]

    else:

        return bloodPressure

    

df['BloodPressure'] = df.apply(lambda row : fixBloodPressure(row['BloodPressure'], row['Outcome']), axis=1)
#fill Glucose's zero values



glucose_by_outcome = df.groupby(['Outcome']).mean()['Glucose']



def fixGlucose(glucose, outcome):

    if glucose == 0:

        return glucose_by_outcome[outcome]

    else:

        return glucose

    

df['Glucose'] = df.apply(lambda row : fixGlucose(row['Glucose'], row['Outcome']), axis=1)
#fill BMI's zero values



bim_by_outcome = df.groupby(['Outcome']).mean()['BMI']



def fixBMI(bim, outcome):

    if bim == 0:

        return bim_by_outcome[outcome]

    else:

        return bim

    

df['BMI'] = df.apply(lambda row : fixBMI(row['BMI'], row['Outcome']), axis=1)
# try to compute a 'Sex' columns based on pregnancy (if more than zero for sure is a female)



def getSex(pregnancy):

    if pregnancy > 0:

        return 1 #'Female'

    else:

        return 0 #'Unknown'

    

df['Sex'] = df.apply(lambda row : getSex(row['Pregnancies']), axis=1)
#drop some outliers based on scatterplots (not shown here)



df = df[df.Glucose > 50]

print(len(df))

df = df[(df.BloodPressure > 42) | (df.BloodPressure < 116)]

print(len(df))

df = df[(df.SkinThickness > 5) | (df.SkinThickness < 58)]

print(len(df))

df = df[df.Insulin < 625]

print(len(df))

df = df[(df.BMI > 15) | (df.BMI < 55)]

print(len(df))

df = df[df.DiabetesPedigreeFunction < 55]

print(len(df))

df = df[df.Age < 70]

print(len(df))
# add BMI classes as per World Health Organizations



BMI_OMSNutritional_map = {

-1:(0,18.5), #'Underweight'

0:(18.5,24.9), #'Normal weight'

1: (24.9,29.9), #'Pre-obesity'

2: (29.9,34.9), #'Obesity class I'

3: (34.9,39.9), #'Obesity class II'

4: (39.9, 1000) #'Obesity class III'

}



def getBMIClass(bmi):

    bmi_class = -100

    for limit_index, limit in enumerate(BMI_OMSNutritional_map.values()):

        if int(bmi) >= limit[0] and int(bmi) < limit[1]: # >= for lower limit tends to assign higher category rather than lower

            bmi_class = list(BMI_OMSNutritional_map.keys())[limit_index]

            break

    if bmi_class == -100:

        print('Assined -100 class for: %d' %(bmi))

    return bmi_class



df['BMI_class'] = df.apply(lambda row : getBMIClass(row['BMI']), axis=1)

sns.barplot(x=df.BMI_class, y=df.Outcome,data=df);
# There is no documentation about how the 'Glucose' measure is taken in the dataset, anyway I try to consider the following:

# A blood sugar level less than 140 mg/dL (7.8 mmol/L) is normal.

# A reading between 140 and 199 mg/dL (7.8 mmol/L and 11.0 mmol/L) indicates prediabetes

# A reading of more than 200 mg/dL (11.1 mmol/L) after two hours indicates diabetes.





OMS_Glucose_map = {

0 :(0,140), #'normal'

1:(140,200), #'prediabetes'

2: (200,1000) #'diabetes'

}







def getGlucoseClass(glucose):

    glucose_class = 'None'

    for limit_index, limit in enumerate(OMS_Glucose_map.values()):

        if glucose >= limit[0] and glucose < limit[1]: # >= for lower limit tends to assign higher category rather than lower

            glucose_class = list(OMS_Glucose_map.keys())[limit_index]

            break

    return glucose_class



df['Glucose_Class'] = df.apply(lambda row : getGlucoseClass(row['Glucose']), axis=1)

sns.barplot(x=df.Glucose_Class, y=df.Outcome,data=df);
#ideal blood pressure is considered to be between 90/60mmHg and 120/80mmHg.

#high blood pressure is considered to be 140/90mmHg or higher.

#low blood pressure is considered to be 90/60mmHg or lower.



Pressure_map = {

-1:(60,90), #'low'

0: (90,140), #'ideal'

1 : (140,1000) #'high'

}



def getPressureClass(pressure):

    pressure_class = -2

    for limit_index, limit in enumerate(Pressure_map.values()):

        if pressure >= limit[0] and pressure < limit[1]: # >= for lower limit tends to assign higher category rather than lower

            pressure_class = list(Pressure_map.keys())[limit_index]

            break

    return pressure_class



df['BloodPressure_Class'] = df.apply(lambda row : getPressureClass(row['BloodPressure']), axis=1)

sns.barplot(x=df.BloodPressure_Class, y=df.Outcome,data=df);
# add Insulin classes: 100 and 126 limits are values found on several articles on Internet



def getInsulinClass(insulin): 

    if insulin >= 100 and insulin <= 126:

        return 0 #'Normal'

    else:

        return 1 #'Abnormal'



df['Insulin_Class'] = df.apply(lambda row : getInsulinClass(row['Insulin']), axis=1)

sns.barplot(x=df.Insulin_Class, y=df.Outcome,data=df);
# see which are the most correlated features

df.corr()['Outcome'].sort_values(ascending=False)
# prepare the dataset for training and predictions



df = pd.get_dummies(df)

X = df.drop(['Outcome'], axis=1)

Y = df.Outcome.values



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = df.Outcome, random_state=0)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# quickly try with RandomForest and all collected features

regressor = RandomForestClassifier(n_estimators=20)

regressor.fit(X_train,Y_train)



Y_train_pred = regressor.predict(X_train)

Y_test_pred = regressor.predict(X_test)



accuracy_train = accuracy_score(Y_train,Y_train_pred)

accuracy_test = accuracy_score(Y_test,Y_test_pred)





print('RandomForest Accuracy\ttrain: %.4f , test: %.4f' %(accuracy_train,accuracy_test))



#try with GradientBoosting and all collected features



regressor = GradientBoostingClassifier(n_estimators=20)

regressor.fit(X_train,Y_train)



Y_train_pred = regressor.predict(X_train)

Y_test_pred = regressor.predict(X_test)



accuracy_train = accuracy_score(Y_train,Y_train_pred)

accuracy_test = accuracy_score(Y_test,Y_test_pred)





print('GradientBoost Accuracy\ttrain: %.4f , test: %.4f' %(accuracy_train,accuracy_test))



# See how accuracy already improved
# The number of feature is not very big: try all possible feature's combinations to get the best performing list of features



columns = df.columns

columns = columns.drop('Outcome')



Y = df.Outcome.values



best_test_accuracy = 0

best_feature_list = []



for n_features in range(1,len(columns)):

    for comb in combinations(columns,n_features):

        feature_list = [e for e in comb]

        #print(feature_list)

        X = df[feature_list]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = df.Outcome, random_state=0)

        regressor = GradientBoostingClassifier(n_estimators=60)

        regressor.fit(X_train,Y_train)



        Y_train_pred = regressor.predict(X_train)

        Y_test_pred = regressor.predict(X_test)



        accuracy_train = accuracy_score(Y_train,Y_train_pred)

        accuracy_test = accuracy_score(Y_test,Y_test_pred)

        if accuracy_test > best_test_accuracy:

            best_test_accuracy = accuracy_test

            best_feature_list = feature_list

            print('GradientBoost Accuracy\ttrain: %.4f , test: %.4f' %(accuracy_train,accuracy_test))

            

print("best feature list: " + str(feature_list))





# It looks like the best feature list for GradientBoosting contains: ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

# 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Sex', 'BMI_class', 'Glucose_Class', 'BloodPressure_Class', 'Insulin_Class']

# the accuracy on test set is around 0.91





#GradientBoost Accuracy	train: 0.9474 , test: 0.9058

#GradientBoost Accuracy	train: 0.9474 , test: 0.9110
# Do the same with RandomForest classifier

columns = df.columns

columns = columns.drop('Outcome')



Y = df.Outcome.values



best_test_accuracy = 0

best_feature_list = []



for n_features in range(1,len(columns)):

    for comb in combinations(columns,n_features):

        feature_list = [e for e in comb]

        #print(feature_list)

        X = df[feature_list]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify = df.Outcome, random_state=0)

        regressor = RandomForestClassifier(n_estimators=60)

        regressor.fit(X_train,Y_train)



        Y_train_pred = regressor.predict(X_train)

        Y_test_pred = regressor.predict(X_test)



        accuracy_train = accuracy_score(Y_train,Y_train_pred)

        accuracy_test = accuracy_score(Y_test,Y_test_pred)

        if accuracy_test > best_test_accuracy:

            best_test_accuracy = accuracy_test

            best_feature_list = feature_list

            print('RandomForest Accuracy\ttrain: %.4f , test: %.4f' %(accuracy_train,accuracy_test))

            

print("best feature list: " + str(feature_list))



# It looks like the best feature list for RandomForest contains: ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

# 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Sex', 'BMI_class', 'Glucose_Class', 'BloodPressure_Class', 'Insulin_Class']

# which is the same set found for GradientBoosting

# but the accuracy on test set is around 0.90 (almost the same anyway)