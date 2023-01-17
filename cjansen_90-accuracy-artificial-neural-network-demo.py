import numpy as np 

import pandas as pd 



from subprocess import check_output

dataset = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

#dataset = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
#removing standard hours and employeeID + attrition as first row

dataset = dataset[['Attrition',

                   'Age',

                   'BusinessTravel',

                   'DailyRate',

                   'Department',

                   'DistanceFromHome',

                   'Education',

                   'EducationField',

                   'EmployeeCount',

                   'EnvironmentSatisfaction',

                   'Gender',

                   'HourlyRate',

                   'JobInvolvement',

                   'JobLevel',

                   'JobRole',

                   'JobSatisfaction',

                   'MaritalStatus',

                   'MonthlyIncome',

                   'MonthlyRate',

                   'NumCompaniesWorked',

                   'OverTime',

                   'PercentSalaryHike',

                   'PerformanceRating',

                   'RelationshipSatisfaction',

                   'StockOptionLevel',

                   'TotalWorkingYears',

                   'TrainingTimesLastYear',

                   'WorkLifeBalance',

                   'YearsAtCompany',

                   'YearsInCurrentRole',

                   'YearsSinceLastPromotion',

                   'YearsWithCurrManager']]
dataset['JobInvolment_On_Salary']= dataset['JobInvolvement'] / dataset['MonthlyIncome'] * 1000

dataset['MarriedAndBad_Worklife_Balance'] = np.where(dataset['MaritalStatus']=='Married', 

                                               dataset['WorkLifeBalance']-2,

                                               dataset['WorkLifeBalance']+1)

dataset['DistanceFromHome_rootedTo_JobSatisfaction'] = dataset['DistanceFromHome']**(1/dataset['JobSatisfaction'])

dataset['TotalJobSatisfaction'] = dataset['EnvironmentSatisfaction'] + dataset['JobSatisfaction'] + dataset['RelationshipSatisfaction'] 

dataset['OldLowEmployeeTendToStay'] = dataset['YearsAtCompany'] / dataset['JobLevel']

dataset['Mothers'] = np.where((dataset['Gender']=='Female') & (dataset['Age']>=36), 1,0)

dataset['Rate'] = dataset['DailyRate'] * 20 + dataset['HourlyRate'] * 8 * 20 + dataset['MonthlyRate']

dataset['RateExtended'] = dataset['Rate'] * (8 - dataset['JobSatisfaction'] - dataset['EnvironmentSatisfaction'])
X = dataset.iloc[:, 1:].values

y = dataset.iloc[:, 0].values
# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_3 = LabelEncoder()

X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])

labelencoder_X_6= LabelEncoder()

X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])

labelencoder_X_9= LabelEncoder()

X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])

labelencoder_X_13= LabelEncoder()

X[:, 13] = labelencoder_X_13.fit_transform(X[:, 13])

labelencoder_X_15= LabelEncoder()

X[:, 15] = labelencoder_X_15.fit_transform(X[:, 15])

labelencoder_X_19= LabelEncoder()

X[:, 19] = labelencoder_X_19.fit_transform(X[:, 19])

X = X.astype(float)

labelencoder_y= LabelEncoder()

y = labelencoder_y.fit_transform(y)
#no dummy trap

onehotencoder1 = OneHotEncoder(categorical_features = [1])

X = onehotencoder1.fit_transform(X).toarray()

X = X[:,1:]

onehotencoder3 = OneHotEncoder(categorical_features = [4])

X = onehotencoder3.fit_transform(X).toarray()

X = X[:,1:]

onehotencoder6 = OneHotEncoder(categorical_features = [8])

X = onehotencoder6.fit_transform(X).toarray()

X = X[:,1:]

onehotencoder13 = OneHotEncoder(categorical_features = [19])

X = onehotencoder13.fit_transform(X).toarray()

X = X[:,1:]

onehotencoder15 = OneHotEncoder(categorical_features = [28])

X = onehotencoder15.fit_transform(X).toarray()

X = X[:,1:]
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
dropout = 0.1

epochs = 100

batch_size = 30

optimizer = 'adam'

k = 20
# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(16, kernel_initializer="truncated_normal", activation = 'relu', input_shape = (X.shape[1],)))

    classifier.add(Dropout(dropout))

    classifier.add(Dense(1, kernel_initializer="truncated_normal", activation = 'sigmoid', )) #outputlayer

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ["accuracy"])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = batch_size, epochs = epochs, verbose=0)

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 30)

max = accuracies.max()
print("Best accuracy: ",max)