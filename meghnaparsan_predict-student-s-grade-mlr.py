import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



dataset = pd.read_csv('../input/student-alcohol-consumption/student-mat.csv')
# df.filter(regex='[A-CEG-I]')



features = dataset[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob',

        'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',

        'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 

        'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]
features
label = dataset[['G3']]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
features = pd.DataFrame(features)

#extracting categorical data from the dataset

categorical_data_set = features.dtypes==object

categorical_list = features.columns[categorical_data_set].tolist()



print (categorical_list)



non_categorical_list = features.columns[~categorical_data_set].tolist()



print (non_categorical_list)
features[categorical_list] = features[categorical_list].apply (lambda cols: labelencoder.fit_transform(cols))
features[categorical_list].head(10)
oneHotEncoder = OneHotEncoder (categorical_features = categorical_data_set, sparse = False)

featuresOHE = oneHotEncoder.fit_transform(features)
type(features)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(featuresOHE, label, test_size = 0.2, random_state = 2)
from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit (X_train, y_train)
grades_predict = regression.predict (X_test)
print (grades_predict)
print (X_test[0], grades_predict[0])