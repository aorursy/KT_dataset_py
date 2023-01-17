import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression
data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
data
encoder = LabelEncoder()



data['gender'] = encoder.fit_transform(data['gender'])

gender_mappings = {index: label for index, label in enumerate(encoder.classes_)}



data['race/ethnicity'] = encoder.fit_transform(data['race/ethnicity'])

ethnicity_mappings = {index: label for index, label in enumerate(encoder.classes_)}



data['parental level of education'] = encoder.fit_transform(data['parental level of education'])

parent_education_mappings = {index: label for index, label in enumerate(encoder.classes_)}



data['lunch'] = encoder.fit_transform(data['lunch'])

lunch_mappings = {index: label for index, label in enumerate(encoder.classes_)}



data['test preparation course'] = encoder.fit_transform(data['test preparation course'])

test_prep_mappings = {index: label for index, label in enumerate(encoder.classes_)}
report = ProfileReport(data)
report.to_notebook_iframe()
X = data[['math score', 'reading score', 'writing score']]



y_math = X['math score']

y_reading = X['reading score']

y_writing = X['writing score']
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_math = X[['reading score', 'writing score']]

X_reading = X[['math score', 'writing score']]

X_writing = X[['math score', 'reading score']]
X_math_train, X_math_test, y_math_train, y_math_test = train_test_split(X_math, y_math, train_size=0.7)

X_reading_train, X_reading_test, y_reading_train, y_reading_test = train_test_split(X_reading, y_reading, train_size=0.7)

X_writing_train, X_writing_test, y_writing_train, y_writing_test = train_test_split(X_writing, y_writing, train_size=0.7)
math_model = LinearRegression()

reading_model = LinearRegression()

writing_model = LinearRegression()



math_model.fit(X_math_train, y_math_train)

reading_model.fit(X_reading_train, y_reading_train)

writing_model.fit(X_writing_train, y_writing_train)



math_R2 = math_model.score(X_math_test, y_math_test)

reading_R2 = reading_model.score(X_reading_test, y_reading_test)

writing_R2 = writing_model.score(X_writing_test, y_writing_test)
print(f"Math R^2: {math_R2}")

print(f"Reading R^2: {reading_R2}")

print(f"Writing R^2: {writing_R2}")