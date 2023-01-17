import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
data_set = pd.read_csv('../input/insurance/insurance.csv')
print(data_set.head(10))
print(data_set.shape)
sns.countplot(x = 'region', hue = 'sex' , data = data_set)
sns.countplot(x = 'smoker', hue = 'sex' , data = data_set)
data_set['bmi'].plot.hist(bins=50,figsize=(15,5))
data_set.sex.replace(['male', 'female'], [1, 0], inplace=True) # if male , replace value to 1 else to 0.
data_set.smoker.replace(['yes', 'no'], [1,0], inplace=True) # if smoker, replace value to 1 else to 0.
data_set.drop(['region'], axis=1, inplace=True)
data_set.isnull().sum()
X = data_set.drop('charges', axis=1) # input column
y = data_set['charges'] # output column
print(X.head(5))
print()
print(y.head(5))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
predictions = linear_model.predict(X_test)
print(mean_squared_error(y_test, predictions))
linear_model.score(X_test, predictions)
val = input('Do you want to predict a value : ( "Y" for yes "N" for no ) : ')
temp_val = []
if val == 'Y' or 'y':
    age = int(input('Enter the age of candidate : '))
    temp_val.append(age)
    sex = int(input('Enter the sex of the candidates : '))
    temp_val.append(sex)
    bmi = float(input('Enter the BMI of the candidate : '))
    temp_val.append(bmi)
    children = int(input('Enter the value of the children : '))
    temp_val.append(children)
    smoker = int(input('Enter the value for smoking : '))
    temp_val.append(smoker)
else :
    print('Have fun !!')
new_val = [temp_val]
predict = linear_model.predict(new_val)
print()
print("Input Data : %s, Predicted Amount : $ %s " % (new_val[0], predict[0]))
#print(data_set.head(10))