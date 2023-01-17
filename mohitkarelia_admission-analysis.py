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
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
data.head()
data.drop('Serial No.',1,inplace = True)
missing_data = data.isnull()
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
data.info()
data.describe()
sb.set_style('whitegrid')
sb.set_palette('pastel')
sb.regplot(data['GRE Score'],data['TOEFL Score'],color='blue')
plt.title('GRE Score vs TOEFL Score')
sb.regplot(data['GRE Score'],data['Chance of Admit '],color='green')
plt.title('GRE Score vs Chance of Admit')
sb.regplot(data['GRE Score'],data['University Rating'],color='darkorange')

sb.regplot(data['TOEFL Score'],data['University Rating'],color='darkorange')
sb.distplot(data.loc[data['University Rating'] == 1,'Chance of Admit '],color = 'orange')
plt.title('University Rating = 1')
sb.distplot(data.loc[data['University Rating'] == 2,'Chance of Admit '],color = 'orange')
plt.title('University Rating = 2')
sb.distplot(data.loc[data['University Rating'] == 3,'Chance of Admit '],color = 'orange')
plt.title('University Rating = 3')
sb.distplot(data.loc[data['University Rating'] == 4,'Chance of Admit '],color = 'orange')
plt.title('University Rating = 4')
sb.distplot(data.loc[data['University Rating'] == 5,'Chance of Admit '],color = 'orange')
plt.title('University Rating = 5')
plt.figure(figsize = (15,5))
sb.heatmap(data.corr(),annot=True)
y = data['Chance of Admit ']
data.drop(['Chance of Admit '],1,inplace = True)
X = data
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)
model1 = LinearRegression()
model1.fit(X_train,y_train)
y_train_pred = model1.predict(X_train)
y_test_pred = model1.predict(X_test)

plt.scatter(y_train_pred,y_train_pred-y_train,c='blue')
plt.scatter(y_test_pred,y_test_pred-y_test,c='darkorange')
plt.legend(labels = ('train data','test data'),loc = 'lower right')

print("Coefficients:",model1.coef_)
print('Intercept:',model1.intercept_)
print("Insample Score: ",model1.score(X_train,y_train))
print("Outsample Score: ",model1.score(X_test,y_test))
model2 = Lasso(alpha = 0.001)
model2.fit(X_train,y_train)
y_train_pred = model2.predict(X_train)
y_test_pred = model2.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue')
plt.scatter(y_test_pred,y_test_pred-y_test,c='darkorange')
plt.legend(labels = ('train data','test data'),loc = 'lower right')
print("Coefficients:",model2.coef_)
print('Intercept:',model2.intercept_)
print("Insample Score: ",model2.score(X_train,y_train))
print("Outsample Score: ",model2.score(X_test,y_test))
model3 = ElasticNet(alpha = 0.001)
model3.fit(X_train,y_train)
y_train_pred = model2.predict(X_train)
y_test_pred = model2.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue')
plt.scatter(y_test_pred,y_test_pred-y_test,c='darkorange')
plt.legend(labels = ('train data','test data'),loc = 'lower right')
print("Coefficients:",model2.coef_)
print('Intercept:',model2.intercept_)
print("Insample Score: ",model3.score(X_train,y_train))
print("Outsample Score: ",model3.score(X_test,y_test))
model4 = SVR(kernel='poly')
model4.fit(X_train,y_train)
y_train_pred = model4.predict(X_train)
y_test_pred = model4.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue')
plt.scatter(y_test_pred,y_test_pred-y_test,c='darkorange')
plt.legend(labels = ('train data','test data'),loc = 'lower right')
print("Insample Score: ",model4.score(X_train,y_train))
print("Outsample Score: ",model4.score(X_test,y_test))

model5 = DecisionTreeRegressor(max_depth=2)
model5.fit(X_train,y_train)
y_train_pred = model5.predict(X_train)
y_test_pred = model5.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue')
plt.scatter(y_test_pred,y_test_pred-y_test,c='darkorange')
plt.legend(labels = ('train data','test data'),loc = 'lower right')
print("Insample Score: ",model5.score(X_train,y_train))
print("Outsample Score: ",model5.score(X_test,y_test))
model6 = RandomForestRegressor(n_estimators=100,criterion='mse',n_jobs = -1)
model6.fit(X_train,y_train)
y_train_pred = model6.predict(X_train)
y_test_pred = model6.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue')
plt.scatter(y_test_pred,y_test_pred-y_test,c='darkorange')
plt.legend(labels = ('train data','test data'),loc = 'lower right')
print("Insample Score: ",model6.score(X_train,y_train))
print("Outsample Score: ",model6.score(X_test,y_test))
model7 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2),n_estimators=600)
model7.fit(X_train,y_train)
y_train_pred = model7.predict(X_train)
y_test_pred = model7.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue')
plt.scatter(y_test_pred,y_test_pred-y_test,c='darkorange')
plt.legend(labels = ('train data','test data'),loc = 'lower right')
print("Insample Score: ",model7.score(X_train,y_train))
print("Outsample Score: ",model7.score(X_test,y_test))
model = LinearRegression()
model.fit(X,y)
gre_score = int(input("Enter GRE Score: "))
toefl_score = int(input("Enter TOEFL Score: "))
university_rating = int(input("Enter University Rating(1-5): "))
sop = float(input("Enter SOP: "))
lor = float(input("Enter LOR: "))
cgpa = float(input("Enter CGPA: "))
Research = int(input("Have you done any Research(0=no,1=yes): "))
params = [[gre_score,toefl_score,university_rating,sop,lor,cgpa,Research]]
for i,chance_of_admit in enumerate(model.predict(params)):
    print("Chance of Admission are:",chance_of_admit )

