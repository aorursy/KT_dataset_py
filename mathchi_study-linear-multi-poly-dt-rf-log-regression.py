# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error, r2_score                # we are using this for model tunning



from warnings import filterwarnings

filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Life_Expectancy_Data = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")

data = Life_Expectancy_Data.copy()

data = data.dropna()            # If there is a missing or empty observation, delete it. Or 'data.fillna(data.mean(), inplace=True)' with this make NaN values take mean



lindata = data.copy()

multidata = data.copy()

polydata = data.copy()

RFdata = data.copy()

logdata = data.copy()
lindata.info()
lindata.head()
lindata.corr()
# plot the heatmap

corr = lindata.corr()

sns.heatmap(corr, 

        xticklabels=corr.columns,

        yticklabels=corr.columns)



linear_reg = LinearRegression()

x = lindata.GDP.values.reshape(-1,1)

y = lindata['percentage expenditure'].values.reshape(-1,1)          



linear_reg.fit(x,y)
b0 = linear_reg.predict(([[10000]]))       

print("b0: ", b0)



b1 = linear_reg.coef_

print("b1: ", b1)
x_array = np.arange(min(lindata.GDP),max(lindata.GDP)).reshape(-1,1)  # this for information about the line to be predicted



plt.scatter(x,y)

y_head = linear_reg.predict(x_array)                                 # this is predict percentage of expenditure

plt.plot(x_array,y_head,color="red")

plt.show()



from sklearn import metrics

print("Mean Absolute Error: ", metrics.mean_absolute_error(x_array,y_head))

print("Mean Squared Error: ", metrics.mean_squared_error(x_array,y_head))

print("Root Mean Squared Error: ", np.sqrt(metrics.mean_squared_error(x_array, y_head)))



print(r2_score(y, linear_reg.predict(x)))
Life_Expectancy_Data = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")

data = Life_Expectancy_Data.copy()

data = data.dropna()



multidata = data.copy()



multidata.drop(["Country", "Status"], axis=1, inplace=True)             # When we look at the data, Country and Status columns are composed of objects. Because we need to be int or float.



x = multidata.iloc[:, [-2,-1]].values                                   # I took the last two columns (Income composition of resources, schooling) as independent variables.

y = multidata["percentage expenditure"].values.reshape(-1,1)            # our independent variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)

lm = LinearRegression()

model = lm.fit(x_train,y_train)
print("b0: ", lm.intercept_)

print("b1,b2: ", lm.coef_)
new_data = [[0.4,8], [0.5,10]]   

new_data = pd.DataFrame(new_data).T       # .T is transfor the chart.

model.predict(new_data) 
rmse = np.sqrt(mean_squared_error(y_train, model.predict(x_train)))

rmse
model.score(x_train, y_train) 
cross_val_score(model, x_train,  y_train, cv= 10, scoring="r2").mean()
y_head = model.predict(x_test)

y_head[0:5]
y_test_1 =np.array(range(0,len(y_test)))
# r2 value: 

r2_degeri = r2_score(y_test, y_head)

print("Test r2 error = ",r2_degeri) 



plt.plot(y_test_1,y_test,color="r")

plt.plot(y_test_1,y_head,color="blue")

plt.show()
from sklearn.preprocessing import PolynomialFeatures     # this gives properties of polynomial



Life_Expectancy_Data = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")

data = Life_Expectancy_Data.copy()

data = data.dropna()        



polydata = data.copy()
linear_reg = LinearRegression()

x = polydata.GDP.values.reshape(-1,1)

y = polydata['percentage expenditure'].values.reshape(-1,1)          



linear_reg.fit(x,y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)
polynomial_regression = PolynomialFeatures(degree = 15)    

x_polynomial = polynomial_regression.fit_transform(x)



linear_reg2 = LinearRegression()

linear_reg2.fit(x_polynomial,y)



y_head = linear_reg2.predict(x_polynomial)



plt.plot(x,y_head,color="green",label="poly")

plt.legend()

plt.show()
pol_reg = PolynomialFeatures(degree = 8)                    



level_poly = pol_reg.fit_transform(x_train)                 # According to the polynomial, x_train is defined



lm = LinearRegression()                                     

lm.fit(level_poly,y_train)
y_head = lm.predict(pol_reg.fit_transform(x_train))

y_test =np.array(range(0,len(y_train)))
r2 = r2_score(y_train, y_head)

print("r2 value: ", r2)                               # percentage of significance





plt.scatter(y_test, y_train, color="red")

plt.scatter(y_test, y_head, color = "g")

plt.xlabel("GDP")

plt.ylabel("percentage expenditure")

plt.show()
plt.plot(y_test,y_train, color="red")

plt.plot(y_test, y_head, color = "blue")

plt.xlabel("GDP")

plt.ylabel("percentage expenditure")

plt.show()
from sklearn.tree import DecisionTreeRegressor               # for our predict model



Life_Expectancy_Data = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")

data = Life_Expectancy_Data.copy()

data = data.dropna()                                         # same is done 



DTdata = data.copy()
x = polydata.GDP.values.reshape(-1,1)

y = polydata['percentage expenditure'].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state= 42)
DT_reg = DecisionTreeRegressor()           # created model

DT_reg.fit(x_train,y_train)                # fitted model according to train values



print(DT_reg.predict([[1000]]))            
x_array = np.arange(min(x),max(x),0.01).reshape(-1,1)   # line information to be drawn as a predict

y_head = DT_reg.predict(x_array)                        # percentage of spend estimate



plt.scatter(x,y, color="red")

plt.plot(x_array,y_head,color="blue")

plt.xlabel("GDP")

plt.ylabel("percentage expenditure")

plt.show()
from sklearn.ensemble import RandomForestRegressor           # for our predict model



Life_Expectancy_Data = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv")

data = Life_Expectancy_Data.copy()

data = data.dropna()                                         # same is done 



RFdata = data.copy()
x = polydata.GDP.values.reshape(-1,1)

y = polydata['percentage expenditure'].values.reshape(-1,1)
RF_reg = RandomForestRegressor(n_estimators=100, random_state=42)          

RF_reg.fit(x,y)                                                # the best fit line is drawn
print(RF_reg.predict([[1000]]))            
x_array = np.arange(min(x),max(x),0.01).reshape(-1,1)   # line information to be drawn as a predict

y_head = RF_reg.predict(x_array)                        # percentage of spend predict



plt.scatter(x,y, color="red")

plt.plot(x_array,y_head,color="blue")

plt.xlabel("GDP")

plt.ylabel("percentage expenditure")

plt.show()
logdata.drop(["Country"], axis=1, inplace=True)  

logdata.head()
logdata["Status"].value_counts()
logdata["Status"].value_counts().plot.barh();
logdata.Status = [1 if each == "Developing" else 0 for each in logdata.Status]   


logdata.describe().T
y = logdata["Status"]

X_data = logdata.drop(["Status"], axis=1)
#*** Normalize ***#



X = (X_data - np.min(X_data))/(np.max(X_data) - np.min(X_data)).values
loj = sm.Logit(y, X)

loj_model= loj.fit()

loj_model.summary()
from sklearn.linear_model import LogisticRegression

loj = LogisticRegression(solver = "liblinear")

loj_model = loj.fit(X,y)

loj_model
# constant value

loj_model.intercept_
loj_model.coef_
y_pred = loj_model.predict(X)              # predict
confusion_matrix(y, y_pred)
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
loj_model.predict(X)[0:10]


loj_model.predict_proba(X)[0:10][:,0:2]                # Top 10


y_probs = loj_model.predict_proba(X)

y_probs = y_probs[:,1]
y_probs[0:10]               # top 10


y_pred = [1 if i > 0.5 else 0 for i in y_probs]


y_pred[0:10]



confusion_matrix(y, y_pred)

accuracy_score(y, y_pred)
print(classification_report(y, y_pred))


loj_model.predict_proba(X)[:,1][0:5]
logit_roc_auc = roc_auc_score(y, loj_model.predict(X))
fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Oranı')

plt.ylabel('True Positive Oranı')

plt.title('ROC')

plt.show()

# test train is subjected to separation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)



loj = LogisticRegression(solver = "liblinear")

loj_model = loj.fit(X_train,y_train)

loj_model



accuracy_score(y_test, loj_model.predict(X_test))

cross_val_score(loj_model, X_test, y_test, cv = 10).mean()
