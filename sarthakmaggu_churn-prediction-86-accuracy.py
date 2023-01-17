import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import QuantileTransformer

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel
data = pd.read_csv("../input/predicting-churn-for-bank-customers/Churn_Modelling.csv")
data.corr()
data.shape
data.head()
data["Geography"].unique() #checking for unique values in Geography
data.describe()
data.dtypes
plt.figure(figsize = (15,15))

sns.catplot(x = 'Geography', kind = 'count', data = data, palette = 'pink')

plt.title('Customers distribution across Countries')

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'Gender', kind = 'count', data = data, palette = 'pastel')

plt.title("Males vs Females")

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'IsActiveMember', kind = 'count', data = data, palette = 'pink')

plt.title("Active VS Non-Active Members")

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'HasCrCard', kind = 'count', palette = 'pastel', data = data)

plt.title("Credit Card VS No Credit Card")

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'Exited', kind = 'count', hue = 'Gender', palette = 'pink', data = data)

plt.title("Gender and Exited")

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'HasCrCard', kind = 'count', hue = 'Gender', palette = 'pastel', data = data)

plt.title("Gender and Credit Card")

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'IsActiveMember', kind = 'count', hue = 'Gender', palette = 'pink', data = data)

plt.title("Gender and Active Members")

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = "NumOfProducts", kind = 'count', palette = 'pastel', data = data )

plt.title('Number of Products')

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'Tenure', kind = 'count', palette = 'pastel', data = data)

plt.title("Tenure of Customer")

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'Exited', kind = 'count', hue = 'IsActiveMember', palette = 'pink', data = data)

plt.title("Exited and Active Members")

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'Exited', kind = 'count', hue = 'HasCrCard', palette = 'pastel', data = data)

plt.title("Exited and Card")

plt.show()
plt.figure(figsize = (15,15))

sns.catplot(x = 'IsActiveMember', kind = 'count', hue = 'HasCrCard', palette = 'pink', data = data)

plt.title('Active Member and Card')

plt.show()
plt.figure(figsize = (15,15))

sns.scatterplot(x = 'Balance', y = 'EstimatedSalary', hue = 'Exited',palette = 'pastel', data = data)

plt.title("Balance vs Estimated Salary")

plt.show()
plt.figure(figsize = (15,15))

sns.scatterplot(x = 'Balance', y = 'CreditScore', hue = 'Exited',palette = 'pink', data = data)

plt.title("Balance vs Credit Score")

plt.show()
plt.figure(figsize = (15,15))

sns.scatterplot(x = 'Balance', y = 'EstimatedSalary', hue = 'Gender',palette = 'pastel', data = data)

plt.title("Estimated Salary vs Credit Score")

plt.show()
plt.figure(figsize = (15,15))

sns.scatterplot(x = 'Balance', y = 'EstimatedSalary', hue = 'IsActiveMember',palette = 'pastel', data = data)

plt.title("Estimated Salary vs Credit Score")

plt.show()
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1, inplace = True)
data.isnull().sum() #checking for null values
plt.figure(figsize = (15,15))

sns.distplot(data['Age'])

plt.title("Age")

plt.show()
plt.figure(figsize = (15,15))

sns.distplot(data["CreditScore"])

plt.title("Credit Score")

plt.show()
plt.figure(figsize = (15,15))

sns.distplot(data["EstimatedSalary"])

plt.title("Estimated Salary")

plt.show()
plt.figure(figsize = (15,15))

sns.distplot(data["Balance"])

plt.title("Balance")

plt.show()
column = ["Age", "Balance", "EstimatedSalary", "CreditScore"]

for i in column:

    plt.figure(figsize = (15,15))

    sns.boxplot(data[i])

    plt.title('Box Plot')

    plt.show()
data = data[(data["Age"] <60)]

data = data[(data["CreditScore"] >400)]
data.describe()
data["Balance"] = QuantileTransformer().fit_transform(data["Balance"].values.reshape(-1,1))

data["CreditScore"] = QuantileTransformer().fit_transform(data["CreditScore"].values.reshape(-1,1))

data["EstimatedSalary"] = QuantileTransformer().fit_transform(data["EstimatedSalary"].values.reshape(-1,1))

data["Age"] = QuantileTransformer().fit_transform(data["Age"].values.reshape(-1,1))
data["Balance"] = StandardScaler().fit_transform(data["Balance"].values.reshape(-1,1))

data["CreditScore"] = StandardScaler().fit_transform(data["CreditScore"].values.reshape(-1,1))

data["EstimatedSalary"] = StandardScaler().fit_transform(data["CreditScore"].values.reshape(-1,1))

data.describe()
data["Geography"] = LabelEncoder().fit_transform(data["Geography"])

data["Gender"] = LabelEncoder().fit_transform(data["Gender"])
data.head()
data.corr()
y = data["Exited"]
y.head()
data.drop(["Exited"], axis = 1, inplace = True)
data.head()
train_x,test_x,train_y,test_y = train_test_split(data,y, test_size = 0.3, random_state = 50)
logistic = LogisticRegression()

logistic.fit(train_x,train_y)

log_y = logistic.predict(test_x)

print(accuracy_score(log_y,test_y))
random_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,'penalty':['l1','l2']}

print(random_parameters)
random_para = RandomizedSearchCV(estimator = logistic, param_distributions = random_parameters, n_iter = 50, cv = 10, verbose=2, random_state= 50, n_jobs = -1)

random_para.fit(train_x,train_y)
random_para.best_params_
logistic2 = LogisticRegression(penalty ='l2', C =1)

logistic2.fit(train_x,train_y)

log_y = logistic2.predict(test_x)

print(accuracy_score(log_y,test_y))
feature = SelectFromModel(LogisticRegression())

feature.fit(train_x,train_y)

feature_support = feature.get_support()

feature_selected = train_x.loc[:,feature_support].columns.tolist()

print(str(len(feature_selected)), 'selected features')
print(feature_selected)
train_x_feature = train_x[["Age", "IsActiveMember"]]

train_x_feature.head()
test_x_feature = test_x[["Age", "IsActiveMember"]]

test_x_feature.head()
logistic.fit(train_x_feature, train_y)

log_y_feature = logistic.predict(test_x_feature)

print(accuracy_score(log_y_feature, test_y))
random = RandomForestClassifier()

random.fit(train_x,train_y)

random_y = random.predict(test_x)

print(accuracy_score(random_y,test_y))
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto','sqrt']

max_depth = [int(x) for x in np.linspace(10,110,num=11)]

max_depth.append(None)

min_samples_split = [2,5,10]

min_samples_leaf = [1,2,4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

'max_features': max_features,

'max_depth': max_depth,

'min_samples_split': min_samples_split,

'min_samples_leaf': min_samples_leaf,

'bootstrap': bootstrap

}

print(random_grid)
random_para = RandomizedSearchCV(estimator = random, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

random_para.fit(train_x,train_y)
random_para.best_params_
random_2 = RandomForestClassifier(n_estimators=1400,min_samples_split =10,min_samples_leaf= 2,max_features = 'sqrt',max_depth=80,bootstrap= True)

random_2.fit(train_x,train_y)

random_2_y = random_2.predict(test_x)

print(accuracy_score(random_2_y,test_y)) 
feature = SelectFromModel(RandomForestClassifier(n_estimators=1400,min_samples_split =10,min_samples_leaf= 2,max_features = 'sqrt',max_depth=80,bootstrap= True))

feature.fit(train_x,train_y)

feature_support = feature.get_support()

feature_selected = train_x.loc[:,feature_support].columns.tolist()

print(str(len(feature_selected)), 'selected features')
feature_selected
train_x_feature = train_x[['Age', 'Balance', 'NumOfProducts']]

train_x_feature.head()
test_x_feature = test_x[['Age', 'Balance', 'NumOfProducts']]

test_x_feature.head()
random_2.fit(train_x_feature,train_y)

random_2_feature_y = random_2.predict(test_x_feature)

print(accuracy_score(random_2_feature_y,test_y))
bayes = GaussianNB()

bayes.fit(train_x,train_y)

bayes_y = bayes.predict(test_x)

print(accuracy_score(bayes_y,test_y))
train_x_feature = train_x[["Age", "Balance"]] #based on correlation values

train_x_feature.head()
test_x_feature = test_x[["Age", "Balance"]] #based on correlation values

test_x_feature.head()
bayes.fit(train_x_feature,train_y)

bayes_feature_y =bayes.predict(test_x_feature)

print(accuracy_score(bayes_feature_y, test_y))
print(str((accuracy_score(random_2_y,test_y)) * 100) + "%")