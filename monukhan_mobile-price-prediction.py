import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv('../input/mobile-price-classification/train.csv')

dataset.head()
dataset.isnull().sum()
dataset.dtypes
dataset.shape
dataset.describe()
value_counts = pd.value_counts(dataset['price_range'])

value_counts.values # converting into numpy array cause other wise we can't plot pie

label  = ['very high', 'high', 'medium', 'low']

colors = ['yellow','turquoise','lightblue', 'pink']

fig1, axarr = plt.subplots()



plt.pie(value_counts.values, autopct = '%0.01f', explode = [0.1,0.1,0.1,0.1], shadow = True, labels = label, colors = colors)



axarr.set_title('balanced or imbalaced?')

plt.show()

sns.jointplot(x = 'ram', y = 'price_range', data = dataset, kind = 'kde', color = 'green')
sns.pointplot(y = 'int_memory', x = 'price_range', data = dataset)
sns.boxplot(x = 'price_range', y = 'battery_power',data = dataset)
values = dataset['four_g'].value_counts()

label = ['4G-supported', 'Not supported']

color = ['lightgreen', 'lightpink']

fig, ax1 = plt.subplots()

plt.pie(values, autopct = '%0.01f', labels = label, startangle = 90, colors  =color, shadow = True)

ax1.set_title('4G supported or not supported?')

plt.show()
values = dataset['three_g'].value_counts()

label = ['3G supported', 'Not supported']

fig, ax1 = plt.subplots()

plt.pie(values, startangle = 70, labels = label, autopct = '%0.01f%%', explode = [0,0.1], shadow  = True)

ax1.set_title('3G supported or not supported?') 

plt.show()
plt.figure(figsize=(10,6))

dataset['fc'].hist(alpha=0.5,color='blue',label='Front camera')

dataset['pc'].hist(alpha=0.5,color='red',label='Primary camera')

plt.legend()

plt.xlabel('MegaPixels')

sns.jointplot(x = 'mobile_wt',y = 'price_range', data = dataset,kind = 'kde', color = 'green')

plt.show()
sns.pointplot(y = 'talk_time',x = 'price_range', data = dataset,kind = 'kde', color = 'gold')

plt.show()
#sns.pairplot(data = dataset, hue = 'price_range')
dataset.corr()
plt.figure(figsize = (20,20))

sns.heatmap(dataset.corr(), annot = True, cmap = 'RdYlGn')
X  = dataset.iloc[:,:-1]

y = dataset.iloc[:,-1]
X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
X_train
X_test
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class='multinomial',solver = 'sag') # (sag = Stochastic Average Gradient)

lr.fit(X_train, y_train)



# Predict the test set

y_pred = lr.predict(X_test)



# evauate the preformance

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = lr,X = X_train, y = y_train)

print('accuracy of validation set :', cvs.mean())

print('accuracy of the training set :', lr.score(X_train,y_train))

print('accuracy of the testset :', lr.score(X_test, y_test))
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'entropy')

dt.fit(X_train,y_train)



# Predict the test set

y_pred = dt.predict(X_test)



# evauate the preformance

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = dt,X = X_train, y = y_train)

print('accuracy of validation set :', cvs.mean())

print('accuracy of the training set :', dt.score(X_train,y_train))

print('accuracy of the testset :', dt.score(X_test, y_test))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

rf.fit(X_train, y_train)



# Predict the test set

y_pred = rf.predict(X_test)



# evauate the preformance

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = rf,X = X_train, y = y_train)

print('accuracy of validation set :', cvs.mean())

print('accuracy of the training set :', rf.score(X_train,y_train))

print('accuracy of the testset :', rf.score(X_test, y_test))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)



# Predict the test set

y_pred = nb.predict(X_test)



# evauate the preformance

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = nb,X = X_train, y = y_train)

print('accuracy of validation set :', cvs.mean())

print('accuracy of the training set :', nb.score(X_train,y_train))

print('accuracy of the testset :', nb.score(X_test, y_test))
parameters ={

'C' : [1,0.1,0.25,0.5,2,0.75],

'kernel' : ["linear","rbf"],

'gamma' : ["auto",0.01,0.001,0.0001,1],

'decision_function_shape' : ["ovo" ,"ovr"]}
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC



grid_search = GridSearchCV(estimator = SVC(),

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           )

grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best Parameters:", best_parameters)
from sklearn.svm import SVC



svc=SVC(C=2,gamma="auto",decision_function_shape="ovo",kernel="linear",random_state=0)

svc.fit(X_train, y_train)





# Predict the test set

y_pred = svc.predict(X_test)



# evaluate the preformance

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))
from sklearn.model_selection import cross_val_score

cvs = cross_val_score(estimator = svc,X = X_train, y = y_train)

print('accuracy of validation set :', cvs.mean())

print('accuracy of the training set :', svc.score(X_train,y_train))

print('accuracy of the testset :', svc.score(X_test, y_test))
plt.figure(figsize = (12,6))

label = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'GaussainNB', 'Support Vector Machine',]

acc_score = [0.95, 0.85, 0.87, 0.83, 0.95]



plt.bar(label,acc_score, color=['lightblue', 'pink', 'lightgrey','gold', 'cyan'])

plt.title('Which model is the most accurate?')

plt.xlabel('')

plt.ylabel('Accuracy Scores')

plt.show()
test_data = pd.read_csv('../input/mobile-price-classification/test.csv')

test_data.head()
test_df  = test_data.drop('id', axis = 1)
test_df


sc = StandardScaler()

test_df1 = sc.fit_transform(test_df)

predicted_price_range = svc.predict(test_df1) 
predicted_price_range
test_df['price_range'] = predicted_price_range
test_df