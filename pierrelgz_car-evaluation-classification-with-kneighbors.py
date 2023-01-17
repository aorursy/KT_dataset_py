import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



data = pd.read_csv("/kaggle/input/car-evaluation-data-set/car_evaluation.csv")
data.head()
data = data.rename(columns = {'vhigh': 'buying', 'vhigh.1': 'maint', '2': 'doors', '2.1': 'persons', 'small': 'lug_boot', 'low': 'safety', 'unacc': 'decision'})

data.describe(include=['O'])
column = data[['buying','maint','doors','persons','lug_boot','safety','decision']]

for x in column:

    print(x," :",set(data[x]))
column = data[['buying','maint','doors','persons','lug_boot','safety','decision']]

for x in column:

    print(x," :")

    print(data[x].value_counts())
plt.hist(data['decision'])
data_label = data.copy()

data_label['decision'].replace(['unacc','acc','good','vgood'],[0,1,1,1], inplace=True) 

data_label['buying'].replace(['vhigh', 'med', 'low', 'high'],[3,1,0,2], inplace=True) 

data_label['maint'].replace(['vhigh', 'med', 'low', 'high'],[3,1,0,2], inplace=True) 

data_label['doors'].replace(['4', '5more', '3', '2'],[2,3,1,0], inplace=True) 

data_label['persons'].replace(['4', 'more', '2'],[1,2,0], inplace=True) 

data_label['lug_boot'].replace(['small', 'med', 'big'],[0,1,2], inplace=True)

data_label['safety'].replace(['low', 'med', 'high'],[0,1,2], inplace=True)

data_label.tail()
sns.barplot(data["safety"],data_label["decision"])
plt.figure("buy")

sns.barplot(data["buying"],data_label["decision"])

plt.figure("maint")

sns.barplot(data["maint"],data_label["decision"])
sns.barplot(data["persons"],data_label["decision"])
sns.barplot(data["doors"],data_label["decision"])
sns.barplot(data["lug_boot"],data_label["decision"])
#https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking

num=data_label.select_dtypes(exclude='object')

numcorr=num.corr()

Num=numcorr['decision'].sort_values(ascending=False).head(10).to_frame()



cm = sns.light_palette("cyan", as_cmap=True)



s = Num.style.background_gradient(cmap=cm)

s
data = data_label

data.head()
model = KNeighborsClassifier()



y= data['decision']

X= data.drop('decision', axis=1)



# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#Traning of the model and R score

model.fit(X_train,y_train)

first_model = model.score(X_test,y_test)

print("The score on the trained data is :",model.score(X_train,y_train))

print("The score on the tested data is :",model.score(X_test,y_test))
from sklearn.model_selection import validation_curve



model = KNeighborsClassifier()

k = np.arange(1, 50)



train_score, val_score = validation_curve(model, X_train, y_train,

                                          'n_neighbors', k, cv=5)# cv = the number of separations



plt.plot(k, val_score.mean(axis=1), label='validation')

plt.plot(k, train_score.mean(axis=1), label='train')



plt.ylabel('score')

plt.xlabel('n_neighbors')

plt.legend()
from sklearn.model_selection import GridSearchCV



param_grid = {'n_neighbors': np.arange(1, 20),

              'metric': ['euclidean', 'manhattan']}



grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5) 

grid.fit(X_train, y_train)



print(grid.best_score_) #best score from the cross-validation : fit on part of the training data, and the score is computed by predicting the rest of the training data.

print(grid.best_params_)



model = grid.best_estimator_

print("The new one:",model.score(X_test, y_test))

print("The old one :", first_model)

def exemple(model,buying =3,maint=3,doors=2,persons=2,lug_boot=1,safety=1):

    x=np.array([buying,maint,doors,persons,lug_boot,safety]).reshape(1,6)

    print(model.predict(x))

    print(model.predict_proba(x))

    

exemple(model)