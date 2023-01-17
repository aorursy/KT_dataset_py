import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score



sns.set_style('darkgrid')
raw_data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

raw_data.head()
data = raw_data.copy()

data.info()
data.describe()
data.columns.values
data.isnull().sum()
plt.figure(figsize = (25,15))

sns.boxplot(data = pd.melt(data) , x = 'variable', y = 'value')

plt.show()
plt.figure(figsize = (8,6))

sns.distplot(data['total sulfur dioxide'])

plt.show()
data = data[data['total sulfur dioxide']<180]

plt.figure(figsize = (8,6))

sns.distplot(data['total sulfur dioxide'])

plt.show()
plt.figure(figsize = (15,15))

sns.boxplot(data = pd.melt(data) , x = 'variable', y = 'value')

plt.show()
plt.figure(figsize = (8,6))

sns.distplot(data['quality'])

plt.show()
plt.figure(figsize =(12,12))

sns.heatmap(data.corr(), cmap = 'Blues', annot = True)

plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor



variables = data[['density', 'citric acid', 'total sulfur dioxide', 'free sulfur dioxide']]

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif['Features'] = variables.columns

vif
data = data.drop('free sulfur dioxide', axis = 1)

data
variables = data[['density', 'citric acid', 'total sulfur dioxide']]

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

vif['Features'] = variables.columns

vif
from sklearn.ensemble import ExtraTreesClassifier 



X = data.drop('quality', axis = 1)

Y = data['quality']



model =  ExtraTreesClassifier()

model.fit(X,Y)



features = pd.DataFrame()

features['Features'] = X.columns

features['Importance'] = model.feature_importances_



plt.figure(figsize =(15,6))

sns.barplot(y='Importance', x='Features', data=features,  order=features.sort_values('Importance',ascending = False).Features)

plt.xlabel("Features", size=15)

plt.ylabel("Importance", size=15)

plt.title("Features Importance(Descending order)", size=18)

plt.tight_layout()
data_new = data.drop('pH', axis = 1)

data_new.shape
from sklearn.preprocessing import StandardScaler



x = data_new.drop('quality', axis = 1)

scaler = StandardScaler()

scaler.fit(x)

x_scaled = scaler.transform(x)

x_scaled.shape
plt.figure(figsize =(16,5))

plt.subplot(1,2,1)

sns.distplot(data['quality'])

plt.subplot(1,2,2)

sns.countplot(data['quality'])#Showing the frequency of occurence of a particular quality rating

plt.show()
category = [] # Defining an empty array

for x in data['quality']:

    if x>=1 and x<=3:

        category.append('Bad')

    elif x>=4 and x<=6:

        category.append('Normal')

    elif x>=7 and x<=10:

        category.append('Good')

        

        

data_new['category'] = category #Assigning a new column

data_new.head()
data_final = data_new.copy()

data_final = data_final.drop('quality',axis =1)

data_final.head()
data_final['category'].value_counts() #Checking the number of ratings in each category
from sklearn.model_selection import train_test_split



#defining inputs(independent) and targets(dependent) variables

inputs = x_scaled

targets = data_final['category']



#splitting into training and testing data



x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size = 0.2, random_state = 42)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
#Defining a method or function that will print the cross validation score and accuracy for each model



from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score



def model_report(cl):

    

    cl.fit(x_train, y_train)



    print('Cross Val Score: ',(cross_val_score(cl,x_train,y_train, cv=5).mean()*100).round(2))#using a 5-Fold cross validation



    y_pred = cl.predict(x_test)



    print('Accuracy Score: ', (accuracy_score(y_test,y_pred)*100).round(2))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()



model_report(lr)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()



model_report(dt)
from sklearn.svm import SVC



svc = SVC()



model_report(svc)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()



model_report(rf)
from sklearn.neighbors import KNeighborsClassifier



kn = KNeighborsClassifier(algorithm ='auto')



model_report(kn)
from sklearn.model_selection import GridSearchCV



#Defining a function that will calculate the best parameters and accuracy of the model based on those parameters

#Using GridSearchCV



def grid_search(classifier,parameters):

    

    grid = GridSearchCV(estimator = classifier,

                        param_grid = parameters,

                        scoring = 'accuracy',

                        cv = 5,

                        n_jobs = -1

                        )

    

    grid.fit(x_train,y_train)



    print('Best parameters: ', grid.best_params_) #Displaying the best parameters of the model



    print("Accuracy: ", ((grid.best_score_)*100).round(2))#Accuracy of the model based on those parameters
param_svc = {

    'C': [0.1, 1, 10, 100],  

    'gamma': [0.0001, 0.001, 0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 

    'kernel': ['linear','rbf']

    }

svc = SVC()



grid_search(svc,param_svc)
#Training the model again with the best parameters we got

svc = SVC(C = 10, gamma = 0.3, kernel='rbf')



model_report(svc)
param_rf = {

    'n_estimators': [10,50,100,500,1000],

    'min_samples_leaf': [1,10,20,50]

    }

rf = RandomForestClassifier(random_state = 0)

grid_search(rf,param_rf)
#Training the model again with the best parameters we got

rf = RandomForestClassifier(n_estimators = 1000, min_samples_leaf = 1,random_state = 0)

model_report(rf)
n_neighbors = list(range(5,10))#This is basically the value of k

                   

param_knn = {

    'n_neighbors' : n_neighbors,

    'p' : [1,2]

    

    }



knn = KNeighborsClassifier(algorithm ='auto', n_jobs = -1)

grid_search(knn,param_knn)
#Training the model again with the best parameters we got

knn = KNeighborsClassifier(n_neighbors = 7, p = 2, algorithm ='auto', n_jobs = -1)

model_report(knn)
from sklearn.ensemble import  AdaBoostClassifier



ab = AdaBoostClassifier(random_state = 42)



model_report(ab)
from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(random_state = 42, learning_rate = 0.2)



model_report(gb)
from xgboost import XGBClassifier



xg = XGBClassifier(random_state = 42, learning_rate = 0.2)



model_report(xg)
features_rf = pd.DataFrame()

x_rf = data_final.drop('category',axis=1)

features_rf['Features'] = x_rf.columns

features_rf['Importance'] = rf.feature_importances_



plt.figure(figsize =(15,6))

sns.barplot(y='Importance', x='Features', data=features_rf,  order=features_rf.sort_values('Importance',ascending = False).Features)

plt.xlabel("Features", size=15)

plt.ylabel("Importance", size=15)

plt.title("Features Importance(Descending order) for Random Forest", size=18)

plt.tight_layout()
features_xg = pd.DataFrame()

x_xg = data_final.drop('category',axis=1)

features_xg['Features'] = x_xg.columns

features_xg['Importance'] = xg.feature_importances_



plt.figure(figsize =(15,6))

sns.barplot(y='Importance', x='Features', data=features_xg,  order=features_xg.sort_values('Importance',ascending = False).Features)

plt.xlabel("Features", size=15)

plt.ylabel("Importance", size=15)

plt.title("Features Importance(Descending order) for XGBoost", size=18)

plt.tight_layout()