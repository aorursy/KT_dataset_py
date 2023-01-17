import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style="whitegrid")

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier
data = pd.read_csv('/kaggle/input/Churn_Modelling.csv')

data.head()
data.shape
data.info()
data.describe()
for column in data.columns : 

    print('Number of unique data for {0} is {1}'.format(column , len(data[column].unique())))

    print('unique data for {0} is {1}'.format(column , data[column].unique()))

    print('=====================================')
data.drop(['RowNumber', 'CustomerId', 'Surname' ], axis=1, inplace=True)
data.head()
for column in data.columns : 

    print('Number of unique data for {0} is {1}'.format(column , len(data[column].unique())))

    print('unique data for {0} is {1}'.format(column , data[column].unique()))

    print('=====================================')
def make_pie(feature) : 

    plt.pie(data[feature].value_counts(),labels=list(data[feature].value_counts().index),

        autopct ='%1.2f%%' , labeldistance = 1.1,explode = [0.05 for i in range(len(data[feature].value_counts()))] )

    plt.show()
def make_countplot(feature) :

    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("prism", 3)) 
def make_kdeplot(feature) : 

    sns.kdeplot(data[feature], shade=True)
def divide_feature(feature,n):

    return round((data[feature]- data[feature].min())/n)
def make_label_encoder(original_feature , new_feature) : 

    enc  = LabelEncoder()

    enc.fit(data[original_feature])

    data[new_feature] = enc.transform(data[original_feature])

    data.drop([original_feature],axis=1, inplace=True)
def make_standardization(feature) : 

    data[feature] =  (data[feature] - data[feature].mean()) / (data[feature].max() - data[feature].min())
def make_report() : 

    print(classification_report(y_test,y_pred))

    print('************************************')

    CM = confusion_matrix(y_test, y_pred)

    print('Confusion Matrix is : \n', CM)

    print('************************************')

    sns.heatmap(CM, center = True)

    plt.show()
make_countplot("Exited")
len(data['CreditScore'].unique())
data['temp'] = divide_feature('CreditScore',100)
data.head()
make_countplot('temp')
data.drop(["temp" ], axis=1, inplace=True)
make_countplot("Geography")
make_pie('Geography')
make_countplot("Gender")
make_pie("Age")
data['temp'] = divide_feature('Age',10)
make_pie('temp')
make_kdeplot('Age')
data.drop(["temp" ], axis=1, inplace=True)
make_countplot("Tenure")
make_kdeplot('Balance')
data['temp'] = divide_feature('Balance',10000)

print('Number of Sectors are {}'.format(len(data['temp'].unique())))
make_pie('temp')
data.drop(["temp" ], axis=1, inplace=True)
make_pie('NumOfProducts')
make_countplot('HasCrCard')
make_pie('IsActiveMember')
len(data['EstimatedSalary'].unique())
data['temp'] = divide_feature('EstimatedSalary',10000)

print('Number of Sectors are {}'.format(len(data['temp'].unique())))
make_pie('temp')
make_kdeplot('temp')
data.drop(["temp"], axis=1, inplace=True)
data.head()
make_label_encoder('Geography' , 'Geography Code')
data.head()
make_label_encoder('Gender' , 'Gender Code')

data.head()
for column in data.columns  : 

    if not column  =='Exited' :

        make_standardization(column)
data.head()
X = data.drop(['Exited'], axis=1, inplace=False)

y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)



print('X_train shape is ' , X_train.shape)

print('X_test shape is ' , X_test.shape)

print('y_train shape is ' , y_train.shape)

print('y_test shape is ' , y_test.shape)
LogisticRegressionModel = LogisticRegression(penalty='l2',solver='sag',C=1.0,random_state=33)

LogisticRegressionModel.fit(X_train, y_train)
print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, y_train))

print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, y_test))

print('LogisticRegressionModel Classes are : ' , LogisticRegressionModel.classes_)

print('LogisticRegressionModel No. of iteratios is : ' , LogisticRegressionModel.n_iter_)

print('----------------------------------------------------')
y_pred = LogisticRegressionModel.predict(X_test)

y_pred_prob = LogisticRegressionModel.predict_proba(X_test)

make_report()
GaussianNBModel = GaussianNB()

GaussianNBModel.fit(X_train, y_train)



print('GaussianNBModel Train Score is : ' , GaussianNBModel.score(X_train, y_train))

print('GaussianNBModel Test Score is : ' , GaussianNBModel.score(X_test, y_test))
y_pred = GaussianNBModel.predict(X_test)

y_pred_prob = GaussianNBModel.predict_proba(X_test)

make_report()
DecisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=33) #criterion can be entropy

DecisionTreeClassifierModel.fit(X_train, y_train)



print('DecisionTreeClassifierModel Train Score is : ' , DecisionTreeClassifierModel.score(X_train, y_train))

print('DecisionTreeClassifierModel Test Score is : ' , DecisionTreeClassifierModel.score(X_test, y_test))

print('DecisionTreeClassifierModel Classes are : ' , DecisionTreeClassifierModel.classes_)

print('DecisionTreeClassifierModel feature importances are : ' , DecisionTreeClassifierModel.feature_importances_)
y_pred = DecisionTreeClassifierModel.predict(X_test)

y_pred_prob = DecisionTreeClassifierModel.predict_proba(X_test)



make_report()
SVCModel = SVC(kernel= 'sigmoid',# it can be also linear,poly,sigmoid,precomputed

               max_iter=1000,C=0.5,gamma='auto')

SVCModel.fit(X_train, y_train)



print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))

print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))
RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini',n_estimators=1000,max_depth=2,random_state=33) #criterion can be also : entropy 

RandomForestClassifierModel.fit(X_train, y_train)



print('RandomForestClassifierModel Train Score is : ' , RandomForestClassifierModel.score(X_train, y_train))

print('RandomForestClassifierModel Test Score is : ' , RandomForestClassifierModel.score(X_test, y_test))

print('RandomForestClassifierModel features importances are : ' , RandomForestClassifierModel.feature_importances_)

GBCModel = GradientBoostingClassifier(n_estimators=100,max_depth=5,random_state=33) 

GBCModel.fit(X_train, y_train)



#Calculating Details

print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))

print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))
y_pred = GBCModel.predict(X_test)

y_pred_prob = GBCModel.predict_proba(X_test)



print('Predicted Value for GBCModel is : ' , y_pred[:10])

print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob[:10])
make_report()
SelectedModel = GradientBoostingClassifier()

SelectedParameters = {'loss':('deviance', 'exponential'), 'max_depth':[1,2,3,4,5] , 'n_estimators':[50,75,100]}



GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters,cv = 5,return_train_score=True)

GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())

GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]
print('All Results are :\n', GridSearchResults )

print('Best Score is :', GridSearchModel.best_score_)

print('Best Parameters are :', GridSearchModel.best_params_)

print('Best Estimator is :', GridSearchModel.best_estimator_)
GBCModel = GridSearchModel.best_estimator_

GBCModel.fit(X_train, y_train)
print('GBCModel Train Score is : ' , GBCModel.score(X_train, y_train))

print('GBCModel Test Score is : ' , GBCModel.score(X_test, y_test))

print('GBCModel features importances are : ' , GBCModel.feature_importances_)
y_pred = GBCModel.predict(X_test)

y_pred_prob = GBCModel.predict_proba(X_test)



print('Predicted Value for GBCModel is : ' , y_pred)

print('Prediction Probabilities Value for GBCModel is : ' , y_pred_prob)
X_test.insert(10,'Predicted Valued',y_pred)
X_test.head(30)