import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

%matplotlib inline

matplotlib.style.use('ggplot')
train=pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

test=pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
train.head()
test.head()
test.info()
test.info()
train.info()
test.info()
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
train.describe()
test.describe()
train['Response'].value_counts()
plt.figure(figsize=(8,6))

sns.countplot(train['Response'])

plt.title('Response Count for Training Data')

plt.xlabel('Response')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(train['Gender'])

plt.title('Gender Count for Training Data')

plt.xlabel('Response')

plt.ylabel('Count')
train['Gender'].value_counts()
plt.figure(figsize=(8,6))

sns.countplot(train['Response'],hue=train['Gender'])

plt.title('Gender Count for Training Data')

plt.xlabel('Gender')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(train['Response'],hue=train['Gender'])

plt.title('Gender Count for Training Data')

plt.xlabel('Response')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.distplot(train['Age'])

plt.title('Age distribution Training Data')

plt.xlabel('Age')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.distplot(test['Age'])

plt.title('Age distribution Test Data')

plt.xlabel('Age')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(train['Driving_License'])

plt.title('Driving License Count for Training Data')

plt.xlabel('Driving License')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(test['Driving_License'])

plt.title('Driving License Count for Test Data')

plt.xlabel('Driving License')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(train['Previously_Insured'])

plt.title('Previously Insured Count for Training Data')

plt.xlabel('Previously Insured')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(test['Previously_Insured'])

plt.title('Previously Insured Count for Test Data')

plt.xlabel('Previously Insured')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(train['Vehicle_Age'])

plt.title('Vehicle Age Count for Train Data')

plt.xlabel('Vehicle Age')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(train['Vehicle_Age'],hue=train['Response'])

plt.title('Vehicle Age Count for Train Data')

plt.xlabel('Vehicle Age')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(test['Vehicle_Age'])

plt.title('Vehicle Age Count for Test Data')

plt.xlabel('Vehicle Age')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(train['Vehicle_Damage'])

plt.title('Vehicle Damage Count for Train Data')

plt.xlabel('Vehicle Damage')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(test['Vehicle_Damage'])

plt.title('Vehicle Damage Count for Test Data')

plt.xlabel('Vehicle Damage')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.countplot(train['Vehicle_Damage'],hue=train['Response'])

plt.title('Vehicle Damage Count for Train Data')

plt.xlabel('Vehicle Damage')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.distplot(train['Annual_Premium'])

plt.title('Annual Premium for Train Data')

plt.xlabel('Annual Premium')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.distplot(test['Annual_Premium'])

plt.title('Annual Premium for Test Data')

plt.xlabel('Annual Premium')

plt.ylabel('Count')
train['Policy_Sales_Channel'].nunique()

plt.figure(figsize=(8,6))

sns.distplot(train['Policy_Sales_Channel'])

plt.title('Policy_Sales_Channel Count for Train Data')

plt.xlabel('Policy Sales Channel ')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.distplot(test['Policy_Sales_Channel'])

plt.title('Policy_Sales_Channel Count for Test Data')

plt.xlabel('Policy Sales Channel ')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.distplot(train['Vintage'])

plt.title('Vintage Count for Train Data')

plt.xlabel('Vintage ')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.distplot(test['Vintage'])

plt.title('Vintage Count for Test Data')

plt.xlabel('Vintage ')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.distplot(train['Region_Code'])

plt.title('Region Code Count for Train Data')

plt.xlabel('Region Code ')

plt.ylabel('Count')
plt.figure(figsize=(8,6))

sns.distplot(test['Region_Code'])

plt.title('Region Code Count for Test Data')

plt.xlabel('Region Code ')

plt.ylabel('Count')
train['Gender']=train['Gender'].map({'Male':1,'Female':0})

train['Vehicle_Age']=train['Vehicle_Age'].map({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

train['Vehicle_Damage']=train['Vehicle_Damage'].map({'Yes':1,'No':0})
test['Gender']=test['Gender'].map({'Male':1,'Female':0})

test['Vehicle_Age']=test['Vehicle_Age'].map({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

test['Vehicle_Damage']=test['Vehicle_Damage'].map({'Yes':1,'No':0})
train.info()
plt.figure(figsize=(10,10))

sns.heatmap(train.corr(),annot=True)
print(train.corr())
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
scalar=MinMaxScaler()
train['Age']=scalar.fit_transform(train[['Age']])

test['Age']=scalar.fit_transform(test[['Age']])
train['Vintage']=scalar.fit_transform(train[['Vintage']])

test['Vintage']=scalar.fit_transform(test[['Vintage']])
train.head()
test.head()
scalar=StandardScaler()
train['Annual_Premium']=scalar.fit_transform(train[['Annual_Premium']])

test['Annual_Premium']=scalar.fit_transform(test[['Annual_Premium']])
train=train.drop(['id'],axis=1)
X=train.drop(['Response'],axis=1)

y=train['Response']
y.head()
print("Before OverSampling, counts of label '1': {}".format(sum(y == 1))) 

print("Before OverSampling, counts of label '0': {} \n".format(sum(y == 0))) 
# import SMOTE module from imblearn library 

# pip install imblearn (if you don't have imblearn in your system) 

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 

X, y = sm.fit_sample(X, y.ravel()) 

  

print('After OverSampling, the shape of X: {}'.format(X.shape)) 

print('After OverSampling, the shape of y: {} \n'.format(y.shape)) 
print("After OverSampling, counts of label '1': {}".format(sum(y == 1))) 

print("After OverSampling, counts of label '0': {}".format(sum(y == 0))) 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from pprint import pprint

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import plot_confusion_matrix,roc_auc_score



import scikitplot as skplt
def model_error(model,X_test,y_test):

    predict = model.predict(X_test)

    print("RMSE of model: ",np.sqrt(mean_squared_error(y_test, predict)))

    print("\nAccuracy: ",accuracy_score(y_test,predict))

    print("\nClassification Report: ",classification_report(y_test,predict))

    print("\nConfusion Matrix: \n",confusion_matrix(y_test,predict))

    print("\nROC_AUC_Score: ",roc_auc_score(y_test,predict))



    fig, ax = plt.subplots(figsize=(10, 10))

    plot_confusion_matrix(model, X_test, y_test,ax=ax,cmap='YlOrBr',normalize='all')

    plt.title("Confusion Matrix")

    #skplt.metrics.plot_confusion_matrix(y_test, predict,figsize=(10,8),cmap='YlOrBr',text_fontsize='medium')

    plt.show()
rf=RandomForestClassifier()
base_model = RandomForestClassifier(n_estimators = 10)

base_model.fit(X_train, y_train)
model_error(base_model,X_test,y_test)
#HyperParameter Tuning



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 20, stop = 50, num = 5)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]

# Minimum number of samples required to split a node

min_samples_split = [5, 10]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split}



pprint(random_grid)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=42, n_jobs = -1, scoring='neg_mean_squared_error')
rf_random.fit(X_train,y_train)
rf_random.best_estimator_
model_error(rf_random.best_estimator_,X_test,y_test)
#knn Model

from sklearn.neighbors import KNeighborsClassifier
error_rate = []



# Will take some time

for i in range(1,20):

 

 knn = KNeighborsClassifier(n_neighbors=i)

 knn.fit(X_train,y_train)

 pred_i = knn.predict(X_test)

 error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,20),error_rate,color='blue', linestyle='dashed', marker='o',

 markerfacecolor='red', markersize=10)

plt.title("Error Rate vs. K Value")

plt.xlabel("K")

plt.ylabel("Error Rate")
# NOW WITH K=14

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

model_error(knn,X_test,y_test)
from catboost import CatBoostClassifier
classifier=CatBoostClassifier()
classifier.fit(X_train,y_train)
model_error(classifier,X_test,y_test)
Prediction = [predict[1] for predict in classifier.predict_proba(test.values)]

submission = pd.DataFrame(data = {'id': test['id'] ,'Response': Prediction})

submission.to_csv('./health-insurance-cross-sell-prediction_v1.csv', index = False)

submission.head()