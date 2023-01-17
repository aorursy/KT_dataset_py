import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/IBM_Employee.csv')
data.head()
data.columns
# #Converting certain features to categorical form
categorical_features = ['Attrition', 'BusinessTravel','Department','Education', 'EducationField',
                        'EnvironmentSatisfaction', 'Gender','JobInvolvement', 'JobLevel', 'JobRole',
                        'JobSatisfaction','MaritalStatus']
data[categorical_features] = data[categorical_features].astype('category')
data.info()
def categorical_eda(df):
    """Given dataframe, generate EDA of categorical data"""
    print("To check: Unique count of non-numeric data")
    print(df.select_dtypes(include=['category']).nunique())
    # Plot count distribution of categorical data
    
    for col in df.select_dtypes(include='category').columns:
        if df[col].nunique() < 20:
            fig = sns.catplot(x=col,hue='Attrition', kind="count", data=df)
            fig.set_xticklabels(rotation=90)
            plt.show()
        
        
categorical_eda(data)
data.columns
# #Converting certain features to categorical form
Numerical_features = ['MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
       'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
       'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager']
data[Numerical_features] = data[Numerical_features].astype('int')
data[Numerical_features]

def Numerical_eda(df):
    """Given dataframe, generate EDA of categorical data"""
    print("To check: Unique count of numeric data")
    print(df.select_dtypes(include=['int64']).nunique())
    # Plot count distribution of categorical data
    
    for col in df.select_dtypes(include='int64').columns:
        if df[col].nunique() < 20:
            g = sns.FacetGrid(data, col='Attrition')
            g = g.map(sns.distplot, col)
#             fig.set_xticklabels(rotation=90)
            plt.show()
        
        
Numerical_eda(data)


# g = sns.FacetGrid(data, col='Attrition')
# g = g.map(sns.distplot, "WorkLifeBalance")
def Numerical_eda(df):
    """Given dataframe, generate EDA of categorical data"""
    print("To check: Unique count of numeric data")
    print(df.select_dtypes(include=['int32']).nunique())
    # Plot count distribution of categorical data
    
    for col in df.select_dtypes(include='int32').columns:
        if df[col].nunique() < 20:
            g = sns.FacetGrid(data, col='Attrition')
            g = g.map(sns.distplot, col)
#             fig.set_xticklabels(rotation=90)
            plt.show()
Numerical_eda(data)
data.info()
# Explore Parch Attrition vs DistanceFromHome
g  = sns.factorplot(y="DistanceFromHome",x="Attrition",hue="Department",data=data,kind="bar", size = 6 , palette = "muted")
# Explore Parch Attrition vs DailyRate
g  = sns.factorplot(y="DailyRate",x="Attrition",data=data,kind="bar", size = 6 , palette = "muted")
# Explore Parch Attrition vs HourlyRate
g  = sns.factorplot(y="HourlyRate",x="Attrition",data=data,kind="bar", hue="Gender",size = 6 , palette = "muted")
# Explore Parch Attrition vs MonthlyIncome
g  = sns.factorplot(y="MonthlyIncome",x="Attrition",data=data,kind="bar", size = 6 , palette = "muted")
# Explore Parch Attrition vs MonthlyRate
g  = sns.factorplot(y="MonthlyRate",x="Attrition",data=data,kind="bar", size = 6 , palette = "muted")
data.info()
data.describe(include=['object', 'bool'])
data['OverTime'] = data['OverTime'].map({'Yes':1, 'No':0}).astype('int')
g = sns.barplot(x="OverTime",y="Attrition",data=data)
g = sns.barplot(y="PerformanceRating",x="Attrition",hue="Gender",data=data)
g = sns.barplot(x="Attrition",y="TotalWorkingYears",hue="Gender",data=data)
# YearsInCurrentRole
g = sns.barplot(x="Attrition",y="YearsInCurrentRole",hue="Gender",data=data)
g = sns.barplot(x="Attrition",y="YearsSinceLastPromotion",hue="BusinessTravel",data=data)
data.isna().sum()
data=data.drop(['EmployeeCount','EmployeeNumber','HourlyRate','Over18','PerformanceRating','StandardHours'],axis=1)
data.head()
data.info()
plt.figure(figsize=(20,20))
g = sns.heatmap(data.corr(),cmap="BrBG",annot=True)
data.head()
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
# Here applying label encoder to categorical attribute by using column key name.
data['Attrition']=labelencoder.fit_transform(data['Attrition'])
data['Attrition'].unique()
data.columns
data['BusinessTravel']=labelencoder.fit_transform(data['BusinessTravel'])
data['Department']=labelencoder.fit_transform(data['Department'])
data['EducationField']=labelencoder.fit_transform(data['EducationField'])
data['Gender']=labelencoder.fit_transform(data['Gender'])
data['JobRole']=labelencoder.fit_transform(data['JobRole'])
data['MaritalStatus']=labelencoder.fit_transform(data['MaritalStatus'])
data['OverTime']=labelencoder.fit_transform(data['OverTime'])
data.head()
data.info()
data['Education'] = data['Education'].astype('int')
data['EnvironmentSatisfaction'] = data['EnvironmentSatisfaction'].astype('int')
data['JobInvolvement'] = data['JobInvolvement'].astype('int')
data['JobLevel'] = data['JobLevel'].astype('int')
data['JobSatisfaction'] = data['JobSatisfaction'].astype('int')
data.info()
y=data['Attrition'].values
x=data.drop(['Attrition'],axis=1).values
print(X_train.shape)
print(y_train.shape)
# dataset split.
train_size=0.80
test_size=0.20
seed=5

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,train_size=train_size,test_size=test_size,random_state=seed)
n_neighbors=5
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# keeping all models in one list
models=[]
models.append(('LogisticRegression',LogisticRegression()))
models.append(('knn',KNeighborsClassifier(n_neighbors=n_neighbors)))
models.append(('SVC',SVC()))
models.append(("decision_tree",DecisionTreeClassifier()))
models.append(('Naive Bayes',GaussianNB()))

# Evaluating Each model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

names=[]
predictions=[]
error='accuracy'
for name,model in models:
    fold=KFold(n_splits=10,random_state=0)
    result=cross_val_score(model,X_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)\
    
# # Visualizing the Model accuracy
fig=plt.figure()
fig.suptitle("Comparing Algorithms")
plt.boxplot(predictions)
plt.show()
# Spot Checking and Comparing Algorithms With StandardScaler Scaler
from sklearn.pipeline import Pipeline
from sklearn. preprocessing import StandardScaler
pipelines=[]
pipelines.append(('scaled Logisitic Regression',Pipeline([('scaler',StandardScaler()),('LogisticRegression',LogisticRegression())])))
pipelines.append(('scaled KNN',Pipeline([('scaler',StandardScaler()),('KNN',KNeighborsClassifier(n_neighbors=n_neighbors))])))
pipelines.append(('scaled SVC',Pipeline([('scaler',StandardScaler()),('SVC',SVC())])))
pipelines.append(('scaled DecisionTree',Pipeline([('scaler',StandardScaler()),('decision',DecisionTreeClassifier())])))
pipelines.append(('scaled naive bayes',Pipeline([('scaler',StandardScaler()),('scaled Naive Bayes',GaussianNB())])))

# # Evaluating Each model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
names=[]
predictions=[]
for name,model in models:
    fold=KFold(n_splits=10,random_state=0)
    result=cross_val_score(model,X_train,y_train,cv=fold,scoring=error)
    predictions.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
    

# # Visualizing the Model accuracy
fig=plt.figure()
fig.suptitle("Comparing Algorithms")
plt.boxplot(predictions)
plt.show()
# Ensemble and Boosting algorithm to improve performance

#Ensemble
# Boosting methods
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Bagging methods
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
ensembles=[]
ensembles.append(('scaledAB',Pipeline([('scale',StandardScaler()),('AB',AdaBoostClassifier())])))
ensembles.append(('scaledGBC',Pipeline([('scale',StandardScaler()),('GBc',GradientBoostingClassifier())])))
ensembles.append(('scaledRFC',Pipeline([('scale',StandardScaler()),('rf',RandomForestClassifier(n_estimators=10))])))
ensembles.append(('scaledETC',Pipeline([('scale',StandardScaler()),('ETC',ExtraTreesClassifier(n_estimators=10))])))

# Evaluate each Ensemble Techinique
results=[]
names=[]
for name,model in ensembles:
    fold=KFold(n_splits=10,random_state=5)
    result=cross_val_score(model,X_train,y_train,cv=fold,scoring=error)
    results.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
    
# Visualizing the compared Ensemble Algorithms
fig=plt.figure()
fig.suptitle('Ensemble Compared Algorithms')
plt.boxplot(results)
plt.show()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                          random_state=0, shuffle=False)
clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
AdaBoostClassifier(n_estimators=100, random_state=0)
clf.feature_importances_
clf.score(X_test,y_test)
import xgboost
classifier=xgboost.XGBClassifier()

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X_train, y_train,cv=10)

score
score.mean()
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()
X_train.shape
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu',input_dim = 28))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, nb_epoch = 100)
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
y_pred

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test, y_pred))
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
