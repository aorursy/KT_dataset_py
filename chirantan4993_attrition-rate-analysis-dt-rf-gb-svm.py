import pandas as pd # package for high-performance, easy-to-use data 

#structures and data analysis

import numpy as np # fundamental package for scientific computing with Python

import matplotlib

import matplotlib.pyplot as plt # for plotting

import seaborn as sns # for making plots with seaborn

import missingno as msno #checking missing values

color = sns.color_palette()

sns.set_style("whitegrid")

plt.style.use("fivethirtyeight")

import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.offline as offline

offline.init_notebook_mode()

from pylab import rcParams





from datetime import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split

from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score



# import cufflinks and offline mode

import cufflinks as cf

cf.go_offline()



# from sklearn import preprocessing

# # Supress unnecessary warnings so that presentation looks clean

import warnings

warnings.filterwarnings("ignore")

data=pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data.head(5)
data.info()

data.describe().T
msno.bar(data, color = 'r', figsize = (10,8))   
#Reassign target

data.Attrition.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)

# Drop useless feat

data = data.drop(columns=['StandardHours', 

                          'EmployeeCount', 

                          'Over18',

                        ])

data.head(5)
attrition = data[(data['Attrition'] != 0)]

no_attrition = data[(data['Attrition'] == 0)]



#COUNT

trace = go.Bar(x = (len(attrition), len(no_attrition)), y = ['Yes_attrition', 'No_attrition'], orientation = 'h', opacity = 0.8, marker=dict(

        color=['gold', 'lightskyblue'],

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  'Attrition Count')

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)



#PERCENTAGE

trace = go.Pie(labels = ['No_attrition', 'Yes_attrition'], values = data['Attrition'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['lightskyblue','gold'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Attrition Distribution')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
plt.figure(figsize=(20,20))

corr = data.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(20,20))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr,  cmap="RdYlGn", annot=True, fmt=".2f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
data_num=data.select_dtypes(include='number')

data_num.head()
data_obj=data.select_dtypes(include='object')

data_obj.head()
plt.subplots(figsize=(10,10))

sns.countplot(data.Age)
sns.scatterplot(x='Age',y='MonthlyIncome',data=data)
sns.countplot(x='Attrition',hue='PerformanceRating',data=data)
plt.subplots(figsize=(10,8))

sns.countplot(data.Education)
plt.subplots(figsize=(10,8))

sns.countplot(data.NumCompaniesWorked)
plt.subplots(figsize=(10,8))

sns.countplot(data.PercentSalaryHike)
plt.subplots(figsize=(6,8))

sns.countplot(x='BusinessTravel', data=data)
plt.subplots(figsize=(10,8))

sns.countplot(x='MaritalStatus', data=data)
print(data.groupby(['Gender','MaritalStatus'])['MaritalStatus'].count())

print(data.groupby('Gender')['Gender'].count())
plt.figure(figsize=(8,8))

plt.pie(data['JobRole'].value_counts(),labels=data['JobRole'].value_counts().index,autopct='%.2f%%');

plt.title('Job Role Distribution',fontdict={'fontsize':22});
plt.subplots(figsize=(10,8))

fig = plt.gcf()

fig.set_size_inches(20,14)

sns.countplot(x='JobRole', hue='Gender',data=data)

plt.title('Job Role Between Male and Female')
plt.figure(figsize=(8,8))

plt.pie(data['EducationField'].value_counts(),labels=data['EducationField'].value_counts().index,autopct='%.2f%%')
plt.subplots(figsize=(10,8))

sns.countplot(x='Department', data=data)
plt.subplots(figsize=(10,8))

sns.countplot(x='Department', hue='Attrition',data=data)
plt.figure(figsize=(12, 9))

sns.boxplot(x='PercentSalaryHike',y='Age',data=data,palette='winter')
sns.set(font_scale=1)

sns.boxplot(x='JobRole',y='MonthlyIncome',data=data)

plt.xticks(rotation=90)
sns.boxplot(x='EducationField',y='MonthlyIncome',data=data)

plt.xticks(rotation=90)
data.corr()['Attrition'].sort_values(ascending=False)
data.groupby(by='JobRole')["PercentSalaryHike","YearsAtCompany","TotalWorkingYears","YearsInCurrentRole","WorkLifeBalance"].mean()
data.head()
data_obj=data.select_dtypes(include='object')

data_obj.head()
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
categorical_col=[]

for col in data.columns:

    if data[col].dtype== object and data[col].nunique()<=50:

        categorical_col.append(col)

print(categorical_col)
for col in categorical_col:

    data[col]=le.fit_transform(data[col])
data.shape
from sklearn.model_selection import train_test_split
X= data.drop('Attrition',axis=1)

y=data['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import roc_auc_score,roc_curve



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 33)

print("Train Set Size : ",X_train.shape)

print("Train Target Set Size : ",y_train.shape)

print("Test  Set Size : ",X_test.shape)

print("Test  Target Set Size : ",y_test.shape)
from sklearn.tree import DecisionTreeClassifier
model= DecisionTreeClassifier()
model.fit(X_train,y_train)
dir(model) #to select which all parameters are important to us
pred= model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
from sklearn.model_selection import RandomizedSearchCV
params={"criterion":("gini", "entropy"),

        "splitter":("best", "random"), 

        "max_depth":(list(range(1, 20))), 

        "min_samples_split":[2, 3, 4], 

        "min_samples_leaf":list(range(1, 20))}
tree_randomized= RandomizedSearchCV(model,params,n_iter=100,n_jobs=-1,cv=5,verbose=2)
tree_randomized.fit(X_train,y_train)
tree_randomized.best_estimator_
model=DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',

                       max_depth=4, max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=11, min_samples_split=4,

                       min_weight_fraction_leaf=0.0, presort='deprecated',

                       random_state=42, splitter='random')
model.fit(X_train,y_train)

pred=model.predict(X_test)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
print(RandomForestClassifier())

print(RandomForestRegressor()) #check HP we can tune
from sklearn.ensemble import RandomForestClassifier 

import pandas as pd

from sklearn.model_selection import GridSearchCV

import warnings

warnings.filterwarnings('ignore', category=FutureWarning) #to let us that the default value for gridsearch is going to change in future release

warnings.filterwarnings('ignore', category=DeprecationWarning) #to let us know tyhe beahviour of gridsearchcv within test
def print_results(results):

    print('BEST PARAMS: {}\n'.format(results.best_params_))



    means = results.cv_results_['mean_test_score']

    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):

        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))
rf = RandomForestClassifier()

parameters = {

    'n_estimators': [5,50,250], 'max_depth':[2,4,8,16,32,None] #none will let it go as deep as it want

}



cv = GridSearchCV(rf, parameters, cv=5) #(modelobject, parameter dictionary, how many folds we want cv=5)

cv.fit(X_train,y_train.values.ravel()) #training lables are stored as vector type, but we need array , hence .ravel()



print_results(cv)
cv.best_estimator_
rf= RandomForestClassifier(n_estimators=50,max_depth=32)

rf.fit(X_train,y_train)

rf_pred= rf.predict(X_test)

print(classification_report(y_test,rf_pred))
print(confusion_matrix(y_test,rf_pred))
from sklearn.svm import SVC
SVC()# we only select ones that are imp - C and kernel
dir(SVC)
from sklearn.model_selection import train_test_split

from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 101)
clf = svm.SVC()

clf.fit(X_train,y_train)



print('Accuracy of SVC on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))



print('Accuracy of SVC on test set: {:.2f}'.format(clf.score(X_test, y_test) * 100))
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)

accuracy_lin_train = linear.score(X_train, y_train)

accuracy_lin_test = linear.score(X_test, y_test)

print('Accuracy Linear Kernel on training set:', accuracy_lin_train*100)

print('Accuracy Linear Kernel on testing set:', accuracy_lin_test*100)
rbf = svm.SVC(kernel='rbf', gamma=0.1, C=0.1, decision_function_shape='ovo').fit(X_train, y_train)

accuracy_rbf_train = rbf.score(X_train, y_train)

accuracy_rbf_test = rbf.score(X_test, y_test)

print('Accuracy Radial Basis Kernel on training set:', accuracy_rbf_train*100)

print('Accuracy Radial Basis Kernel on testing set:', accuracy_rbf_test*100)
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
# May take awhile!

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
data.columns
X= data.drop(['Attrition','BusinessTravel','DailyRate','Department','DistanceFromHome','Education','EmployeeNumber','Gender',

             'HourlyRate','JobInvolvement','JobLevel','JobRole','JobSatisfaction','MaritalStatus',

             'MonthlyRate','NumCompaniesWorked','OverTime','RelationshipSatisfaction','StockOptionLevel',

              'TrainingTimesLastYear'],axis=1)

y=data['Attrition']
data.head()
X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import roc_auc_score,roc_curve



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)

print("Train Set Size : ",X_train.shape)

print("Train Target Set Size : ",y_train.shape)

print("Test  Set Size : ",X_test.shape)

print("Test  Target Set Size : ",y_test.shape)
# Applying Scaling Standardiztion to all of the features in order to bring them into common scale .

# Standardiztion : is preferred when most of the featues are not following gaussian distribution . 



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = pd.DataFrame(sc.fit_transform(X_train))

X_test  = pd.DataFrame(sc.fit_transform(X_test))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 42 )



# Setting Parameters for Logistic Regression . 



params = {    # Regularization Params

             'penalty' : ['l1','l2','elasticnet'],

              # Lambda Value 

             'C' : [0.01,0.1,1,10,100]

         }



log_reg = GridSearchCV(lr,param_grid = params,cv = 10)

log_reg.fit(X_train,y_train)

log_reg.best_params_
# Make Prediction of test data 

y_pred = log_reg.predict(X_test)

print(classification_report(y_test,y_pred))
plt.rcParams['figure.figsize'] = (6,4)

class_names = [1,0]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(confusion_matrix(y_test,y_pred)), annot = True, cmap = 'BuGn_r',

           fmt = 'g')

plt.tight_layout()

plt.title('Confusion matrix for Logistic Regression  Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
plt.rcParams['figure.figsize'] = (10,6)



# Get predicted probabilites from the model

y_proba = log_reg.predict_proba(X_test)[:,1]



# display auc value for log_reg

auc_log_reg = roc_auc_score(y_test,y_pred)

print("roc_auc_score value for log reg is : ",roc_auc_score(y_test,y_pred))



# Create true and false positive rates

fpr_log_reg,tpr_log_reg,thershold_log_reg_model = roc_curve(y_test,y_proba)

plt.plot(fpr_log_reg,tpr_log_reg)

plt.plot([0,1],ls='--')

#plt.plot([0,0],[1,0],c='.5')

#plt.plot([1,1],c='.5')

plt.title('Reciever Operating Characterstic For Logistic Regregression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
from sklearn.tree import DecisionTreeClassifier 



dt = DecisionTreeClassifier(random_state = 42)





# Setting Parameters for DecisionTreeClassifier . 



params = {  

             'criterion'    : ["gini", "entropy"],

             'max_features' : ["auto", "sqrt", "log2"],

              'min_samples_split' :[i for i in range(4,16)],

              'min_samples_leaf' : [i for i in range(4,16)]

         }



dt_clf = GridSearchCV(dt,param_grid = params,cv = 10)

dt_clf.fit(X_train,y_train)

dt_clf.best_params_
# Make Prediction of test data 

y_pred = dt_clf.predict(X_test)

print(classification_report(y_test,y_pred))
plt.rcParams['figure.figsize'] = (6,4)

class_names = [1,0]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(confusion_matrix(y_test,y_pred)), annot = True, cmap = 'BuGn_r',

           fmt = 'g')

plt.tight_layout()

plt.title('Confusion matrix for DecisionTreeClassifier   Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
plt.rcParams['figure.figsize'] = (10,6)



# Get predicted probabilites from the model

y_proba = dt_clf.predict_proba(X_test)[:,1]



dt_clf_auc_score = roc_auc_score(y_test,y_pred)

# display auc value for DecisionTreeClassifier

print("roc_auc_score value for log reg is : ",roc_auc_score(y_test,y_pred))



# Create true and false positive rates

fpr_dt_clf,tpr_dt_clf,thershold_dt_clf_model = roc_curve(y_test,y_proba)

plt.plot(fpr_dt_clf,tpr_dt_clf)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.title('Reciever Operating Characterstic For DecisionTreeClassifier ')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators = 150,min_samples_split = 20,min_samples_leaf = 5,random_state = 42)

rf_clf.fit(X_train,y_train)

y_pred = rf_clf.predict(X_test)

# Make Prediction of test data 

y_pred = rf_clf.predict(X_test)

print(classification_report(y_test,y_pred))
plt.rcParams['figure.figsize'] = (6,4)

class_names = [1,0]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(confusion_matrix(y_test,y_pred)), annot = True, cmap = 'BuGn_r',

           fmt = 'g')

plt.tight_layout()

plt.title('Confusion matrix for RandomForestClassifier   Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
plt.rcParams['figure.figsize'] = (10,6)



# Get predicted probabilites from the model

y_proba = dt_clf.predict_proba(X_test)[:,1]



rf_auc_score = roc_auc_score(y_test,y_pred)



# display auc value for RandomForestClassifier

print("roc_auc_score value for log reg is : ",roc_auc_score(y_test,y_pred))



# Create true and false positive rates

fpr_rf_clf,tpr_rf_clf,thershold_rf_clf_model = roc_curve(y_test,y_proba)

plt.plot(fpr_rf_clf,tpr_rf_clf)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.title('Reciever Operating Characterstic For RandomForestClassifier ')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_jobs = -1)



# set params



params = {

             "n_neighbors" : [i for i in range(15)],

               'p' : [1,2] ,

              'leaf_size' : [i for i in range(15)],

               

          }

knn = GridSearchCV(knn,param_grid = params, cv = 5)

knn.fit(X_train,y_train)

knn.best_params_
# Make Prediction of test data 

y_pred = knn.predict(X_test)

print(classification_report(y_test,y_pred))
plt.rcParams['figure.figsize'] = (6,4)

class_names = [1,0]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(confusion_matrix(y_test,y_pred)), annot = True, cmap = 'BuGn_r',

           fmt = 'g')

plt.tight_layout()

plt.title('Confusion matrix for KNN Algorithm   Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
plt.rcParams['figure.figsize'] = (10,6)



# Get predicted probabilites from the model

y_proba = knn.predict_proba(X_test)[:,1]



knn_auc_score = roc_auc_score(y_test,y_pred)





# display auc value for KNN Algorithm

print("roc_auc_score value for log reg is : ",roc_auc_score(y_test,y_pred))



# Create true and false positive rates

fpr_KNN,tpr_KNN,thershold_KNN_model = roc_curve(y_test,y_proba)

plt.plot(fpr_KNN,tpr_KNN)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.title('Reciever Operating Characterstic For KNN Algorithm ')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
plt.figure(figsize=(10,6))

plt.title('Reciever Operating Characterstic Curve')

plt.plot(fpr_log_reg,tpr_log_reg,label='LogisticRegression')

plt.plot(fpr_dt_clf,tpr_dt_clf,label='DecisionTreeClassifier')

plt.plot(fpr_rf_clf,tpr_rf_clf,label='RandomForestClassifier')

plt.plot(fpr_KNN,tpr_KNN,label='KNearestNeighbors ')

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.legend()

plt.show()
print("Area Under Curve Score values for Different algorithms : ")

print("LogisticRegression          : ",auc_log_reg)

print("DecisionTreeClassfier       : ",dt_clf_auc_score)

print("RandomForest Classifier     : ",rf_auc_score)

print("KnearestNeighborsClassifier : ",knn_auc_score)
from sklearn.svm import SVC
scaler=StandardScaler()

scaled_data=scaler.fit_transform(data.drop('Attrition',axis=1))

X=scaled_data

y=data['Attrition']
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
SVC()# we only select ones that are imp - C and kernel
from sklearn.model_selection import train_test_split

from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


clf = svm.SVC()

clf.fit(X_train,y_train)



print('Accuracy of SVC on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))



print('Accuracy of SVC on test set: {:.2f}'.format(clf.score(X_test, y_test) * 100))

from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train,y_train)
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=3)

grid.fit(X_train, y_train)
from sklearn.model_selection import train_test_split

from sklearn import svm

import matplotlib.pyplot as plt

import numpy as np







X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=150)



clf = svm.SVC(C=1,gamma=0.01)

clf.fit(X_train,y_train)



print('Accuracy of SVC on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))



print('Accuracy of SVC on test set: {:.2f}'.format(clf.score(X_test, y_test) * 100))
from sklearn.svm import SVC

svm = SVC(kernel='linear')

svm.fit(X_train,y_train)
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(kernel='linear'), param_grid, refit = True, verbose=3)

grid.fit(X_train, y_train)
from sklearn.model_selection import train_test_split

from sklearn import svm

import matplotlib.pyplot as plt

import numpy as np







X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=150)



clf = svm.SVC(kernel='linear',C=1,gamma=0.01)

clf.fit(X_train,y_train)



print('Accuracy of SVC on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))



print('Accuracy of SVC on test set: {:.2f}'.format(clf.score(X_test, y_test) * 100))
from sklearn.svm import SVC

svm = SVC(kernel='rbf')

svm.fit(X_train,y_train)
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, refit = True, verbose=3)

grid.fit(X_train, y_train)
from sklearn.model_selection import train_test_split

from sklearn import svm

import matplotlib.pyplot as plt

import numpy as np







X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=150)



clf = svm.SVC(kernel='rbf',C=1,gamma=0.01)

clf.fit(X_train,y_train)



print('Accuracy of SVC on training set: {:.2f}'.format(clf.score(X_train, y_train) * 100))



print('Accuracy of SVC on test set: {:.2f}'.format(clf.score(X_test, y_test) * 100))
from sklearn.model_selection import train_test_split

from sklearn import svm

import matplotlib.pyplot as plt

import numpy as np







X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=150)
clf = svm.SVC(kernel='precomputed')
gram_train = np.dot(X_train, X_train.T)

clf.fit(gram_train, y_train)
gram_test = np.dot(X_test, X_train.T)

clf.predict(gram_test)
print('Accuracy of SVC on training set: {:.2f}'.format(clf.score(gram_train, y_train) * 100))

print('Accuracy of SVC on training set: {:.2f}'.format(clf.score(gram_test, y_test) * 100))
plt.rcParams['figure.figsize'] = (6,4)

class_names = [1,0]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(confusion_matrix(y_test,y_pred)), annot = True, cmap = 'BuGn_r',

           fmt = 'g')

plt.tight_layout()

plt.title('Confusion matrix for Logistic Regression  Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
print(GradientBoostingClassifier())

print(GradientBoostingRegressor())
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore', category=FutureWarning) #to let us that the default value for gridsearch is going to change in future release

warnings.filterwarnings('ignore', category=DeprecationWarning) #to let us know tyhe beahviour of gridsearchcv within test

gb = GradientBoostingClassifier()

parameters = {

    'n_estimators': [5,50,250,500], 'max_depth':[2,4,8,16,32],'learning_rate': [0.01,0.1,1,10,100]

}





cv = GridSearchCV(gb, parameters, cv=5) #(modelobject, parameter dictionary, how many folds we want cv=5)

cv.fit(X_train,y_train.values.ravel()) #training lables are stored as vector type, but we need array , hence .ravel()



print_results(cv)
cv.best_estimator_