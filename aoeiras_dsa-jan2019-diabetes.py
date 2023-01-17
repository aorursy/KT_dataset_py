import os

path = os.getcwd()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import warnings

warnings.filterwarnings('ignore')

from sklearn import metrics

%matplotlib inline
pima = pd.read_csv('../input/diabetes/diabetes.csv', 

                   header = None, sep = ",",

                   names=['Pregnancy', 'Glucose', 'BloodPressure' ,'SkinfoldThickness', 'Insulin', 'BodyMassIndex', 'DiabetesPedigreeFunction', 'Age', 'Class'])   
pima.head(5)
pima.info()
pima.describe() 
pima[pima.isnull().any(axis=1)] 
pima.isnull().values.any() 
# Análise dos dados

columns=pima.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in zip(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    pima[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
# Análise de casos de diabetes

pima1=pima[pima['Class']==1]

columns=pima.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in zip(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    pima1[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
pima.plot(kind= 'box' , subplots=True, layout=(3,3),figsize=(14,10))
pima['Class'].value_counts().plot(kind='bar', figsize=(6,6))

plt.title('pima_indians_diabetes - Class')

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.show()
pima.boxplot(column='Insulin',by='Class')

plt.show()
pima.boxplot(column='Pregnancy',by='Class')

plt.show()
pima.boxplot(column='Glucose',by='Class')

plt.show()
plt.figure(figsize=(14,3))

Insulin_plt = pima.groupby(pima['Insulin']).Class.count().reset_index()

sns.distplot(pima[pima.Class == 0]['Insulin'], color='red', kde=False, label='Diabetic')

sns.distplot(pima[pima.Class == 1]['Insulin'], color='green', kde=False, label='Non-Diabetic')

plt.legend()

plt.title('Histogram of Insulin values depending in the class')

plt.show()
plt.figure(figsize=(20,5))

glucose_plt = pima.groupby('Glucose').Class.mean().reset_index()

sns.barplot(glucose_plt.Glucose, glucose_plt.Class)

plt.title('Percentual de chance de ser diagnosticado com diabetes por leitura de glicose')

plt.show()
corr = pima.corr()

_ , ax = plt.subplots( figsize =( 12 , 10 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
import seaborn as sns

sns.pairplot(pima,hue='Class',palette='coolwarm')
# Check for zero values

print("Number of missing values : " + repr(pima[pima.Glucose == 0].shape[0]))

print(pima[pima.Glucose == 0].groupby('Class')['Class'].count())
# Replace zero value with the mean value of the classes

Glucose_0 = pima[(pima['Glucose']== 0)]

pima[(pima['Glucose']== 0) & (pima['Class'] == 0)] = Glucose_0[Glucose_0['Class']== 0].replace(0, pima[(pima['Class']== 0)].mean())

pima[(pima['Glucose']== 0) & (pima['Class'] == 1)] = Glucose_0[Glucose_0['Class']== 1].replace(0, pima[(pima['Class']== 1)].mean())
# Check for zero values

print("Number of missing values : " + repr(pima[pima.BloodPressure == 0].shape[0]))

print(pima[pima.BloodPressure == 0].groupby('Class')['Class'].count())
# Replace zero value with the mean value of the classes

BloodPressure_0 = pima[(pima['BloodPressure']== 0)]

pima[(pima['BloodPressure']== 0) & (pima['Class'] == 0)] = BloodPressure_0[BloodPressure_0['Class']== 0].replace(0, pima[(pima['Class']== 0)].mean())

pima[(pima['BloodPressure']== 0) & (pima['Class'] == 1)] = BloodPressure_0[BloodPressure_0['Class']== 1].replace(0, pima[(pima['Class']== 1)].mean())
# Check for zero values

print("Number of missing values : " + repr(pima[pima.SkinfoldThickness == 0].shape[0]))

print(pima[pima.SkinfoldThickness == 0].groupby('Class')['Class'].count())
# Replace zero value with the mean value of the classes

SkinfoldThickness_0 = pima[(pima['SkinfoldThickness']== 0)]

pima[(pima['SkinfoldThickness']== 0) & (pima['Class'] == 0)] = SkinfoldThickness_0[SkinfoldThickness_0['Class']== 0].replace(0, pima[(pima['Class']== 0)].mean())

pima[(pima['SkinfoldThickness']== 0) & (pima['Class'] == 1)] = SkinfoldThickness_0[SkinfoldThickness_0['Class']== 1].replace(0, pima[(pima['Class']== 1)].mean())
# Check for zero values

print("Number of abnormal cases in skinfold thickness : " + repr(pima[pima.SkinfoldThickness > 60].shape[0]))

print(pima[pima.SkinfoldThickness > 60]['SkinfoldThickness'])

print(pima[pima.SkinfoldThickness > 60].groupby('Class')['Class'].count())
# imputing impossible value with mean value

pima['SkinfoldThickness'].iloc[579] = pima['SkinfoldThickness'].mean()
# Check for zero values

print("Number of missing values : " + repr(pima[pima.Insulin == 0].shape[0]))

print(pima[pima.Insulin == 0].groupby('Class')['Class'].count())
# Replace zero value with the mean value of the classes

Insulin_0 = pima[(pima['Insulin']== 0)]

pima[(pima['Insulin']== 0) & (pima['Class'] == 0)] = Insulin_0[Insulin_0['Class']== 0].replace(0, pima[(pima['Class']== 0)].mean())

pima[(pima['Insulin']== 0) & (pima['Class'] == 1)] = Insulin_0[Insulin_0['Class']== 1].replace(0, pima[(pima['Class']== 1)].mean())
# Check for zero values

print("Number of missing values : " + repr(pima[pima.BodyMassIndex == 0].shape[0]))

print(pima[pima.BodyMassIndex == 0].groupby('Class')['Class'].count())
# Replace zero value with the mean value of the classes

BodyMassIndex_0 = pima[(pima['BodyMassIndex']== 0)] 

pima[(pima['BodyMassIndex']== 0) & (pima['Class'] == 0)] = BodyMassIndex_0[BodyMassIndex_0['Class']== 0].replace(0, pima[(pima['Class']== 0)].mean())

pima[(pima['BodyMassIndex']== 0) & (pima['Class'] == 1)] = BodyMassIndex_0[BodyMassIndex_0['Class']== 1].replace(0, pima[(pima['Class']== 1)].mean())
pima.describe() 
columns=pima.columns[:8]

plt.subplots(figsize=(18,15))

length=len(columns)

for i,j in zip(columns,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    pima[i].hist(bins=20,edgecolor='black')

    plt.title(i)

plt.show()
pima.plot(kind= 'box' , subplots=True, layout=(3,3),figsize=(14,10))
pima.Class.value_counts()
random_state=12342
#data = pima.copy().iloc[:, 0:8].values

#target = pima.copy().iloc[:, 8:9].Class.values
!pip install imblearn
import imblearn

# Obs: pode ser necessário reiniciar o Jupyter para poder carregar o pacote
# Oversampling data is given with a subscript of 'o'

np.random.seed(75)

from imblearn.over_sampling import SMOTE, ADASYN

data_o, target_o = SMOTE().fit_sample(pima, pima.Class)
data_o.shape
target_o.shape
import collections

collections.Counter(target_o)
from sklearn.model_selection import train_test_split
Xo_train, Xo_test, yo_train, yo_test = train_test_split(data_o, target_o, test_size=0.20, random_state=4)
Xo_train.shape
yo_train.shape
Xo_test.shape
yo_test.shape
data=pima[pima.columns[:8]]

target=pima['Class']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols = data.iloc[:, 0:8].columns

data[cols] = scaler.fit_transform(data)

data.head()
#data[cols] = preprocessing.scale(data)
train,test=train_test_split(pima,test_size=0.20,random_state=437,stratify=pima['Class'])# stratify the outcome



X_train=train[train.columns[:8]]

X_test=test[test.columns[:8]]

y_train=train['Class']

y_test=test['Class']
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xo_train, yo_train)
pred = knn.predict(Xo_test)
pred.shape
from sklearn.metrics import confusion_matrix

print (confusion_matrix(yo_test,pred))
from sklearn.metrics import classification_report
print (classification_report(yo_test,pred))
from sklearn import preprocessing

data_on = preprocessing.scale(data_o)

Xon_train, Xon_test, yon_train, yon_test = train_test_split(data_on, target_o, test_size=0.20, random_state=4)
knn.fit(Xon_train, yon_train)

pred = knn.predict(Xon_test)



print (confusion_matrix(yon_test,pred))
print (classification_report(yon_test,pred))
import numpy as np 

import matplotlib.pyplot as plt



error_rate = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(Xo_train,yo_train)

    pred_i = knn.predict(Xo_test)

    error_rate.append(np.mean(pred_i != yo_test))



plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# NOW WITH K=1



knn = KNeighborsClassifier(n_neighbors=1, weights='distance',p=1)



knn.fit(Xo_train,yo_train)

pred = knn.predict(Xo_test)



print('WITH K=1')

print('\n')

print('Confusion Matrix')

cm_knn = confusion_matrix(yo_test,pred)

print(cm_knn)

print('\n')

rpt_knn = classification_report(yo_test,pred)

print(rpt_knn)
from sklearn.utils import shuffle

new_Ind = []
cur_MaxScore = 0.0
col_num = 8
col_Ind_Random = shuffle(range(0,col_num), random_state=13)


for cur_f in range(0, col_num):

    new_Ind.append(col_Ind_Random[cur_f])

    newData = data.values[:, new_Ind]

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(newData, target, test_size=0.2, random_state=1987)

    clf = KNeighborsClassifier(1)

    fit = clf.fit(Xs_train, ys_train)

    cur_Score = clf.score(Xs_test, ys_test)

    if cur_Score < cur_MaxScore:

        new_Ind.remove(col_Ind_Random[cur_f])

    else:

        cur_MaxScore = cur_Score

        print ("Score with " + str(len(new_Ind)) + " selected features: " + str(cur_Score))



error_rate = []

random_state=19

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(Xs_train,ys_train)

    pred_i = knn.predict(Xs_test)

    error_rate.append(np.mean(pred_i != ys_test))



plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=9, weights='distance',p=1)



knn.fit(Xs_train,ys_train)

pred = knn.predict(Xs_test)



print('WITH K=31')

print('\n')

print('Confusion Matrix')

cm_knn = confusion_matrix(ys_test,pred)

print(cm_knn)

print('\n')

rpt_knn = classification_report(ys_test,pred)

print(rpt_knn)
error_rate = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))



plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=30, weights='distance',p=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=31')

print('\n')

print('Confusion Matrix')

cm_knn = confusion_matrix(y_test,pred)

print(cm_knn)

print('\n')

rpt_knn = classification_report(y_test,pred)

print(rpt_knn)
from sklearn.tree import DecisionTreeClassifier

random_state=234

dtree = DecisionTreeClassifier(random_state=998)
dtree.fit(X_train,y_train)
pred = dtree.predict(X_test)

print("Accuracy for Decision treeclassifier is",metrics.accuracy_score(pred,y_test))
print('Confusion Matrix')

cm_dtree = confusion_matrix(y_test,pred)

print(cm_dtree)

print('\n')

rpt_dtree = classification_report(y_test,pred)

print(rpt_dtree)
from IPython.display import Image

from sklearn import tree



dot_data = tree.export_graphviz(dtree, out_file='tree.dot', 

                         filled=True, rounded=True,  

                         special_characters=True) 
feat_names = pima.copy().iloc[:, 0:8].columns

targ_names = ['Yes','No']
!pip install graphviz
import graphviz

import os

from sklearn import tree

os.environ["PATH"] += os.pathsep + path

from sklearn.tree import DecisionTreeClassifier,export_graphviz



import graphviz



data1 = export_graphviz(dtree,out_file=None,feature_names=feat_names,class_names=targ_names,   

                         filled=True, rounded=True,  

                         special_characters=True)

graph = graphviz.Source(data1)

graph
from sklearn.ensemble import RandomForestClassifier
# number of base decision tree estimators

n_est = 100

# maximum depth of any given decision tree estimator

max_depth = 5

# random state variable

rstate = 42

# initialize a random forest algorithm



rf = RandomForestClassifier(n_estimators=n_est, 

                             max_depth=max_depth,

                             random_state=rstate)
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
print('Confusion Matrix')

cm_rf = confusion_matrix(y_test,pred)

print(cm_rf)

print('\n')

rpt_rf = classification_report(y_test,pred)

print(rpt_rf)
# list of columns to be used for training each model

features = [col for col in list(X_train) ]

print('%i features: %s' % (len(features), features))
# report the most important featuers for predicting each target



# collect ranking of most "important" features for E

importances =  rf.feature_importances_

descending_indices = np.argsort(importances)[::-1]

sorted_importances = [importances[idx] for idx in descending_indices]

sorted_features = [features[idx] for idx in descending_indices]

print('Most important feature for diabetes energy is %s' % sorted_features[0])
# plot the feature importances



def plot_importances(X_train, sorted_features, sorted_importances):

    """

    Args:

        X_train (nd-array) - feature matrix of shape (number samples, number features)

        sorted_features (list) - feature names (str)

        sorted_importances (list) - feature importances (float)

    Returns:

        matplotlib bar chart of sorted importances

    """

    axis_width = 1.5

    maj_tick_len = 6

    fontsize = 14

    bar_color = 'lightblue'

    align = 'center'

    label = '__nolegend__'

    ax = plt.bar(range(X_train.shape[1]), sorted_importances,

                 color=bar_color, align=align, label=label)

    ax = plt.xticks(range(X_train.shape[1]), sorted_features, rotation=90)

    ax = plt.xlim([-1, X_train.shape[1]])

    ax = plt.ylabel('Feature Importance', fontsize=fontsize)

    ax = plt.tick_params('both', length=maj_tick_len, width=axis_width, 

                         which='major', right=True, top=True)

    ax = plt.xticks(fontsize=fontsize)

    ax = plt.yticks(fontsize=fontsize)

    ax = plt.tight_layout()

    return ax



fig1 = plt.figure(1, figsize=(10,8))



ax = plot_importances(X_train, sorted_features, sorted_importances)



# plt.tight_layout()

plt.show()

plt.close()
temp=[]

classifier=['Decision Tree','Random Forest','KNN','KNN (Smote)']

models=[DecisionTreeClassifier(random_state=998),RandomForestClassifier(n_estimators=n_est, 

                             max_depth=max_depth,

                             random_state=rstate),KNeighborsClassifier(n_neighbors=6),"SMOTE"]

for i in models:

    model = i

    if model == "SMOTE":

        model = KNeighborsClassifier(n_neighbors=1,weights='distance',p=1)

        model.fit(Xo_train,yo_train)

        pred1=model.predict(Xo_test)

        temp.append(metrics.accuracy_score(pred1,yo_test))

    else:

        model.fit(X_train,y_train)

        prediction=model.predict(X_test)    

        temp.append(metrics.accuracy_score(prediction,y_test))

        

models_dataframe=pd.DataFrame(temp,index=classifier)   

models_dataframe.columns=['Accuracy']

models_dataframe
diab2=pima[['Pregnancy','Glucose','SkinfoldThickness','Insulin','BodyMassIndex','Age','Class']]



train1,test1=train_test_split(diab2,test_size=0.20,random_state=437,stratify=diab2['Class'])



X_train=train1[train1.columns[:6]]

X_test=test1[test1.columns[:6]]

y_train=train1['Class']

y_test=test1['Class']



# SMOTE

np.random.seed(795)

data1, target1 = SMOTE().fit_sample(diab2, diab2.Class)

Xo_train, Xo_test, yo_train, yo_test = train_test_split(data1, target1, test_size=0.20, random_state=4)
temp=[]

classifier=['Decision Tree','Random Forest','KNN','KNN (Smote)']

models=[DecisionTreeClassifier(random_state=998),RandomForestClassifier(n_estimators=n_est, 

                             max_depth=max_depth,

                             random_state=rstate),KNeighborsClassifier(n_neighbors=6),"SMOTE"]

for i in models:

    model = i

    if model == "SMOTE":

        model = KNeighborsClassifier(n_neighbors=1,weights='distance',p=1)

        model.fit(Xo_train,yo_train)

        pred1=model.predict(Xo_test)

        temp.append(metrics.accuracy_score(pred1,yo_test))

    else:

        model.fit(X_train,y_train)

        prediction=model.predict(X_test)    

        temp.append(metrics.accuracy_score(prediction,y_test))

        

models_dataframe=pd.DataFrame(temp,index=classifier)   

models_dataframe.columns=['Accuracy']

models_dataframe
from sklearn.model_selection import KFold 

from sklearn.model_selection import cross_val_score 

kfold = KFold(n_splits=10, random_state=998) 
temp=[]

accuracy=[]

classifiers=['KNN','KNN (SMOTE)','Decision Tree','Random Forest']

models=[KNeighborsClassifier(n_neighbors=6),"SMOTE", DecisionTreeClassifier(),RandomForestClassifier(n_estimators=n_est, 

                             max_depth=max_depth,

                             random_state=938)]

for i in models:

    model = i

    

    if model == "SMOTE":

        model = KNeighborsClassifier(n_neighbors=1,weights='distance',p=1)

        cv_result = cross_val_score(model,data_o,target_o, cv = kfold,scoring = "accuracy")

        temp.append(cv_result.mean())

        accuracy.append(cv_result)

       

    else:

        cv_result = cross_val_score(model,data,target, cv = kfold,scoring = "accuracy")

        temp.append(cv_result.mean())

        accuracy.append(cv_result)

new_models_dataframe2=pd.DataFrame(temp,index=classifiers)   

new_models_dataframe2.columns=['CV Mean']    

new_models_dataframe2
box=pd.DataFrame(accuracy,index=[classifiers])

fig3 = plt.figure(1, figsize=(12,8))

sns.boxplot(data=box.T, orient="h", palette="Set1")

plt.show()
from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf.get_params())
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)
rf_random.best_params_
def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    errors = abs(predictions - y_test)

    mape = 100 * np.mean(errors)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    

    return accuracy
base_model = RandomForestClassifier(random_state = 82)

base_model.fit(X_train, y_train)

base_accuracy = evaluate(base_model, X_test, y_test)
best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(dtree.get_params())
from sklearn.model_selection import RandomizedSearchCV



# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Method of selecting samples for training each tree

bootstrap = [True, False]



random_state=294



# Create the random grid

random_grid = {

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               }

pprint(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

dtree = DecisionTreeClassifier()

random_state=194

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

dtree_random = RandomizedSearchCV(estimator = dtree, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=294, n_jobs = -1)

# Fit the random search model

dtree_random.fit(X_train, y_train)
dtree_random.best_params_
base_model =  DecisionTreeClassifier(min_samples_leaf=2, min_samples_split=3,random_state=474)

base_model.fit(X_train, y_train)

base_accuracy = evaluate(base_model, X_test, y_test)
pprint(base_model.get_params())
best_dtree_random = dtree_random.best_estimator_

best_dtree_random.fit(X_train, y_train)

random_accuracy = evaluate(best_dtree_random, X_test, y_test)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
best_dtree_random
dot_data = tree.export_graphviz(best_dtree_random, out_file='tree2.dot', 

                         filled=True, rounded=True,  

                         special_characters=True) 
temp=[]

accuracy=[]

classifiers=['KNN','KNN (SMOTE)','Decision Tree','Random Forest']

models=[KNeighborsClassifier(n_neighbors=6),"SMOTE", best_dtree_random,best_random]

for i in models:

    model = i

    

    if model == "SMOTE":

        model = KNeighborsClassifier(n_neighbors=1,weights='distance',p=1)

        cv_result = cross_val_score(model,data_o,target_o, cv = kfold,scoring = "accuracy")

        temp.append(cv_result.mean())

        accuracy.append(cv_result)

       

    else:

        cv_result = cross_val_score(model,data,target, cv = kfold,scoring = "accuracy")

        temp.append(cv_result.mean())

        accuracy.append(cv_result)

new_models_dataframe2=pd.DataFrame(temp,index=classifiers)   

new_models_dataframe2.columns=['CV Mean']    

new_models_dataframe2
box=pd.DataFrame(accuracy,index=[classifiers])

fig3 = plt.figure(1, figsize=(12,8))

sns.boxplot(data=box.T, orient="h", palette="Set1")

plt.show()