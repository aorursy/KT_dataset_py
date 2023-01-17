import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Some visualization libraries

from matplotlib import pyplot as plt

import seaborn as sns



## Some other snippit of codes to get the setting right 

## This is so that the chart created by matplotlib can be shown in the jupyter notebook. 

%matplotlib inline 

%config InlineBackend.figure_format = 'retina' ## This is preferable for retina display. 



#import warnings ## importing warnings library. 

#warnings.filterwarnings('ignore') ## Ignore warning



import os ## imporing os

print(os.listdir("../input/"))
## Importing the datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.head()
print ("The shape of the train data is (row, column):"+ str(train.shape))

print (train.info())

print ("The shape of the test data is (row, column):"+ str(test.shape))

print (test.info())
##!!  this block is for delete. I think.



## saving passenger id in advance in order to submit later. 

passengerid = test.PassengerId

## We will drop PassengerID and Ticket since it will be useless for our data. 

#train.drop(['PassengerId'], axis=1, inplace=True)

#test.drop(['PassengerId'], axis=1, inplace=True)
total = train.isnull().sum().sort_values(ascending = False)

percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)

pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
total = test.isnull().sum().sort_values(ascending = False)

percent = round(test.isnull().sum().sort_values(ascending = False)/len(test)*100, 2)

pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])
## Concat train and test into a variable "all_data"

survivers = train.Survived



train.drop(["Survived"],axis=1, inplace=True)



all_data = pd.concat([train,test], ignore_index=False)
## were carefull with all_data. data leackage possible if use mean() and so on







## assign most appropriate value according to charts

all_data.Embarked.fillna("C", inplace=True)







## we do not touch "Age" feature with Null here, as it is very important
## Assign all the null values to N

all_data.Cabin.fillna("N", inplace=True)



# multiple entries in Cabin grouped into one by first letter

all_data.Cabin = [i[0] for i in all_data.Cabin]



with_N = all_data[all_data.Cabin == "N"]



without_N = all_data[all_data.Cabin != "N"]



all_data.groupby("Cabin")['Fare'].mean().sort_values()





def cabin_estimator(i):

    a = 0

    if i<16:

        a = "G"

    elif i>=16 and i<27:

        a = "F"

    elif i>=27 and i<38:

        a = "T"

    elif i>=38 and i<47:

        a = "A"

    elif i>= 47 and i<53:

        a = "E"

    elif i>= 53 and i<54:

        a = "D"

    elif i>=54 and i<116:

        a = 'C'

    else:

        a = "B"

    return a



##applying cabin estimator function. 

with_N['Cabin'] = with_N.Fare.apply(lambda x: cabin_estimator(x))



## getting back train. 

all_data = pd.concat([with_N, without_N], axis=0)
## getting back train. 



## PassengerId helps us separate train and test. 

all_data.sort_values(by = 'PassengerId', inplace=True)



## Separating train and test from all_data. 

train = all_data[:891]



test = all_data[891:]



## replace "fare" null values with mean vallue for similar records

missing_value = test[(test.Pclass == 3) & (test.Embarked == 'S') & (test.Sex == 'male')].Fare.mean()

test.Fare.fillna(missing_value, inplace=True)



# adding saved target variable with train. 

train['Survived'] = survivers



# Placing 0 for female and 

# 1 for male in the "Sex" column. 

train['Sex'] = train['Sex'].apply(lambda x: 0 if x == "female" else 1)

test['Sex'] = test['Sex'].apply(lambda x: 0 if x == "female" else 1)

print(train.shape,test.shape)
## a lot of charts here
train.describe()
#many other statistics here
pd.DataFrame(abs(train.corr()['Survived']).sort_values(ascending = False))
## get the most important variables. 

corr = train.corr()**2

corr.Survived.sort_values(ascending=False)
## heatmeap to see the correlation between features. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)

mask = np.zeros_like(train.corr(), dtype=np.bool)

#mask[np.triu_indices_from(mask)] = True



plt.subplots(figsize = (15,12))

sns.heatmap(train.corr(), 

            annot=True,

            #mask = mask,

            cmap = 'RdBu_r',

            linewidths=0.1, 

            linecolor='white',

            vmax = .9,

            square=True)

plt.title("Correlations Among Features", y = 1.03,fontsize = 20);
# Hypothesis testing (null hypothesis) and p-value
train['name_length'] = [len(i) for i in train.Name]

test['name_length'] = [len(i) for i in test.Name]



def name_length_group(size):

    a = ''

    if (size <=20):

        a = 'short'

    elif (size <=35):

        a = 'medium'

    elif (size <=45):

        a = 'good'

    else:

        a = 'long'

    return a





train['nLength_group'] = train['name_length'].map(name_length_group)

test['nLength_group'] = test['name_length'].map(name_length_group)



print(train.shape,test.shape)
## get the title from the name

train["title"] = [i.split('.')[0] for i in train.Name]

train["title"] = [i.split(',')[1] for i in train.title]

test["title"] = [i.split('.')[0] for i in test.Name]

test["title"]= [i.split(',')[1] for i in test.title]



#rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']

#train.Name = ['rare' for i in train.Name for j in rare_title if i == j]

## train Data

train["title"] = [i.replace('Ms', 'Miss') for i in train.title]

train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]

train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]

train["title"] = [i.replace('Dr', 'rare') for i in train.title]

train["title"] = [i.replace('Col', 'rare') for i in train.title]

train["title"] = [i.replace('Major', 'rare') for i in train.title]

train["title"] = [i.replace('Don', 'rare') for i in train.title]

train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]

train["title"] = [i.replace('Sir', 'rare') for i in train.title]

train["title"] = [i.replace('Lady', 'rare') for i in train.title]

train["title"] = [i.replace('Capt', 'rare') for i in train.title]

train["title"] = [i.replace('the Countess', 'rare') for i in train.title]

train["title"] = [i.replace('Rev', 'rare') for i in train.title]







#rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']

#train.Name = ['rare' for i in train.Name for j in rare_title if i == j]

## test data

test['title'] = [i.replace('Ms', 'Miss') for i in test.title]

test['title'] = [i.replace('Dr', 'rare') for i in test.title]

test['title'] = [i.replace('Col', 'rare') for i in test.title]

test['title'] = [i.replace('Dona', 'rare') for i in test.title]

test['title'] = [i.replace('Rev', 'rare') for i in test.title]
print(train.shape,test.shape)
## Family_size seems like a good feature to create

train['family_size'] = train.SibSp + train.Parch+1

test['family_size'] = test.SibSp + test.Parch+1



def family_group(size):

    a = ''

    if (size <= 1):

        a = 'loner'

    elif (size <= 4):

        a = 'small'

    else:

        a = 'large'

    return a



train['family_group'] = train['family_size'].map(family_group)

test['family_group'] = test['family_size'].map(family_group)



print(train.shape,test.shape)
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]

test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]



print(train.shape,test.shape)
#don't know how to manage this features, so drop it.



train.drop(['Ticket'], axis=1, inplace=True)

test.drop(['Ticket'], axis=1, inplace=True)



train.drop(['PassengerId'], axis=1, inplace=True)

test.drop(['PassengerId'], axis=1, inplace=True)



print(train.shape,test.shape)
## Calculating fare based on family size. 

train['calculated_fare'] = train.Fare/train.family_size

test['calculated_fare'] = test.Fare/test.family_size



print(train.shape,test.shape)
def fare_group(fare):

    a= ''

    if fare <= 4:

        a = 'Very_low'

    elif fare <= 10:

        a = 'low'

    elif fare <= 20:

        a = 'mid'

    elif fare <= 45:

        a = 'high'

    else:

        a = "very_high"

    return a



train['fare_group'] = train['calculated_fare'].map(fare_group)

test['fare_group'] = test['calculated_fare'].map(fare_group)



#train['fare_group'] = pd.cut(train['calculated_fare'], bins = 5, labels=('Very_low','low','mid','high','very_high'))



print(train.shape,test.shape)
#way of moving from categorical variables to numbers



train = pd.get_dummies(train, columns=['title',"Pclass", 'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)

test = pd.get_dummies(test, columns=['title',"Pclass",'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)

train.drop(['family_size','Name', 'Fare','name_length'], axis=1, inplace=True)

test.drop(['Name','family_size',"Fare",'name_length'], axis=1, inplace=True)
print(train.shape,test.shape)

train.head()
## rearranging the columns so that I can easily use the dataframe to predict the missing age values. 

train = pd.concat([train[["Survived", "Age", "Sex","SibSp","Parch"]], train.loc[:,"is_alone":]], axis=1)

test = pd.concat([test[["Age", "Sex"]], test.loc[:,"SibSp":]], axis=1)



## Importing RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor



## writing a function that takes a dataframe with missing values and outputs it by filling the missing values. 

def completing_age(df):

    ## gettting all the features except survived

    age_df = df.loc[:,"Age":] 

    

    temp_train = age_df.loc[age_df.Age.notnull()] ## df with age values

    temp_test = age_df.loc[age_df.Age.isnull()] ## df without age values

    

    y = temp_train.Age.values ## setting target variables(age) in y 

    x = temp_train.loc[:, "Sex":].values

    

    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)

    rfr.fit(x, y)

    

    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])

    

    df.loc[df.Age.isnull(), "Age"] = predicted_age

    



    return df



## Implementing the completing_age function in both train and test dataset. 

completing_age(train)

completing_age(test)
## create bins for age

def age_group_fun(age):

    a = ''

    if age <= 1:

        a = 'infant'

    elif age <= 4: 

        a = 'toddler'

    elif age <= 13:

        a = 'child'

    elif age <= 18:

        a = 'teenager'

    elif age <= 35:

        a = 'Young_Adult'

    elif age <= 45:

        a = 'adult'

    elif age <= 55:

        a = 'middle_aged'

    elif age <= 65:

        a = 'senior_citizen'

    else:

        a = 'old'

    return a

        

## Applying "age_group_fun" function to the "Age" column.

train['age_group'] = train['Age'].map(age_group_fun)

test['age_group'] = test['Age'].map(age_group_fun)



## Creating dummies for "age_group" feature. 

train = pd.get_dummies(train,columns=['age_group'], drop_first=True)

test = pd.get_dummies(test,columns=['age_group'], drop_first=True);



print(train.shape,test.shape)
X = train.drop(['Survived'], axis = 1)

y = train["Survived"]





print(X.shape,y.shape)
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X,y,test_size = .33, random_state = 0)
print(train_x.shape, test_x.shape)
headers = train_x.columns 



train_x.head()




# Feature Scaling

## We will be using standardscaler to transform

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



## transforming "train_x"

train_x = sc.fit_transform(train_x)

## transforming "test_x"

test_x = sc.transform(test_x)



## transforming "The testset"

test = sc.transform(test)
pd.DataFrame(train_x, columns=headers).head()
#why this?...

train.calculated_fare = train.calculated_fare.astype(float)
# import LogisticRegression model in python. 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error, accuracy_score



## call on the model object

logreg = LogisticRegression(solver='liblinear')



## fit the model with "train_x" and "train_y"

logreg.fit(train_x,train_y)



## Once the model is trained we want to find out how well the model is performing, so we test the model. 

## we use "test_x" portion of the data(this data was not used to fit the model) to predict model outcome. 

y_pred = logreg.predict(test_x)



## Once predicted we save that outcome in "y_pred" variable.

## Then we compare the predicted value( "y_pred") and actual value("test_y") to see how well our model is performing. 



print ("So, Our accuracy Score is: {}".format(round(accuracy_score(y_pred, test_y),4)))
from sklearn.metrics import roc_curve, auc

#plt.style.use('seaborn-pastel')

y_score = logreg.decision_function(test_x)



FPR, TPR, _ = roc_curve(test_y, y_score)

ROC_AUC = auc(FPR, TPR)

print (ROC_AUC)



plt.figure(figsize =[11,9])

plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)

plt.plot([0,1],[0,1], 'k--', linewidth = 4)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate', fontsize = 18)

plt.ylabel('True Positive Rate', fontsize = 18)

plt.title('ROC for Titanic survivors', fontsize= 18)

plt.show()
from sklearn.metrics import precision_recall_curve



y_score = logreg.decision_function(test_x)



precision, recall, _ = precision_recall_curve(test_y, y_score)

PR_AUC = auc(recall, precision)



plt.figure(figsize=[11,9])

plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)

plt.xlabel('Recall', fontsize=18)

plt.ylabel('Precision', fontsize=18)

plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)

plt.legend(loc="lower right")

plt.show()
## Using StratifiedShuffleSplit

## We can use KFold, StratifiedShuffleSplit, StratiriedKFold or ShuffleSplit, They are all close cousins. look at sklearn userguide for more info.   

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

## Using standard scale for the whole dataset.

X = sc.fit_transform(X)

accuracies = cross_val_score(LogisticRegression(), X,y, cv  = cv)

print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),5)))
from sklearn.model_selection import GridSearchCV, StratifiedKFold

## C_vals is the alpla value of lasso and ridge regression(as alpha increases the model complexity decreases,)

## remember effective alpha scores are 0<alpha<infinity 

C_vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,16.5,17,17.5,18]

## Choosing penalties(Lasso(l1) or Ridge(l2))

penalties = ['l1','l2']

## Choose a cross validation strategy. 

cv = StratifiedShuffleSplit(n_splits = 10, test_size = .25)



## setting param for param_grid in GridSearchCV. 

param = {'penalty': penalties, 'C': C_vals}



logreg = LogisticRegression(solver='liblinear')

## Calling on GridSearchCV object. 

grid = GridSearchCV(estimator=LogisticRegression(), 

                           param_grid = param,

                           scoring = 'accuracy',

                            n_jobs =-1,

                           cv = cv

                          )

## Fitting the model

grid.fit(X, y)
## Getting the best of everything. 

print (grid.best_score_)

print (grid.best_params_)

print(grid.best_estimator_)
### Using the best parameters from the grid-search.

logreg_grid = grid.best_estimator_

logreg_grid.score(X,y)
## Importing the model. 

from sklearn.neighbors import KNeighborsClassifier

## calling on the model oject. 

knn = KNeighborsClassifier(metric='minkowski', p=2)

## knn classifier works by doing euclidian distance 





## doing 10 fold staratified-shuffle-split cross validation 

cv = StratifiedShuffleSplit(n_splits=10, test_size=.25, random_state=2)



accuracies = cross_val_score(knn, X,y, cv = cv, scoring='accuracy')

print ("Cross-Validation accuracy scores:{}".format(accuracies))

print ("Mean Cross-Validation accuracy score: {}".format(round(accuracies.mean(),3)))
## Search for an optimal value of k for KNN.

k_range = range(1,31)

k_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X,y, cv = cv, scoring = 'accuracy')

    k_scores.append(scores.mean())

print("Accuracy scores are: {}\n".format(k_scores))

print ("Mean accuracy score: {}".format(np.mean(k_scores)))
from matplotlib import pyplot as plt

plt.plot(k_range, k_scores)
from sklearn.model_selection import GridSearchCV

## trying out multiple values for k

k_range = range(1,31)

## 

weights_options=['uniform','distance']

# 

param = {'n_neighbors':k_range, 'weights':weights_options}

## Using startifiedShufflesplit. 

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available processors. 

grid = GridSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1)

## Fitting the model. 

grid.fit(X,y)
print (grid.best_score_)

print (grid.best_params_)

print(grid.best_estimator_)
### Using the best parameters from the grid-search.

knn_grid= grid.best_estimator_

knn_grid.score(X,y)
from sklearn.model_selection import RandomizedSearchCV

## trying out multiple values for k

k_range = range(1,31)

## 

weights_options=['uniform','distance']

# 

param = {'n_neighbors':k_range, 'weights':weights_options}

## Using startifiedShufflesplit. 

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

# estimator = knn, param_grid = param, n_jobs = -1 to instruct scikit learn to use all available processors. 

## for RandomizedSearchCV, 

grid = RandomizedSearchCV(KNeighborsClassifier(), param,cv=cv,verbose = False, n_jobs=-1, n_iter=40)

## Fitting the model. 

grid.fit(X,y)
print (grid.best_score_)

print (grid.best_params_)

print(grid.best_estimator_)
### Using the best parameters from the grid-search.

knn_ran_grid = grid.best_estimator_

knn_ran_grid.score(X,y)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(X, y)

y_pred = gaussian.predict(test_x)

gaussian_accy = round(accuracy_score(y_pred, test_y), 3)

print(gaussian_accy)
from sklearn.svm import SVC

Cs = [0.001, 0.01, 0.1, 1,1.5,2,2.5,3,4,5, 10] ## penalty parameter C for the error term. 

gammas = [0.0001,0.001, 0.01, 0.1, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)

grid_search = GridSearchCV(SVC(kernel = 'rbf', probability=True), param_grid, cv=cv) ## 'rbf' stands for gaussian kernel

grid_search.fit(X,y)
print(grid_search.best_score_)

print(grid_search.best_params_)

print(grid_search.best_estimator_)
# using the best found hyper paremeters to get the score. 

svm_grid = grid_search.best_estimator_

svm_grid.score(X,y)
from sklearn.tree import DecisionTreeClassifier

max_depth = range(1,30)

max_feature = [21,22,23,24,25,26,28,29,30,'auto']

criterion=["entropy", "gini"]



param = {'max_depth':max_depth, 

         'max_features':max_feature, 

         'criterion': criterion}

grid = GridSearchCV(DecisionTreeClassifier(), 

                                param_grid = param, 

                                 verbose=False, 

                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),

                                n_jobs = -1)

grid.fit(X, y) 
print( grid.best_params_)

print (grid.best_score_)

print (grid.best_estimator_)
dectree_grid = grid.best_estimator_

## using the best found hyper paremeters to get the score. 

dectree_grid.score(X,y)
import graphviz



from sklearn import tree



dot_data = tree.export_graphviz(dectree_grid, out_file=None)



graph = graphviz.Source(dot_data)



graph.render("house")



graph
from sklearn.ensemble import BaggingClassifier

BaggingClassifier = BaggingClassifier()

BaggingClassifier.fit(X, y)

y_pred = BaggingClassifier.predict(test_x)

bagging_accy = round(accuracy_score(y_pred, test_y), 3)

print(bagging_accy)
from sklearn.ensemble import RandomForestClassifier

n_estimators = [90,95,100,105,110]

max_depth = range(1,30)

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)





parameters = {'n_estimators':n_estimators, 

         'max_depth':max_depth, 

        }

grid = GridSearchCV(RandomForestClassifier(),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

grid.fit(X,y) 
print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
rf_grid = grid.best_estimator_

rf_grid.score(X,y)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gradient = GradientBoostingClassifier()

gradient.fit(X, y)

y_pred = gradient.predict(test_x)

gradient_accy = round(accuracy_score(y_pred, test_y), 3)

print(gradient_accy)
from xgboost import XGBClassifier

XGBClassifier = XGBClassifier()

XGBClassifier.fit(X, y)

y_pred = XGBClassifier.predict(test_x)

XGBClassifier_accy = round(accuracy_score(y_pred, test_y), 3)

print(XGBClassifier_accy)
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier()

adaboost.fit(X, y)

y_pred = adaboost.predict(test_x)

adaboost_accy = round(accuracy_score(y_pred, test_y), 3)

print(adaboost_accy)
from sklearn.ensemble import ExtraTreesClassifier

ExtraTreesClassifier = ExtraTreesClassifier()

ExtraTreesClassifier.fit(X, y)

y_pred = ExtraTreesClassifier.predict(test_x)

extraTree_accy = round(accuracy_score(y_pred, test_y), 3)

print(extraTree_accy)
from sklearn.gaussian_process import GaussianProcessClassifier

GaussianProcessClassifier = GaussianProcessClassifier()

GaussianProcessClassifier.fit(X, y)

y_pred = GaussianProcessClassifier.predict(test_x)

gau_pro_accy = round(accuracy_score(y_pred, test_y), 3)

print(gau_pro_accy)
from sklearn.ensemble import VotingClassifier



voting_classifier = VotingClassifier(estimators=[

    ('logreg_grid', logreg_grid),

    ('svc', svm_grid),

    ('random_forest', rf_grid),

    ('gradient_boosting', gradient),

    ('decision_tree_grid',dectree_grid),

    ('knn_grid', knn_grid),

    ('XGB Classifier', XGBClassifier),

    ('BaggingClassifier', BaggingClassifier),

    ('ExtraTreesClassifier', ExtraTreesClassifier),

    ('gaussian',gaussian),

    ('gaussian process classifier', GaussianProcessClassifier)

    ], voting='soft')



voting_classifier = voting_classifier.fit(train_x,train_y)



y_pred = voting_classifier.predict(test_x)

voting_accy = round(accuracy_score(y_pred, test_y), 3)

print(voting_accy)
all_models = [voting_classifier

              ,knn_grid

              ,GaussianProcessClassifier, gaussian, ExtraTreesClassifier, BaggingClassifier

              , XGBClassifier,  dectree_grid, gradient, rf_grid, svm_grid, logreg_grid

             ]



c = {}

for i in all_models:

    a = i.predict(test_x)

    b = accuracy_score(a, test_y)

    c[i] = b
test_prediction = (max(c, key=c.get)).predict(test)

submission = pd.DataFrame({

        "PassengerId": passengerid,

        "Survived": test_prediction

    })



submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)



submission.to_csv("titanic1_submission.csv", index=False)