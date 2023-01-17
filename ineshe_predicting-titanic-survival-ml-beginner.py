#import

#pandas
import pandas as pd
from pandas import Series,DataFrame

#numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

import time
import random
import sklearn


#machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train = train_df.copy(deep = True)

train.head(5)
#conserve test passengerID in variable

test_passengerid = test_df.PassengerId

#drop unnecessary columns

train = train.drop(['PassengerId','Ticket'], axis = 1)

test_df = test_df.drop(['PassengerId','Ticket'], axis = 1)
print("The null value of each column in train data:\n ", train.isnull().sum())
print("--------------------------------")
print("The null value of each column in test data:\n ",test_df.isnull().sum())


  #Embarked:fill in with mode
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)
  
  #Fare
train['Fare'].fillna(train['Fare'].median(), inplace = True)
test_df['Fare'].fillna(test_df['Fare'].median(),inplace = True)

#age
train['Age'].fillna(train['Age'].median(), inplace = True)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)



train.drop('Cabin', axis = 1, inplace = True)
test_df.drop('Cabin',axis = 1, inplace = True)




#check the data again
train.info()
test_df.info()
pal = {'male':"green", 'female':"Pink"}
plt.subplots(figsize = (15,8))
ax = sns.barplot(x = "Sex", 
            y = "Survived", 
            data=train, 
            palette = pal,
            linewidth=2 )
plt.title("Survived Passenger Gender Distribution", fontsize = 25)
plt.ylabel("% of passenger survived", fontsize = 15)
plt.xlabel("Sex",fontsize = 15);
pal = {1:"seagreen", 0:"gray"}
sns.set(style="darkgrid")
plt.subplots(figsize = (15,8))
ax = sns.countplot(x = "Sex", 
                   hue="Survived",
                   data = train, 
                   linewidth=2, 
                   palette = pal
)

## Fixing title, xlabel and ylabel
plt.title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 25)
plt.xlabel("Sex", fontsize = 15);
plt.ylabel("# of Passenger Survived", fontsize = 15)

## Fixing xticks
#labels = ['Female', 'Male']
#plt.xticks(sorted(train.Sex.unique()), labels)

## Fixing legends
leg = ax.get_legend()
leg.set_title("Survived")
legs = leg.texts
legs[0].set_text("No")
legs[1].set_text("Yes")
plt.show()
plt.subplots(figsize = (15,10))
sns.barplot(x = "Pclass", 
            y = "Survived", 
            data=train, 
            linewidth=2)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
plt.xlabel("Socio-Economic class", fontsize = 15);
plt.ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper', 'Middle', 'Lower']
#val = sorted(train.Pclass.unique())
val = [0,1,2] ## this is just a temporary trick to get the label right. 
plt.xticks(val, labels);
# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
## I have included to different ways to code a plot below, choose the one that suites you. 
ax=sns.kdeplot(train.Pclass[train.Survived == 0] , 
               color='gray',
               shade=True,
               label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'] , 
               color='g',
               shade=True, 
               label='survived')
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Class", fontsize = 15)
## Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(train.Pclass.unique()), labels);
#Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')
plt.title('Fare Distribution Survived vs Non Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Fare", fontsize = 15)
# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Age Distribution - Survivors V.S. Non Survivors', fontsize = 25)
plt.xlabel("Age", fontsize = 15)
plt.ylabel('Frequency', fontsize = 15);
g = sns.FacetGrid(train,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",
                  palette = pal
                  )
g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend();
g.fig.suptitle("Survived by Embarked,Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)
g = sns.FacetGrid(train, size=5,hue="Survived", col ="Sex", margin_titles=True,
                palette=pal,)
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
g.fig.suptitle("Survived by Sex, Fare and Age", size = 25)
plt.subplots_adjust(top=0.85)
## dropping the three outliers where Fare is over $500 
train = train[train.Fare < 500]
## factor plot
sns.factorplot(x = "Parch", y = "Survived", data = train,kind = "point",size = 8)
plt.title("Factorplot of Parents/Children survived", fontsize = 25)
plt.subplots_adjust(top=0.85)
sns.factorplot(x =  "SibSp", y = "Survived", data = train,kind = "point",size = 8)
plt.title('Factorplot of Sibilings/Spouses survived', fontsize = 25)
plt.subplots_adjust(top=0.85)
train.describe()
# Overview(Survived vs non survied)
survived_summary = train.groupby("Survived")
survived_summary.mean().reset_index()
# Placing 0 for female and 
# 1 for male in the "Sex" column. 
train['Sex'] = train.Sex.apply(lambda x: 0 if x == 'female' else 1)
test_df['Sex'] = test_df.Sex.apply(lambda x: 0 if x == 'female' else 1)
train.Sex
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
male_mean = train[train['Sex'] == 'male'].Survived.mean()

female_mean = train[train['Sex'] == 'female'].Survived.mean()
print ("Male survival mean: " + str(male_mean))
print ("female survival mean: " + str(female_mean))

print ("The mean difference between male and female survival rate: " + str(female_mean - male_mean))
train.Sex.head()
# separating male and female dataframe. 
male = train[train['Sex'] == 'male']
female = train[train['Sex'] == 'female']

# getting 50 random sample for male and female. 
import random
male_sample = random.sample(list(male['Survived']),50)
female_sample = random.sample(list(female['Survived']),50)

# Taking a sample means of survival feature from male and female
male_sample_mean = np.mean(male_sample)
female_sample_mean = np.mean(female_sample)

# Print them out
print ("Male sample mean: " + str(male_sample_mean))
print ("Female sample mean: " + str(female_sample_mean))
print ("Difference between male and female sample mean: " + str(female_sample_mean - male_sample_mean))
import scipy.stats as stats

print (stats.ttest_ind(male_sample, female_sample))
print ("This is the p-value when we break it into standard form: " + format(stats.ttest_ind(male_sample, female_sample).pvalue, '.32f'))
## Family_size seems like a good feature to create
train['family_size'] = train.SibSp + train.Parch+1
test_df['family_size'] = test_df.SibSp + test_df.Parch+1


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
test_df['family_group'] = test_df['family_size'].map(family_group)
train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
test_df['is_alone'] = [1 if i<2 else 0 for i in test_df.family_size]

train['Title'] = train['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
test_df['Title'] = test_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (train['Title'].value_counts() < stat_min)

train['Title'] = train['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



title_names_test = (test_df['Title'].value_counts() < stat_min)

test_df['Title'] = test_df['Title'].apply(lambda x: 'Misc' if title_names_test.loc[x] == True else x)
print(train['Title'].value_counts())
train['FareBin'] = pd.qcut(train['Fare'], 4)
test_df['FareBin'] = pd.qcut(test_df['Fare'], 4)
train['AgeBin'] = pd.cut(train['Age'].astype(int), 5)
test_df['AgeBin'] = pd.cut(test_df['Age'].astype(int), 5)
train.head()
train = pd.get_dummies(train, columns=['Sex','Title',"Pclass",'Embarked', 'family_group', 'FareBin','AgeBin'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Sex','Title',"Pclass",'Embarked', 'family_group', 'FareBin','AgeBin'], drop_first=True)

train.drop([ 'family_size','Name', 'Fare'], axis=1, inplace=True)
test_df.drop(['Name','family_size',"Fare"], axis=1, inplace=True)
X = train.drop(['Survived'], axis = 1)
Y = train["Survived"]
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size = .33, random_state = 0)
# Feature Scaling
## We will be using standardscaler to transform
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

## transforming "train_x"
train_x = sc.fit_transform(train_x)
## transforming "train_x"
test_x = sc.transform(test_x)

## transforming "The testset"
test_df = sc.transform(test_df)
train.head()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score

## call on the model object
logreg = LogisticRegression()

## fit the model with "train_x" and "train_y"
logreg.fit(train_x,train_y)

## Once the model is trained we want to find out how well the model is performing, so we test the model. 
## we use "test_x" portion of the data(this data was not used to fit the model) to predict model outcome. 
y_pred = logreg.predict(test_x)

## Once predicted we save that outcome in "y_pred" variable.
## Then we compare the predicted value( "y_pred") and actual value("test_y") to see how well our model is performing. 

print ("Accuracy Score: {}".format(accuracy_score(y_pred, test_y)))
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = logreg, X = train_x, y = train_y, cv = 10, n_jobs = -1)
logreg_accy = accuracies.mean()
print (round((logreg_accy),3))
#note: this is an alternative to train_test_split
from sklearn import model_selection
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
cv_results = model_selection.cross_validate(logreg, X,Y, cv  = cv_split)
print (cv_results)
cv_results['train_score'].mean()
from sklearn.model_selection import GridSearchCV
C_vals = [0.099,0.1,0.2,0.5,12,13,14,15,16,16.5,17,17.5,18]
penalties = ['l1','l2']

param = {'penalty': penalties, 
         'C': C_vals 
        }
grid_search = GridSearchCV(estimator=logreg, 
                           param_grid = param,
                           scoring = 'accuracy', 
                           cv = 10
                          )
grid_search = grid_search.fit(train_x, train_y)
print (grid_search.best_params_)
print (grid_search.best_score_)
logreg_grid = grid_search.best_estimator_
logreg_accy = logreg_grid.score(test_x, test_y)
logreg_accy
from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(test_y, y_pred, labels=logreg_grid.classes_))
print (confusion_matrix(y_pred, test_y))
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')
y_score = logreg_grid.decision_function(test_x)

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

plt.style.use('seaborn-pastel')

y_score = logreg_grid.decision_function(test_x)

precision, recall, _ = precision_recall_curve(test_y, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)
plt.legend(loc="lower right")
plt.show()
from sklearn.neighbors import KNeighborsClassifier
## choosing the best n_neighbors
nn_scores = []
best_prediction = [-1,-1]
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance', metric='minkowski', p =2)
    knn.fit(train_x,train_y)
    score = accuracy_score(test_y, knn.predict(test_x))
    #print i, score
    if score > best_prediction[1]:
        best_prediction = [i, score]
    nn_scores.append(score)
    
print (best_prediction)
plt.plot(range(1,100),nn_scores)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
#n_neighbors: specifies how many neighbors will vote on the class
#weights: uniform weights indicate that all neighbors have the same weight while "distance" indicates
        # that points closest to the 
#metric and p: when distance is minkowski (the default) and p == 2 (the default), this is equivalent to the euclidean distance metric
knn.fit(train_x, train_y)
y_pred = knn.predict(test_x)
knn_accy = round(accuracy_score(test_y, y_pred), 3)
print (knn_accy)
from sklearn.model_selection import StratifiedKFold

n_neighbors=[1,2,3,4,5,6,7,8,9,10]
weights=['uniform','distance']
param = {'n_neighbors':n_neighbors, 
         'weights':weights}
grid2 = GridSearchCV(knn, 
                     param,
                     verbose=False, 
                     cv=StratifiedKFold(n_splits=5, random_state=15, shuffle=True)
                    )
grid2.fit(train_x, train_y)
print (grid2.best_params_)
print (grid2.best_score_)
## using grid search to fit the best model.
knn_grid = grid2.best_estimator_
##accuracy_score =(knn_grid.predict(x_test), y_test)
knn_accy = knn_grid.score(test_x, test_y)
knn_accy
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(train_x, train_y)
y_pred = gaussian.predict(test_x)
gaussian_accy = round(accuracy_score(y_pred, test_y), 3)
print(gaussian_accy)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC(kernel = 'rbf', probability=True, random_state = 1, C = 3)
svc.fit(train_x, train_y)
y_pred = svc.predict(test_x)
svc_accy = round(accuracy_score(y_pred, test_y), 3)
print(svc_accy)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier()
dectree.fit(train_x, train_y)
y_pred = dectree.predict(test_x)
dectree_accy = round(accuracy_score(y_pred, test_y), 3)
print(dectree_accy)
from sklearn.ensemble import BaggingClassifier
BaggingClassifier = BaggingClassifier()
BaggingClassifier.fit(train_x, train_y)
y_pred = BaggingClassifier.predict(test_x)
bagging_accy = round(accuracy_score(y_pred, test_y), 3)
print(bagging_accy)
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=100,max_depth=9,min_samples_split=6, min_samples_leaf=4)
#randomforest = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
randomforest.fit(train_x, train_y)
y_pred = randomforest.predict(test_x)
random_accy = round(accuracy_score(y_pred, test_y), 3)
print (random_accy)
n_estimators = [100,120]
max_depth = range(1,30)



parameters = {'n_estimators':n_estimators, 
         'max_depth':max_depth, 
        }
randomforest_grid = GridSearchCV(randomforest,
                                 param_grid=parameters,
                                 cv=StratifiedKFold(n_splits=20, random_state=15, shuffle=True),
                                 n_jobs = -1
                                )
randomforest_grid.fit(train_x, train_y) 
randomforest_grid.score(test_x, test_y)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier()
gradient.fit(train_x, train_y)
y_pred = gradient.predict(test_x)
gradient_accy = round(accuracy_score(y_pred, test_y), 3)
print(gradient_accy)
from xgboost import XGBClassifier
XGBClassifier = XGBClassifier()
XGBClassifier.fit(train_x, train_y)
y_pred = XGBClassifier.predict(test_x)
XGBClassifier_accy = round(accuracy_score(y_pred, test_y), 3)
print(XGBClassifier_accy)
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier()
adaboost.fit(train_x, train_y)
y_pred = adaboost.predict(test_x)
adaboost_accy = round(accuracy_score(y_pred, test_y), 3)
print(adaboost_accy)
from sklearn.ensemble import ExtraTreesClassifier
ExtraTreesClassifier = ExtraTreesClassifier()
ExtraTreesClassifier.fit(train_x, train_y)
y_pred = ExtraTreesClassifier.predict(test_x)
extraTree_accy = round(accuracy_score(y_pred, test_y), 3)
print(extraTree_accy)
from sklearn.gaussian_process import GaussianProcessClassifier
GaussianProcessClassifier = GaussianProcessClassifier()
GaussianProcessClassifier.fit(train_x, train_y)
y_pred = GaussianProcessClassifier.predict(test_x)
gau_pro_accy = round(accuracy_score(y_pred, test_y), 3)
print(gau_pro_accy)
all_models = [GaussianProcessClassifier, gaussian, ExtraTreesClassifier, BaggingClassifier, XGBClassifier,knn_grid, knn,  dectree, gradient, randomforest, svc, logreg, logreg_grid  ]

c = {}
for i in all_models:
    a = i.predict(test_x)
    b = accuracy_score(a, test_y)
    c[i] = b
MLA_columns = ['MLA Name','Accuracy_score']
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
for alg in all_models:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'Accuracy_score'] = c[alg]
    row_index+=1


  
MLA_compare.sort_values(by = ['Accuracy_score'], ascending = False, inplace = True)
MLA_compare
import matplotlib.pyplot as plt

#barplot 
sns.barplot(x='Accuracy_score', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
test_prediction = (max(c, key=c.get)).predict(test_df)
submission = pd.DataFrame({
        "PassengerId": test_passengerid,
        "Survived": test_prediction
    })

submission.PassengerId = submission.PassengerId.astype(int)
submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic1_submission.csv", index=False)
#calculate the predicted survival rate

survival = submission[submission.Survived == 0]

survival_count =len(survival.index)

passenger_count = len(submission.index)

survival_rate = survival_count/passenger_count

survival_rate