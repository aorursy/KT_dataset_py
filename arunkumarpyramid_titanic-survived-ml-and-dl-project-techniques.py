# Importing the dataset

import pandas as pd

train_dataset = pd.read_csv("../input/titanic/train.csv")

print(train_dataset.head())
# Load dataset for test set

test_dataset = pd.read_csv("../input/titanic/test.csv")

print(test_dataset.head())
# Displaying Shape of training set

train_dataset.shape
# Displaying Shape of test dataset

test_dataset.shape
# Displaying datatype of each attributes for train dataset

train_dataset.dtypes
# Displaying datatype of each attributes for test dataset

test_dataset.dtypes
# Displaying descritive statistical analysis for train dataset



train_dataset.describe()
# Displaying descritive statistical analysis for test dataset

test_dataset.describe()
# Histogram plot statistical visualisation analysis for train dataset

import matplotlib.pyplot as plt

train_dataset.hist(bins=20,figsize=(12,12),layout=(3,3))

plt.show()
# Density plot statistical visualisation analysis for train datasets



train_dataset.plot(kind='density',figsize=(10,10),layout=(3,3),subplots=True,sharex=False,sharey=False)
# Histogram plot statstical visualisation analysis for test dataset



test_dataset.hist(bins=20,figsize=(12,12),layout=(3,3))

plt.show()
# Density plot Stastistical Visualisation analysis for test dataset

test_dataset.plot(kind='density',figsize=(10,10),layout=(3,3),sharex=False,sharey=False,subplots=True)

plt.show()
# extracting insights of "Pclass" attribute using statistical visualisation distribution.



# kde--- kind distribution wave

# color--- color of graphical plot

# norm_hist --- True or false makes normal distribution graph should be there or not.





import seaborn as sb

sb.distplot(train_dataset['Pclass'],bins=5,hist=True,norm_hist=True,color='green',vertical=False,kde=True,label='Pclass')

plt.legend()

plt.show()
# Survived output attribute( its optional to visualise )



sb.distplot(train_dataset['Survived'],color='red',bins=5,hist=True,norm_hist=True,vertical=False,kde=True,label='Survied')

plt.legend()

plt.show()
# Age attribute



sb.distplot(train_dataset['Age'],bins=5,hist=True,norm_hist=True,vertical=False,label='Age')

plt.legend()
# Fare Attribute



sb.distplot(train_dataset['Fare'],bins=35,vertical=False,hist=True,norm_hist=True,label='Fare')

plt.legend()
# SibSp attribute



sb.distplot(train_dataset['SibSp'],label='SibSp', bins=10, hist=True, vertical=False,norm_hist=True)

plt.legend()
# Displaying the null values list for each attribute



# train dataset

train_dataset.info()
# test dataset



test_dataset.info()
# Displaying the sum of null value count in each attribute from train dataset



train_dataset.isna().sum()
train_dataset_null=(train_dataset.isna().sum()/len(train_dataset))*100.0



# Now we are droping all values ratio is equal zero in all_data_null variable

all_data_null= train_dataset_null.drop(train_dataset_null[train_dataset_null==0].index).sort_values(ascending=False)



print(all_data_null)
sb.barplot(x=all_data_null.index,y=all_data_null)

plt.xlabel('Feature')

plt.xticks(rotation='90')

plt.xlabel('percentage of missing ratio')

plt.title('percentage of missing ratio on barplot')
# Displaying the sum of null value count in each attribute from test dataset



test_dataset.isna().sum()
# Removing the Cabin column in both train and test dataset



train_dataset=train_dataset.drop('Cabin',axis=1)
train_dataset=train_dataset.fillna(train_dataset.mean())
train_dataset.isna().sum()


train_dataset=train_dataset.dropna(axis=0)
train_dataset.isna().sum()
test_null_ratio=(test_dataset.isna().sum()/len(test_dataset))*100.0



# Now we are droping all values ratio is equal zero in all_data_null variable



all_null_value_ratio=test_null_ratio.drop(test_null_ratio[test_null_ratio==0].index).sort_values(ascending=False)



print(all_null_value_ratio)
# Missing value ratio using visualising the barplot

sb.barplot(x=all_null_value_ratio.index,y=all_null_value_ratio)

plt.xlabel('Percentage')

plt.xticks(rotation='90')

plt.ylabel('missing value ratio')

plt.title('visualising the Missing value ratio on barplot')
# Removing the null values from test dataset

test_dataset=test_dataset.drop('Cabin',axis=1)
test_dataset=test_dataset.fillna(test_dataset.mean())
# Checking the null values removed or not from test dataset



test_dataset.isna().sum()
train_dataset=train_dataset.drop('PassengerId',axis=1)

train_dataset=train_dataset.drop('Name',axis=1)

train_dataset.head()
# Making duplicate of test_dataset

test_dataset1=test_dataset



test_dataset=test_dataset.drop('PassengerId',axis=1)

test_dataset=test_dataset.drop('Name',axis=1)

test_dataset.head()
train_dataset['Sex'].unique()
train_dataset['Sex']=train_dataset['Sex'].map({'female':1,'male':0})

train_dataset.head()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

train_dataset['Ticket']=label.fit_transform(train_dataset['Ticket'])

train_dataset['Embarked']=label.fit_transform(train_dataset['Embarked'])

train_dataset.head()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

test_dataset['Ticket']=label.fit_transform(test_dataset['Ticket'])

test_dataset['Embarked']=label.fit_transform(test_dataset['Embarked'])

test_dataset.head()
test_dataset['Sex']=label.fit_transform(test_dataset['Sex'])

test_dataset.head()
# outlier checking between Survived and Pclass attributes....



plt.scatter(train_dataset['Pclass'],train_dataset['Survived'])

plt.xlabel("Pclass")

plt.ylabel("Survived")

plt.title("Outlier between Pclass and Survived")

plt.show()
# outlier checking between Survived and Sex attributes....



plt.scatter(train_dataset['Sex'],train_dataset['Survived'])

plt.xlabel("Sex")

plt.ylabel("Survived")

plt.title("Outlier between Sex and Survived")

plt.show()
# outlier checking between Survived and Age attributes....



plt.scatter(train_dataset['Age'],train_dataset['Survived'])

plt.xlabel('Age')

plt.ylabel('Survived')

plt.title('Outlier between Survived and Age')

plt.show()
# removing outlier between age and survived...



# Deleting outliers



train_dataset=train_dataset.drop(train_dataset[(train_dataset['Age']>70)].index)



plt.scatter(train_dataset['Age'],train_dataset['Survived'])

plt.xlabel('Age')

plt.ylabel('Survived')

plt.show()
# Outlier checking between SibSp and Survived



plt.scatter(train_dataset['SibSp'],train_dataset['Survived'])

plt.xlabel('SibSp')

plt.ylabel('Survived')

plt.title('Outlier between SibSp and Survived')

plt.show()
# We found outlier between SibSp and Survived, We are going to remove all values after 5 sbisp



train_dataset=train_dataset.drop(train_dataset[(train_dataset['SibSp'])>5].index)



plt.scatter(train_dataset['SibSp'],train_dataset['Survived'])

plt.xlabel('SibSp')

plt.ylabel('Survived')

plt.title('After removed outlier between SibSp and Survived')

plt.show()
# Outlier check between Parch and Survived



plt.scatter(train_dataset['Parch'],train_dataset['Survived'])

plt.xlabel('Parch')

plt.ylabel('Survived')

plt.title('Outlier between Parch and Survived')

plt.show()
# Checking outlier between Ticket and survived



plt.scatter(train_dataset['Ticket'],train_dataset['Survived'])

plt.xlabel('Ticket')

plt.ylabel('Survived')

plt.title('Outlier between Ticket and Survived')

plt.show()
# Checking outliers between Fare and Survived



plt.scatter(train_dataset['Fare'],train_dataset['Survived'])

plt.xlabel('Fare')

plt.ylabel('Survived')

plt.title('Outliers between Fare snd Survived')

plt.show()
# Removing the outliers between Fare and Survived



train_dataset=train_dataset.drop(train_dataset[train_dataset['Fare']>300].index)



plt.scatter(train_dataset['Fare'],train_dataset['Survived'])

plt.xlabel('Fare')

plt.ylabel('Survived')

plt.title('Outliers between Fare and Survived')

plt.show()
# Checking Outliers between Embarked and Survived



plt.scatter(train_dataset['Embarked'],train_dataset['Survived'])

plt.xlabel('Embarked')

plt.ylabel('Survived')

plt.title('Outlier Between Embarked and Survived')

plt.show()
servived_data= train_dataset['Survived'].value_counts()[1]/len(train_dataset)*100.0

non_servived_data=train_dataset['Survived'].value_counts()[0]/len(train_dataset)*100.0



print("servived dataset",servived_data)

print("non_servived dataset",non_servived_data)

print("servived data rows ",train_dataset['Survived'].value_counts()[1])

print("non servived data rows ",train_dataset['Survived'].value_counts()[0])
# Visualise the class distribution.



sb.countplot('Survived',data=train_dataset)

plt.title('class Distribution \n(0 : Non Servived || 1: Servived)',fontsize=12)
# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.



#train_dataset=train_dataset.sample(frac=1)



servived_data=train_dataset.loc[train_dataset['Survived'] == 1]

non_servived_data=train_dataset.loc[train_dataset['Survived'] == 0][:336]



nomal_distribution_dataset=pd.concat([servived_data,non_servived_data])



# Shuffle dataframe rows

balanced_train_dataset=nomal_distribution_dataset.sample(frac=1,random_state=5)



balanced_train_dataset.head()
print("Distribution of the classes in the subsample dataset")

print(balanced_train_dataset['Survived'].value_counts()/len(train_dataset))



sb.countplot('Survived',data=balanced_train_dataset)

plt.title("Equally Distribution classes",fontsize=12)

plt.show()
'''apply SelectKBest class to extract top 10 best features... 

if you have more than 10 features it will show top 10 and if you have less than 10 features then,

we have use k=all or else it will through an error (ValueError: k should be >=0, <= n_features = 8; got 10. Use k='all' to return all features), 

we still can keep 10 as max features'''



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



x=balanced_train_dataset.iloc[:,1:]

y=balanced_train_dataset.iloc[:,0:1]



# Applying SelectKBest class to extract

# k-- No of features we want to display or top features we want to visualise

bestfeature=SelectKBest(score_func=chi2,k=8)

fit_selectkbest=bestfeature.fit(x,y)



dataframe_scores=pd.DataFrame(fit_selectkbest.scores_)

dataframe_columns=pd.DataFrame(x.columns)



# concat two data frame for better visualization

feature_score=pd.concat([dataframe_columns,dataframe_scores],axis=1)



# Naming the dataframe columns

feature_score.columns=['Specs','Score']



# print top 10 best features

print(feature_score.nlargest(8,'Score'))
# Feature importance using extratreesclassifier



from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier()

model.fit(x,y)



# user inbuild class feature_importances of tree based classifier

print(model.feature_importances_)



# plotting the graph of feature importances for better visualization

importance=pd.Series(model.feature_importances_,index=x.columns)

importance.nlargest(8).plot(kind='barh')

plt.show()
# correlation matrix is used to identify the relation percentage between each other...



correlation_matrix=balanced_train_dataset.corr()

sb.heatmap(correlation_matrix,square=True)
#1.

# correlation coefficient value for survived output attribute by dataframe technique

import numpy as np

k=10 # selecting top 10 correlation matrix to visualise

columns=correlation_matrix.nlargest(k,'Survived')['Survived'].index



correlation_coefficient=np.corrcoef(balanced_train_dataset[columns].values.T)

sb.set(font_scale=1.25)



Zoomin_heatmap=sb.heatmap(correlation_coefficient,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=columns.values,

                         xticklabels=columns.values)

plt.show()
#2.

# Correlatio coefficient value for survived attribute by after convert from dataframe to array of matrix technique



#get correlations of each features in dataset

feature_maps=balanced_train_dataset.corr()

top_corr_features=feature_maps.index

plt.figure(figsize=(15,15))



# plot for heatmap

g=sb.heatmap(balanced_train_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# Correlation input attribute relation with output attribute



correlation=balanced_train_dataset.corr()['Survived']

correlation.abs().sort_values(ascending=False)
# Removing less correlated values Age and SibSp



columns=['Age','SibSp']

balanced_train_dataset=balanced_train_dataset.drop(columns,axis=1)

balanced_train_dataset.head()
# Removing less correlated values Age and SibSp



columns=['Age','SibSp']

test_dataset=test_dataset.drop(columns,axis=1)

test_dataset.head()
# 1. first technique

array_data=balanced_train_dataset.values

x1=array_data[:,1:]

y1=array_data[:,0:1]

print(x1[:5,])

print(y1[:5,])
# 2. Second technique is used iloc funcation to slice the dataframe from certian range



x2=balanced_train_dataset.iloc[:,1:].values

y2=balanced_train_dataset.iloc[:,0:1].values

print(x2[:5,:])

print(y2[:5,])
# Importing classifier algorithms



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

neighbors=4
# Keep all model in one pipeline



models=[]

models.append(('Logistic Regression',LogisticRegression()))

models.append(('KNN',KNeighborsClassifier(n_neighbors=neighbors)))

models.append(('SVC',SVC()))

models.append(('Naive Bayes',GaussianNB()))
# Evaluate each model

import warnings; 

warnings.simplefilter('ignore')

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

classifier_names=[]

classifier_predictions=[]

classifier_scoring='accuracy'



for name,model in models:

    fold=KFold(n_splits=10,random_state=5)

    result=cross_val_score(model,x2,y2,cv=fold,scoring=classifier_scoring)

    classifier_predictions.append(result)

    classifier_names.append(name)

    msg="%s: %f and %f"%(name,result.mean(),result.std())

    print(msg)
# plotting the traning accuracy using boxplot



fig=plt.figure()

plt.suptitle('Compare algorithm accuracies')

plt.boxplot(classifier_predictions)

plt.show()
# creating pipeline with feature scaling followed by algorithm



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import Normalizer
# Algorithms



scaled_models=[]

scaled_models.append(("Scaled Logisitic Regression",Pipeline([('normalizer',Normalizer()),

                                                              ('scaled Logistic Regression',LogisticRegression())])))

scaled_models.append(('Scaled KNN',Pipeline([('normalizer',Normalizer()),

                                             ('scaled Knn',KNeighborsClassifier(n_neighbors=neighbors))])))

scaled_models.append(('Scaled Naive Bayes',Pipeline([('Normalizer',Normalizer()),

                                                     ('Scaled Naive bayes',GaussianNB())])))

scaled_models.append(('scaled SVC',Pipeline([('normalizer',Normalizer()),

                                             ('Scaled SVC',SVC())])))
# Evaluate each model



scaled_names=[]

scaled_predictions=[]

for name,model in scaled_models:

    fold=KFold(n_splits=10,random_state=5)

    result=cross_val_score(model,x2,y2,cv=fold,scoring=classifier_scoring)

    scaled_predictions.append(result)

    scaled_names.append(name)

    msg="Algorithm : %s, mean: %f, Std_dev:%f"%(name,result.mean(),result.std())

    print(msg)
fig=plt.figure()

plt.suptitle("Scaled Algorithm Accuracies")

plt.boxplot(scaled_predictions)

plt.show()
# Navie Bayes doesn't have parameters to tune, So accuracy remains same.

GaussianNB().get_params().keys()
# Regularization hyperparameter tunning for KNN.

KNeighborsClassifier().get_params().keys()
# Regularization Hyperparameter Tunning



from sklearn.model_selection import GridSearchCV

scaled_xtrain=Normalizer().fit_transform(x2)
# Hyper parameters for KNN algorithm

param_grid=dict(n_neighbors=[2,3,4,5,6,7,8,9,10],weights=['uniform', 'distance'],p=[1,2])



# GridSearchCV to tune and identify proper parameter.

model=KNeighborsClassifier()

fold=KFold(n_splits=10,random_state=5)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=classifier_scoring,cv=fold,n_jobs=-1)

grid_result=grid.fit(scaled_xtrain,y2)



print("best parameters: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Importing the ensemble boosting and bagging algorithms



# Bagging 

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier



# Boosting

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

'''import xgboost as xgb''' # Beacuse i didnt installed XGBoost in my system i will update once i installed
# holding feature scaling and algorithm as a pipeline technqiue



estimators=10

ensemble_models=[]

ensemble_models.append(('Scaled Random Forest',Pipeline([('Scaling',Normalizer()),

                                                         ('Random Forest',RandomForestClassifier(n_estimators=estimators))])))

ensemble_models.append(('Scaled Decision Tree',Pipeline([('Scaling',Normalizer()),

                                                         ('Decision Tree',DecisionTreeClassifier())])))

ensemble_models.append(('Scaled Extra Tree',Pipeline([('SCaling',Normalizer()),

                                                      ('Extra Tree',ExtraTreesClassifier(n_estimators=estimators))])))

ensemble_models.append(('Scaled Ada Boost',Pipeline([('Scaling',Normalizer()),

                                                     ('Ada Boost',AdaBoostClassifier())])))

ensemble_models.append(('Scaled Gradient Boost',Pipeline([('Scaling',Normalizer()),

                                                          ('Gradient Boosting',GradientBoostingClassifier())])))
# Evaluating each ensemble models



ensemble_names=[]

ensemble_predictions=[]



for name,model in ensemble_models:

    fold=KFold(n_splits=10,random_state=5)

    result=cross_val_score(model,x2,y2,cv=fold,scoring=classifier_scoring)

    ensemble_predictions.append(result)

    ensemble_names.append(name)

    msg= " algorithm: %s , mean accuracy: %f and Std_dev: %f"%(name,result.mean(),result.std())

    print(msg)
# Visualize the result

fig=plt.figure()

plt.suptitle('Ensemble Algorithms')

plt.boxplot(ensemble_predictions)

plt.show()
# Regularization hyper parameter tunning for Random Forest using GridSearchCV

scaled_xtrain=Normalizer().fit_transform(x2)

model=RandomForestClassifier()

fold=KFold(n_splits=10,random_state=5)



param_grid=dict(n_estimators=[2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,100],criterion=['gini', 'entropy'])

grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=fold,scoring=classifier_scoring,n_jobs=-1)



grid_result=grid.fit(scaled_xtrain,y2)

print("Best:%f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Regularization hyper parameter tunning for Random Forest using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

param_grid=dict(n_estimators=[2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,100],criterion=['gini', 'entropy'])

grid=RandomizedSearchCV(estimator=model,param_distributions=param_grid,cv=fold,scoring=classifier_scoring,n_jobs=-1)



grid_result=grid.fit(scaled_xtrain,y2)

print("Best:%f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Regularization hyper parameter tunning for Extra Tree using GridSearchCV

scaled_xtrain=Normalizer().fit_transform(x2)

model=ExtraTreesClassifier()

fold=KFold(n_splits=10,random_state=5)



param_grid=dict(n_estimators=[2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,100],criterion=['gini', 'entropy'])

grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=fold,scoring=classifier_scoring,n_jobs=-1)



grid_result=grid.fit(scaled_xtrain,y2)

print("Best:%f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Regularization hyper parameter tunning for Extra Tree using RandomizedSearchCV

param_grid=dict(n_estimators=[2,4,6,8,10,12,14,16,18,20,25,30,35,40,45,50,100],criterion=['gini', 'entropy'])

grid=RandomizedSearchCV(estimator=model,param_distributions=param_grid,cv=fold,scoring=classifier_scoring,n_jobs=-1)



grid_result=grid.fit(scaled_xtrain,y2)

print("Best:%f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Regualarization hyper parameter tunning for Ada Boost using GridSearchCV

scaled_xtrain=Normalizer().fit_transform(x2)

model=AdaBoostClassifier()

fold=KFold(n_splits=10,random_state=5)



param_grid=dict(n_estimators=[10,15,20,25,30,35,40,45,50,100,150,200,250,300],learning_rate=[0.01,0.1,0.2,0.3,0.4,0.5,0.6])

grid=GridSearchCV(estimator=model,cv=fold,param_grid=param_grid,n_jobs=-1,scoring=classifier_scoring)



grid_result=grid.fit(scaled_xtrain,y2)

print("Best %f using %s"%(grid_result.best_score_,grid_result.best_params_))
# Regularization hyper parameter tunning for Ada Boost using RandomizedSearchCV

param_grid=dict(n_estimators=[10,15,20,25,30,35,40,45,50,100,150,200,250,300],learning_rate=[0.01,0.1,0.2,0.3,0.4,0.5,0.6])

grid=RandomizedSearchCV(estimator=model,param_distributions=param_grid,cv=fold,scoring=classifier_scoring,n_jobs=-1)



grid_result=grid.fit(scaled_xtrain,y2)

print("Best:%f using %s "%(grid_result.best_score_,grid_result.best_params_))
# Regularisation hyper prameter tunning for Gradient boosting using GridSearchCV



model=GradientBoostingClassifier()

fold=KFold(n_splits=10,random_state=5)



param_grid=dict(learning_rate=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7],

            n_estimators=[10,20,30,50,100,150,200,250])

grid=GridSearchCV(estimator=model,cv=fold,scoring=classifier_scoring,param_grid=param_grid,n_jobs=-1)



grid_result=grid.fit(scaled_xtrain,y2)

print("Best : %f using %s"%(grid_result.best_score_,grid_result.best_params_))
#Regularisation hyper prameter tunning for Gradient boosting using RandomizedSearchCV

param_grid=dict(learning_rate=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7],

            n_estimators=[10,20,30,50,100,150,200,250])

grid=RandomizedSearchCV(estimator=model,param_distributions=param_grid,cv=fold,scoring=classifier_scoring,n_jobs=-1)



grid_result=grid.fit(scaled_xtrain,y2)

print("Best:%f using %s "%(grid_result.best_score_,grid_result.best_params_))
# importing libraries



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
# Rescaling training and test set

scaled=Normalizer().fit(x2)

scaled_xtrain=scaled.transform(x2)

scaled_xtest=scaled.transform(test_dataset)
# Algorithm fitting

model=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1)

model_fit=model.fit(scaled_xtrain,y2)
# Predicting Algorithm 

y_pred=model_fit.predict(scaled_xtest)

print(y_pred)
# output

output=pd.DataFrame({'PassengerId':test_dataset1.PassengerId,'Survived':y_pred})

output.to_csv("my_submission.csv",index=False)

print("Submission sucessfully")