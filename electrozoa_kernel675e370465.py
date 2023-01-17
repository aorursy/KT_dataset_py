



#1.1 Importing the required packages.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve

from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#from sklearn.cross_validation import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve

from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



import warnings

warnings.filterwarnings('ignore')

#1.2 Reading the required files

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()

#Looking at the data types available

print (train_df.info(),"\n\n\n")

print (test_df.info())

print (train_df.describe(),'\n\n', test_df.describe())

#2.1 Data Manipulation

#Since it can be observed that there are lots of Names which have a surname before them, hence extracting them



my_data=[train_df,test_df]

for i in my_data:

    i['Title'] = i.Name.str.extract(' ([A-Za-z]+)\.', expand=False)    

    

train_df.Title.value_counts()

#Now replacing the extra titles with male and female titles for better clarity

for i in my_data:

    i['Title'] = i['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    i['Title'] = i['Title'].replace('Mlle', 'Miss')

    i['Title'] = i['Title'].replace('Ms', 'Miss')

    i['Title'] = i['Title'].replace('Mme', 'Mrs')

    i['Title'] = pd.factorize(i['Title'])[0]

    



pd.crosstab(train_df['Title'], train_df['Sex'])

pd.crosstab(train_df['Title'], train_df['Sex'])

#'Passenger id, Name, Ticket and Cabin are nominal variables, and hence we decide to drop them'

drop_list=['PassengerId','Name','Ticket','Cabin']

train_new=train_df.drop(drop_list, axis=1)

test_new=test_df.drop(drop_list, axis=1)

new_set = [train_new,test_new]



train_new.shape, test_new.shape

#Checking the missing values in our dataset

pd.DataFrame(train_new.isnull().sum())

#2.2 Replacing the missing values

#Creating a function which replaces the missing age of that particular person with mean age of his/her title group

def age_imputer(df):

    for i in range(1,4):

        mean_age=df[df["Title"]==i]["Age"].mean()

        df["Age"]=df["Age"].fillna(mean_age)

        return df



train_new = age_imputer(train_new)

# Looking at the abscence of embarked cases

train_new[train_new['Embarked'].isnull()]

train_new.Embarked.value_counts()

#Since S is the maximum repeatance, hence we are replacing it with S for missing values

train_new["Embarked"] = train_new["Embarked"].fillna('S')

test_new.isnull().sum()

test_new[test_new['Age'].isnull()].head()

#Replacing the age in test dataset using our function

test_new = age_imputer(test_new)

test_new[test_new['Fare'].isnull()]

#Replacing the Missing fare with median value for that particular case

test_new.Fare = test_new.Fare.fillna(test_new[(test_new["Pclass"]==3) & (test_new["Embarked"]=="S")]["Fare"].median())

## Checking for any missing value in train

train_new.isnull().any()

## Checking for any missing value in Test

test_new.isnull().any()

#Creating flags for age in our dataset



for i in new_set:

    i.loc[i["Age"] <= 9, "Age"] = 0

    i.loc[(i["Age"] > 9) & (i["Age"] <= 19), "Age"] = 1

    i.loc[(i["Age"] > 19) & (i["Age"] <= 29), "Age"] = 2

    i.loc[(i["Age"] > 29) & (i["Age"] <= 39), "Age"] = 3

    i.loc[(i["Age"] > 29) & (i["Age"] <= 39), "Age"] = 3

    i.loc[i["Age"] > 39, "Age"] = 4

pd.qcut(train_new["Fare"], 8).value_counts()

#Creating flags for fare in our dataset



for i in new_set:

    i.loc[i["Fare"] <= 7.75, "Fare"] = 0

    i.loc[(i["Fare"] > 7.75) & (i["Fare"] <= 7.91), "Fare"] = 1

    i.loc[(i["Fare"] > 7.91) & (i["Fare"] <= 9.841), "Fare"] = 2

    i.loc[(i["Fare"] > 9.841) & (i["Fare"] <= 14.454), "Fare"] = 3   

    i.loc[(i["Fare"] > 14.454) & (i["Fare"] <= 24.479), "Fare"] = 4

    i.loc[(i["Fare"] >24.479) & (i["Fare"] <= 31), "Fare"] = 5   

    i.loc[(i["Fare"] > 31) & (i["Fare"] <= 69.487), "Fare"] = 6

    i.loc[i["Fare"] > 69.487, "Fare"] = 7

#Creating flags for sex and embarked in datset

for i in new_set:

    i['Sex'] = pd.factorize(i['Sex'])[0]

    i['Embarked']= pd.factorize(i['Embarked'])[0]

train_new.head()

train_new.head()

#Checking the correlation among the variables



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(train_new.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

#3.1 Building the classifier

#Selection of the X and Y variables for our classification model

x_var = train_new.drop("Survived", axis=1)

y_var = train_new["Survived"]

train_new.head()

#Splitting the datset into test and train

x_train, x_test, y_train, y_test = train_test_split(x_var,y_var,test_size=.25,random_state=1)

#Creating a list of all the classification algorithms to be used in this case



Classifier = [

    gaussian_process.GaussianProcessClassifier(),

    ensemble.BaggingClassifier(),

    tree.DecisionTreeClassifier() ,

    ensemble.RandomForestClassifier(),

    ensemble.GradientBoostingClassifier(),

    linear_model.LogisticRegressionCV(),

    ensemble.AdaBoostClassifier(),

    linear_model.SGDClassifier(),

    svm.SVC(probability=True),

    naive_bayes.BernoulliNB(),

    neighbors.KNeighborsClassifier(),

    linear_model. RidgeClassifierCV(),

    svm.LinearSVC()

    

    ]

#Preparing a dataframe to get the Accuracy, Precision, Recall and AUC

Classifier_columns = []

Classifier_compare = pd.DataFrame(columns = Classifier_columns)



row_index = 0

for alg in Classifier:

    

    

    predicted = alg.fit(x_train, y_train).predict(x_test)

    fp, tp, th = roc_curve(y_test, predicted)

    Classifier_name = alg.__class__.__name__

    Classifier_compare.loc[row_index,'Classifier Name'] = Classifier_name

    Classifier_compare.loc[row_index, 'Classifier Train Accuracy'] = round(alg.score(x_train, y_train), 4)

    Classifier_compare.loc[row_index, 'Classifier Test Accuracy'] = round(alg.score(x_test, y_test), 4)

    Classifier_compare.loc[row_index, 'Classifier Precission'] = precision_score(y_test, predicted)

    Classifier_compare.loc[row_index, 'Classifier Recall'] = recall_score(y_test, predicted)

    Classifier_compare.loc[row_index, 'Classifier AUC'] = auc(fp, tp)



    row_index+=1

    

Classifier_compare.sort_values(by = ['Classifier Test Accuracy'], ascending = False, inplace = True)    

Classifier_compare

Classifier_compare.columns[1:]

#Plotting the attributes to check the best classifier

for i in Classifier_compare.columns[1:]:

    plt.subplots(figsize=(15,6))

    sns.barplot(x="Classifier Name", y=i,data=Classifier_compare)

    plt.xticks(rotation=90)

    plt.title(i+ ' Comparison')

    plt.show()

#Comparing the ROC Curve for our classifiers

index = 1

for module in Classifier:

    

    

    predicted = module.fit(x_train, y_train).predict(x_test)

    fp, tp, th = roc_curve(y_test, predicted)

    roc_auc_Classifier = auc(fp, tp)

    Classifier_name = module.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (Classifier_name, roc_auc_Classifier))

   

    index+=1



plt.title('ROC Curve comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('TPR')

plt.xlabel('FPR')    

plt.show()

la = linear_model.LogisticRegressionCV()

la.fit(x_train, y_train)



print('Early Parameters ', la.get_params())

print("Early score on train: {:.2f}". format(la.score(x_train, y_train))) 

print("Early Score on test: {:.2f}". format(la.score(x_test, y_test)))



#Tuning our parameter to find the best classifier by using grid search

param_grid = {

              'Cs': [10,20,30,40,50],

              'intercept_scaling':  [1,2,3,4,5], 

              'max_iter': [50,100,150,200,250], 

             }



tune_model = model_selection.GridSearchCV(linear_model.LogisticRegressionCV(), param_grid=param_grid, scoring = 'roc_auc')

tune_model.fit (x_train, y_train)



print('Tuned Parameters: ', tune_model.best_params_)

print("Tuned Training score: {:.2f}". format(tune_model.score(x_train, y_train))) 

print("Tuned Test score: {:.2f}". format(tune_model.score(x_test, y_test)))

#4. Submitting on our tuned model

yhat_test = tune_model.predict(test_new)

# Submitting

submission = test_df.copy()

submission['Survived'] = yhat_test

submission.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)



submission[['PassengerId', 'Survived']].head(15)

train_df.head()
