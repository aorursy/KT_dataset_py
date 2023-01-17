import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

data.head()
# data shape

print('Data Shape is : ', data.shape)
#check null values

data.isna().sum()
# Features correlation

plt.figure(figsize=(12,8))

sns.heatmap(data.corr(),annot = True)
def Get_Features_types(data):

    categorical_features = []

    numerical_features = []

    for col in data.iloc[:,:-1]:

        if data[col].nunique() < 10:

            categorical_features.append(col)

        else : 

            numerical_features.append(col)

    return categorical_features , numerical_features



Get_Features_types(data)



categorical_features = Get_Features_types(data)[0]

numerical_features = Get_Features_types(data)[1]



print(categorical_features)

print(numerical_features)
sns.countplot( x = 'DEATH_EVENT' , data = data)
def CountPlot_hue_Categorical_data(data):

    for feature in categorical_features:

        plt.figure()

        sns.countplot(x = "DEATH_EVENT" , data = data , hue=feature)

        plt.title(feature)



CountPlot_hue_Categorical_data(data)
def Get_Percentage_data(data):

    percentage_death_true_case = []

    percentage_death_false_case = []

    for col in categorical_features:

        true_case = round(data['DEATH_EVENT'][data[col] == 1].value_counts(normalize = True)[1] * 100 , 2)

        false_case = round(data['DEATH_EVENT'][data[col] == 0].value_counts(normalize = True)[1] * 100,2)

        percentage_death_true_case.append(true_case)

        percentage_death_false_case.append(false_case)

    Percentage = pd.DataFrame(list(zip(percentage_death_true_case , percentage_death_false_case)) ,

                              index = categorical_features ,

                              columns = ['% Percentage Death (IF 1)' , '% Percentage Death (IF 0)'])

    return Percentage

        

Get_Percentage_data(data)
def BOX_Plot_numerical_features(data):

    for col in numerical_features:

        plt.figure()

        sns.boxplot(data[col])
BOX_Plot_numerical_features(data)
print("Numerical Features are " , numerical_features)
data[data['ejection_fraction'] > 65]
## Age 

sns.distplot(data[data['DEATH_EVENT'] == 0]['age'] , label ="Death")

sns.distplot(data[data['DEATH_EVENT'] == 1]['age'] , label = "Survived")

plt.legend()
plt.figure()

sns.countplot(data[(data['age'] < 70)]['DEATH_EVENT'])

print(data[(data['age'] < 70)]['DEATH_EVENT'].value_counts())

plt.title("Age Under than 70")

plt.show()

####

plt.figure()

sns.countplot(data[(data['age'] > 70)]['DEATH_EVENT'])

print(data[(data['age'] > 70)]['DEATH_EVENT'].value_counts())

plt.title("Age Higher than 70")

plt.show()
## Creatinine_phosphokinase

sns.distplot(data[data['DEATH_EVENT'] == 0]['creatinine_phosphokinase'] , label ="Death")

sns.distplot(data[data['DEATH_EVENT'] == 1]['creatinine_phosphokinase'] , label = "Survived")

plt.legend()
## Platelets

sns.distplot(data[data['DEATH_EVENT'] == 0]['platelets'] , label ="Death")

sns.distplot(data[data['DEATH_EVENT'] == 1]['platelets'] , label = "Survived")

plt.legend()
## Serum_creatinine

sns.distplot(data[data['DEATH_EVENT'] == 0]['serum_creatinine'] , label ="Death")

sns.distplot(data[data['DEATH_EVENT'] == 1]['serum_creatinine'] , label = "Survived")

plt.legend()
## Serum_sodium

sns.distplot(data[data['DEATH_EVENT'] == 0]['serum_sodium'] , label ="Death")

sns.distplot(data[data['DEATH_EVENT'] == 1]['serum_sodium'] , label = "Survived")

plt.legend()
# Time

sns.distplot(data[data['DEATH_EVENT'] == 0]['time'] , label ="Death")

sns.distplot(data[data['DEATH_EVENT'] == 1]['time'] , label = "Survived")

plt.legend()
plt.figure()

sns.countplot(data[data['time'] > 90]['DEATH_EVENT'])

print(data[data['time'] > 90]['DEATH_EVENT'].value_counts())

plt.title("Time Higher than 90")

plt.show()

####

plt.figure()

sns.countplot(data[data['time'] < 90]['DEATH_EVENT'])

print(data[data['time'] < 90]['DEATH_EVENT'].value_counts())

plt.title("Time lower than 90")

plt.show()

# Ejection_fraction

sns.distplot(data[data['DEATH_EVENT'] == 0]['ejection_fraction'] , label ="Death")

sns.distplot(data[data['DEATH_EVENT'] == 1]['ejection_fraction'] , label = "Survived")

plt.legend()
sns.countplot(data[data['ejection_fraction'] > 28]['DEATH_EVENT'])

print(data[data['ejection_fraction'] > 28]['DEATH_EVENT'].value_counts())

plt.title("Ejection Fraction more than 30")

plt.show()

#####

sns.countplot(data[data['ejection_fraction'] < 30]['DEATH_EVENT'])

print(data[data['ejection_fraction'] < 30]['DEATH_EVENT'].value_counts())

plt.title("Ejection Fraction lower than 30")

plt.show()
data_copy= data.copy()
data_copy['platelets/age'] = data['platelets'] / data['age']

data_copy['time/age'] = data['time'] / data['age']

data_copy.drop(['platelets' , 'time'] , axis = 1 , inplace = True ) 

data_copy.head()
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier ,AdaBoostClassifier

from sklearn.metrics import f1_score as f1

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest , f_classif
#Split the data

X = data_copy.drop(['DEATH_EVENT'] , axis = 1)

y = data_copy['DEATH_EVENT']

x_train , x_test , y_train , y_test = train_test_split(X,y,test_size = 0.2 ,random_state=1)
# We will use KNN with differents values of k in order to have an idea about our score accuracy and f1-score

def KNN_best_scores(n_iterations):

    acc_score = []

    f1_score = []

    for k in range(1,n_iterations):

        model = KNeighborsClassifier(n_neighbors= k)

        model.fit(x_train , y_train)

        y_pred = model.predict(x_test)

        acc_score.append(model.score(x_test , y_test))

        f1_score.append(f1(y_pred , y_test))

    Knn_scores = pd.DataFrame(list(zip(acc_score , f1_score)) , index = range(1,n_iterations) ,

                              columns = ['Accuracy Score' , 'F1_Score'])

    print('Best values for f1_score are : \n',Knn_scores.nlargest(5, ['F1_Score']))

    return  Knn_scores.nlargest(5, ['Accuracy Score'])
#Selecting a range of value between (1,20)

KNN_best_scores(20)
## RandomForest

def RandomForest_best_score():

    model = RandomForestClassifier()

    model.fit(x_train,y_train)

    acc_score = model.score(x_test,y_test)

    y_pred = model.predict(x_test)

    f1_score = f1(y_pred , y_test)

    return pd.DataFrame([[acc_score , f1_score]] , columns = ['Accuracy Score' , 'F1_Score'])

    
print("RandomForest : Accuracy Score / F1_Score")

RandomForest_best_score()
# Gradient Boosting Classifier

def GradientBoosting_best_score():

    model = GradientBoostingClassifier()

    model.fit(x_train , y_train)

    y_pred = model.predict(x_test)

    acc_score = model.score(x_test , y_test)

    f1_score = f1(y_pred , y_test)

    

    return pd.DataFrame([[acc_score , f1_score]] , columns = ['Accuracy Score' , 'F1_Score'])



    

    
print("GradientBoosting : Accuracy Score / F1_Score")



GradientBoosting_best_score()
# It combines multiple classifiers to increase the accuracy of classifiers. AdaBoost is an iterative ensemble method

def AdaBoostClassifier_best_score(list_values):

    acc_scores = []

    f1_scores = []

    for n in list_values:

        model = AdaBoostClassifier(n_estimators=n,learning_rate=0.01)

        model.fit(x_train , y_train)

        acc_score = model.score(x_test , y_test)

        y_pred = model.predict(x_test)

        f1_score = f1(y_pred,y_test)

        acc_scores.append(acc_score)

        f1_scores.append(f1_score)

    return pd.DataFrame(list(zip(acc_scores,f1_scores)) , columns = ['Accuracy Score' , 'F1_Score'])
AdaBoostClassifier_best_score([i for i in range(100,1000,100)])
## SelectKBest with range of values (5,6,7,8) and try to find difference between models with and without feature selection

def KNN_With_Feature_Selection():

    acc = []

    f1_scores = []

    features_names = []

    for n in range(5,9):

        feature_selection = SelectKBest(f_classif , k = n)

        features_name = feature_selection.fit_transform(data_copy[X.columns], data_copy['DEATH_EVENT'])

        cols = feature_selection.get_support(indices=True)

        features_name = data_copy.columns[cols]

        features_names.append(features_name)

        knn_processor = make_pipeline(StandardScaler() , feature_selection )

        for i in range(1,20):

            KNN_model = make_pipeline(knn_processor , KNeighborsClassifier(n_neighbors = i))

            KNN_model.fit(x_train , y_train)

            acc_score = KNN_model.score(x_test,y_test)

            y_pred = KNN_model.predict(x_test)

            f1_score = f1(y_pred , y_test)

            acc.append(acc_score)

            f1_scores.append(f1_score)

    DF1 = pd.DataFrame(list(zip(acc[:19] , f1_scores[:19])) , columns = ['Accuracy(n = 5)' , 'F1_Score(n = 5)'])

    DF2 = pd.DataFrame(list(zip(acc[20:37] , f1_scores[20:37])) , columns = ['Accuracy(n = 6)' , 'F1_Score(n = 6)'])

    DF3 = pd.DataFrame(list(zip(acc[38:56] , f1_scores[38:56])) , columns = ['Accuracy(n = 7)' , 'F1_Score(n = 7)'])

    DF4 = pd.DataFrame(list(zip(acc[57:] , f1_scores[57:])) , columns = ['Accuracy(n = 8)' , 'F1_Score(n = 8)'])

    

    return (list(features_names[0]) , DF1) , (list(features_names[1]) , DF2) , (list(features_names[2]) , DF3) , (list(features_names[3]) , DF2)

   
(FN0,DF1),(FN1,DF2) ,(FN2,DF3) ,(FN3,DF4) = KNN_With_Feature_Selection()

print("Feature Selection with n =5 " , FN0)

DF1
print("Feature Selection with n =6" , FN1)

DF2
print("Feature Selection with n =7 " , FN2)

DF3
print("Feature Selection with n =8 " , FN3)

DF4
def AdaBoostClassifier_best_score_with_feature_selection(list_values):

    train_score = []

    acc = []

    f1_scores = []

    feature_selection = SelectKBest(f_classif , k = 5)

    adaboost_processor = make_pipeline(StandardScaler() , feature_selection )

    for i in list_values:

        adabooost_model = make_pipeline(adaboost_processor , AdaBoostClassifier(n_estimators= i,learning_rate=0.09))

        adabooost_model.fit(x_train , y_train)

        train_sc= adabooost_model.score(x_train,y_train)

        train_score.append(train_sc)

        acc_score = adabooost_model.score(x_test,y_test)

        y_pred = adabooost_model.predict(x_test)

        f1_score = f1(y_pred , y_test)

        acc.append(acc_score)

        f1_scores.append(f1_score)

    return pd.DataFrame(list(zip(train_score,acc,f1_scores)) , columns = ['Train Score','Accuracy Score' , 'F1_Score'])

   
AdaBoostClassifier_best_score_with_feature_selection([i for i in range(100,1000,100)])
AdaBoostClassifier_best_score_with_feature_selection([800])
### Final Model :

feature_selection = SelectKBest(f_classif , k = 5)

adaboost_processor = make_pipeline(StandardScaler() , feature_selection )

adaboost_model = make_pipeline(adaboost_processor , AdaBoostClassifier(n_estimators= 800,learning_rate=0.09))

adaboost_model.fit(X , y)

my_predictions = adaboost_model.predict(X)

submission = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

submission['Predictions'] = my_predictions

submission.to_csv("submission.csv", index=False)