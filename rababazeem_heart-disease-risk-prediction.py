

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #plots

import matplotlib.pyplot as plt #more plots

import sklearn as sk #machine learning



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#loading the data in

data = pd.read_csv("/kaggle/input/logistic-regression-heart-disease-prediction/framingham_heart_disease.csv")
data.head()
print(data.columns)
data.describe()
# listing how many null values are in the dataset

data.isna().sum()
# creating histograms to visualize all the data

fig = plt.figure(figsize = (40,40))

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)



ax = fig.gca()

data.hist(ax = ax)
# creating a heatmap to find the correlation between each feature

sns.set(font_scale=3)

def plot_corr( df ):

    corr = df.corr()

    _, ax=plt.subplots( figsize=(50,25) )

    cmap = sns.diverging_palette( 240 , 10 , as_cmap = True)

    _ = sns.heatmap(corr,cmap=cmap,square=True, cbar_kws = {'shrink': .9}, fmt= '.1f', ax=ax, annot=True)

    

plot_corr(data)

data = data.drop(['education'], axis=1)
data.head()
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

my_imputer.fit(data)

imputed_data = pd.DataFrame(my_imputer.transform(data))



# Imputation removed column names; put them back

imputed_data.columns = data.columns
imputed_data.isna().sum()
from sklearn.model_selection import train_test_split



y = imputed_data['TenYearCHD']

X = imputed_data.drop(['TenYearCHD'], axis = 1)



# divide

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=14)
#Creating a set of features that are standardized

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train_std = pd.DataFrame(scaler.fit_transform(X_train))

X_test_std = pd.DataFrame(scaler.transform(X_test))



X_train_std.columns = X_train.columns

X_test_std.columns = X_test.columns



X_train_std
# Shuffle df

shuffled_df = imputed_data.sample(frac=1,random_state=4)



# Put all the fraud class in a separate dataset.

CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 1]



#Randomly select 492 observations from the non-fraud (majority class)

non_CHD_df = shuffled_df.loc[shuffled_df['TenYearCHD'] == 0].sample(n=611,random_state=42)



# Concatenate(join) both dataframes again

normalized_df = pd.concat([CHD_df, non_CHD_df])



# plot new count

sns.countplot(normalized_df.TenYearCHD, palette="OrRd")

plt.box(False)

plt.xlabel('Heart Disease No/Yes',fontsize=11)

plt.ylabel('Patient Count',fontsize=11)

plt.title('Count Outcome Heart Disease after Resampling\n')

#plt.savefig('Balance Heart Disease.png')

plt.show()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



# find best scored 6 features

select_feature = SelectKBest(chi2, k=10).fit(X_train, y_train)



dfscores = pd.DataFrame(select_feature.scores_)

dfcolumns = pd.DataFrame(X_train.columns)



#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

top_featureScores = featureScores.nlargest(10,'Score')

print(top_featureScores)
# visualizing feature selection

plt.figure(figsize=(20,5))

sns.barplot(x='Specs', y='Score', data=featureScores, palette = "GnBu_d")

plt.title('Feature importance', fontsize=16)

plt.xlabel('\n Features', fontsize=14)

plt.ylabel('Importance \n', fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
#Making a dataset with only the top 10 features

best_feature_names = top_featureScores['Specs']



X_best_train = X_train[best_feature_names]

X_best_test = X_test[best_feature_names]

y_best_train = y_train[best_feature_names]

y_best_test = y_test[best_feature_names]



X_best_train
from sklearn.linear_model import LogisticRegression



LR_model= LogisticRegression()



tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,

              'penalty':['l1','l2']

                   }
# Using a gridsearch to tune parameters

from sklearn.model_selection import GridSearchCV



GS = GridSearchCV(LR_model, tuned_parameters,cv=10)



GS.fit(X_train_std, y_train)
print(GS.best_params_)
# Making a new model with the best parameters

LR_model= LogisticRegression(C=0.1, penalty='l2')
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score



#Fitting the model to the standardized data

LR_model.fit(X_train_std, y_train)



#Using the model to predict

y_pred = LR_model.predict(X_test_std)

y_pred



#Getting the accuracies



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for LogReg is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for LogReg is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, y_pred)

print(f"The precision score for LogReg is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, y_pred)

print(f"The recall score for LogReg is: {round(recall,3)*100}%")
from sklearn.metrics import accuracy_score



#Fitting the model to the non standardized data

LR_model.fit(X_train, y_train)



#Using the model to predict

y_pred = LR_model.predict(X_test)



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for LogReg is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for LogReg is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, y_pred)

print(f"The precision score for LogReg is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, y_pred)

print(f"The recall score for LogReg is: {round(recall,3)*100}%")
from sklearn.metrics import accuracy_score



#Fitting the model to the standardized data

LR_model.fit(X_best_train, y_train)



#Using the model to predict

y_pred = LR_model.predict(X_best_test)

y_pred



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for LogReg is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for LogReg is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, y_pred)

print(f"The precision score for LogReg is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, y_pred)

print(f"The recall score for LogReg is: {round(recall,3)*100}%")
from sklearn.ensemble import RandomForestClassifier



#initiating a new model

rf = RandomForestClassifier(random_state=1)



#making a list of parameters for the grid search to compare

tuned_parameters = {'n_estimators': [100, 500, 1000]}



#Making a grid search model

GS_rf=GridSearchCV(rf, tuned_parameters, cv=10)



#Fitting the grid search

# GS_rf.fit(X_train_std, y_train)



#Printing the best features

# print(GS_rf.best_params_)



# I commented the last lines of code out so that I wouldn't have to run it again since I already know the best paramters
# adding the best parameters

rf = RandomForestClassifier(random_state=1, n_estimators = 1000)
#fitting the model

rf.fit(X_train_std, y_train)



#making predictions

y_pred = rf.predict(X_test_std)



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for RandomForest is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for RandomForest is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, y_pred)

print(f"The precision score for RandomForest is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, y_pred)

print(f"The recall score for RandomForest is: {round(recall,3)*100}%")
#fitting the model

rf.fit(X_train, y_train)



#making predictions

y_pred = rf.predict(X_test)



# check accuracy

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for RandomForest is: {round(acc,3)*100}%")



# f1 score

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for RandomForest is: {round(f1,3)*100}%")



# Precision score

precision = precision_score(y_test, y_pred)

print(f"The precision score for RandomForest is: {round(precision,3)*100}%")



# recall score

recall = recall_score(y_test, y_pred)

print(f"The recall score for RandomForest is: {round(recall,3)*100}%")
#fitting the model

rf.fit(X_best_train, y_train)



#making predictions

y_pred = rf.predict(X_best_test)



# check accuracy

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for RandomForest is: {round(acc,3)*100}%")



# f1 score

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for RandomForest is: {round(f1,3)*100}%")



# Precision score

precision = precision_score(y_test, y_pred)

print(f"The precision score for RandomForest is: {round(precision,3)*100}%")



# recall score

recall = recall_score(y_test, y_pred)

print(f"The recall score for RandomForest is: {round(recall,3)*100}%")
from sklearn.neighbors import KNeighborsClassifier



#Initializing the model

knn = KNeighborsClassifier()



#The parameters

my_params={'n_neighbors': [1,2,5,10,15, 20, 21, 25, 30]}



#creating the gridsearch

GS_knn = GridSearchCV(knn, my_params, cv=10)



#fitting the model

GS_knn.fit(X_train_std, y_train)



#finding the best parameters

print(GS_knn.best_params_)
#making a tuned model

knn = KNeighborsClassifier(n_neighbors=20)
#fitting the model

knn.fit(X_train_std, y_train)



#making predictions

y_pred = knn.predict(X_test_std)



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for KNN is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for KNN is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, y_pred)

print(f"The precision score for KNN is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, y_pred)

print(f"The recall score for KNN is: {round(recall,3)*100}%")
#fitting the model

knn.fit(X_train, y_train)



#making predictions

y_pred = knn.predict(X_test)



# check accuracy

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for KNN is: {round(acc,3)*100}%")



# f1 score

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for KNN is: {round(f1,3)*100}%")



# Precision score

precision = precision_score(y_test, y_pred)

print(f"The precision score for KNN is: {round(precision,3)*100}%")



# recall score

recall = recall_score(y_test, y_pred)

print(f"The recall score for KNN is: {round(recall,3)*100}%")
#fitting the model

knn.fit(X_best_train, y_train)



#making predictions

y_pred = knn.predict(X_best_test)



# check accuracy

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for KNN is: {round(acc,3)*100}%")



# f1 score

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for KNN is: {round(f1,3)*100}%")



# Precision score

precision = precision_score(y_test, y_pred)

print(f"The precision score for KNN is: {round(precision,3)*100}%")



# recall score

recall = recall_score(y_test, y_pred)

print(f"The recall score for KNN is: {round(recall,3)*100}%")
from xgboost import XGBRegressor



#creating a new model with experimental parameters

XGB = XGBRegressor(n_estimators=1000, early_stopping_rounds=5, learning_state=0.01, objective='binary:hinge')



#fitting the model

XGB.fit(X_train_std, y_train)



#making predcitions

y_pred = XGB.predict(X_test_std)



# check accuracy: Accuracy: Overall, how often is the classifier correct? Accuracy = (True Pos + True Negative)/total

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for XGB is: {round(acc,3)*100}%")



# f1 score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for XGB is: {round(f1,3)*100}%")



# Precision score: When it predicts yes, how often is it correct? Precision=True Positive/predicted yes

precision = precision_score(y_test, y_pred)

print(f"The precision score for XGB is: {round(precision,3)*100}%")



# recall score: True Positive Rate(Sensitivity or Recall): When it’s actually yes, how often does it predict yes? True Positive Rate = True Positive/actual yes

recall = recall_score(y_test, y_pred)

print(f"The recall score for XGB is: {round(recall,3)*100}%")
#creating a new model with experimental parameters

XGB = XGBRegressor(n_estimators=1000, early_stopping_rounds=5, learning_state=0.05, objective='binary:hinge')



#fitting the model

XGB.fit(X_train, y_train)



#making predcitions

y_pred = XGB.predict(X_test)



# check accuracy

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for XGB is: {round(acc,3)*100}%")



# f1 score

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for XGB is: {round(f1,3)*100}%")



# Precision score

precision = precision_score(y_test, y_pred)

print(f"The precision score for XGB is: {round(precision,3)*100}%")



# recall score

recall = recall_score(y_test, y_pred)

print(f"The recall score for XGB is: {round(recall,3)*100}%")
#creating a new model with experimental parameters

XGB = XGBRegressor(n_estimators=1000, early_stopping_rounds=5, learning_state=0.05, objective='binary:hinge')



#fitting the model

XGB.fit(X_best_train, y_train)



#making predcitions

y_pred = XGB.predict(X_best_test)



# check accuracy

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for XGB is: {round(acc,3)*100}%")



# f1 score

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for XGB is: {round(f1,3)*100}%")



# Precision score

precision = precision_score(y_test, y_pred)

print(f"The precision score for XGB is: {round(precision,3)*100}%")



# recall score

recall = recall_score(y_test, y_pred)

print(f"The recall score for XGB is: {round(recall,3)*100}%")
'''

y = normalized_df['TenYearCHD']

X = normalized_df.drop(['TenYearCHD'], axis = 1)



# divide

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=14)



#Creating a set of features that are standardized

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train_std = pd.DataFrame(scaler.fit_transform(X_train))

X_test_std = pd.DataFrame(scaler.transform(X_test))



X_train_std.columns = X_train.columns

X_test_std.columns = X_test.columns



X_train_std

'''
from sklearn.svm import SVC



#initialize model

svm = SVC()



#fit model

svm.fit(X_train_std, y_train)



#make predictions

y_pred = svm.predict(X_test_std)



# check accuracy

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for SVM is: {round(acc,3)*100}%")



# f1 score

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for SVM is: {round(f1,3)*100}%")



# Precision score

precision = precision_score(y_test, y_pred)

print(f"The precision score for SVM is: {round(precision,3)*100}%")



# recall score

recall = recall_score(y_test, y_pred)

print(f"The recall score for SVM is: {round(recall,3)*100}%")
#initialize model

svm = SVC()



#fit model

svm.fit(X_train, y_train)



#make predictions

y_pred = svm.predict(X_test)



# check accuracy

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for SVM is: {round(acc,3)*100}%")



# f1 score

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for SVM is: {round(f1,3)*100}%")



# Precision score

precision = precision_score(y_test, y_pred)

print(f"The precision score for SVM is: {round(precision,3)*100}%")



# recall score

recall = recall_score(y_test, y_pred)

print(f"The recall score for SVM is: {round(recall,3)*100}%")
#initialize model

svm = SVC()



#fit model

svm.fit(X_best_train, y_train)



#make predictions

y_pred = svm.predict(X_best_test)



# check accuracy

acc = accuracy_score(y_test, y_pred)

print(f"The accuracy score for SVM is: {round(acc,3)*100}%")



# f1 score

f1 = f1_score(y_test, y_pred)

print(f"The f1 score for SVM is: {round(f1,3)*100}%")



# Precision score

precision = precision_score(y_test, y_pred)

print(f"The precision score for SVM is: {round(precision,3)*100}%")



# recall score

recall = recall_score(y_test, y_pred)

print(f"The recall score for SVM is: {round(recall,3)*100}%")
import matplotlib.font_manager

matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')



# Making a dataframe of the accuracies

a = {'Random Forest Classifier': [84.89], 'K-Nearest Neighbours': [84.89], 'Logistic Regression': [85.5], 'XGBoost':[83.3], 'SVM':[84.7]}

accuracies = pd.DataFrame(data=a)



# making bar plot comparing the accuracies of the models

sns.set(font_scale=1)

ax = accuracies.plot.bar(

    figsize= (10, 5),

    fontsize=14)

plt.xticks(rotation=0, fontsize=14)

plt.xlabel('Models', fontsize=14)

plt.ylabel('Accuracy', fontsize=14)

x_labels = ['']

xticks = [-0.20, -0.1, 0.02, 0.14, ]

ax.set_xticks(xticks)

ax.set_xticklabels(x_labels, rotation=0)

axbox = ax.get_position()

plt.legend(loc = (axbox.x0 + 0.65, axbox.y0 + 0.70), fontsize=14)

plt.title('Accuracies of Each Model Predicting CHD')

ax.set_facecolor('xkcd:white')

ax.set_facecolor(('#ffffff'))

ax.spines['left'].set_color('black')

ax.spines['bottom'].set_color('black')
my_predictors = []

parameters=['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds','male']



age = 17 ##"Patient's age: >>>

my_predictors.append(age)

male = 0#"Patient's gender. male=1, female=0:

my_predictors.append(male)

cigsPerDay =0 ##"Patient's smoked cigarettes per day:

my_predictors.append(cigsPerDay)

sysBP = 100##"Patient's systolic blood pressure:

my_predictors.append(sysBP)

diaBP = 80##"Patient's diastolic blood pressure

my_predictors.append(diaBP)

totChol = 130##"Patient's cholesterin level:

my_predictors.append(totChol)

prevalentHyp =0 ##"Was Patient hypertensive? Yes=1, No= 0

my_predictors.append(prevalentHyp)

diabetes = 0##"Did Patient have diabetes? Yes=1, No=0

my_predictors.append(diabetes)

glucose = 100##"What is the Patient's glucose level?

my_predictors.append(diabetes)

BPMeds = 0##"Has Patient been on Blood Pressure Medication? Yes=1, No=0

my_predictors.append(BPMeds)



#adding the data to the dataset

my_data = dict(zip(parameters, my_predictors))

my_df = pd.DataFrame(my_data, index=[0])



my_y_pred = knn.predict(my_df)

print('Result:')

if my_y_pred == 1:

    print("The patient will develop a heart disease.")

if my_y_pred == 0:

    print("The patient will not develop a heart disease.")