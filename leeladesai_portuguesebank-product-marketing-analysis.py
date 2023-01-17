# To enable plotting graphs in Jupyter notebook

%matplotlib inline 
# To starts with lets import necessary libraries 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder as le

from scipy.stats import zscore

sns.set(color_codes=True)

sns.set(style = 'darkgrid')



import warnings

warnings.filterwarnings('ignore')

print("Packages LOADED")
# import the data using pandas readcsv function



original_df = pd.read_csv("../input/bank-full.csv")



print('DATA MATRIX = ',original_df.shape)
print('********************* Top 10 rows of the data *********************')

original_df.head()
print('********************* Last 10 rows of the data ********************')

original_df.tail()
original_df.isnull().sum()
print('**************** Basic Infomation about of the data Types ***************')

original_df.info()
original_df.describe().T
cleansed_df = original_df.copy()
# Impute unknowns function

def impute_unknowns(df, column):

    col_values = df[column].values

    df[column] = np.where(col_values=='unknown', dataset[column].mode(), col_values)

    return df
cleansed_df["poutcome"].value_counts()
cleansed_df[['poutcome']] = cleansed_df[['poutcome']].replace(['unknown','other'],'nonexistent')
# duration

cleansed_df = cleansed_df.drop(["duration"],axis=1)
# Fill 'unknown' in job and education to 'other'

cleansed_df[['job','education']] = cleansed_df[['job','education']].replace(['unknown'],'other')
#Encoding Categorical data into digits form.



cleansed_df['Target'] = cleansed_df.Target.map({'no':0, 'yes':1})



#Job

cleansed_df.loc[:,['job']] = cleansed_df.loc[:,['job']].apply(le().fit_transform)



#Marital

cleansed_df['marital'] = cleansed_df.marital.map({'single':0, 'married':1, 'divorced':2})



#Education

cleansed_df.loc[:,['education']] = cleansed_df.loc[:,['education']].apply(le().fit_transform)



#Default

cleansed_df.loc[:,['default']] = cleansed_df.loc[:,['default']].apply(le().fit_transform)



cleansed_df.loc[:,['housing']] = cleansed_df.loc[:,['housing']].apply(le().fit_transform)



cleansed_df.loc[:,['loan']] = cleansed_df.loc[:,['loan']].apply(le().fit_transform)



cleansed_df.loc[:,['contact']] = cleansed_df.loc[:,['contact']].apply(le().fit_transform)



cleansed_df['month'] = cleansed_df.month.map(

    {'jan':1,

     'feb':2,

     'mar':3,

     'apr':4,

     'may':5,

     'jun':6,

     'jul':7,

     'aug':8,

     'sep':9,

     'oct':10,

     'nov':11,

     'dec':12

     })



cleansed_df.loc[:,['poutcome']] = cleansed_df.loc[:,['poutcome']].apply(le().fit_transform)

negative_bal_count = cleansed_df[cleansed_df['balance'] < 0]['balance'].count()

print("Number of records in the dataset have negative balance : ", negative_bal_count)
cleansed_df['balance']= zscore(cleansed_df['balance'])
cleansed_df[original_df['pdays'] == -1]['pdays'].value_counts()
cleansed_df['pdays'] = cleansed_df['pdays'].replace(-1,999)
cleansed_df.info()
cleansed_df.groupby(["Target"]).count()
cleansed_df.describe().T
#Numeric data plot

plt.figure(figsize=(20, 20))

index = 0

for column_index, column in enumerate(cleansed_df.columns):

    if column == 'age' or column == 'balance' or column == 'day' or column == 'duration' or column == 'campaign' or column == 'pdays' or column == 'poutccome' or column == 'previous' :

        index = index+1

        plt.subplot(3, 3, index)

        sns.distplot(cleansed_df[column],bins = 20)
#categorical data plot

plt.figure(figsize=(20, 20))

index = 0

for column_index, column in enumerate(cleansed_df.columns):

    if column == 'Target' or  column == 'marital' or column == 'default' or column == 'job' or column == 'contact' or column == 'education' or column == 'month' or column == 'poutccome' or column == 'housing' or column == 'loan' :

        index = index+1;

        plt.subplot(5, 4, index)

        sns.countplot(cleansed_df[column])
#Pair plot all the variables in the dataset

sns.pairplot(cleansed_df, hue='Target')
# probably not a great feature since lot of outliers

cleansed_df.boxplot(column='age', by='Target')
#Heat Map to understand the correlation between the features

mask = np.zeros_like(cleansed_df.corr())

mask[np.triu_indices_from(mask, 1)] = True



plt.figure(figsize=(10, 10))

sns.heatmap(cleansed_df.corr(), mask=mask,annot=True,square=True)
def remove_outliers(df, column , minimum, maximum):

    col_values = df[column].values

    df[column] = np.where(np.logical_or(col_values<minimum, col_values>maximum), col_values.mean(), col_values)

    return df


plt.figure(figsize=(20, 20))

index = 0

for column_index, column in enumerate(cleansed_df.columns):

    if column == 'duration' or  column == 'education' or column == 'housing' or column == 'age' or column == 'job' or column == 'campaign':

        index = index+1;

        plt.subplot(5, 2, index)

        sns.boxplot(x="Target", y=column, data=cleansed_df)


min_val = cleansed_df["age"].min()

max_val = 80

cleansed_df = remove_outliers(df=cleansed_df, column='age' , minimum=min_val, maximum=max_val)



min_val = cleansed_df["campaign"].min()

max_val = 6

cleansed_df = remove_outliers(df=cleansed_df, column='campaign' , minimum=min_val, maximum=max_val)
#Import Libraries

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import preprocessing
#defining feature matrix(X) and response vector(y) from the cleansed data

X = cleansed_df.drop(['Target'], axis = 1)

y = cleansed_df['Target']
#Split Train and Test dataset in ration 70:30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
scaled_X = X.copy()

cat_cols = ['job','education','month']

for col in cat_cols:

    scaled_X = scaled_X.join(pd.get_dummies(scaled_X[col],prefix = col))

    scaled_X = scaled_X.drop([col],axis=1)

   

scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test = train_test_split(scaled_X, y, test_size=0.3, random_state=1)
# Importing libraries from the SCIKIT LEARN

from sklearn.linear_model import LogisticRegression

logit_reg = LogisticRegression()
# Apply training data to the logistic regression model

logit_reg.fit(X_train, y_train)



#Predict the test results

y_predict = logit_reg.predict(X_test)



#generate model prediction score

#logit_reg_score = logit_reg.score(X_test, y_test)

logit_reg_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)



#Calculate the coefficients of logistic regression model

coef_df = pd.DataFrame(logit_reg.coef_, columns= list(X_train.columns))

coef_df['intercept'] = logit_reg.intercept_

print(coef_df)

print()

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print(metrics.classification_report(y_test, y_predict))

print()

print("*"*54)

print("* Logistic Regression Accuracy Score (in %) =",logit_reg_accr_score, "*")

print("*"*54)
logit_reg_for_scaled = LogisticRegression()



# Apply training data to the logistic regression model

logit_reg_for_scaled.fit(scaled_X_train, scaled_y_train)



#Predict the test results

scaled_y_predict = logit_reg_for_scaled.predict(scaled_X_test)



#logit_reg_for_scaled_score = logit_reg_for_scaled.score(scaled_X_test, scaled_y_test)

scaled_logit_reg_accr_score = (metrics.accuracy_score(scaled_y_test, scaled_y_predict)*100).round(3)

#Calculate the coefficients of logistic regression model

scaled_coef_df = pd.DataFrame(logit_reg_for_scaled.coef_, columns= list(scaled_X_train.columns))

scaled_coef_df['intercept'] = logit_reg_for_scaled.intercept_

scaled_logit_confusion_matrix = metrics.confusion_matrix(scaled_y_test, scaled_y_predict)

print(scaled_coef_df)

print()

print("Confusion Matrix:")

print("-"*18)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print()

print("*"*69)

print("* Logistic Regression Accuracy Score(in %) for Scaled data =",scaled_logit_reg_accr_score, "*")

print("*"*69)

from sklearn.neighbors import KNeighborsClassifier

NNH_3= KNeighborsClassifier(n_neighbors= 3 , weights = 'distance')

NNH_3.fit(X_train, y_train)

y_predict = NNH_3.predict(X_test)

#NNH_3_score = NNH_3.score(X_test, y_test)

NNH_3_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(y_test, y_predict))

print("*"*55)

print("* K-NN(3) Accuracy Score(in %) for Base data =",NNH_3_accr_score, "*")

print("*"*55)
NNH_5= KNeighborsClassifier(n_neighbors= 5, weights = 'distance')

NNH_5.fit(X_train, y_train)

y_predict = NNH_5.predict(X_test)

#NNH_5_score = NNH_5.score(X_test, y_test)

NNH_5_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(y_test, y_predict))

print("*"*55)

print("* K-NN(5) Accuracy Score(in %) for Base data =",NNH_5_accr_score, "*")

print("*"*55)

scaled_NNH_3= KNeighborsClassifier(n_neighbors= 3 , weights = 'distance')

scaled_NNH_3.fit(scaled_X_train, scaled_y_train)

scaled_y_predict = scaled_NNH_3.predict(scaled_X_test)

#scaled_NNH_3_score = scaled_NNH_3.score(scaled_X_test, scaled_y_test)

scaled_NNH_3_accr_score = (metrics.accuracy_score(scaled_y_test, scaled_y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print("*"*55)

print("* K-NN(3) Accuracy Score(in %) for Scaled data =",scaled_NNH_3_accr_score, "*")

print("*"*55)
scaled_NNH_5= KNeighborsClassifier(n_neighbors= 5 , weights = 'distance')

scaled_NNH_5.fit(scaled_X_train, scaled_y_train)

scaled_y_predict = scaled_NNH_5.predict(scaled_X_test)

#scaled_NNH_5_score = scaled_NNH_5.score(scaled_X_test, scaled_y_test)

scaled_NNH_5_accr_score = (metrics.accuracy_score(scaled_y_test, scaled_y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print("*"*57)

print("* K-NN(5) Accuracy Score(in %) for Scaled data =",scaled_NNH_5_accr_score, "*")

print("*"*57)
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()

NB.fit(X_train,y_train)

y_predict = NB.predict(X_test)

#NB_score = NB.score(X_test, y_test)

NB_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(y_test, y_predict))

print("*"*59)

print("* Naive Bayes Accuracy Score(in %) for Base data =",NB_accr_score, "*")

print("*"*59)
scaled_NB = GaussianNB()

scaled_NB.fit(scaled_X_train,scaled_y_train)

scaled_y_predict = scaled_NB.predict(scaled_X_test)

#scaled_NB_score = scaled_NB.score(scaled_X_test, scaled_y_test)

scaled_NB_accr_score = (metrics.accuracy_score(scaled_y_test, scaled_y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print("*"*59)

print("* Naive Bayes Accuracy Score(in %) for Scaled data =",scaled_NB_accr_score, "*")

print("*"*59)
from sklearn import svm
clf = svm.SVC(gamma=0.02, C=3)

clf.fit(X_train , y_train)

y_predict = clf.predict(X_test)

svm_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(y_test, y_predict))

print("*"*69)

print("* Support Vector Machines Accuracy Score(in %) for Base data =",svm_accr_score, "*")

print("*"*69)
clf = svm.SVC(gamma=0.02, C=3)

clf.fit(scaled_X_train , scaled_y_train)

scaled_y_predict = clf.predict(scaled_X_test)

scaled_svm_accr_score = (metrics.accuracy_score(scaled_y_test, scaled_y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print("*"*73)

print("* Support Vector Machines Accuracy Score(in %) for Scaled data =",scaled_svm_accr_score, "*")

print("*"*73)
from sklearn.tree import DecisionTreeClassifier as DT_Class
dt_model = DT_Class(criterion = 'entropy')

dt_model.fit(X_train , y_train)

y_predict = dt_model.predict(X_test)

dt_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(y_test, y_predict))

print("*"*69)

print("* Decision Tree (Entropy) Accuracy Score(in %) for Base data =",dt_accr_score, "*")

print("*"*69)
dt_model = DT_Class(criterion = 'entropy')

dt_model.fit(scaled_X_train , scaled_y_train)

scaled_y_predict = dt_model.predict(scaled_X_test)

scaled_dt_accr_score = (metrics.accuracy_score(scaled_y_test, scaled_y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print("*"*69)

print("* Decision Tree (Entropy) Accuracy Score(in %) for Scaled data =",scaled_dt_accr_score, "*")

print("*"*69)
from sklearn.ensemble import BaggingClassifier

bgcl = BaggingClassifier(n_estimators=100, max_samples=.50 , oob_score=True)
bgcl.fit(X_train , y_train)

y_predict = bgcl.predict(X_test)

bgcl_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(y_test, y_predict))

print("*"*69)

print("* Ensemble - Bagging Classifier Accuracy Score(in %) for Base data =",bgcl_accr_score, "*")

print("*"*69)
bgcl.fit(scaled_X_train , scaled_y_train)

scaled_y_predict = bgcl.predict(scaled_X_test)

scaled_bgcl_accr_score = (metrics.accuracy_score(scaled_y_test, scaled_y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print("*"*69)

print("* Ensemble - Bagging Classifier Accuracy Score(in %) for Base data =",scaled_bgcl_accr_score, "*")

print("*"*69)
from sklearn.ensemble import AdaBoostClassifier

abcl = AdaBoostClassifier(base_estimator=dt_model, n_estimators=50)
abcl.fit(X_train , y_train)

y_predict = abcl.predict(X_test)

abcl_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(y_test, y_predict))

print("*"*69)

print("* Ensemble - AdaBoost Classifier Accuracy Score(in %) for Base data =",abcl_accr_score, "*")

print("*"*69)
abcl.fit(scaled_X_train , scaled_y_train)

scaled_y_predict = abcl.predict(scaled_X_test)

scaled_abcl_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print("*"*69)

print("* Ensemble - AdaBoost Classifier Accuracy Score(in %) for scaled data =",scaled_abcl_accr_score, "*")

print("*"*69)
from sklearn.ensemble import GradientBoostingClassifier

gbcl = GradientBoostingClassifier(n_estimators = 50, learning_rate = 0.09, max_depth=5)
gbcl.fit(X_train , y_train)

y_predict = gbcl.predict(X_test)

gbcl_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(y_test, y_predict))

print("*"*69)

print("* Ensemble - GradientBoosting Classifier Accuracy Score(in %) for Base data =",gbcl_accr_score, "*")

print("*"*69)
gbcl.fit(scaled_X_train , scaled_y_train)

scaled_y_predict = gbcl.predict(scaled_X_test)

scaled_gbcl_accr_score = (metrics.accuracy_score(scaled_y_test, scaled_y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print("*"*69)

print("* Ensemble - GradientBoosting Classifier Accuracy Score(in %) for scled_ data =",scaled_gbcl_accr_score, "*")

print("*"*69)
from sklearn.ensemble import RandomForestClassifier

rfcl = RandomForestClassifier(n_estimators = 6)
rfcl.fit(X_train , y_train)

y_predict = rfcl.predict(X_test)

rfcl_accr_score = (metrics.accuracy_score(y_test, y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(y_test, y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(y_test, y_predict))

print("*"*69)

print("* Ensemble - RandomForest Classifier Accuracy Score(in %) for Base data =",rfcl_accr_score, "*")

print("*"*69)
rfcl.fit(scaled_X_train , scaled_y_train)

scaled_y_predict = rfcl.predict(scaled_X_test)

scaled_rfcl_accr_score = (metrics.accuracy_score(scaled_y_test, scaled_y_predict)*100).round(3)

print("Confusion Matrix:")

print("-"*20)

print(metrics.confusion_matrix(scaled_y_test, scaled_y_predict))

print()

print("Classification Report:")

print("-"*23)

print(metrics.classification_report(scaled_y_test, scaled_y_predict))

print("*"*69)

print("* Ensemble - RandomForest Classifier Accuracy Score(in %) for scaled data =",scaled_rfcl_accr_score, "*")

print("*"*69)
model_accurcay_list = [["Logistic Regression",logit_reg_accr_score, scaled_logit_reg_accr_score],

                       ["K-Nearest Neighbors (3)",NNH_3_accr_score, scaled_NNH_3_accr_score],

                       ["K-Nearest Neighbors (5)",NNH_5_accr_score, scaled_NNH_5_accr_score],

                       ["Naïve Bayes’",NB_accr_score, scaled_NB_accr_score],

                       ["Support Vector Machines",svm_accr_score, scaled_svm_accr_score],

                       ["Decision Tree (Entrpoy)",dt_accr_score, scaled_dt_accr_score],

                       ["Ensemble - Bagging",bgcl_accr_score, scaled_bgcl_accr_score],

                       ["Ensemble - AdaBoost",abcl_accr_score, scaled_abcl_accr_score],

                       ["Ensemble - GradientBoosting",gbcl_accr_score, scaled_gbcl_accr_score],

                       ["Ensemble - RandomForest",rfcl_accr_score, scaled_rfcl_accr_score]]
print("-"*65)

print('| Overall Model Accuracy(in %) | Before Scaling | After Scaling |')

print("-"*65)

for item in model_accurcay_list :

    print("|",item[0]," "*(27-len(item[0])), "|",

         item[1]," "*(13-len(str(item[1]))), "|",

         item[2]," "*(12-len(str(item[2]))), "|")

print("-"*65)