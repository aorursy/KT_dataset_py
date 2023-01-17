import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing libraries



# for data manipulation

import pandas as pd

import numpy as np



# for data visualization  

import seaborn as sns

import matplotlib.pyplot as plt
# Importing training data



train = pd.read_csv("/kaggle/input/credit-default-prediction-ai-big-data/train.csv", index_col = 'Id')



train.dropna(how = 'all')



print("Train Data has been read")



# Printing transpose of dataset to view all columns

train.T
# Importing test data



test = pd.read_csv("/kaggle/input/credit-default-prediction-ai-big-data/test.csv", index_col = 'Id')



test.dropna(how = 'all')

print("Train Data has been read")



# Printing transpose of dataset to view all columns

test.T
# Info about the train dataset



train.info()
# Stats about the train dataset



train.describe().T
# Info about the test dataset



test.info()
# Stats about the test dataset



test.describe().T
#Deleting spaces between names and interchanging with underscore



new_cols = [str(i).lower().replace(" ", "_") for i in (list(train.columns))]

new_cols
# Assigning new names to columns



train.columns = new_cols

test.columns = new_cols[:-1]
# Printing transpose of dataset to view all columns



train.T
# Printing transpose of dataset to view all columns



test.T
## Since renewable energy has no value hence analyzing it



print(train.purpose.value_counts(dropna = False)), print('*' * 80, '\n'), print(test.purpose.value_counts(dropna = False))
train = train[train.purpose != 'renewable energy']

train.T
# Creating a dataframe that contains various values to corresponding columns



mask = train.isnull()                  # calculate total null values

total = mask.sum()                    # calculate the sum

percent = 100 * mask.mean()           # calculate the percent missing values

dtype = train.dtypes                   # getting the data types of the columns

unique = train.nunique()               # getting the all unique values of the columns





# creating a dataframe



null_train = pd.concat([total, percent, dtype, unique], 

                       axis = 1, 

                       keys = ['Total Count', 'Percent Missing', 'dtype', 'Unique Values'])



null_train.sort_values(by = 'Total Count', ascending = False)
mask1 = test.isnull()                          # calculate total null values

total1 = mask1.sum()                           # calculate the sum

percent1 = 100 * mask1.mean()                  # calculate the percent missing values

dtype1 = test.dtypes                           # getting the data types of the columns

unique1 = test.nunique()                       # getting the all unique values of the columns



null_test = pd.concat([total1, percent1, dtype1, unique1], 

                      axis = 1, 

                      keys = ['Total Count', 'Percent Missing', 'dtype', 'Unique Values'])





# Soritng the values by count



null_test.sort_values(by = 'Total Count', ascending = False)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize = (20,10))

sns.heatmap(train.corr(), annot = True, vmax = 1, vmin = -1, square = True)
plt.figure(figsize = (17,7))

sns.barplot(y = train.months_since_last_delinquent, x = train.bankruptcies, data = train)
plt.figure(figsize = (17,7))

sns.barplot(y = train.months_since_last_delinquent, x = train.number_of_credit_problems, data = train)
plt.figure(figsize = (17,7))

sns.barplot(y = train.months_since_last_delinquent, x = train.purpose, data = train)
print('Tax Liens Value Counts ::\n\n', train.tax_liens.value_counts(dropna = False))

print('*' * 80)

print('\nbankruptcies Value Counts ::\n\n', train.bankruptcies.value_counts(dropna = False))
print('purpose Value Counts ::\n\n', train.purpose.value_counts(dropna = False))

print('*' * 80)

print('\number_of_credit_problems Value Counts ::\n\n', train.number_of_credit_problems.value_counts(dropna = False))
train.bankruptcies.value_counts(dropna = False)
from scipy.stats import mode



# Filling with mode value



train.bankruptcies =  train.bankruptcies.agg(lambda x : x.fillna(value = 2.0))



# Checking for NaN values present



print('Total NaN values present after filling :: ', train.bankruptcies.isna().sum())

train.bankruptcies.value_counts(dropna = False)
# Checking values before the filled values 



train['months_since_last_delinquent'].value_counts(dropna = False)
# Grouping months_since_last_delinquent columns as per  'purpose', 'home_ownership'



mean_score = train.groupby(['tax_liens','bankruptcies','number_of_credit_problems'])['months_since_last_delinquent']

mean_score.agg([np.mean, np.median, np.size])
train.loc[: , 'months_since_last_delinquent'] = mean_score.transform(lambda x : x.fillna(x.median()))



#Using ffill to fill any left missing value



train.months_since_last_delinquent = train.months_since_last_delinquent.fillna(method = 'ffill')

# Checking for NaN values present



print('Total NaN values present after filling :: ', train.months_since_last_delinquent.isna().sum())
train['months_since_last_delinquent'].value_counts(dropna = False)
print('Total NaN values present before filling :: ', train.years_in_current_job.isna().sum())



# Displaying total unique values present 



train.years_in_current_job.value_counts(dropna = False)
### Since NaN values present hence filling it using ffill and bfill 



train.years_in_current_job.fillna(method = 'ffill', inplace = True)

train.years_in_current_job.fillna(method = 'bfill', inplace = True)





# Checking for NaN values present

print('Total NaN values present after filling :: ', train.years_in_current_job.isna().sum())



train.years_in_current_job.value_counts(dropna = False)
# Visualizing to see its relationship with other columns



plt.figure(figsize = (20,8))

sns.barplot(x = train.number_of_credit_problems, y = train.annual_income, data = train)
plt.figure(figsize = (20,8))

sns.barplot(x = train.purpose, y = train.annual_income, data = train)
# Again visualizing 



plt.figure(figsize = (20,8))

sns.barplot(x = train.years_in_current_job, y = train.annual_income, data = train)
plt.figure(figsize = (20,8))

sns.barplot(x = train.number_of_open_accounts, y = train.annual_income, data = train)
# Displaying total unique values



train.annual_income.value_counts(dropna = False)
val = train.groupby(['purpose','years_in_current_job'])['annual_income']

val.agg([np.mean, np.median, np.size])
#tranforming/applying mean values to corresponding rows with missing values



train.annual_income = val.transform(lambda x : x.fillna(x.median()))

train.annual_income = train.annual_income.fillna(method = 'ffill')

# Checking for NaN values present



print('Total NaN values present after filling :: ', train.annual_income.isna().sum())



#Displaying values



train.annual_income.value_counts(dropna = False)
plt.figure(figsize = (20,8))

sns.boxplot(train.annual_income, data = train)
# Displaying total unique values



train.credit_score.value_counts(dropna = False)
plt.figure(figsize = (17,7))

sns.barplot(x = train.tax_liens, y = 'credit_score', data = train)
plt.figure(figsize = (17,7))

sns.barplot(x = train.term, y = 'credit_score', hue = train.years_in_current_job,data = train)
plt.figure(figsize = (17,7))

sns.barplot(x = 'years_in_current_job', y = 'credit_score', hue = train.tax_liens,data = train)
plt.figure(figsize = (17,7))

sns.barplot(x = 'home_ownership', y = 'credit_score', data = train)
plt.figure(figsize = (17,7))

sns.barplot(x = train.bankruptcies, y = 'credit_score',data = train)
plt.figure(figsize = (20,8))

sns.boxplot(x = train.credit_score, y = train.years_in_current_job 	, data = train)
plt.figure(figsize = (17,7))

sns.barplot(x = train.number_of_credit_problems, y = 'credit_score', data = train)
train.credit_score.value_counts(dropna = False)
mean_score2 = train.groupby('purpose')['credit_score']



mean_score2.agg(np.median)             # Since value are close so using this
train.credit_score = pd.Series(mean_score2.transform(lambda x : x.fillna(value = x.median())))



# Since one category has no values in barchart hence using ffill to fill it

#train.credit_score.fillna(method = 'ffill', inplace = True)



# Checking for NaN values present



print('Total NaN values present after filling :: ', train.credit_score.isna().sum())
#Displaying values



train.credit_score.value_counts(dropna = False)
# Checking for NaN values



train.agg(lambda x : x.isna().sum())
train_int_cols = list(train.select_dtypes(include = np.number).columns)

train_obj_cols = list(train.select_dtypes(include = np.object).columns) 
# Plotting all int/float columns boxplot



plt.figure(figsize = (20,15))

sns.boxplot(data = train[train_int_cols], orient = 'h')
train_int_cols
# dropping last item

train_int_cols_temp = train_int_cols[ :-1]

train_int_cols_temp



train[train_int_cols_temp]= train[train_int_cols_temp].transform(lambda x : np.log(x + 1))

train[train_int_cols_temp].isna().sum()
# Plotting all int/float columns boxplot



plt.figure(figsize = (20,15))

sns.boxplot(data = train[train_int_cols], orient = 'h')
cols = ['annual_income', 'maximum_open_credit', 'current_loan_amount', 'current_credit_balance', 'monthly_debt']



train[cols].plot(kind = 'box', figsize = (20,15))
train[train_int_cols].describe().T
train[train_obj_cols].isna().sum()
dummy = pd.get_dummies(train[train_obj_cols])

dummy.T
train_final = pd.concat([train[train_int_cols], dummy], axis = 1)

train_final
x = train_final.drop('credit_default', axis = 1)

y = train_final.credit_default
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble  import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
model_dt = DecisionTreeClassifier(random_state = 10)

model_rf = RandomForestClassifier(random_state = 11)

model_knn = KNeighborsClassifier(n_neighbors = 10, n_jobs = -1)

#model_lr = LogisticRegression(random_state = 12)

model_nb = GaussianNB()

model_svc = SVC(random_state = 14)
train_final.credit_default.value_counts()
from sklearn.metrics import accuracy_score, classification_report



score_values = []

def modelling(models, x_train, y_train, x_test, y_test):

    for model in models:

        model.fit(x_train, y_train)

        predict = model.predict(x_test)

        score = accuracy_score(y_test, predict)

        score_values.append(score)

        print(f"Accurace score of {model} is :: ", score)

        print(f"Classification Report is :: \n")

        print(classification_report(y_test, predict))

        print('*' * 90)



        

# using the function



models = [model_dt, model_knn, model_nb, model_rf, model_svc]

modelling(models, x_train, y_train, x_test, y_test)
from sklearn.metrics import roc_auc_score, roc_curve



# calculating log probabilities of models used



prob_dt = model_dt.predict_proba(x_test)

prob_rf = model_rf.predict_proba(x_test)

prob_knn = model_knn.predict_proba(x_test)

prob_nb = model_nb.predict_proba(x_test)

# Keep Probabilities of the positive class only.



prob_dt = prob_dt[:, 1]

prob_rf = prob_rf[:, 1]

prob_knn = prob_knn[:, 1]

prob_nb = prob_nb[:, 1]
# Calculating the auc score and displaying it



auc_dt = roc_auc_score(y_test, prob_dt)

auc_rf = roc_auc_score(y_test, prob_rf)

auc_knn = roc_auc_score(y_test, prob_knn)

auc_nb = roc_auc_score(y_test, prob_nb)



print('ROC- AUC Score for Decision Tree is :: %0.3f'%auc_dt)

print('ROC- AUC Score for Random Forest is :: %0.3f'%auc_rf)

print('ROC- AUC Score for KNN is :: %0.3f'%auc_knn)

print('ROC- AUC Score for Niave Bayes is :: %0.3f'%auc_nb)
# Getting the ROC Curve.



fpr_dt, tpr_dt, threshold_dt = roc_curve(y_test, prob_dt)

fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test, prob_rf)

fpr_knn, tpr_knn, threshold_knn = roc_curve(y_test, prob_knn)

fpr_nb, tpr_nb, threshold_nb = roc_curve(y_test, prob_nb)
# Plotting the ROC Curve for all models 



import matplotlib.pyplot as plt

%matplotlib inline



plt.figure(figsize = (15,7))

plt.plot(fpr_dt, tpr_dt, linewidth = 2, linestyle = 'dotted', label = 'Decision Tree')

plt.plot(fpr_rf, tpr_rf, linewidth = 2, linestyle = 'dashdot', label = 'Random Forest')

plt.plot(fpr_knn, tpr_knn, linewidth = 2, linestyle = 'dashed', label = 'K-Neighbours')

plt.plot(fpr_nb, tpr_nb, linewidth = 2, linestyle = '-', label = 'Naive Bayes')



plt.title('Receiver Operating Characteristic Curve (ROC AUC) Curve', fontsize = 25)

plt.xlabel('False Positive Rate', fontsize = 20)

plt.ylabel('True Positive Rate', fontsize = 20)

plt.legend(fontsize = 16)

plt.show()
from scipy.stats import mode



# Filling with mode value



test.bankruptcies =  test.bankruptcies.agg(lambda x : x.fillna(value = mode(x).mode[0]))



# Checking for NaN values present



print('Total NaN values present after filling :: ', test.bankruptcies.isna().sum())

# Checking values before the filled values 



test['months_since_last_delinquent'].value_counts(dropna = False)
# Grouping months_since_last_delinquent columns as per  'purpose', 'home_ownership'



mean_score_t = test.groupby(['purpose', 'home_ownership'])['months_since_last_delinquent']

mean_score_t.agg([np.median])
test.loc[: , 'months_since_last_delinquent'] = mean_score_t.transform(lambda x : x.fillna(x.median()))



#Using ffill to fill any left missing value



test.months_since_last_delinquent = test.months_since_last_delinquent.fillna(method = 'ffill')

# Checking for NaN values present



print('Total NaN values present after filling :: ', test.months_since_last_delinquent.isna().sum())
print('Total NaN values present before filling :: ', test.years_in_current_job.isna().sum())



# Displaying total unique values present 



test.years_in_current_job.value_counts(dropna = False)
### Since NaN values present hence filling it using ffill and bfill 



test.years_in_current_job.fillna(method = 'ffill', inplace = True)

test.years_in_current_job.fillna(method = 'bfill', inplace = True)





# Checking for NaN values present

print('Total NaN values present after filling :: ', test.years_in_current_job.isna().sum())



test.years_in_current_job.value_counts(dropna = False)
# Displaying total unique values



test.annual_income.value_counts(dropna = False)
#tranforming/applying mean values to corresponding rows with missing values



test.annual_income = test.groupby('years_in_current_job')['annual_income'].transform(lambda x : x.fillna(x.mean()))



# Checking for NaN values present



print('Total NaN values present after filling :: ', test.annual_income.isna().sum())



#Displaying values



test.annual_income.value_counts(dropna = False)
# Displaying total unique values



test.credit_score.value_counts(dropna = False)
mean_score_t2 = test.groupby('purpose')['credit_score']



mean_score_t2.agg(np.median)             # Since value are close so using this
test.credit_score = mean_score_t2.transform(lambda x : x.fillna(value = x.median()))



# Since one category has no values in barchart hence using ffill to fill it

#train.credit_score.fillna(method = 'ffill', inplace = True)



# Checking for NaN values present



print('Total NaN values present after filling :: ', test.credit_score.isna().sum())
#Displaying values



test.credit_score.value_counts(dropna = False)
# Checking for NaN values



test.agg(lambda x : x.isna().sum())
test_int_cols = list(test.select_dtypes(include = np.number).columns)

test_obj_cols = list(test.select_dtypes(include = np.object).columns) 
# Plotting all int/float columns boxplot



plt.figure(figsize = (20,15))

sns.boxplot(data = test[test_int_cols], orient = 'h')
test_int_cols
test[test_int_cols]= test[test_int_cols].transform(lambda x : np.log(x + 1))

test[test_int_cols].isna().sum()
# Plotting all int/float columns boxplot



plt.figure(figsize = (20,15))

sns.boxplot(data = test[test_int_cols], orient = 'h')
test[test_int_cols].describe().T
test[test_obj_cols].isna().sum()
dummy_t = pd.get_dummies(test[test_obj_cols])

dummy_t.T
test_final = pd.concat([test[test_int_cols], dummy_t], axis = 1)

test_final.T