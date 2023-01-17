# Loading libraries needed for analysis

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate

import time

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline

random_seed = 316

pd.set_option('display.max_columns', None)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('../input/shopee-workshop-building-ann/HR-Employee-Attrition_train.csv')

test_df = pd.read_csv('../input/shopee-workshop-building-ann/HR-Employee-Attrition_test.csv')

print('Train data:', train_df.shape)

print('Test data:', test_df.shape)
train_df.head()
train_df.info()
test_df.info()
train_df.describe()
sns.countplot(train_df['Attrition'])

plt.show()
train_df['Attrition'].value_counts(normalize = True)
col_summary = pd.DataFrame(train_df.columns, columns = ['Column'])

na_list = []

unique_list = []

dtype_list = []



for col in train_df.columns:

    na_list.append(train_df[col].isna().sum())

    unique_list.append(train_df[col].nunique())

    dtype_list.append(train_df[col].dtype)

    

col_summary['Missing values'] = na_list

col_summary['Unique values'] = unique_list

col_summary['Data type'] = dtype_list

col_summary
train_df.drop(['StandardHours', 'Over18', 'EmployeeCount'], axis = 1, inplace = True)

test_df.drop(['StandardHours', 'Over18', 'EmployeeCount'], axis = 1, inplace = True)
num_cols = train_df._get_numeric_data()

num_cols
cat_cols = train_df[[col for col in train_df.columns if col not in num_cols.columns]]

cat_cols
id_col = 'EmployeeNumber'

target_col = 'Attrition'
target_col_mapper = {

    'Yes': 1,

    'No': 0

}



train_df['Attrition'] = train_df['Attrition'].map(target_col_mapper)
# add 'age range' column

train_df['age_band'] = pd.cut(train_df['Age'], 4, labels = False)

test_df['age_band'] = pd.cut(test_df['Age'], 4, labels = False)



# add 'daily rate range' column

train_df['daily_rate_band'] = pd.qcut(train_df['DailyRate'], 5, labels = False)

test_df['daily_rate_band'] = pd.qcut(test_df['DailyRate'], 5, labels = False)



# add 'distance from home range' column

train_df['distance_from_home_band'] = pd.qcut(train_df['DistanceFromHome'], 5, labels = False)

test_df['distance_from_home_band'] = pd.qcut(test_df['DistanceFromHome'], 5, labels = False)



# add 'hourly rate range' column

train_df['hourly_rate_band'] = pd.qcut(train_df['HourlyRate'], 5, labels = False)

test_df['hourly_rate_band'] = pd.qcut(test_df['HourlyRate'], 5, labels = False)



# add 'monthly income range' column

train_df['monthly_income_band'] = pd.qcut(train_df['MonthlyIncome'], 5, labels = False)

test_df['monthly_income_band'] = pd.qcut(test_df['MonthlyIncome'], 5, labels = False)



# add 'monthly rate range' column

train_df['monthly_rate_band'] = pd.qcut(train_df['MonthlyRate'], 5, labels = False)

test_df['monthly_rate_band'] = pd.qcut(test_df['MonthlyRate'], 5, labels = False)



# add 'num companies worked range' column

train_df['num_companies_worked_band'] = pd.qcut(train_df['NumCompaniesWorked'], 3, labels = False)

test_df['num_companies_worked_band'] = pd.qcut(test_df['NumCompaniesWorked'], 3, labels = False)



# add 'percent salary hike range' column

train_df['percent_salary_hike_band'] = pd.qcut(train_df['PercentSalaryHike'], 3, labels = False)

test_df['percent_salary_hike_band'] = pd.qcut(test_df['PercentSalaryHike'], 3, labels = False)



# add 'total working years range' column

train_df['total_working_years_band'] = pd.qcut(train_df['TotalWorkingYears'], 3, labels = False)

test_df['total_working_years_band'] = pd.qcut(test_df['TotalWorkingYears'], 3, labels = False)



# add 'training times last year range' column

train_df['training_times_last_year_band'] = pd.qcut(train_df['TrainingTimesLastYear'], 3, labels = False)

test_df['training_times_last_year_band'] = pd.qcut(test_df['TrainingTimesLastYear'], 3, labels = False)



# add 'years at company range' column

train_df['years_at_company_band'] = pd.qcut(train_df['YearsAtCompany'], 3, labels = False)

test_df['years_at_company_band'] = pd.qcut(test_df['YearsAtCompany'], 3, labels = False)



# add 'years in current role range' column

train_df['years_in_current_role_band'] = pd.qcut(train_df['YearsInCurrentRole'], 3, labels = False)

test_df['years_in_current_role_band'] = pd.qcut(test_df['YearsInCurrentRole'], 3, labels = False)



# add 'years since last promotion range' column

train_df['years_since_last_promotion_band'] = pd.qcut(train_df['YearsSinceLastPromotion'], 3, duplicates='drop', labels = False)

test_df['years_since_last_promotion_band'] = pd.qcut(test_df['YearsSinceLastPromotion'], 3, duplicates='drop', labels = False)



# add 'years with curr manager range' column

train_df['years_with_curr_manager_band'] = pd.qcut(train_df['YearsWithCurrManager'], 3, labels = False)

test_df['years_with_curr_manager_band'] = pd.qcut(test_df['YearsWithCurrManager'], 3, labels = False)
# add 'no_company_before' column

train_df['no_company_before'] = train_df['NumCompaniesWorked'].apply(lambda x: 'Yes' if x == 0 else 'No')

test_df['no_company_before'] = test_df['NumCompaniesWorked'].apply(lambda x: 'Yes' if x == 0 else 'No')



# add 'worked_less_than_a_year' column

train_df['worked_less_than_a_year'] = train_df['TotalWorkingYears'].apply(lambda x: 'Yes' if x == 0 else 'No')

test_df['worked_less_than_a_year'] = test_df['TotalWorkingYears'].apply(lambda x: 'Yes' if x == 0 else 'No')



# add 'no_training_last_year' column

train_df['no_training_last_year'] = train_df['TrainingTimesLastYear'].apply(lambda x: 'Yes' if x == 0 else 'No')

test_df['no_training_last_year'] = test_df['TrainingTimesLastYear'].apply(lambda x: 'Yes' if x == 0 else 'No')



# add 'less_than_a_year_at_company' column

train_df['less_than_a_year_at_company'] = train_df['YearsAtCompany'].apply(lambda x: 'Yes' if x == 0 else 'No')

test_df['less_than_a_year_at_company'] = test_df['YearsAtCompany'].apply(lambda x: 'Yes' if x == 0 else 'No')



# add 'never_promoted' column

train_df['never_promoted'] = train_df['YearsSinceLastPromotion'].apply(lambda x: 'Yes' if x == 0 else 'No')

test_df['never_promoted'] = test_df['YearsSinceLastPromotion'].apply(lambda x: 'Yes' if x == 0 else 'No')



# add 'less_than_a_year_with_manager' column

train_df['less_than_a_year_with_manager'] = train_df['YearsWithCurrManager'].apply(lambda x: 'Yes' if x == 0 else 'No')

test_df['less_than_a_year_with_manager'] = test_df['YearsWithCurrManager'].apply(lambda x: 'Yes' if x == 0 else 'No')



# add 'no_business_travel' column

train_df['no_business_travel'] = train_df['BusinessTravel'].apply(lambda x: 'Yes' if x == 'Non-Travel' else 'No')

test_df['no_business_travel'] = test_df['BusinessTravel'].apply(lambda x: 'Yes' if x == 'Non-Travel' else 'No')



# drop columns

drop_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

train_df.drop(drop_cols, axis=1, inplace=True)

test_df.drop(drop_cols, axis=1, inplace=True)



print(train_df.shape, test_df.shape)
# nominal data

nom_cols = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']



# ordinal data

ord_cols = ['BusinessTravel', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']



# additional features

nom_cols.append('no_company_before')

nom_cols.append('worked_less_than_a_year')

nom_cols.append('no_training_last_year')

nom_cols.append('less_than_a_year_at_company')

nom_cols.append('never_promoted')

nom_cols.append('less_than_a_year_with_manager')

nom_cols.append('no_business_travel')



binned_feats = ['age_band', 'daily_rate_band', 'distance_from_home_band', 'hourly_rate_band', 'monthly_income_band', 'monthly_rate_band', 'num_companies_worked_band', 'percent_salary_hike_band', 'total_working_years_band', 'training_times_last_year_band', 'years_at_company_band', 'years_in_current_role_band', 'years_since_last_promotion_band', 'years_with_curr_manager_band']

for feat in binned_feats:

    ord_cols.append(feat)
for feat in nom_cols + ord_cols:

    print('Attrition Correlation by:', feat)

    print(train_df[[feat, target_col]].groupby(feat, as_index=False).mean().sort_values(by=target_col, ascending=False))

    print('-'*50)



# for feat in :

#     display(train_df.groupby(feat)[target_col].mean().to_frame().style.background_gradient(cmap='summer_r'))
train_df.corr()[target_col]
# encode nominal features

def cat_to_dummy(train, test):

    train_d = pd.get_dummies(train)

    test_d = pd.get_dummies(test)

    return train_d, test_d



train_nom, test_nom = cat_to_dummy(train_df[nom_cols], test_df[nom_cols])



# drop original nominal columns

train_df.drop(nom_cols, axis = 1, inplace = True)

test_df.drop(nom_cols, axis = 1, inplace = True)



# concatenate encoded nominal columns

train_df = pd.concat([train_df, train_nom], axis = 1)

test_df = pd.concat([test_df, test_nom], axis = 1)



train_nom
train_df['BusinessTravel'].value_counts()
business_travel_mapper = {

    'Travel_Rarely': 1,

    'Travel_Frequently': 2,

    'Non-Travel': 0

}



train_df['BusinessTravel'] = train_df['BusinessTravel'].map(business_travel_mapper)

test_df['BusinessTravel'] = test_df['BusinessTravel'].map(business_travel_mapper)
X_train, X_test, y_train, y_test = train_test_split(train_df.drop([target_col, id_col], axis=1), train_df[target_col], test_size=0.3, random_state=random_seed, stratify=train_df[target_col])

print('Train data:', X_train.shape)

print('Test data:', X_test.shape)
# scaler = StandardScaler()

# # scaler = MinMaxScaler(feature_range=(-1,1))

# # numerical_cols = [ncol for ncol in train_df.columns if ncol not in nom_cols + ord_cols + [target_col, id_col]]

# # numerical_cols = [ncol for ncol in train_df.columns if ncol not in nom_cols + [target_col, id_col]]

# numerical_cols = ord_cols



# # fit only on training data

# X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

# X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])



# X_train.head()
train_df.to_csv('train_scaled.csv', index = False)

test_df.to_csv('test_scaled.csv', index = False)
%%time

start_time = time.time()



NN = MLPClassifier(max_iter=1000, random_state=random_seed)

parameter_space = {

    # 'hidden_layer_sizes': [(64, 64, 64), (63, 44, 63), (63, 63, 63), (50,50,50), (49, 34, 49), (49, 49, 49)], # based on the number of features + 1

    'hidden_layer_sizes': [(102, 102, 102), (101, 67, 101), (38, 38, 38), (37, 26, 37)],

    'activation': ['tanh', 'relu','logistic'],

    'solver': ['adam', 'lbfgs'],

    'alpha': [0.001, 0.01],

    'learning_rate': ['constant','adaptive']

}



scoring = ['f1', 'recall', 'precision', 'accuracy']

clf = GridSearchCV(NN, parameter_space, n_jobs=-1, scoring=scoring, refit='f1', cv=10)

clf.fit(X_train, y_train)

print('Time taken for training the model:', "{:.2f}".format((time.time() - start_time) / 60))
print('Best parameters found:\n', clf.best_params_)
clf.best_estimator_
clf.best_score_



# baseline

# 0.5919221341710098
# best scores

cv_results = pd.DataFrame.from_dict(clf.cv_results_)

cv_results = cv_results[['mean_test_f1', 'rank_test_f1', 'mean_test_recall', 'rank_test_recall', 'mean_test_precision', 'mean_test_accuracy', 'rank_test_accuracy', 'params', 'std_test_f1', 'std_test_accuracy']]
# f1

cv_results.loc[cv_results['rank_test_f1']==1]
# recall

cv_results[cv_results['rank_test_recall']==1]
# accuracy

cv_results[cv_results['rank_test_accuracy']==1]
%%time

start_time = time.time()



NN = clf.best_estimator_

cv_results = cross_validate(NN, X_train, y_train, scoring=scoring, cv=10, return_train_score=True)



print('CV scores on training set:')

print("Average Recall score: %0.4f (+/- %0.4f)" % (cv_results['train_recall'].mean(), cv_results['train_recall'].std() * 2))

print("Average F1 score: %0.4f (+/- %0.4f)" % (cv_results['train_f1'].mean(), cv_results['train_f1'].std() * 2))

print("Average Precision score: %0.4f (+/- %0.4f)" % (cv_results['train_precision'].mean(), cv_results['train_precision'].std() * 2))

print("Average Accuracy score: %0.4f (+/- %0.4f)" % (cv_results['train_accuracy'].mean(), cv_results['train_accuracy'].std() * 2))

print()

print('CV scores on test set:')

print("Average Recall score: %0.4f (+/- %0.4f)" % (cv_results['test_recall'].mean(), cv_results['test_recall'].std() * 2))

print("Average F1 score: %0.4f (+/- %0.4f)" % (cv_results['test_f1'].mean(), cv_results['test_f1'].std() * 2))

print("Average Precision score: %0.4f (+/- %0.4f)" % (cv_results['test_precision'].mean(), cv_results['test_precision'].std() * 2))

print("Average Accuracy score: %0.4f (+/- %0.4f)" % (cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std() * 2))

print()

print('Time taken for cross-validation:', "{:.2f}".format((time.time() - start_time) / 60))



# baseline

# CV scores on test set:

# Average Recall score: 0.5029 (+/- 0.2099)

# Average F1 score: 0.5919 (+/- 0.2097)

# Average Precision score: 0.7506 (+/- 0.3197)

# Average Accuracy score: 0.8832 (+/- 0.0732)
NN.fit(X_train, y_train)
def metrics(true, pred):

    recall = recall_score(true, pred)

    f1score = f1_score(true, pred)

    precision = precision_score(true, pred)

    accuracy = accuracy_score(true, pred)

    print(f'recall: {recall}, f1-score: {f1score}, precision: {precision}, accuracy: {accuracy}')
y_true, y_pred = y_train, NN.predict(X_train)

print('Results on train set:')

print(confusion_matrix(y_true, y_pred))

metrics(y_true, y_pred)
y_true, y_pred = y_test, NN.predict(X_test)

print('Results on test set:')

print(confusion_matrix(y_true, y_pred))

metrics(y_true, y_pred)



# baseline

# recall: 0.43478260869565216, f1-score: 0.4878048780487805, precision: 0.5555555555555556, accuracy: 0.8467153284671532
pred = NN.predict(test_df.drop([id_col], axis=1))
submission = pd.DataFrame({

    'EmployeeNumber': test_df[id_col],

    'Attrition': ['Yes' if i == 1 else 'No' for i in pred]

})

submission['Attrition'].value_counts()
submission.to_csv('submission.csv', index = False)