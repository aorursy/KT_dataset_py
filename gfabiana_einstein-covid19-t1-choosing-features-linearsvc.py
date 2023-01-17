import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# packages 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import ExtraTreesRegressor
# loading file

df = pd.read_excel(r'/kaggle/input/covid19/dataset.xlsx')
# initial exploring data

df.head()
df.columns
# Patient ID isn't necessary

df = df.drop(columns='Patient ID')
# Are there blank?

np.where(df.applymap(lambda x: x == ''))
# Searching for NaNs

df.info(verbose=True, null_counts=True)
def intitial_eda_checks(df):

    '''

    Thanks to: https://github.com/FredaXin/blog_posts/blob/master/creating_functions_for_EDA.md

    Takes df

    Checks nulls

    '''

    if df.isnull().sum().sum() > 0:

        mask_total = df.isnull().sum().sort_values(ascending=False) 

        total = mask_total[mask_total > 0]



        mask_percent = df.isnull().mean().sort_values(ascending=False) 

        percent = mask_percent[mask_percent > 0] 



        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    

        print(f'Total and Percentage of NaN:\n {missing_data}')

    else: 

        print('No NaN found.')

        

intitial_eda_checks(df)
def view_columns_w_many_nans(df, missing_percent):

    '''

    Thanks to: https://github.com/FredaXin/blog_posts/blob/master/creating_functions_for_EDA.md

    Checks which columns have over specified percentage of missing values

    Takes df, missing percentage

    Returns columns as a list

    '''

    mask_percent = df.isnull().mean()

    series = mask_percent[mask_percent > missing_percent]

    columns = series.index.to_list()

    print(columns) 

    return columns
list_of_cols = view_columns_w_many_nans(df, .9)

df0 = df.drop(columns=list_of_cols)

#Here they are: 
# the dataset for task 1

df1 = df0.drop(columns=['Patient addmited to regular ward (1=yes, 0=no)', 

                        'Patient addmited to semi-intensive unit (1=yes, 0=no)',

                       'Patient addmited to intensive care unit (1=yes, 0=no)'])
# replacing text data to numbers - negative: 0, positive: 1.

df1['SARS-Cov-2 exam result'] = df1['SARS-Cov-2 exam result'].replace({'negative': 0, 'positive': 1})
# get dummies because machine learning algorithms prefers numbers!

df1_dummy = pd.get_dummies(df1)
# correlation 

corr_matrix_df1 = df1_dummy.corr()



corr_matrix_df1['SARS-Cov-2 exam result'].sort_values(ascending=False)
# New DataFrame with target and selected features to plot correlation

df1_new = df1_dummy[['SARS-Cov-2 exam result','Monocytes', 'Red blood Cells', 'Mean platelet volume ',

                    'Hemoglobin', 'Hematocrit','Basophils','Eosinophils', 'Platelets', 'Leukocytes']]
def heatmap_numeric_w_dependent_variable(df, dependent_variable):

    '''

    thanks to: https://github.com/FredaXin/blog_posts/blob/master/creating_functions_for_EDA.md

    Takes df, a dependant variable as str

    Returns a heatmap of all independent variables' correlations with dependent variable 

    '''

    plt.figure(figsize=(8, 10))

    g = sns.heatmap(df.corr()[[dependent_variable]].sort_values(by=dependent_variable), 

                    annot=True, 

                    cmap='coolwarm', 

                    vmin=-1,

                    vmax=1) 

    return g
# plotting correlations with 'SARS-Cov-2 exam result' column

heatmap_numeric_w_dependent_variable(df1_new, 'SARS-Cov-2 exam result')
# defining features and target

X1 = df1_new.drop(['SARS-Cov-2 exam result'], axis=1)



y1 = df1_new['SARS-Cov-2 exam result']
# imputation of missing values with multivariate imputation algorithm

imp = IterativeImputer(max_iter=10, random_state=0)



imp.fit(X1)



X1 = imp.transform(X1)
# scaling X1

scaler = MinMaxScaler()



scaler.fit(X1)



X1 = scaler.transform(X1)
# train test split



X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=33)     
# LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)



clf.fit(X1_train, y1_train)



y1_pred = clf.predict(X1_test)
# cross-validation 

all_accuracies_clf_1 = cross_val_score(estimator=clf, X=X1_train, 

                                 y=y1_train, cv=5)
# mean accuracy and standard deviation

mean_acc_corrint = all_accuracies_clf_1.mean()*100

std_acc_corrint = all_accuracies_clf_1.std()
# defining features and target

X2 = df1_dummy.drop(['SARS-Cov-2 exam result'], axis=1)



y2 = df1_dummy['SARS-Cov-2 exam result']
# I'll need this to apply SelectKBest algorithm 

X2_columns = X2.columns
# imputation of missing values with multivariate imputation algorithm

imp = IterativeImputer(max_iter=10, random_state=0)



imp.fit(X2)



X2 = imp.transform(X2)
# scaling X2

scaler = MinMaxScaler()



scaler.fit(X2)



X2 = scaler.transform(X2)
# transforming X and y to DataFrame to apply SelectKbest and return the selected columns

y2 = pd.DataFrame(data=y2, columns=['SARS-Cov-2 exam result'])



X2 = pd.DataFrame(data=X2, columns=X2_columns) 
# applying

selector = SelectKBest(chi2, k=5)

selector.fit(X2, y2)
# selected columns

# thanks to: https://stackoverflow.com/questions/46927545/get-feature-names-of-selectkbest-function-python

X_new = selector.transform(X2)

print(X_new.shape)



X2.columns[selector.get_support(indices=True)]



vector_names = list(X2.columns[selector.get_support(indices=True)])

print(vector_names)



X2.columns[selector.get_support(indices=True)].tolist()
# assigning target

y_new = np.ravel(y2)
# train test split



X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, 

                                                                    random_state=33)                                                                     

                                                                   
# LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)



clf.fit(X_new_train, y_new_train)



y_new_pred = clf.predict(X_new_test)
# cross-validation 

all_accuracies_clf_2 = cross_val_score(estimator=clf, X=X_new_train, 

                                 y=y_new_train, cv=5)
# mean accuracy and standard deviation

mean_acc_selector = all_accuracies_clf_2.mean()*100

std_acc_selector = all_accuracies_clf_2.std()
print('The average accuracy of the first model is', mean_acc_corrint, '%')

print('and the standard deviation is', std_acc_corrint,'.')

print()

print('The average accuracy of the second model is', mean_acc_selector, '%')

print('and the standard deviation is', std_acc_selector, '.')
print('STAY HOME, IF YOU CAN.')