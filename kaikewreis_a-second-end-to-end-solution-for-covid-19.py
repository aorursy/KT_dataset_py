# Standard modules

import numpy as np

import pandas as pd



# Missing Analysis & Pre-processing

from sklearn.preprocessing import StandardScaler

import missingno as msno



# Missing imputation

from sklearn.experimental import enable_iterative_imputer 

from sklearn.impute import IterativeImputer as MICE



# Model development

from imblearn.over_sampling import SMOTE

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split 

from sklearn.metrics import classification_report, confusion_matrix



# Graphical modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Display module (show all dataset)

pd.options.display.max_columns = None

pd.options.display.max_rows = None

from IPython.display import display
# Importing raw dataset

dataset_raw = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
# Drop unused columns

dataset_raw.drop('Patient ID', axis=1, inplace=True)
# Change column names: uppercase become lowercase and '_' becomes ' '

new_columns = list()

# Loop over all columns

for col in dataset_raw.columns:

    new_columns.append(col.lower().replace(' ','_'))

# Modify dataset columns

dataset_raw.columns = new_columns
# List to keep all categorical feature

cat_vars = list()

# Loop to evaluate if it's categorical

for j in dataset_raw.columns:

    if len(dataset_raw[j].unique()) <= 5:

        cat_vars.append(j)

        print('Discrete variable - ',j)

print('This dataset have ',len(cat_vars), 'of ', len(dataset_raw.columns),' discrete variables with 5 or less categories.')
# Print categories for 'patient_age_quantile'

dataset_raw['patient_age_quantile'].unique()
# Eval categories for all discrete features

for c in cat_vars:

    print(c, ' - ', dataset_raw[c].unique())
# Conclusion 1, 2, 3 and 5 - Dropp columns

categorical_features_to_drop = ['mycoplasma_pneumoniae', 'd-dimer', 'partial_thromboplastin_time\xa0(ptt)\xa0','prothrombin_time_(pt),_activity', 'urine_-_sugar', 'urine_-_yeasts',

                                'urine_-_granular_cylinders', 'urine_-_hyaline_cylinders', 'myeloblasts', 'fio2_(venous_blood_gas_analysis)', 'urine_-_esterase','parainfluenza_2',

                                'urine_-_bile_pigments', 'urine_-_ketone_bodies', 'urine_-_protein', 'urine_-_urobilinogen', 'urine_-_nitrite']

dataset_raw.drop(categorical_features_to_drop, axis=1, inplace=True)
# Conclusion 4 - Input NaN for some columns in place of not_done

dataset_raw['strepto_a'].replace('not_done',np.NaN, inplace=True)

dataset_raw['urine_-_hemoglobin'].replace('not_done',np.NaN, inplace=True)
# Verify discrete features columns after drop process

cat_vars = list()

for j in dataset_raw.columns:

    if len(dataset_raw[j].unique()) <= 5:

        cat_vars.append(j)

        print('Discrete variable - ',j)

print('This dataset have ',len(cat_vars), 'of ', len(dataset_raw.columns),' discrete variables with 5 or less categories.\n\n')



# Verify discrete classes for each discrete var

for c in cat_vars:

    print(c, ' - ', dataset_raw[c].unique())
# Verify psedo discrete variables

for i in ['promyelocytes', 'myelocytes', 'metamyelocytes', 'vitamin_b12']:

    print(i, ' - ', 100*round(dataset_raw[i].isna().sum()/len(dataset_raw),3), ' % of missing, having ',len(dataset_raw)-dataset_raw[i].isna().sum(),' complete rows.')
dataset_raw.drop(['promyelocytes', 'myelocytes', 'metamyelocytes', 'vitamin_b12'], axis=1, inplace=True)
# How many nan have each column

nan_per_column = pd.DataFrame(dataset_raw.isna().sum(),columns=['nanValues']).reset_index()



# Calculate NaN % for each feature

for i in range(0,len(nan_per_column)):

    nan_per_column.loc[i, 'nanValuesPct'] = 100*round(nan_per_column.loc[i, 'nanValues']/len(dataset_raw),3)
# Plot - % of missing rows for each column

plt.figure(figsize=(30,15))

sns.barplot(x="index", y="nanValuesPct", data=nan_per_column)

plt.xlabel('Variables', fontsize=20)

plt.ylabel('Missing %', fontsize=20)

plt.title('Missing Data Plot', fontsize=30)

plt.yticks([0,10,20,30,40,50,60,70,80,90,100])

plt.xticks(rotation=90);
len(dataset_raw.dropna(how='any'))
# Print missing pct per column uniques

print(np.unique(nan_per_column['nanValuesPct']))
# conters

t1 = 0

t2 = 0

t3 = 0



for i in range(0,len(nan_per_column)):

    if nan_per_column.loc[i, 'nanValuesPct'] <= 76:

        t1 += 1

    elif nan_per_column.loc[i, 'nanValuesPct'] > 76 and nan_per_column.loc[i, 'nanValuesPct'] < 90.9:

        t2 += 1

    elif nan_per_column.loc[i, 'nanValuesPct'] >= 90.9:

        t3 += 1

print('If I keep respectively based in T1, T1+T2, T3 and without threshold: ',t1,t1+t2,t3,t1+t2+t3)
# threshold proposed for features, keep in mind that this dataset have 4 target values

rows_50_threshold = int((len(dataset_raw.columns)-4)/2)



# Eval row limit

print('50% of a sample in this dataset is: ', rows_50_threshold)
# Before removing

rc = 0

possible_rows = list()

for i in range(0, len(dataset_raw)):

    if dataset_raw.iloc[i].isnull().sum() >= rows_50_threshold:

        rc += 1

    else:

        possible_rows.append(i)

print('Of ', len(dataset_raw), ' total rows, I have ', rc,' rows with more than 50% of missing data, giving a ', len(dataset_raw)-rc,' rows that follows the recommendation for imputation\n')



# Number of columns with at least 10% of missing data to impute (being generous)

df_aux = dataset_raw.loc[possible_rows]

miss_less_10 = 0

for c in dataset_raw.columns:

    if (df_aux[c].isna().sum()/len(possible_rows)) == 0.0:

        pass

    elif (df_aux[c].isna().sum()/len(possible_rows)) < 0.10:

        miss_less_10 += 1

        print(c, 'with ',df_aux[c].isna().sum(),' of missing data could be imputed')



# Case proportions

print('\n\nPositive samples: ', sum(df_aux['sars-cov-2_exam_result'] == 'positive'),' and Negative samples:', sum(df_aux['sars-cov-2_exam_result'] == 'negative'), 

      ' - proportion [%]: ', round(100*sum(df_aux['sars-cov-2_exam_result'] == 'positive')/len(df_aux),2))
# Possible variables to keep

cols_to_keep = list()

for i in range(0,len(nan_per_column)):

    if nan_per_column.loc[i, 'nanValuesPct'] < 90.9:

        cols_to_keep.append(nan_per_column.loc[i,'index'])

# Print how many variables I will keep  

print('Filter 1 - columns: ', len(cols_to_keep))


# Creating a aux dataframe with filtered columns, keeping control over raw dataset

df_aux = dataset_raw[cols_to_keep]

# Eval limit

rows_50_threshold = int((len(df_aux.columns)-4)/2)

print('50% of a sample in this dataset is: ', rows_50_threshold)
# Threshold T1 + T2 applying...

rc = 0

possible_rows = list()

for i in range(0, len(df_aux)):

    if df_aux.iloc[i].isnull().sum() >= rows_50_threshold:

        rc += 1

    else:

        possible_rows.append(i)

print('Of ', len(df_aux), ' total rows, I have ', rc,' rows with more than 50% of missing data, giving a ', len(df_aux)-rc,' rows that follows the recommendation for imputation\n')

# Number of columns with at least 10% of missing data to impute (being generous)

df_aux = df_aux.loc[possible_rows]

miss_less_15 = 0

for c in df_aux.columns:

    if (df_aux[c].isna().sum()/len(possible_rows)) == 0.0:

        pass

    elif (df_aux[c].isna().sum()/len(possible_rows)) < 0.10:

        miss_less_15 += 1

        print(c, 'with ',df_aux[c].isna().sum(),' of missing data could be imputed')

# Case proportions

print('\n\nPositive samples: ', sum(df_aux['sars-cov-2_exam_result'] == 'positive'),' and Negative samples:', sum(df_aux['sars-cov-2_exam_result'] == 'negative'), 

      ' - proportion: ', round(100*sum(df_aux['sars-cov-2_exam_result'] == 'positive')/len(df_aux),2))
# Renaming df

dataset = df_aux

dataset.index = range(0,len(dataset))
# Counter

rc = 0

# Loop to verify complete rows

for i in range(0, len(dataset)):

    if dataset.iloc[i].isnull().sum() == 0.0:

        rc += 1

print('complete rows: ', rc)
# Verify categorical columns after dropping

cat_vars = list()

for j in dataset.columns:

    if len(dataset[j].unique()) <= 5:

        cat_vars.append(j)

print('This dataset have now',len(cat_vars), ' categorical variables of ',len(dataset),'\n')



for c in cat_vars:

    print(c, ' - ', dataset[c].unique())
dataset.drop('bordetella_pertussis', axis=1, inplace=True)
# Verify categorical columns after dropping

cat_vars = list()

for j in dataset.columns:

    if len(dataset[j].unique()) <= 5:

        cat_vars.append(j)
# Changing string values to 

for j in cat_vars:

    if 'positive' in list(dataset[j].unique()):

        dataset[j].replace('positive',1, inplace=True)

    elif 'detected' in list(dataset[j].unique()):

        dataset[j].replace('detected',1, inplace=True)

    if 'negative' in list(dataset[j].unique()):

        dataset[j].replace('negative',0, inplace=True) 

    elif 'not_detected' in list(dataset[j].unique()):

        dataset[j].replace('not_detected',0, inplace=True)
# Verify categorical columns after dropping

cat_vars = list()

for j in dataset.columns:

    if len(dataset[j].unique()) <= 5:

        cat_vars.append(j)

print('This dataset have now',len(cat_vars), ' categorical variables of ',len(dataset),'\n')



# Print results

for c in cat_vars:

    print(c, ' - ', dataset[c].unique())
# If a value is

for i in range(0, len(dataset)):

    if pd.isna(dataset.loc[i, 'influenza_a,_rapid_test']) is False and pd.isna(dataset.loc[i, 'influenza_a']) is False:

        if dataset.loc[i, 'influenza_a,_rapid_test'] != dataset.loc[i, 'influenza_a']:

            print(i, 'sample have different for influenza A: ',dataset.loc[i, 'influenza_a'],' and test A: ', dataset.loc[i, 'influenza_a,_rapid_test'])

    if pd.isna(dataset.loc[i, 'influenza_b,_rapid_test']) is False and pd.isna(dataset.loc[i, 'influenza_b']) is False:    

        if dataset.loc[i, 'influenza_b,_rapid_test'] != dataset.loc[i, 'influenza_b']:

            print(i, 'sample have different for influenza B: ',dataset.loc[i, 'influenza_b'],' and test B: ', dataset.loc[i, 'influenza_b,_rapid_test'])

    
# Dropping rapid test columns

dataset.drop(['influenza_a,_rapid_test', 'influenza_b,_rapid_test'], axis=1, inplace=True)
# Defining lists for each family column

detection_adenoviridae = ['adenovirus']

detection_coronaviridae = ['coronavirusoc43', 'coronavirus_hku1', 'coronavirusnl63', 'coronavirus229e']

detection_orthomyxoviridae = ['influenza_a', 'influenza_b', 'inf_a_h1n1_2009'] 

detection_paramyxoviridae = ['parainfluenza_1', 'parainfluenza_3', 'parainfluenza_4']

detection_picornaviridae = ['rhinovirus/enterovirus']

detection_pneumoviridae = ['respiratory_syncytial_virus', 'metapneumovirus']

groups_list = [detection_adenoviridae, detection_coronaviridae, detection_orthomyxoviridae, detection_paramyxoviridae, detection_picornaviridae, detection_pneumoviridae]

groups_cols = ['detection_adenoviridae', 'detection_coronaviridae', 'detection_orthomyxoviridae', 'detection_paramyxoviridae', 'detection_picornaviridae', 'detection_pneumoviridae']



# Create the new columns 

for family_group,family_col in zip(groups_list, groups_cols):

    for i in range(0, len(dataset)):

        for j in family_group:

            if pd.isna(dataset.loc[i, j]) is False: # If it's nan can crash the comparison

                if dataset.loc[i, j] == 1: # I need only one column to say if will be a 1 (family detection)

                    dataset.loc[i,family_col] = 1

                    break

                else:

                    dataset.loc[i,family_col] = 0
# Dropping old species columns

drop_species_cols = detection_adenoviridae + detection_coronaviridae + detection_orthomyxoviridae + detection_paramyxoviridae + detection_picornaviridae + detection_pneumoviridae

dataset.drop(drop_species_cols, inplace=True, axis=1)
# Verify if new columns are statical features

cat_vars = list()

for j in dataset.columns:

    if len(dataset[j].unique()) <= 5:

        print(j, ' - ', dataset[j].unique())

        cat_vars.append(j)

print('This dataset have now',len(cat_vars), ' discrete variables of ',len(dataset),'\n')
# Num features

num_features = ['hematocrit',

                'hemoglobin',

                'platelets',

                'mean_platelet_volume_',

                'red_blood_cells',

                'lymphocytes',

                'mean_corpuscular_hemoglobin_concentration\xa0(mchc)',

                'leukocytes',

                'basophils',

                'mean_corpuscular_hemoglobin_(mch)',

                'eosinophils',

                'mean_corpuscular_volume_(mcv)',

                'monocytes',

                'red_blood_cell_distribution_width_(rdw)']

# scaler object

scaler = StandardScaler()

dataset[num_features] = scaler.fit_transform(dataset[num_features])
# Eval missing value for each column again

nan_per_column = pd.DataFrame(dataset.isna().sum(),columns=['nanValues']).reset_index()



# Calculate NaN %

for i in range(0,len(nan_per_column)):

    nan_per_column.loc[i, 'nanValuesPct'] = 100*round(nan_per_column.loc[i, 'nanValues']/len(dataset),3)

    

# Plot - % of missing rows for each column

plt.figure(figsize=(20,10))

sns.barplot(x="index", y="nanValuesPct", data=nan_per_column)

plt.xlabel('Variables', fontsize=20)

plt.ylabel('Missing %', fontsize=20)

plt.title('Missing Data Plot after a cleaning phase', fontsize=30)

plt.yticks([0,10,20,30,40,50,60,70,80,90,100])

plt.xticks(rotation=90);
# Drop any NaN sample, creating a copy of my dataset

dataset_complete = dataset.dropna(how='any').copy()

dataset_complete.index = range(0, len(dataset_complete.index))



# Show number of rows

print('Complete rows: ', len(dataset_complete), '| Keeped % rows:',100*round(len(dataset_complete)/len(dataset),2))
print('Positive cases: ', sum(dataset_complete['sars-cov-2_exam_result'] == 1),' | Negative cases: ', sum(dataset_complete['sars-cov-2_exam_result'] == 0),'\n###')

print('Previous Positive cases: ', sum(dataset['sars-cov-2_exam_result'] == 1), ' | Keeped % for Positive cases: ',

      100*round(sum(dataset_complete['sars-cov-2_exam_result'] == 1)/sum(dataset['sars-cov-2_exam_result'] == 1), 2))
# Nullity Matrix

msno.matrix(dataset, sort='ascending', figsize=(15, 10));
# Heatmap plot

msno.heatmap(dataset);
# Create a copy of my original dataset

dataset_impute = dataset.copy()
# Apply MICE

dataset_impute_complete = MICE(max_iter=150, verbose=1, random_state=1206).fit_transform(dataset_impute.values)



# Turning into df again

dataset_impute = pd.DataFrame(data=dataset_impute_complete, columns=dataset_impute.columns, index=dataset_impute.index)
# Create a copy of choice dataset

dataset_choice = dataset_impute.copy()
# Data Task 1

dataset_task1 = dataset_choice.drop(['patient_addmited_to_semi-intensive_unit_(1=yes,_0=no)',

                                       'patient_addmited_to_intensive_care_unit_(1=yes,_0=no)',

                                       'patient_addmited_to_regular_ward_(1=yes,_0=no)'], axis=1)

# Data Task 2

dataset_task2 = dataset_choice.drop(['sars-cov-2_exam_result'], axis=1)
# Create a new unique target column for Task 2

targets_task2 = ['patient_addmited_to_regular_ward_(1=yes,_0=no)',

                 'patient_addmited_to_semi-intensive_unit_(1=yes,_0=no)',

                 'patient_addmited_to_intensive_care_unit_(1=yes,_0=no)']



## Evaluate the number of possibilities for three targets in a single column

patient_addmited_possibilities = list()

for i in range(0, len(dataset_task2)):

    possibility=str(int(dataset_task2.loc[i,targets_task2[0]])) + str(int(dataset_task2.loc[i,targets_task2[1]])) + str(int(dataset_task2.loc[i,targets_task2[2]]))

    patient_addmited_possibilities.append(possibility)



## Print result

print(sorted(set(patient_addmited_possibilities)))
## Create the new column

dataset_task2['patient_addmited_cats'] = patient_addmited_possibilities



## Change the new column to num values

for i in range(0, len(dataset_task2)):

    if dataset_task2.loc[i, 'patient_addmited_cats'] == '000':

        dataset_task2.loc[i, 'patient_addmited_cats'] = 0

    elif dataset_task2.loc[i, 'patient_addmited_cats'] == '100':

        dataset_task2.loc[i, 'patient_addmited_cats'] = 1

    elif dataset_task2.loc[i, 'patient_addmited_cats'] == '010':

        dataset_task2.loc[i, 'patient_addmited_cats'] = 2

    elif dataset_task2.loc[i, 'patient_addmited_cats'] == '001':

        dataset_task2.loc[i, 'patient_addmited_cats'] = 3

## See class distribution

dataset_task2['patient_addmited_cats'].value_counts()
# Drop residual cols in task 2 dataset

dataset_task2.drop(targets_task2, axis=1, inplace=True)
# Create list of feature columns for each dataset

features = list(dataset_task1.drop(['sars-cov-2_exam_result'], axis=1).columns)
# Separate numerical columns from cat columns to help some plots

num_features = ['hematocrit',

                'hemoglobin',

                'platelets',

                'mean_platelet_volume_',

                'red_blood_cells',

                'lymphocytes',

                'mean_corpuscular_hemoglobin_concentration\xa0(mchc)',

                'leukocytes',

                'basophils',

                'mean_corpuscular_hemoglobin_(mch)',

                'eosinophils',

                'mean_corpuscular_volume_(mcv)',

                'monocytes',

                'red_blood_cell_distribution_width_(rdw)']



cat_features = ['patient_age_quantile',

                'chlamydophila_pneumoniae',

                'detection_adenoviridae',

                'detection_coronaviridae',

                'detection_orthomyxoviridae',

                'detection_paramyxoviridae',

                'detection_picornaviridae',

                'detection_pneumoviridae']
# Eval 'sars-cov-2_exam_result' proportions

print('Positive case proportion - original dataset [%]: ', round(100*dataset_raw['sars-cov-2_exam_result'].value_counts()[1]/dataset_raw['sars-cov-2_exam_result'].value_counts().sum(),2))

print('Positive case proportion - complete dataset [%]: ', round(100*dataset_complete['sars-cov-2_exam_result'].value_counts()[1]/dataset_complete['sars-cov-2_exam_result'].value_counts().sum(),2))
sns.pairplot(dataset_task1[['sars-cov-2_exam_result']+num_features], hue='sars-cov-2_exam_result');
# Correlation calculation

spearman_corr = dataset_task1[num_features].corr('spearman')

# Plot

plt.figure(figsize=(20,10))

sns.heatmap(spearman_corr, annot = True);
# Atualize both datasets

dataset_task1.drop(['hematocrit'], axis=1, inplace=True)

dataset_task2.drop(['hematocrit'], axis=1, inplace=True)
# Update features columns

features = list(set(features).difference(set(['hematocrit'])))

num_features = list(set(num_features).difference(set(['hematocrit'])))
print('For both tasks I have ',len(features),' features.')
# PLOT - Barplots over our variables

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,15))

r = 0 # Index row

c = 0 # Index col

for f in cat_features:

    # Count Plot

    sns.countplot(x=f, hue='sars-cov-2_exam_result', data=dataset_task1,ax=axes[r][c])

    # Plot configs

    axes[r][c].legend(title='sars-cov-2_exam_result', loc='upper right')

    # Index control

    c += 1

    if c > 1:

        c = 0

        r += 1



plt.tight_layout()
dataset_task1['detection_adenoviridae'].value_counts()
dataset_task1['chlamydophila_pneumoniae'].value_counts()
dataset_task1.drop(['chlamydophila_pneumoniae', 'detection_adenoviridae'], axis=1, inplace=True)
# PLOT - Barplots over our variables

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,15))

r = 0 # Index row

c = 0 # Index col

for f in cat_features:

    # Count Plot

    sns.countplot(x=f, hue='patient_addmited_cats', data=dataset_task2,ax=axes[r][c])

    # Plot configs

    axes[r][c].legend(title='patient_addmited_cats', loc='upper right')

    # Index control

    c += 1

    if c > 1:

        c = 0

        r += 1



plt.tight_layout()
dataset_task2.drop(['chlamydophila_pneumoniae', 'detection_adenoviridae'], axis=1, inplace=True)
# Update cat features

cat_features = list(set(cat_features).difference(set(['chlamydophila_pneumoniae', 'detection_adenoviridae'])))
# update total features

features = num_features + cat_features
# targets

target1 = 'sars-cov-2_exam_result'

target2 = 'patient_addmited_cats'
cat_features
num_features
# Print features for models

print('Features Num - ',len(num_features), ' | Features Cat - ', len(cat_features), ' | Total - ', len(cat_features)+len(num_features))
# TTSplit

x_train, x_test, y_train, y_test = train_test_split(dataset_task1[num_features], dataset_task1[target1], test_size = 0.20, random_state = 1206, stratify=dataset_task1[target1])
print(len(x_train))
# create smote object

smt = SMOTE(k_neighbors=5, random_state=1206)



# Do the process

x_train, y_train = smt.fit_sample(x_train, y_train)
print(len(x_train))
# Defining parameter range to grid search

param_gridSVM = {'C': [0.1, 1, 10, 100, 1000],

                 'shrinking':[True, False],

                 'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001], 

                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  



# Best result found after grid search! This trick is to improve commit speed

param_gridSVM = {'C': [10], 'gamma': [0.1], 'kernel': ['rbf'], 'shrinking': [True]}



# Define grid instance

gridSVM = GridSearchCV(cv=5, estimator=SVC(class_weight='balanced', random_state=101), param_grid=param_gridSVM, refit = True, verbose = 1, scoring='balanced_accuracy', n_jobs=3) 



# Initialize grid search, fitting the best model

gridSVM.fit(x_train, y_train);
# print best parameter after tuning svm

print(gridSVM.best_params_)
# print how our best model looks after hyper-parameter tuning 

print(gridSVM.best_estimator_) 
# Make predictions over test set

y_pred_svm = gridSVM.predict(x_test)
# print classification report SVM

print(classification_report(y_test, y_pred_svm))
# Confusion Matrix SVM

## original binary labels

labels = np.unique(y_test)

## DF with C.M.

cm = pd.DataFrame(confusion_matrix(y_test, y_pred_svm, labels=labels), index=labels, columns=labels)

# Visualize labels

cm.index = ['real: 0', 'real: 1']

cm.columns = ['pred: 0', 'pred: 1']



# CM visualization

cm
# TTSplit

x_train, x_test, y_train, y_test = train_test_split(dataset_task1[features], dataset_task1[target1], test_size = 0.20, random_state = 1206, stratify=dataset_task1[target1])



# Create smote object

smt = SMOTE(k_neighbors=5, random_state=1206)



# Do the process

x_train, y_train = smt.fit_sample(x_train, y_train)



# Defining parameter range to grid search

param_gridSVM = {'C': [0.1, 1, 10, 100, 1000],

                 'shrinking':[True, False],

                 'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001], 

                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  



# Best result found after grid search! This trick is to improve commit speed

param_gridSVM = {'C': [10], 'gamma': [0.1], 'kernel': ['rbf'], 'shrinking': [True]}



# Define grid instance

gridSVM = GridSearchCV(cv=5, estimator=SVC(class_weight='balanced', random_state=101), param_grid=param_gridSVM, refit = True, verbose = 1, scoring='balanced_accuracy', n_jobs=3) 



# Initialize grid search, fitting the best model

gridSVM.fit(x_train, y_train);
# print best parameter after tuning svm

print(gridSVM.best_params_)
# Make predictions over test set

y_pred_svm = gridSVM.predict(x_test)



# print classification report SVM

print(classification_report(y_test, y_pred_svm))
# Confusion Matrix SVM

## original binary labels

labels = np.unique(y_test)

## DF with C.M.

cm = pd.DataFrame(confusion_matrix(y_test, y_pred_svm, labels=labels), index=labels, columns=labels)

# Visualize labels

cm.index = ['real: 0', 'real: 1']

cm.columns = ['pred: 0', 'pred: 1']



# CM visualization

cm
# ttsplit

x_train, x_test, y_train, y_test = train_test_split(dataset_task2[num_features], dataset_task2[target2], test_size = 0.20, random_state = 1206, stratify=dataset_task2[target2])
# create smote object

smt = SMOTE(k_neighbors=5, random_state=1206)



# Do the process

x_train, y_train = smt.fit_sample(x_train, y_train)
# Defining parameter range to grid search

param_gridSVM = {'C': [0.1, 1, 10, 100, 1000], 

                'shrinking':[True, False],

                 'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001], 

                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']} 



# Best result found after grid search! This trick is to improve commit speed

param_gridSVM = {'C': [100], 'gamma': ['scale'], 'kernel': ['rbf'], 'shrinking': [True]}



# Define grid instance

gridSVM = GridSearchCV(cv=5, estimator=SVC(class_weight='balanced', random_state=101), param_grid=param_gridSVM, refit = True, verbose = 1, scoring='balanced_accuracy', n_jobs=3) 



# Initialize grid search, fitting the best model

gridSVM.fit(x_train, y_train)
# print best parameter after tuning 

print(gridSVM.best_params_)
# print how our best model looks after hyper-parameter tuning 

print(gridSVM.best_estimator_) 
# Make predictions over test set for both models

y_pred_svm = gridSVM.predict(x_test)
# print classification report SVM

print(classification_report(y_test, y_pred_svm))
# Confusion Matrix SVM

## original binary labels

labels = np.unique(y_test)

## DF with C.M.

cm = pd.DataFrame(confusion_matrix(y_test, y_pred_svm, labels=labels), index=labels, columns=labels)



# CM visualization

cm
# TTSplit

x_train, x_test, y_train, y_test = train_test_split(dataset_task2[features], dataset_task2[target2], test_size = 0.20, random_state = 1206, stratify=dataset_task2[target2])



# Create smote object

smt = SMOTE(k_neighbors=5, random_state=1206)



# Do the process

x_train, y_train = smt.fit_sample(x_train, y_train)



# Defining parameter range to grid search

param_gridSVM = {'C': [0.1, 1, 10, 100, 1000],

                 'shrinking':[True, False],

                 'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001], 

                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  



# Best result found after grid search! This trick is to improve commit speed

param_gridSVM = {'C': [100], 'gamma': [0.1], 'kernel': ['rbf'], 'shrinking': [True]}



# Define grid instance

gridSVM = GridSearchCV(cv=5, estimator=SVC(class_weight='balanced', random_state=101), param_grid=param_gridSVM, refit = True, verbose = 1, scoring='balanced_accuracy', n_jobs=3) 



# Initialize grid search, fitting the best model

gridSVM.fit(x_train, y_train);
# print best parameter after tuning svm

print(gridSVM.best_params_)
# Make predictions over test set

y_pred_svm = gridSVM.predict(x_test)



# print classification report SVM

print(classification_report(y_test, y_pred_svm))
# Confusion Matrix SVM

## original binary labels

labels = np.unique(y_test)

## DF with C.M.

cm = pd.DataFrame(confusion_matrix(y_test, y_pred_svm, labels=labels), index=labels, columns=labels)



# CM visualization

cm