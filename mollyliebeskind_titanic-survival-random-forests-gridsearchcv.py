import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import math

from collections import Counter



import os



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

geneder_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
train_data.shape
def detect_outliers(data,n,features):

    outlier_indices = []  

    for col in features:

        Q1 = np.percentile(data[col], 25)

        Q3 = np.percentile(data[col],75)

        IQR = Q3 - Q1  

        outlier_step = 1.5 * IQR

        outlier_list_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step )].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers   



Outliers_to_drop = detect_outliers(train_data,2,["Age","SibSp","Parch","Fare"])

train_data = train_data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
train_data.isnull().sum()
#Removing 'Cabin' and organizing by dtype

original_columns = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Sex', 'Embarked', 'Ticket', 'Name', 'Cabin']

reorg_columns = ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Sex', 'Embarked', 'Ticket', 'Name'] 

train_data_reorg = train_data[reorg_columns]

train_data_reorg.head(20)
def FillNumerical(data, column_names):

    mean_per_column = data[column_names].apply(lambda x: x.mean(),axis=0)

    numerical_fill = data[column_names].fillna(mean_per_column, axis=0)

    return numerical_fill



def GetAgeMeans(data, age_column_name, name_column_name):

    miss_list = []

    mr_list = []

    mrs_list = []

    other_list = []

    for row in data.index:

        if 'Miss' in data[name_column_name][row]:

            miss_list.append(data[age_column_name][row])

        elif 'Mrs' in data[name_column_name][row]:

            mrs_list.append(data[age_column_name][row])

        elif 'Mr' in data[name_column_name][row]:

            mr_list.append(data[age_column_name][row])

        else: 

            other_list.append(data[age_column_name][row])

    mean_dict = {'miss_mean': np.nanmean(miss_list), 'mrs_mean': np.nanmean(mrs_list), 'mr_mean': np.nanmean(mr_list), 'other_mean': np.nanmean(other_list)}

    return mean_dict



def FillAges(data, age_column_name, name_column_name):

    mean_dict = GetAgeMeans(data, age_column_name, name_column_name)

    for row in data.index:

        if math.isnan(data[age_column_name][row]):

            if 'Miss' in data[name_column_name][row]:

                data.loc[row, age_column_name] = mean_dict.get('miss_mean')

            elif 'Mrs' in data[name_column_name][row]:

                data.loc[row, age_column_name] = mean_dict.get('mrs_mean')

            elif 'Mr' in data[name_column_name][row]:

                data.loc[row, age_column_name] = mean_dict.get('mr_mean')

            else: 

                data.loc[row, age_column_name] = mean_dict.get('other_mean')

        else:

            continue

    age_data = data[age_column_name]

    return age_data



def get_most_frequent_value(columns):

    return columns.value_counts().index[0]



def FillCategorical(data, column_names):

    most_common_val = data[column_names].apply(get_most_frequent_value,axis=0)

    categorical_fill = data[column_names].fillna(most_common_val,axis=0)

    return categorical_fill
numerical_cols = FillNumerical(train_data_reorg,reorg_columns[:6])

age_col = FillAges(train_data_reorg, 'Age', 'Name')

cat_cols = FillCategorical(train_data_reorg, reorg_columns[7:])

cleaned_data = pd.concat((numerical_cols, age_col, cat_cols), sort=False, axis=1)



cleaned_data.tail()
#Confirm that there are no zero values

cleaned_data.isnull().sum()
cleaned_data = pd.get_dummies(cleaned_data, columns=reorg_columns[7:9])

cleaned_data.head()
final_columns = [column for column in cleaned_data.columns]

cleaned_data[final_columns].nunique()
correlation = cleaned_data.corr()

f, axes = plt.subplots(nrows=1, ncols=2, figsize = (16,6))

x1 = sns.heatmap(correlation, center=0, vmin=-1, vmax=1, ax=axes[0]).set_title('All Data Correlations')

x2 = sns.heatmap(correlation[['Survived']], center=0, vmin=-1, vmax=1, ax=axes[1]).set_title('Survival Correlations')
survived_by_pclass = cleaned_data.groupby('Pclass')['Survived'].agg([np.sum])

f,axes = plt.subplots(1,3, figsize=(16,6))

axes[0].pie(cleaned_data['Pclass'].value_counts()/cleaned_data.shape[0]*100, labels=['Pclass3','Pclass1','Pclass2'], colors=['lightsteelblue','cornflowerblue','royalblue'], autopct='%1.1f%%')

axes[0].axis('equal')

axes[0].set_title('Distribution of Passengers\n By Pclass')

axes[1].hist(cleaned_data['Pclass'], bins=np.arange(5), align='left', rwidth=.9, color='cornflowerblue', stacked=True, label='Total Onboard')

axes[1].bar(survived_by_pclass.index, survived_by_pclass[survived_by_pclass.columns[0]], color='navy', label='Survived')

axes[1].set_title('PClass Distrobution and Survival Rate')

axes[1].set_xlabel('Pclass')

axes[1].set_ylabel('Number of Passengers')

axes[1].legend(loc='upper_left')

axes[1].set_xticks(range(4))

axes[1].set_xlim(.5,3.5)

axes[2].bar(survived_by_pclass.index, cleaned_data.groupby('Pclass')['Survived'].mean()*100, color='royalblue', alpha=.6)

axes[2].set_title('Survival Percentage by Pclass')

axes[2].set_xlabel('Pclass')

axes[2].set_ylabel('Percentage of Surviving Passengers')

axes[2].set_xticks(range(4))

axes[2].set_xlim(.5,3.5)

plt.show()
pclass_distro = cleaned_data['Pclass'].value_counts()/cleaned_data.shape[0]*100

print('Percentage of passengers in Pclass 1:', round(pclass_distro[1],2))

print('Percentage of passengers in Pclass 2:', round(pclass_distro[2],2))

print('Percentage of passengers in Pclass 3:', round(pclass_distro[3],2))
survived_by_gender = cleaned_data.groupby('Sex_female')['Survived'].agg([np.sum])

f,axes = plt.subplots(1,3, figsize=(16,6))

axes[0].pie(cleaned_data['Sex_female'].value_counts()/cleaned_data.shape[0]*100, labels=['Male','Female'], colors=['olivedrab','yellowgreen'], autopct='%1.1f%%')

axes[0].axis('equal') 

axes[0].set_title('Distribution of Passengers\n By Sex')

axes[1].hist(cleaned_data['Sex_female'], bins=np.arange(5), align='left', rwidth=.9, color='darkgreen', alpha=.4, stacked=True, label='Total Onboard')

axes[1].bar(survived_by_gender.index, survived_by_gender[survived_by_gender.columns[0]], color='darkgreen', label='Survived')

axes[1].set_title('Sex Distrobution and Survival Rate')

axes[1].set_xlabel('Sex')

axes[1].set_ylabel('Number of Passengers')

axes[1].legend(loc='upper_left')

axes[1].set_xticks([0,1])

axes[1].set_xticklabels(['male', 'female'])

axes[1].set_xlim(-1,2)

axes[2].bar(survived_by_gender.index, cleaned_data.groupby('Sex_female')['Survived'].mean()*100, color='darkseagreen', alpha=.6)

axes[2].set_title('Survival Percentage by Sex')

axes[2].set_xlabel('Sex')

axes[2].set_ylabel('Percentage of Surviving Passengers')

axes[2].set_xticks([0,1])

axes[2].set_xticklabels(['male', 'female'])

axes[2].set_xlim(-1,2)

plt.show()
class1_female = cleaned_data.groupby(['Sex_female','Pclass'])['Survived'].agg([np.mean])

class1_female
test_columns = ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Sex', 'Embarked', 'Ticket', 'Name'] 

test_data_reorg = test_data[test_columns]

test_data_reorg.dtypes
numerical_cols_test = FillNumerical(test_data_reorg,test_columns[:5])

age_col_test = FillAges(test_data_reorg, 'Age', 'Name')

cat_cols_test = FillCategorical(test_data_reorg, test_columns[6:])

cleaned_test_data = pd.concat((numerical_cols_test, age_col_test, cat_cols_test), sort=False, axis=1)



cleaned_test_data.head()
cleaned_test_data.isnull().sum()
y = cleaned_data["Survived"]

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

X = cleaned_data[features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X).astype(float)
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 100, num = 10)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [8, 10, 12]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
rf = RandomForestClassifier()



rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'accuracy', n_iter = 100, cv = 6, verbose=2, random_state=1234, n_jobs = -1)

rf_random.fit(X_scaled, y)



print(rf_random.best_params_)

print(rf_random.best_score_)
# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [50, 100, 200],

    'max_features': [2, 3],

    'min_samples_leaf': [2, 3],

    'min_samples_split': [10, 12],

    'n_estimators': [300, 600, 2000, 3000]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 10, n_jobs = -1, verbose = 2)

grid_search.fit(X_scaled, y)

print(grid_search.best_params_)

print(grid_search.best_score_)
X_test = pd.get_dummies(cleaned_test_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare','Sex', 'Embarked']])

test_scaler = StandardScaler()

X_test_scaled = test_scaler.fit_transform(X_test).astype(float)



predictions = grid_search.predict(X_test_scaled)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_5.csv', index=False)

print("Your submission was successfully saved!")