# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import pickle
import gzip

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.shape
#checking for duplicates
train_df.drop_duplicates()
train_df.shape
# Checking number of rows and columns in test data
test_df.shape
# preview the data
train_df.head(10)
# Target varibale distribution
sns.countplot(x="target", data=train_df)
# Handling imbalanced dataset by undersampling
desired_apriori=0.10

# Get the indices per target value
idx_0 = train_df[train_df.target == 0].index
idx_1 = train_df[train_df.target == 1].index

# Get original number of records per target value
records_0 = len(train_df.loc[idx_0])
records_1 = len(train_df.loc[idx_1])

# Calculate the undersampling rate and resulting number of records with target=0
undersampling_rate = ((1-desired_apriori)*records_1)/(records_0*desired_apriori)
undersampled_records_0 = int(undersampling_rate*records_0)

# Randomly select records with target=0 to get at the desired a priori
undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_records_0)

# Construct list with remaining indices
idx_list = list(undersampled_idx) + list(idx_1)

# Return undersample data frame
train_df = train_df.loc[idx_list].reset_index(drop=True)
# Random shuffle
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.shape
# Frequency Distribution of each binary variable
bin_col = [col for col in train_df.columns if '_bin' in col]
for feature in bin_col:
    print (train_df[feature].value_counts())
# Dropping ps_ind_09_bin, ps_ind_10_bin, ps_ind_11_bin and ps_ind_12_bin as they are completely dominated by zeros
train_df = train_df.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin','ps_ind_13_bin'], axis=1)
test_df = test_df.drop(['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin','ps_ind_13_bin'], axis=1)
# checking for missing values
for feature in train_df.columns:
    missings = train_df[train_df[feature] == -1][feature].count()
    if missings > 0:
        print (str(feature)+"\t\t"+str(missings))
# Dropping ps_car_03_cat and ps_car_05_cat as they have a large proportion of records with missing values
train_df = train_df.drop(['ps_car_03_cat', 'ps_car_05_cat'], axis=1)
test_df = test_df.drop(['ps_car_03_cat', 'ps_car_05_cat'], axis=1)
# Replacing missing values of features other than categorical
# Imputing with the mean or mode
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
train_df['ps_reg_03'] = mean_imp.fit_transform(train_df[['ps_reg_03']]).ravel()
train_df['ps_car_14'] = mean_imp.fit_transform(train_df[['ps_car_14']]).ravel()
train_df['ps_car_11'] = mode_imp.fit_transform(train_df[['ps_car_11']]).ravel()
train_df.dtypes
train_float = train_df.select_dtypes(include=['float64'])
# correlation matrix for float features
colormap = plt.cm.magma
plt.figure(figsize=(16,12))
plt.title('Pearson correlation of continuous features', y=1.05, size=15)
sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True)
# plotting ps_ind_02_cat against target variable
sns.barplot(x="ps_ind_02_cat", y="target", data=train_df)
# plotting ps_ind_04_cat against target variable
sns.barplot(x="ps_ind_04_cat", y="target", data=train_df)
# plotting ps_ind_05_cat against target variable
sns.barplot(x="ps_ind_05_cat", y="target", data=train_df)
# plotting ps_car_01_cat against target variable
sns.barplot(x="ps_car_01_cat", y="target", data=train_df)
# plotting ps_car_02_cat against target variable
sns.barplot(x="ps_car_02_cat", y="target", data=train_df)
# plotting ps_car_04_cat against target variable
sns.barplot(x="ps_car_04_cat", y="target", data=train_df)
# plotting ps_car_06_cat against target variable
sns.barplot(x="ps_car_06_cat", y="target", data=train_df)
# plotting ps_car_07_cat against target variable
sns.barplot(x="ps_car_07_cat", y="target", data=train_df)
# plotting ps_car_08_cat against target variable
sns.barplot(x="ps_car_08_cat", y="target", data=train_df)
# plotting ps_car_09_cat against target variable
sns.barplot(x="ps_car_09_cat", y="target", data=train_df)
# plotting ps_car_10_cat against target variable
sns.barplot(x="ps_car_10_cat", y="target", data=train_df)
# plotting ps_car_11_cat against target variable
sns.barplot(x="ps_car_11_cat", y="target", data=train_df)
# Dropping irrevalent feature
train_df = train_df.drop(['ps_car_10_cat'], axis=1)
test_df = test_df.drop(['ps_car_10_cat'], axis=1)
# checking for missing values in test data
for feature in test_df.columns:
    missings = test_df[test_df[feature] == -1][feature].count()
    if missings > 0:
        print (str(feature)+"\t\t"+str(missings))
# Replacing missing values of features other than categorical
# Imputing with the mean or mode
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
test_df['ps_reg_03'] = mean_imp.fit_transform(test_df[['ps_reg_03']]).ravel()
test_df['ps_car_14'] = mean_imp.fit_transform(test_df[['ps_car_14']]).ravel()
test_df['ps_car_11'] = mode_imp.fit_transform(test_df[['ps_car_11']]).ravel()
X_train = train_df.drop(["target","id"], axis=1)
Y_train = train_df["target"]
X_test  = test_df.drop("id", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
# calculating the coefficient of the features
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
# Dropping less important feature from each pair of highly correlated feature  
train_df = train_df.drop(["ps_reg_03","ps_car_12","ps_car_15"], axis=1)
test_df = test_df.drop(["ps_reg_03", "ps_car_12", "ps_car_15"], axis=1)
X_train = train_df.drop(["target","id"], axis=1)
Y_train = train_df["target"]
X_test  = test_df.drop("id", axis=1).copy()
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train.shape, Y_train.shape, X_test.shape