import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_colwidth', -1)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate



import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

sample_submission = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")
print(f'Shape of train data set is {train.shape} \n')

print(f'Shape of test data set is {test.shape} \n')

print(f'Shape of the sample submission is {sample_submission.shape} \n')



target = set(train.columns) - set(test.columns)

print(f'The target variable is called: {target} \n') # Test data does not contain target column



id_col = set(sample_submission.columns) - set(target)

print(f'The id is: {id_col} \n')



features = set(train.columns) - set(id_col) - set(target)

print(f'The features are: {features} \n') # Test data only contains features



high_cardinality  = [col for col in features if train[col].nunique() > 100]

print(f'High cardinality columns: {high_cardinality} \n')

#low_cardinality  = [col for col in features if train[col].nunique() <= 100]

#print(f'Low cardinality columns: {low_cardinality} \n')



#pd.DataFrame({'train_unique' : train.nunique(), 

#              'test_unique' : test.nunique(),

#              'diff' : (train.nunique() - test.nunique())})
print(f'There is missing data in train: {train.isnull().sum().any()}')

print(f'There is missing data in test: {test.isnull().sum().any()}')

print(f'Columns with missing data are: {[col for col in train.columns if train[col].isnull().sum() > 0]}')
f, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(train.isnull())

plt.show()



f, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(test.isnull())

plt.show()
relevant_cols =['bin_1', 'bin_2', 'bin_3', 'bin_4', 'ord_0', 'ord_1', 'ord_2', 'ord_3','ord_4', 'ord_5', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']



def encode_by_mapping(df):

    bin_3_mapping = {'F' : 0, 

                     'T' : 1}

    df['bin_3'].replace(bin_3_mapping, inplace = True)



    bin_4_mapping = {'N' : 0, 

                     'Y' : 1}

    df['bin_4'].replace(bin_4_mapping, inplace = True)

    

    ord_1_mapping = {'Novice' : 0, 

                 'Contributor' : 1, 

                 'Expert' : 2, 

                 'Master' : 3, 

                 'Grandmaster' : 4}

    df['ord_1'].replace(ord_1_mapping, inplace = True)



    ord_2_mapping = {'Freezing' : -2, 

                     'Cold' : -1, 

                     'Warm' : 1, 

                     'Hot' : 2, 

                     'Boiling Hot' : 3, 

                     'Lava Hot' : 4}

    df['ord_2'].replace(ord_2_mapping, inplace = True)

    return df



def encode_by_mapping_nom(df):

    dict_nom_0 = {'Green': 0.135, 'Red': 0.14, 'Blue': 0.15}

    dict_nom_1 = {'Star': 0.1, 'Triangle': 0.12, 'Square': 0.12, 'Circle': 0.14, 'Polygon': 0.16, 'Trapezoid': 0.18}

    dict_nom_2 = {'Snake': 0.1, 'Cat': 0.135, 'Hamster': 0.135, 'Dog': 0.14, 'Axolotl': 0.155, 'Lion': 0.18}

    dict_nom_3 = {'China': 0.1, 'Canada': 0.115, 'India': 0.12, 'Finland': 0.14, 'Costa Rica': 0.155, 'Russia': 0.17}

    dict_nom_4 = {'Piano': 0.1, 'Theremin': 0.14, 'Oboe': 0.14, 'Bassoon': 0.16}

    df['nom_0'].replace(dict_nom_0, inplace = True)

    df['nom_1'].replace(dict_nom_1, inplace = True)

    df['nom_2'].replace(dict_nom_2, inplace = True)

    df['nom_3'].replace(dict_nom_3, inplace = True)

    df['nom_4'].replace(dict_nom_4, inplace = True)



    return df



def oh_encoding(df_train, df_test=[]):

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    

    oh_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']



    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(df_train[oh_cols]))

    if len(df_test) > 0:

        OH_cols_test = pd.DataFrame(OH_encoder.transform(df_test[oh_cols]))



    # Rename columns

    #new_columns_names = train_df_mod['event_code_oh'].unique()#['test1', 'test2', 'test3', 'test4']    

    #OH_cols_train.columns = new_columns_names

    #OH_cols_test.columns = new_columns_names 



    OH_cols_train.index = df_train.index

    if len(df_test) > 0:

        OH_cols_test.index = df_test.index



    df_train = df_train.drop(oh_cols, axis=1)

    if len(df_test) > 0:

        df_test = df_test.drop(oh_cols, axis=1)



    df_train = pd.concat([df_train, OH_cols_train], axis=1)

    if len(df_test) > 0:

        df_test = pd.concat([df_test, OH_cols_test], axis=1)

    return df_train #, df_test



def label_encoding(df):

    # Alphabetical Label Encoding

    df['ord_3'] = LabelEncoder().fit_transform(df['ord_3']) 

    df['ord_4'] = LabelEncoder().fit_transform(df['ord_4']) 

    df['ord_5'] = LabelEncoder().fit_transform(df['ord_5']) 

    return df



def encode_categorical_features(train, test):

    train = encode_by_mapping(train)

    test = encode_by_mapping(test)



    train, test = oh_encoding(train, test)



    train = label_encoding(train)

    test = label_encoding(test)

    # Standard scaling for regression

    train[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']] = StandardScaler().fit_transform(train[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']])

    test[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']] = StandardScaler().fit_transform(test[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']])

    return train, test



def evaluate_features(X_train, y_train):

    # Model and evaluation copied from https://www.kaggle.com/dkomyagin/cat-in-the-dat-0-80285-private-lb-solution

    clf = LogisticRegression(solver='lbfgs', max_iter=1000, verbose=0, n_jobs=-1, random_state = 1)

    #C=0.12, 

    score = cross_validate(clf, X_train, y_train, cv=3, scoring="roc_auc")

    mean = score['test_score'].mean()

    print(f'Mean ROC AUC: {mean:.8f}\n')
features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 

            'nom_0', 'nom_1','nom_2', 'nom_3', 'nom_4', 

            'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 

            'day', 'month', 'target']

# Create a copy of training and test data

X_train = train[features].copy()

y_train = train.target.copy()



# First encode by mapping from dictionaries

X_train = encode_by_mapping(X_train)



###############################################################

##### Handle missing data: Imputation with outlier values #####

###############################################################

# fillna for bin_x with 0.5

X_train[['bin_1', 'bin_2', 'bin_3', 'bin_4']] = X_train[['bin_1', 'bin_2', 'bin_3', 'bin_4']].fillna(0.5)



X_train['ord_0'] = X_train['ord_0'].fillna(X_train.ord_0.mean())

X_train['ord_1'] = X_train['ord_1'].fillna(X_train.ord_1.mean())

X_train['ord_2'] = X_train['ord_2'].fillna(X_train.ord_2.mean())



# Label encoding

# fillna with dummy for initial label encoding

categorical_cols = [cols for cols in ['ord_3', 'ord_4', 'ord_5', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'] if X_train[cols].dtype == 'object']

X_train[categorical_cols] = X_train[categorical_cols].fillna('zz')



# Label encoding

X_train = label_encoding(X_train)



# Standard scaling for regression

X_train[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']] = StandardScaler().fit_transform(X_train[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']])



# Replace dummy fillna with 0 after StandardScaler

X_train[['ord_3', 'ord_4', 'ord_5']] = X_train[['ord_3', 'ord_4', 'ord_5']].where(~train[['ord_3', 'ord_4', 'ord_5']].isna(), 0)



j=1

fig = plt.figure(figsize=(20, 20))

for nom_var in ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']:

    for ord_var in ['ord_0', 'ord_1', 'ord_2']:# 'ord_3', 'ord_4', 'ord_5']:

        fig.add_subplot(5, 3, j)

        sns.lineplot(y='target', x=ord_var, hue=nom_var, data=X_train.groupby([nom_var, ord_var]).agg({'target': 'mean'}).reset_index())

        plt.ylim(0, 0.5)

        #plt.show()

        j = j+1
pd.set_option('display.max_columns', None)  

train[relevant_cols+['target']].head()
def handle_missing_data_with_outliers(outlier_val_num, outlier_val_cat):

    print(f'Outlier value for missing numerical columns: {outlier_val_num}\nOutlier value for missing categorical columns: {outlier_val_cat}')

    # Create a copy of training and test data

    X_train = train[relevant_cols].copy()

    y_train = train.target.copy()



    # First encode by mapping from dictionaries

    X_train = encode_by_mapping(X_train)

    X_train = encode_by_mapping_nom(X_train)



    ###############################################################

    ##### Handle missing data: Imputation with outlier values #####

    ###############################################################

    numerical_cols = [cols for cols in relevant_cols if X_train[cols].dtype != 'object']

    categorical_cols = [cols for cols in relevant_cols if X_train[cols].dtype == 'object']



    X_train[numerical_cols] = X_train[numerical_cols].fillna(outlier_val_num)

    X_train[categorical_cols] = X_train[categorical_cols].fillna(outlier_val_cat)



    # Label encoding

    X_train = label_encoding(X_train)



    # Standard scaling for regression

    X_train[['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']] = StandardScaler().fit_transform(X_train[['ord_0', 'ord_1', 'ord_2','ord_3', 'ord_4', 'ord_5']])

    X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']] = StandardScaler().fit_transform(X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']])



    evaluate_features(X_train, y_train)



handle_missing_data_with_outliers(9999, 'zz')
# Create a copy of training and test data

X_train = train[relevant_cols].copy()

y_train = train.target.copy()



# First encode by mapping from dictionaries

X_train = encode_by_mapping(X_train)

X_train = encode_by_mapping_nom(X_train)



############################################################

##### Handle missing data: Imputation with mean values #####

############################################################



# fillna for bin_x with 0.5

X_train[['bin_1', 'bin_2', 'bin_3', 'bin_4']] = X_train[['bin_1', 'bin_2', 'bin_3', 'bin_4']].fillna(0.5)



X_train['ord_0'] = X_train['ord_0'].fillna(X_train.ord_0.mean())

X_train['ord_1'] = X_train['ord_1'].fillna(X_train.ord_1.mean())

X_train['ord_2'] = X_train['ord_2'].fillna(X_train.ord_2.mean())



X_train['nom_0'] = X_train['nom_0'].fillna(X_train.nom_0.mean())

X_train['nom_1'] = X_train['nom_1'].fillna(X_train.nom_1.mean())

X_train['nom_2'] = X_train['nom_2'].fillna(X_train.nom_2.mean())

X_train['nom_3'] = X_train['nom_3'].fillna(X_train.nom_3.mean())

X_train['nom_4'] = X_train['nom_4'].fillna(X_train.nom_4.mean())



# Label encoding

# fillna with dummy for initial label encoding

categorical_cols = ['ord_3', 'ord_4', 'ord_5'] 



X_train[categorical_cols] = X_train[categorical_cols].fillna('zz')

X_train = label_encoding(X_train)

# Replace dummy fillna with median after StandardScaler

X_train['ord_3'] = X_train['ord_3'].where(~train['ord_3'].isna(), X_train.ord_3.mean())

X_train['ord_4'] = X_train['ord_4'].where(~train['ord_4'].isna(), X_train.ord_4.mean())

X_train['ord_5'] = X_train['ord_5'].where(~train['ord_5'].isna(), X_train.ord_5.mean())



# Standard scaling for regression

X_train[['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']] = StandardScaler().fit_transform(X_train[['ord_0', 'ord_1', 'ord_2','ord_3', 'ord_4', 'ord_5']])



X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']] = StandardScaler().fit_transform(X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']])



evaluate_features(X_train, y_train)
# Create a copy of training and test data

X_train = train[relevant_cols].copy()

y_train = train.target.copy()



# First encode by mapping from dictionaries

X_train = encode_by_mapping(X_train)

X_train = encode_by_mapping_nom(X_train)



############################################################

##### Handle missing data: Imputation with mean values #####

############################################################



# fillna for bin_x with 0.5

X_train[['bin_1', 'bin_2', 'bin_3', 'bin_4']] = X_train[['bin_1', 'bin_2', 'bin_3', 'bin_4']].fillna(0.5)



X_train['ord_0'] = X_train['ord_0'].fillna(X_train.ord_0.median())

X_train['ord_1'] = X_train['ord_1'].fillna(X_train.ord_1.median())

X_train['ord_2'] = X_train['ord_2'].fillna(X_train.ord_2.median())



X_train['nom_0'] = X_train['nom_0'].fillna(X_train.nom_0.median())

X_train['nom_1'] = X_train['nom_1'].fillna(X_train.nom_1.median())

X_train['nom_2'] = X_train['nom_2'].fillna(X_train.nom_2.median())

X_train['nom_3'] = X_train['nom_3'].fillna(X_train.nom_3.median())

X_train['nom_4'] = X_train['nom_4'].fillna(X_train.nom_4.median())



# Label encoding

# fillna with dummy for initial label encoding

categorical_cols = ['ord_3', 'ord_4', 'ord_5'] 



X_train[categorical_cols] = X_train[categorical_cols].fillna('zz')

X_train = label_encoding(X_train)



# Replace dummy fillna with median after StandardScaler

X_train['ord_3'] = X_train['ord_3'].where(~train['ord_3'].isna(), X_train.ord_3.median())

X_train['ord_4'] = X_train['ord_4'].where(~train['ord_4'].isna(), X_train.ord_4.median())

X_train['ord_5'] = X_train['ord_5'].where(~train['ord_5'].isna(), X_train.ord_5.median())



# Standard scaling for regression

X_train[['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']] = StandardScaler().fit_transform(X_train[['ord_0', 'ord_1', 'ord_2','ord_3', 'ord_4', 'ord_5']])



X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']] = StandardScaler().fit_transform(X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']])



evaluate_features(X_train, y_train)
# Create a copy of training and test data

X_train = train[relevant_cols].copy()

y_train = train.target.copy()



# First encode by mapping from dictionaries

X_train = encode_by_mapping(X_train)

X_train = encode_by_mapping_nom(X_train)



####################################################################

##### Handle missing data: Imputation with reconstructed value #####

####################################################################

# Interpolation technique copied from https://www.kaggle.com/kaushal2896/cat-in-dat-ii-simple-logistic-regression

X_train = X_train.apply(lambda group: group.interpolate(limit_direction='both'))

# Label encoding

# fillna with dummy for initial label encoding

categorical_cols = ['ord_3', 'ord_4', 'ord_5']

X_train[categorical_cols] = X_train[categorical_cols].fillna('zz')

X_train = label_encoding(X_train)



# Standard scaling for regression

X_train[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']] = StandardScaler().fit_transform(X_train[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']])

# Interpolate dummy fillna with 0 after StandardScaler

X_train[['ord_3', 'ord_4', 'ord_5']] = X_train[['ord_3', 'ord_4', 'ord_5']].where(~train[['ord_3', 'ord_4', 'ord_5']].isna()).apply(lambda group: group.interpolate(limit_direction='both'))



X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']] = StandardScaler().fit_transform(X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']])



evaluate_features(X_train, y_train)
# Create a copy of training and test data

train_temp = train[relevant_cols].copy()



# Split into complete training data and training data to predict

train_temp['to_predict'] = train_temp.isnull().sum(axis=1)



X_train = train_temp[train_temp.to_predict == 0]

X_train_predict = train_temp[train_temp.to_predict > 0]



print(f'Splitted X_train ({train_temp.shape[0]}) into {X_train.shape[0]} non-missing and {X_train_predict.shape[0]} missing data points.')



# First encode by mapping from dictionaries

X_train = encode_by_mapping(X_train)

X_train = encode_by_mapping_nom(X_train)



# OH encoding

#OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

#oh_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

#OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[oh_cols]))

#OH_cols_train.index = X_train.index

#X_train = X_train.drop(oh_cols, axis=1)

#X_train = pd.concat([X_train, OH_cols_train], axis=1)



# Label encoding

X_train = label_encoding(X_train)



# Standard scaling for regression

X_train[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']] = StandardScaler().fit_transform(X_train[['ord_0', 'ord_1', 'ord_3', 'ord_4', 'ord_5']])

X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']] = StandardScaler().fit_transform(X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']])
from xgboost import XGBClassifier, XGBRegressor

from xgboost import plot_importance

model = XGBRegressor()

target='ord_5'

feat = [x for x in relevant_cols if x not in (target)]# to_remove +



model.fit(X_train[feat], X_train[target])

# plot feature importance



#plt.rcParams["figure.figsize"] = (10, 30)

plot_importance(model)

plt.show()



#'bin_1', 'ord_5'

# bin _2, ord5

#bin 3, ord5

#bin 4_ ord5



#ord1_ ord5, prd4, ord3



corrmat = X_train[relevant_cols].corr()

f, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()



bin_features = ['bin_1', 'bin_2', 'bin_3', 'bin_4']

for c in bin_features:

    print(c)

    features_temp = set(relevant_cols) - set([c])

    X_train_temp = X_train[features_temp].copy()

    y_train_temp = X_train[c].copy()

    # Model and evaluation copied from https://www.kaggle.com/dkomyagin/cat-in-the-dat-0-80285-private-lb-solution

    clf = LogisticRegression(solver='lbfgs', max_iter=500, verbose=0, n_jobs=-1, random_state = 1)

    score = cross_validate(clf, X_train_temp, y_train_temp, cv=3, scoring="roc_auc")

    last_score = score['test_score'].mean()

    print(f'Mean ROC AUC: {last_score:.8f}\n')
# Create a copy of training and test data

X_train = train[relevant_cols].copy()

y_train = train.target.copy()



# First encode by mapping from dictionaries

X_train = encode_by_mapping(X_train)

X_train = encode_by_mapping_nom(X_train)



############################################################

##### Handle missing data: Imputation with mean values #####

############################################################



# fillna for bin_x with 0.5

X_train[['bin_1', 'bin_2', 'bin_3', 'bin_4']] = X_train[['bin_1', 'bin_2', 'bin_3', 'bin_4']].fillna(0.5)



X_train['ord_0'] = X_train['ord_0'].fillna(X_train.ord_0.mean())

X_train['ord_1'] = X_train['ord_1'].fillna(X_train.ord_1.mean())

X_train['ord_2'] = X_train['ord_2'].fillna(X_train.ord_2.mean())



X_train['nom_0'] = X_train['nom_0'].fillna(X_train.nom_0.mean())

X_train['nom_1'] = X_train['nom_1'].fillna(X_train.nom_1.mean())

X_train['nom_2'] = X_train['nom_2'].fillna(X_train.nom_2.mean())

X_train['nom_3'] = X_train['nom_3'].fillna(X_train.nom_3.mean())

X_train['nom_4'] = X_train['nom_4'].fillna(X_train.nom_4.mean())



# Label encoding

# fillna with dummy for initial label encoding

categorical_cols = ['ord_3', 'ord_4', 'ord_5'] 



X_train[categorical_cols] = X_train[categorical_cols].fillna('zz')

X_train = label_encoding(X_train)



# Standard scaling for regression

X_train[['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']] = StandardScaler().fit_transform(X_train[['ord_0', 'ord_1', 'ord_2','ord_3', 'ord_4', 'ord_5']])

X_train[['ord_3', 'ord_4', 'ord_5']] = X_train[['ord_3', 'ord_4', 'ord_5']].where(~train[['ord_3', 'ord_4', 'ord_5']].isna(), 0)



X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']] = StandardScaler().fit_transform(X_train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']])



print('Baseline from approach 3:')

evaluate_features(X_train, y_train)



############################################################

##### Add extension for missing data #####

############################################################

for col in X_train.columns:

    X_train[f'{col}_ext'] = train[col].isnull()

print('Added column-wise extension to baseline from approach 3:')

evaluate_features(X_train, y_train)
