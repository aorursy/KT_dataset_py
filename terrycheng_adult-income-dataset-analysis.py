import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# default libraries

import numpy as np

import pandas as pd



# for data preprocessing

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



# for classifier models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold, cross_val_score

import xgboost as xgb



# for models evaluation

from sklearn.metrics import confusion_matrix, accuracy_score
dataset = pd.read_csv("/kaggle/input/adult-income-dataset/adult.csv")

dataset.head()
dataset.shape
dataset.info()
dataset.isnull().sum()
dataset.nunique()
dataset['income'] = dataset['income'].map({'<=50K':0, '>50K':1})
def init_check(df):

    """

    A function to make initial check for the dataset including the name, data type, 

    number of null values and number of unique varialbes for each feature.

    

    Parameter: dataset(DataFrame)

    Output : DataFrame

    """

    columns = df.columns    

    lst = []

    for feature in columns : 

        dtype = df[feature].dtypes

        num_null = df[feature].isnull().sum()

        num_unique = df[feature].nunique()

        lst.append([feature, dtype, num_null, num_unique])

    

    check_df = pd.DataFrame(lst)

    check_df.columns = ['feature','dtype','num_null','num_unique']

    check_df = check_df.sort_values(by='dtype', axis=0, ascending=True)

    

    return check_df
#init_check?
init_check(df=dataset)
def categorical_encoding(df, categorical_cloumns, encoding_method):

    """

    A function to encode categorical features to a one-hot numeric array (one-hot encoding) or 

    an array with value between 0 and n_classes-1 (label encoding).

    

    Parameters:

        df (pd.DataFrame) : dataset

        categorical_cloumns  (string) : list of features 

        encoding_method (string) : 'one-hot' or 'label'

    Output : pd.DataFrame

    """

    

    if encoding_method == 'label':

        print('You choose label encoding for your categorical features')

        encoder = LabelEncoder()

        encoded = df[categorical_cloumns].apply(encoder.fit_transform)

        return encoded

    

    elif encoding_method == 'one-hot':

        print('You choose one-hot encoding for your categorical features') 

        encoded = pd.DataFrame()

        for feature in categorical_cloumns:

            dummies = pd.get_dummies(df[feature], prefix=feature)

            encoded = pd.concat([encoded, dummies], axis=1)

        return encoded
#categorical_encoding?
categorical_encoding(df=dataset, categorical_cloumns=['workclass','education'], encoding_method='one-hot')
categorical_encoding(dataset, categorical_cloumns=['workclass','education'], encoding_method='label')
y = dataset['income']
X = dataset.drop(columns='income', axis=1)

categorical_columns = X.select_dtypes(include=['object']).columns

print(categorical_columns)
encoded = categorical_encoding(X, categorical_cloumns=categorical_columns, encoding_method='one-hot')

encoded.head()
X = X.drop(columns=categorical_columns, axis=1)

X = pd.concat([X, encoded], axis=1)

X.head()
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)



print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
scaler=MinMaxScaler()

X_train= pd.DataFrame(scaler.fit_transform(X_train))

X_test = pd.DataFrame(scaler.transform(X_test))
X_train.describe()
def data_preprocessing(df, features, target, encoding_method, test_size, random_state):

    y = df[target]

    

    X = df[features]

    

    categorical_columns = X.select_dtypes(include=['object']).columns

    

    if len(categorical_columns) != 0 :

        encoded = categorical_encoding(X, categorical_cloumns=categorical_columns, encoding_method=encoding_method)

        X = X.drop(columns=categorical_columns, axis=1)

        X = pd.concat([X, encoded], axis=1)

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    

    scaler=MinMaxScaler()

    X_train= pd.DataFrame(scaler.fit_transform(X_train))

    X_test = pd.DataFrame(scaler.transform(X_test))

    

    return X_train, X_test, y_train, y_test
features = dataset.columns.drop('income')



X_train, X_test, y_train, y_test = data_preprocessing(df=dataset, features=features, 

                                                      target='income', encoding_method = 'label',

                                                      test_size=0.2, random_state=123)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
def classifiers_estimator(models, scoring, X_train, y_train, k_fold, shuffle, random_state):

    """

    A function to estimate the performance of each classification model.

    

    Parameters:

        models (string) : list of classificaton models

        scoring (string) : quantifying the quality of predictions 

        X_train (np.array or pd.dataframe) : features variable of training data

        y_train (np.array or pd.dataframe) : target of training data

        k_fold (int) : number of folds

        shuffle (boolean) : whether to shuffle the data before splitting into batches

        random_state (int) : it is the seed used by the random number generator

    

    Output (pd.DataFrame) : model performance

    """

    kf = KFold(n_splits=k_fold, shuffle=shuffle, random_state=random_state)

    

    results = []

    for model in models:



        if model == 'RF':

            estimator = RandomForestClassifier()

        elif model == 'LR':

            estimator = RandomForestClassifier()

        elif model == 'KNN':

            estimator == KNeighborsClassifier(n_neighbors = 5)

        elif model == 'XGB':

            estimator == xgb.XGBClassifier()

            

        cv_results = cross_val_score(estimator=estimator, X=X_train, y=y_train, cv=kf, scoring=scoring, n_jobs=-1)

        cv_mean_accuracy = cv_results.mean()

        cv_std_accuracy = cv_results.std()

        cv_max = cv_results.max()

        cv_min = cv_results.min()

        results.append([model, cv_mean_accuracy, cv_std_accuracy, cv_max, cv_min])

        print('Finish %s model' %model)

    

    results_df = pd.DataFrame(results)

    results_df.columns = ['Models','Mean','Std','Max','Min']

    

    return results_df
#classifiers_estimator?
classifiers_estimator(['LR', 'RF', 'XGB'], 'accuracy', X_train, y_train, k_fold=6, shuffle=True, random_state=123)
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

print(rf.score(X_test, y_test))
import eli5

from eli5.sklearn import PermutationImportance



# Make a small change to the code below to use in this problem. 

perm = PermutationImportance(rf, random_state=123).fit(X_test, y_test)



# uncomment the following line to visualize your results

eli5.show_weights(perm, feature_names = features.tolist(), top=150)
features  = ['education','capital-gain']



X_train, X_test, y_train, y_test = data_preprocessing(df=dataset, features=features, 

                                                      target='income', encoding_method = 'label',

                                                      test_size=0.2, random_state=123)



classifiers_estimator(['LR', 'RF', 'XGB'], 'accuracy', X_train, y_train, k_fold=10, shuffle=True, random_state=123)