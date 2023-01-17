import csv

import numpy as np 

import pandas as pd 



import category_encoders as ce



from catboost import CatBoostClassifier, Pool



from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator



from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression, BayesianRidge, RidgeClassifier, RidgeClassifierCV

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier 

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.neural_network import MLPClassifier



from sklearn.model_selection import GridSearchCV, train_test_split, KFold, StratifiedKFold



from sklearn.metrics import roc_auc_score, auc

FILE_TRAIN = "/kaggle/input/cat-in-the-dat-ii/train.csv"

FILE_TEST = "/kaggle/input/cat-in-the-dat-ii/test.csv"



NAN_STRING_TO_REPLACE = 'zz'

NAN_VALUE_FLOAT = 8888.0

NAN_VALUE_INT = 8888

NAN_VALUE_STRING = '8888'



PARAMS_CATBOOST = dict()

PARAMS_CATBOOST['logging_level'] = 'Silent'

PARAMS_CATBOOST['eval_metric'] = 'AUC'

PARAMS_CATBOOST['loss_function'] = 'Logloss'

PARAMS_CATBOOST['iterations'] = 1000

PARAMS_CATBOOST['od_type'] = 'Iter' # IncToDec, Iter

PARAMS_CATBOOST['l2_leaf_reg'] = 300 # lambda, default 3

PARAMS_CATBOOST['learning_rate'] = 0.1 # alpha, default 0.3 if no l2_leaf_reg

PARAMS_CATBOOST['task_type'] = 'CPU'

PARAMS_CATBOOST['use_best_model']: True


#Description: Read Data from CSV file into Pandas DataFrame

def readData(inFile, sep=','):

    df_op = pd.read_csv(filepath_or_buffer=inFile, low_memory=False, encoding='utf-8', sep=sep)

    return df_op



# Description: Write Pandas DataFrame into CSV file

def writeData(df, outFile):

    f = open(outFile+'.csv', 'w')

    r = df.to_csv(index=False, path_or_buf=f)

    f.close()



# Write submission into file    

def print_submission_into_file(y_pred, df_test_id):

    l = []

    for myindex in range(y_pred.shape[0]):

        Y0 = y_pred[myindex]

        l.insert(myindex, Y0)

    

    df_pred = pd.DataFrame(pd.Series(l), columns=["target"])

    df_result = pd.concat([df_test_id, df_pred], axis=1, sort=False)

     

    f = open('submission.csv', 'w')

    r = df_result.to_csv(index=False, path_or_buf=f)

    f.close()

    return df_result

    

# Create a column with the number of missing values in each row:    

def set_missing_values_column(df):

    df['nb_missing_values'] = df.shape[1] - df.count(axis=1)

    return df





# Description; Ballance data for a pd.DataFrame, from scratch

# Input: df : pd.DataFrame without the Id column, containing the label column

# Output: X_res: pd.Dataframe for training ballanced containing the label column



def ballance_data_with_y(df):

    df_1 =  df[df["target"]==1]

    df_0 =  df[df["target"]==0]

    len1 = df_1.shape[0]

    len0 = df_0.shape[0]

    

    vmax = 0

    vmin = 1

    if len1 > len0:

        vmax = 1

        vmin = 0

        df_max = df_1

        df_min = df_0

    elif len1 < len0:

        vmax = 0

        vmin = 1

        df_max = df_0

        df_min = df_1

    else:

        return (df, Y)

    

    len_max = df_max.shape[0]

    len_min = df_min.shape[0]

    

    to_multiply = int(round(len_max/len_min))

    df_to_append = pd.concat([df_min] * to_multiply, ignore_index=True)

    

    len_append = df_to_append.shape[0]

    

    X_res = pd.concat([df_max, df_to_append], ignore_index=True)

    

    to_add = len_max - len_append

    if to_add > 0:

        df_to_add = df_min.sample(n=to_add, random_state=1)

        X_res = pd.concat([X_res, df_to_add], ignore_index=True)

    

    X_res = X_res.reset_index(drop=True)

    return X_res



# Convert categories to numeric

def convert_data_to_numeric_2(df, df_all, ar_train_transformed=None, enc=None):

    columns_for_ordinal_encoder = [cn for cn in df.columns if cn not in ['id', 'target', 'bin_0', 'bin_1', 'day', 'month', 'ord_0', 'ord_1', 'ord_2', 'nb_missing_values']]

    #print(columns_for_ordinal_encoder)

    enc_new = enc

    

    # Fillna with ZZ which will be the last value in alphabetical order. 

    df_ordinal = df[columns_for_ordinal_encoder].fillna(NAN_STRING_TO_REPLACE).applymap(lambda x: str(x))

    df_ordinal_all = df_all[columns_for_ordinal_encoder].fillna(NAN_STRING_TO_REPLACE).applymap(lambda x: str(x))

    ar_train_transformed_new = ar_train_transformed

    if enc == None and ar_train_transformed == None:

        enc_new = OrdinalEncoder(dtype=np.int16)

        enc_new.fit(df_ordinal_all)

        #print(enc_new.categories_)

        

        ar_train_transformed_new = enc_new.transform(df_ordinal_all)

    

    ar_ordinal_transformed = enc_new.transform(df_ordinal)

    count_columns = 0

    for cn in columns_for_ordinal_encoder:

        this_col_all = ar_train_transformed_new[:,count_columns]

        mx = this_col_all.max() # Find the index of ZZ, last one always, in our case, as OrdinalEncoder encodes by alphabetic order

        

        this_col_train = ar_ordinal_transformed[:,count_columns]

        this_col_nan_train = np.where(this_col_train==mx, NAN_VALUE_INT, this_col_train)

    

        ar_ordinal_transformed[:,count_columns] = this_col_nan_train

        count_columns = count_columns+1



    df_ordinal_transformed = pd.DataFrame(ar_ordinal_transformed, columns=columns_for_ordinal_encoder)

    

    df.update(df_ordinal_transformed)      



    ord_1_mapping = {'Novice' : 0, 'Contributor' : 1, 'Expert' : 2, 'Master': 3, 'Grandmaster': 4}

    ord_2_mapping = { 'Freezing': 0, 'Cold': 1, 'Warm' : 2, 'Hot': 3, 'Boiling Hot' : 4, 'Lava Hot' : 5}

    

    df['ord_1'] = df.loc[df.ord_1.notnull(), 'ord_1'].map(ord_1_mapping)

    df['ord_2'] = df.loc[df.ord_2.notnull(), 'ord_2'].map(ord_2_mapping)

    

    df['ord_0'] = df.loc[df.ord_0.notnull(), 'ord_0'].apply(lambda x: x-1)

    df['day'] = df.loc[df.day.notnull(), 'day'].apply(lambda x: x-1)

    df['month'] = df.loc[df.month.notnull(), 'month'].apply(lambda x: x-1)



    # Fill NaN with -1 first : 

    df = df.fillna(-1)

     

    # Optimize : 

    columns_only_16 = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']

    

    # columns with more than 127 and less than 3000 unique values:

    columns_int16 =  [cn for cn in df.columns if cn not in ['id', 'target'] and cn in columns_only_16]

    

    # columns with less than 127 unique values:

    columns_int8 = [cn for cn in df.columns if cn not in ['id', 'target', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']]



    df[columns_int8] = df[columns_int8].astype('int8')

    df[columns_int16] = df[columns_int16].astype('int16')

    

    df = df.applymap(lambda x: NAN_VALUE_INT if x == -1 else x)

    

    return (df, ar_train_transformed_new, enc_new)



# Parse OneHotEncoding for train and test, and for the features in cols : 

def get_ohe(df_train, df_test, cols):

    new_cols = ["ohe_"+col for col in cols]

    

    size_train = df_train.shape[0]

    size_test = df_test.shape[0]    

        

    df = df_train.append(df_test, ignore_index=True, sort=False)

    df_ohe = df.loc[:, cols].astype('category')

    

    df_ohe = df_ohe.applymap(lambda x: np.nan if x==NAN_VALUE_INT else x) # replace NAN_VALUE_INT with nan

    

    df_ohe = pd.get_dummies(df_ohe, prefix=new_cols, sparse=True, columns=cols)

     

    index_test_start = size_train

    index_test_end = size_train+size_test-1

    

    df_train = df_ohe.loc[0:size_train-1,:].astype('int16')

    df_test = df_ohe.loc[index_test_start:index_test_end,:].astype('int16')

    return (df_train, df_test)



# Encodes with simple imputer for DataFrame, cols, strategy, and returns the encoder

def encode_missing_simple_imputer(df, cols, strategy, si=None):

    df_to_transform = df[cols]

    

    if si==None:

        si = SimpleImputer(strategy=strategy, missing_values=NAN_VALUE_INT)

        ar = si.fit_transform(df_to_transform)

    else:

        ar = si.transform(df_to_transform)

    

    df_transformed = pd.DataFrame(ar, columns=cols)

    df.update(df_transformed)

    return (df, si)





# Encode with SimpleImputer

def encode_missing_simple_imputer(df, cols, strategy, si=None):

    df_to_transform = df[cols]

    

    if si==None:

        si = SimpleImputer(strategy=strategy, missing_values=NAN_VALUE_INT)

        ar = si.fit_transform(df_to_transform)

    else:

        ar = si.transform(df_to_transform)

    

    df_transformed = pd.DataFrame(ar, columns=cols)

    if strategy == 'most_frequent':

        df_transformed = df_transformed.applymap(lambda x: str(x)).astype('category')

       

    df.update(df_transformed)

    

    return (df, si)



def parse_all_kernels(X, Y, cat_features):

    names = [

        "RidgeClassifier",

        "RidgeClassifierCV",

        "CatBoostClassifier",        "GradientBoostingClassifier",

        "HistGradientBoostingClassifier",

        "ExtraTreesClassifier",

        "LinearDiscriminantAnalysis",

        "QuadraticDiscriminantAnalysis",

        "DecisionTreeClassifier",

        "RandomForestClassifier",

        "AdaBoostClassifier",

        "GaussianNB",

        "LogisticRegression",

        "MLPClassifier"

    ]



    classifiers = [

        RidgeClassifier(),

        RidgeClassifierCV(),

        CatBoostClassifier(**PARAMS_CATBOOST),

        GradientBoostingClassifier(),

        HistGradientBoostingClassifier(),

        ExtraTreesClassifier(),

        LinearDiscriminantAnalysis(),

        QuadraticDiscriminantAnalysis(),

        DecisionTreeClassifier(max_depth=5),

        RandomForestClassifier(max_depth=5, n_estimators=10),

        AdaBoostClassifier(),

        GaussianNB(), # Naive Bayes :-( 

        LogisticRegression(max_iter=10000),

        MLPClassifier(alpha=0.1, max_iter=1000, batch_size=1000, tol=0.0001, verbose=False)

    ]



    SPLITS = 3

    for name, clf in zip(names, classifiers):

        kf = StratifiedKFold(n_splits=SPLITS, shuffle=True)

        current_roc = 0.0



        count = 0

        for train_index, test_index in kf.split(X, Y):

            count = count+1

            #print("Split "+str(count)+" ... ")



            X_train, X_test = X.iloc[train_index], X.iloc[test_index]

            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]



            if name == "CatBoostClassifier":

                clf = CatBoostClassifier(**PARAMS_CATBOOST)



                train_dataset = Pool(data=X_train,

                     label=y_train,

                     cat_features=cat_features)



                eval_dataset = Pool(data=X_test,

                     label=y_test,

                     cat_features=cat_features)



                clf.fit(train_dataset,

                    use_best_model=True,

                    eval_set=eval_dataset)



                print("Count of trees in model = {}".format(clf.tree_count_))

            else:

                clf.fit(X_train, y_train)



            if name in ["RidgeClassifier", "RidgeClassifierCV"]:

                y_test_predict = clf.decision_function(X_test)

            else:

                y_test_predict = clf.predict_proba(X_test)[:,1]



            oof_auc_score_test = roc_auc_score(y_test, y_test_predict)

            current_roc += oof_auc_score_test

        final_roc = current_roc/float(SPLITS)



        print(name+": "+str(final_roc))

        print("-------------------")

# Read data :

df_train = readData(FILE_TRAIN)

df_test = readData(FILE_TEST)



# Add nb_missing_values

df_train = set_missing_values_column(df_train)

df_test = set_missing_values_column(df_test)



# Convert data to numeric. 

# Be careful tu use the same encoder for both train and test! The encoder is fit on both train + test.

df_all = df_train.append(df_test, sort=True)



(df_train, ar_train, enc_ord) = convert_data_to_numeric_2(df=df_train, df_all=df_all, ar_train_transformed=None, enc=None)

(df_test, ar_train, enc_ord) = convert_data_to_numeric_2(df=df_test, df_all=df_all, ar_train_transformed=ar_train, enc=enc_ord)



# Ballance data:

df_train = ballance_data_with_y(df_train)

Y = df_train["target"]



# I prefer to split the datasets, rather than making several copies, so as not to charge the RAM:

df_train = df_train.drop("id", axis=1) # we do not longer need this one

df_train = df_train.drop("target", axis=1) # saved already

df_train = df_train.reset_index(drop=True)

Y = Y.reset_index(drop=True)



df_test_id = df_test["id"] # we need this for the submission, so save ... 

df_test = df_test.drop("id", axis=1) # ... and drop afterwards

df_test = df_test.reset_index(drop=True)



cols = df_train.columns



X = df_train

X_testset = df_test



print(X.shape)
cols_bin = ["bin_0",  "bin_1", "bin_2", "bin_3", "bin_4"]

cols_ord_dm = ['day', 'month', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']

cols_ord_nom_04 = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

cols_ord_nom_59 = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



# Encode F for ord, dm, nom_5 -> nom_9:

strategy_F = 'most_frequent'

cols_F = cols_ord_dm + cols_ord_nom_59

(df_train, si) = encode_missing_simple_imputer(df_train, cols_F, strategy_F)

(df_test, si) = encode_missing_simple_imputer(df_test, cols_F, strategy_F, si)

    

# Encode OHE bin + nom_0 -> nom_4

cols_ohe = cols_bin + cols_ord_nom_04

(X_ohe_train, X_ohe_test) = get_ohe(df_train, df_test, cols_ohe)

    

df_train.drop(cols_ohe, axis=1, inplace=True)

df_test.drop(cols_ohe, axis=1, inplace=True)



df_train = df_train.reset_index(drop=True)

df_test = df_test.reset_index(drop=True)

X_ohe_train = X_ohe_train.reset_index(drop=True)

X_ohe_test = X_ohe_test.reset_index(drop=True)

Y = Y.reset_index(drop=True)



X = pd.concat([df_train, X_ohe_train], axis=1)

X_testset = pd.concat([df_test, X_ohe_test], axis=1)

cat_features = cols_F

columns = X.columns



print(X.shape)

print(X_testset.shape)

print("End")
# parse_all_kernels(X, Y, cat_features)

        

    