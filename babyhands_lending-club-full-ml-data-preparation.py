!pip install feature-engine 

# Feature Engine is an extremely useful module for feature pre-processing. It saves me a lot of time. 

# It is worth your time to check it out: https://feature-engine.readthedocs.io/en/latest/#

# There is no affiliation with the author of the package.
import pandas as pd

import numpy as np



from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



from feature_engine import outlier_removers 

from feature_engine.categorical_encoders import OneHotCategoricalEncoder, RareLabelCategoricalEncoder



# Display options



#pd.options.mode.chained_assignment = None #set it to None to remove SettingWithCopyWarning

pd.options.display.float_format = '{:.4f}'.format #set it to convert scientific noations such as 4.225108e+11 to 422510842796.00

pd.set_option('display.max_columns', 100) #  display all the columns

pd.set_option('display.max_rows', 100) # display all the rows

np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})





def remove_single_unique_value_features(dataframe):

    

    """

    Drop all the columns that only contain one unique value.

    not optimized for categorical features yet.

    

    """    

    cols_to_drop = dataframe.nunique()

    cols_to_drop = cols_to_drop.loc[cols_to_drop.values==1].index

    dataframe = dataframe.drop(cols_to_drop,axis=1)

    return dataframe
df = pd.read_csv('../input/lending-club-loan-data/loan.csv', low_memory=False)

print('df is loaded')



# the list contains features that are either proven useless or introduce look-ahead bias into data.

list_to_remove = ['last_pymnt_amnt','total_rec_prncp','total_pymnt',

                  'total_pymnt_inv','total_rec_int','total_rec_late_fee','total_rec_prncp',

                  'issue_d','earliest_cr_line','last_pymnt_d',

                  'last_credit_pull_d','id','member_id','settlement_date',

                  'next_pymnt_d','zip_code']



df.drop(list_to_remove,axis='columns',inplace=True)



df = df.infer_objects()

# drop any features that have more than 30% of NaN values in them.

df.dropna(axis=1,how='any',thresh=int(0.3*len(df)),inplace=True)

print(df.shape)
labels_to_drop = ['Current','Late (31-120 days)','Late (16-30 days)','In Grace Period','Default']

df = df[~df.loan_status.isin(labels_to_drop)]



dictionary = {'Does not meet the credit policy. Status:Fully Paid':'Fully Paid',

             'Does not meet the credit policy. Status:Charged Off':'Charged Off'}



df['loan_status'].replace(dictionary,inplace=True)

df['loan_status'].value_counts(normalize=True)
y = df[['loan_status']].values.ravel()

X = df.drop('loan_status',axis='columns')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 

                                            random_state=42, stratify=None)
# dividing training and testing data into categorical and numerical parts

nmrcl_X_train = X_train.select_dtypes(exclude=['object'])

nmrcl_X_test = X_test.select_dtypes(exclude=['object'])



ctgrcl_X_train = X_train.select_dtypes(include=['object'])

ctgrcl_X_test = X_test.select_dtypes(include=['object'])





print('Numerical part:')

print(nmrcl_X_train.shape)

print(nmrcl_X_test.shape)

print('Categorical part:')

print(ctgrcl_X_train.shape)

print(ctgrcl_X_test.shape)
""" 

Training df medians have to be saved as a pd.Series object othervise replace() 

method does not work when replacing NaN in testing df.

"""



training_medians = pd.Series(nmrcl_X_train.median()) # get the training medians 



nmrcl_X_train = nmrcl_X_train.fillna(training_medians) # fillna first

nmrcl_X_test = nmrcl_X_test.fillna(training_medians)



nmrcl_X_train = remove_single_unique_value_features(nmrcl_X_train) # remove features with 

nmrcl_X_test = remove_single_unique_value_features(nmrcl_X_test)   # single unique 
# 1.outlier replacement

# 2.discretization

# 3.scaling



# discretization is done using KBins algorithm, it's not an optimal solution as it can be seen from the warning output. 

#CAIMD is a much better way to bin the data. It's very intensive computationally though



capper = outlier_removers.Winsorizer(distribution='skewed', tail='both', fold=1.5)

discretizer = KBinsDiscretizer(n_bins=12, encode='ordinal', strategy='kmeans')

scaler = MinMaxScaler()



numerical_pipeline = Pipeline([('capper',capper),

                    ('discretizer',discretizer),

                    ('scaler',scaler)])



nmrcl_X_train = numerical_pipeline.fit_transform(nmrcl_X_train)

nmrcl_X_test = numerical_pipeline.transform(nmrcl_X_test)
# this is the way to access names of the columns, it's needed to convert 

# pipeline-produced np.array back to pd.DataFrame.

nmrc_feature_cols = numerical_pipeline.named_steps['capper'].variables



nmrcl_X_train = pd.DataFrame(nmrcl_X_train, columns=nmrc_feature_cols)

nmrcl_X_test = pd.DataFrame(nmrcl_X_test, columns=nmrc_feature_cols)
ctgrcl_X_train.fillna('other',inplace=True)

ctgrcl_X_test.fillna('other',inplace=True)



# two step pipeline:

# 1. rare labels (frequency below 1% are changed to 'rare')

# 2. n-1 OneHot encoding



encoder = RareLabelCategoricalEncoder(tol=0.01)

ohe_enc = OneHotCategoricalEncoder(top_categories=None,drop_last=True)



categorical_pipeline = Pipeline([('rare_label',encoder),('onehot',ohe_enc)])



ctgrcl_X_train = categorical_pipeline.fit_transform(ctgrcl_X_train)

ctgrcl_X_test = categorical_pipeline.transform(ctgrcl_X_test)



# reseting the index so all the dfs are alinable

ctgrcl_X_train.reset_index(drop=True,inplace=True)

ctgrcl_X_test.reset_index(drop=True,inplace=True)
le = LabelEncoder()

le.fit(y_train)

y_train = le.transform(y_train)

y_test = le.transform(y_test)
"""

LabelEncoder() output is a numpy array, it's missing the index which is later used for

concatanation of categorical, numercial and label data together. The following is a 

primitive solution but it works and there is no missalignment in the final df.



"""

y_train = pd.DataFrame(y_train)

y_test = pd.DataFrame(y_test)



y_train.columns = ['training labels']

y_test.columns = ['testing labels']
final_train = pd.concat([nmrcl_X_train,ctgrcl_X_train,y_train],axis=1)

final_test = pd.concat([nmrcl_X_test,ctgrcl_X_test,y_test],axis=1)



print(final_train.shape)

print(final_test.shape)
#final_train.to_csv('Data/pipeline_K_bins_train.csv')

#final_test.to_csv('Data/pipeline_K_bins_test.csv')