# Wasting time on Kaggle competitions

import pandas as pd

# load the data

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

print(test_df.info())
print(train_df.info())
# convert all of the AgeuponOutcome values into weeks

def convert_AgeuponOutcome_to_weeks(df):
    result = {}
    for k in df['AgeuponOutcome'].unique():
        if type(k) != type(""):
            result[k] = -1
        else:
            v1, v2 = k.split()
            if v2 in ["year", "years"]:
                result[k] = int(v1) * 52
            elif v2 in ["month", "months"]:
                result[k] = int(v1) * 4.5
            elif v2 in ["week", "weeks"]:
                result[k] = int(v1)
            elif v2 in ["day", "days"]:
                result[k] = int(v1) / 7
                
    df['_AgeuponOutcome'] = df['AgeuponOutcome'].map(result).astype(float)
    df = df.drop('AgeuponOutcome', axis = 1)
                
    return df

train_df = convert_AgeuponOutcome_to_weeks(train_df)
test_df = convert_AgeuponOutcome_to_weeks(test_df)
print("AgeuponOutcome conversion done.")
# add a column with a Name frequency count

names = pd.concat([test_df['Name'], train_df['Name']])
values = dict(names.value_counts())

train_df['_NameFreq'] = train_df['Name'].map(values)
test_df['_NameFreq'] = test_df['Name'].map(values)

train_df['_NameFreq'] = train_df['_NameFreq'].fillna(99999)
test_df['_NameFreq'] = test_df['_NameFreq'].fillna(99999)

print(test_df.info())
print(train_df.info())

print("Name frequency count done.")
# convert all of the remaining features to numeric values

def convert_to_numeric(df):
    for col in ['Name', 'AnimalType', 'SexuponOutcome',
                'Breed', 'Color', 'OutcomeType']:
        if col in df.columns:
            _col = "_%s" % (col)
            values = df[col].unique()
            _values = dict(zip(values, range(len(values))))
            df[_col] = df[col].map(_values).astype(int)
            df = df.drop(col, axis = 1)
    return df

train_df = convert_to_numeric(train_df)
test_df = convert_to_numeric(test_df)

print("Numerical conversion of features done.")
# fix the DateTime column

def fix_date_time(df):
    def extract_field(_df, start, stop):
        return _df['DateTime'].map(lambda dt: int(dt[start:stop]))
    df['Year'] = extract_field(df,0,4)
    df['Month'] = extract_field(df,5,7)
    df['Day'] = extract_field(df,8,10)
    df['Hour'] = extract_field(df,11,13)
    df['Minute'] = extract_field(df,14,16)
    
    return df.drop(['DateTime'], axis = 1)

train_df = fix_date_time(train_df)
test_df = fix_date_time(test_df)

print("DateTime column split into parts done.")
# re-index train_df so that ID is first and Target (_OutcomeType) is last

train_df = train_df.reindex(columns = ['AnimalID', '_Name', '_NameFreq',
                                       '_AnimalType', '_SexuponOutcome',
                                       '_AgeuponOutcome', '_Breed', '_Color',
                                       'Year', 'Month', 'Day', 'Hour', 'Minute',
                                       '_OutcomeType'])
print(train_df.info())
# split the data into a training set (80%) and a validation set (20%)

cut = int(len(train_df) * 0.8)
_validation_df = train_df[cut:]
_train_df = train_df[:cut]

print(len(_train_df))
print(len(_validation_df))
# build a classifier with scikit-learn

import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

A1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2),
                        n_estimators = 100,
                        learning_rate = 0.1)

classifiers = [c.fit(_train_df.values[:,1:-1],
                     _train_df.values[:,-1].astype(int)) \
               for c in [A1]]
results = [c.predict_proba(_validation_df.values[:,1:-1]) \
           for c in classifiers]
print(results[0])
# calculate the log loss of result

from sklearn.metrics import log_loss

print([log_loss(_validation_df.values[:,-1].astype(int), r) for r in results])
# re-build the selected classifier on the entire training set

ab = classifiers[0].fit(train_df.values[:,1:-1],
                        train_df.values[:,-1].astype(int))

# and use the classifier on test_df

ab_result = ab.predict_proba(test_df.values[:,1:])
ab_sub_df = pd.DataFrame(ab_result, columns=['Adoption', 'Died', 'Euthanasia',
                                             'Return_to_owner', 'Transfer'])
ab_sub_df.insert(0, 'ID', test_df.values[:,0].astype(int))

print(ab_sub_df)

# write to submission files

ab_sub_df.to_csv("submission.csv", index = False)

print("Done.")