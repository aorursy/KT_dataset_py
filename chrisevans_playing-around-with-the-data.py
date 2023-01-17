import numpy as np

import pandas as pd
train_df = pd.read_csv('../input/train/train.csv')

train_df.head()
# Key here is that we use the values from train_df => we can use the same encoding for test_df

def make_dummies(full_df, col_name):

    vals = sorted(set(train_df[col_name].values))

    df = full_df.copy(deep=True)

    for val in vals:

        df[col_name + "_is_" + str(val)] = df[col_name].apply(lambda x: 1 if x==val else 0)

    df = df.drop([col_name], axis=1)

    return(df)

    
make_dummies(train_df[['Color1']], 'Color1').head()


# Put all the features in one place. This logic is applied the same to train_df and test_df

def make_features(full_df):

    df = full_df.copy(deep=True)

    if 'AdoptionSpeed' in df.columns:

        df = df.drop(['AdoptionSpeed'], axis=1) # Remove label

    # Columns to drop

    df = df.drop(['Name', 'RescuerID', 'PetID'], axis=1)

    # New feats

    df['DescriptionLength'] = df['Description'].apply(lambda x: len(str(x)))

    df = df.drop(['Description'], axis=1)

    df['IsYear'] = df['Age'].apply(lambda x: 1 if x%12==0 else 0)

    # Dummies

    #df = make_dummies(df, 'Color1')

    #df = make_dummies(df, 'Color2')

    #df = make_dummies(df, 'Color3')

    #df = make_dummies(df, 'Vaccinated')

    #df = make_dummies(df, 'Dewormed')

    #df = make_dummies(df, 'Sterilized')

    #df = make_dummies(df, 'State')

    #df = make_dummies(df, 'Breed1')

    #df = make_dummies(df, 'Breed2')

    return df

from sklearn.model_selection import train_test_split



# Make some features into factors

X = make_features(train_df).values

Y = train_df[['AdoptionSpeed']].values.ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)
# Try Random Forest Boosting Classifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, Y_train)

pd.crosstab(Y_test, rfc.predict(X_test))
# Try Gradient Boosting Classifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train, Y_train)

pd.crosstab(Y_test, gbc.predict(X_test))
# Try the popular LightGBM

import lightgbm as lgb



d_train = lgb.Dataset(X_train, label=Y_train)

clf = lgb.train({

    'objective': 'multiclass',

    'num_class' : 5,

    }, d_train, 100)

preds = np.array([np.argmax(pred) for pred in clf.predict(X_test)])

pd.crosstab(Y_test, preds)

# Prepare submission

test_df = pd.read_csv('../input/test/test.csv')

test_X = make_features(test_df).values



# Combine predictions from RFC, GBC and LGBM

rfc_preds = rfc.predict(test_X)

gbc_preds = gbc.predict(test_X)

lgbm_preds = np.array([np.argmax(pred) for pred in clf.predict(test_X)])

combined_preds = np.round((rfc_preds + gbc_preds + lgbm_preds) / 3).astype(int)



submission_df = pd.DataFrame(data={'PetID': test_df['PetID'].values, 'AdoptionSpeed': combined_preds})
submission_df.to_csv('submission.csv', index=False)