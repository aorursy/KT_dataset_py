import os

import numpy as np

import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, cross_val_score



pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 100)



print(os.listdir('../input'))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print("Train has {} samples and {} variables.".format(train.shape[0],train.shape[1]))

print("Test has {} samples and {} variables.".format(test.shape[0],test.shape[1]))
train.head()
train.columns.values
for df in [train,test]:

    for c in df:

        if (df[c].dtype=='object'):

            lbl = LabelEncoder() 

            lbl.fit(list(df[c].values))

            df[c] = lbl.transform(list(df[c].values))
train.head()
ntrain = train.shape[0]

ntest = test.shape[0]

SEED = 2019 # for reproducibility

NFOLDS = 5

# Define Cross Validation

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)



# Define evaluation function (Root Mean Square Error)

def cv_rmse(model, X, y):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=folds))

    return (rmse)
cols_to_exclude = ['id', 'Overall Probability']

y_train = train['Overall Probability'].ravel() #ravel coverts a series to a numpy array

df_train_columns = [c for c in train.columns if c not in cols_to_exclude]





x_train = train[df_train_columns].values # converts a dataframe to a numpy array

x_test = test[df_train_columns].values
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=1000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )



score = cv_rmse(lightgbm, x_train, y_train)

print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()) )
lgb_model = lightgbm.fit(x_train, y_train)

prediction = lgb_model.predict(x_test)
sample_submission = pd.read_csv('../input/sample_submission.csv')

sub_df = pd.DataFrame({"id":sample_submission["id"].values})

sub_df["Overall Probability"] = prediction

sub_df["Overall Probability"] = sub_df["Overall Probability"].apply(lambda x: 1 if x>1 else 0 if x<0 else x)

sub_df.to_csv("submission.csv", index=False)