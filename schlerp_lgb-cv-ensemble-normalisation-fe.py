import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import random
# set up dataset
number_classes = 7
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# lets take a look...
train_df.head()
# create train datasets
X_train = train_df.drop(['Id', 'Cover_Type'], axis=1)
Y_train = train_df[['Cover_Type']].values
Y_train = Y_train.reshape(len(Y_train))

# create test dataset and ID's
X_test = test_df.drop(['Id'], axis=1)
ID_test = test_df['Id'].values
ID_test = ID_test.reshape(len(ID_test))

# concatenate both together for feature engineering and normalisation
X_all = pd.concat([X_train, X_test], axis=0)
# mean hillshade
def mean_hillshade(df):
    df['mean_hillshade'] = (df['Hillshade_9am'] + df['Hillshade_Noon'] + df['Hillshade_3pm']) / 3
    return df

# calculate the distance to hydrology using pythagoras theorem
def distance_to_hydrology(df):
    df['distance_to_hydrology'] = np.sqrt(np.power(df['Horizontal_Distance_To_Hydrology'], 2) + \
                                          np.power(df['Vertical_Distance_To_Hydrology'], 2))
    return df

# calculate diagnial distance down to sea level?
def diag_to_sealevl(df):
    df['diag_to_sealevel'] = np.divide(df['Elevation'], np.cos(180-df['Slope']))
    return df

# calculate mean distance to features
def mean_dist_to_feature(df):
    df['mean_dist_to_feature'] = (df['Horizontal_Distance_To_Hydrology'] + \
                                  df['Horizontal_Distance_To_Roadways'] + \
                                  df['Horizontal_Distance_To_Fire_Points']) / 3
    return df

def binned_columns(df):
    bin_defs = [
        # col name, bin size, new name
        ('Elevation', 200, 'Binned_Elevation'),
        ('Aspect', 45, 'Binned_Aspect'),
        ('Slope', 6, 'Binned_Slope'),
        ('Horizontal_Distance_To_Hydrology', 140, 'Binned_Horizontal_Distance_To_Hydrology'),
        ('Horizontal_Distance_To_Roadways', 712, 'Binned_Horizontal_Distance_To_Roadways'),
        ('Hillshade_9am', 32, 'Binned_Hillshade_9am'),
        ('Hillshade_Noon', 32, 'Binned_Hillshade_Noon'),
        ('Hillshade_3pm', 32, 'Binned_Hillshade_3pm'),
        ('Horizontal_Distance_To_Fire_Points', 717, 'Binned_Horizontal_Distance_To_Fire_Points')
    ]
    
    for col_name, bin_size, new_name in bin_defs:
        df[new_name] = np.floor(df[col_name]/bin_size)
    
    return df

X_all = mean_hillshade(X_all)
X_all = distance_to_hydrology(X_all)
X_all = diag_to_sealevl(X_all)
X_all = mean_dist_to_feature(X_all)
X_all = binned_columns(X_all)

X_all.head()
# normalise dataset
def normalise_df(df):
    df_mean = df.mean()
    df_std = df.std()    
    df_norm = (df - df_mean) / (df_std)
    return df_norm, df_mean, df_std

# define columsn to normalise
cols_non_onehot = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                   'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                   'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                   'Horizontal_Distance_To_Fire_Points', 'mean_hillshade',
                   'distance_to_hydrology', 'diag_to_sealevel', 'mean_dist_to_feature']

X_all_norm, df_mean, df_std = normalise_df(X_all[cols_non_onehot])

# replace columns with normalised versions
X_all = X_all.drop(cols_non_onehot, axis=1)
X_all = pd.concat([X_all_norm, X_all], axis=1)
# split back into test and train sets
X_train = np.array(X_all[:len(X_train)])
X_test = np.array(X_all[len(X_train):])
# set up Kfolds
n_splits = 12
kfolds = StratifiedKFold(n_splits=n_splits, shuffle=True)
scores = []
models = []

lrs = [0.1, 0.03, 0.01]
nls = [46, 48, 50]
n_ests = [200, 225, 250]

current_fold = 1
for train, test in kfolds.split(X_train, Y_train):
    print('commencing fold {}'.format(current_fold))
    # prepare data
    print('  preparing data...')
    Xt = X_train[train]
    Yt = Y_train[train]
    Xv = X_train[test]
    Yv = Y_train[test]

    # create, fit and test model
    
    n_est = random.choice(n_ests)
    lr = random.choice(lrs)
    nl = random.choice(nls)
    
    print('  building model with ests={}, lr={}, nl={}...'.format(n_est, lr, nl))
    
    classifier = lgb.LGBMClassifier(n_estimators=n_est,
                                    boosting_type='dart', 
                                    learning_rate=lr,
                                    num_leaves=nl,
                                    )
    print('  fitting model...')
    classifier.fit(Xt, Yt, eval_set=(Xv, Yv), early_stopping_rounds=50, verbose=False)
    print('  evaluating model...')
    y_pred = classifier.predict(Xv)
    score = accuracy_score(Yv, y_pred)
    scores.append(score)
    models.append(classifier)
    print('  fold {} accuracy: {} %'.format(current_fold, score*100))
    current_fold += 1

print('ensemble average accuracy: {} % (+/- {} %)'.format(np.mean(scores)*100, np.std(scores)*100))
print('testing ensemble accuracy on whole training set...')
y_preds = []
for index, classifier in enumerate(models):
    print('getting predictions from model {}...'.format(index+1))
    y_pred = classifier.predict(X_train)
    y_preds.append(y_pred)

print('taking average and rounding...')
y_pred = np.rint(np.mean(y_preds, axis=0))
y_pred = y_pred.astype(int)

print('calcualting accuracy...')
ensemble_accuracy = accuracy_score(Y_train, y_pred)

print('ensemble accuracy: {} %'.format(ensemble_accuracy*100))
print('producing test data predictions...')
y_preds = []
for index, classifier in enumerate(models):
    print('getting predictions from model {}...'.format(index+1))
    y_pred = classifier.predict(X_test)
    y_preds.append(y_pred)

print('taking average and rounding...')
y_pred = np.rint(np.mean(y_preds, axis=0))
y_pred = y_pred.astype(int)

print(max(y_pred))
print(min(y_pred))
sub = pd.DataFrame()
sub['Id'] = ID_test
sub['Cover_Type'] = y_pred
sub.to_csv('my_submission.csv', index=False)
print('good luck!')