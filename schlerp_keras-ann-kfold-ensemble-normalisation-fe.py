import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras import Sequential
from keras.layers import Dense, Dropout, ELU, Softmax
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import StratifiedKFold
# set up dataset
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

X_all = mean_hillshade(X_all)
X_all = distance_to_hydrology(X_all)
X_all = diag_to_sealevl(X_all)
X_all = mean_dist_to_feature(X_all)
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
n_splits = 3
kfolds = StratifiedKFold(n_splits=n_splits, shuffle=True)

num_classes = 7
num_features = X_train.shape[-1]
def build_model(ELU_alpha=1.0, dropout=0.3):
    model = Sequential()

    model.add(Dense(1024, input_shape=(num_features,)))
    model.add(ELU(ELU_alpha))
    if dropout:
        model.add(Dropout(dropout))

    model.add(Dense(1024))
    model.add(ELU(ELU_alpha))

    model.add(Dense(512))
    model.add(ELU(ELU_alpha))

    model.add(Dense(256))
    model.add(ELU(ELU_alpha))
    if dropout:
        model.add(Dropout(dropout))
        
    model.add(Dense(num_classes))
    model.add(Softmax())
    
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    
    return model
scores = []
models = []
current_fold = 1
for train, test in kfolds.split(X_train, Y_train):
    print('commencing fold {}'.format(current_fold))
    # prepare data
    print('  preparing data...')
    Xt = X_train[train]
    Yt = to_categorical(Y_train[train]-1)
    Xv = X_train[test]
    Yv = to_categorical(Y_train[test]-1)

    # create, fit and test model
    print('  building model...')
    classifier = build_model()
    print('  fitting model...')
    classifier.fit(Xt, Yt, epochs=120, batch_size=512, verbose=False)
    print('  evaluating model...')
    score = classifier.evaluate(Xv, Yv, batch_size=1024, verbose=False)
    scores.append(score[-1])
    models.append(classifier)
    print('  fold {} accuracy: {}'.format(current_fold, score[-1]*100))
    current_fold += 1
    
print('ensemble average accuracy: {} % (+/- {} %)'.format(np.mean(scores)*100, np.std(scores)*100))
print('testing ensemble accuracy on whole training set...')
y_preds = []
for index, classifier in enumerate(models):
    print('getting predictions from model {}...'.format(index+1))
    y_onehot = classifier.predict(X_train, batch_size=1024)
    y_pred = np.argmax(y_onehot, axis=1)
    y_preds.append(y_pred)
  
print('taking average and rounding...')
y_pred = np.rint(np.mean(y_preds, axis=0)) + 1
y_pred = y_pred.astype(int)

print('calcualting accuracy...')
ensemble_accuracy = accuracy_score(Y_train, y_pred)

print('ensemble accuracy: {} %'.format(ensemble_accuracy*100))
y_preds = []
for index, classifier in enumerate(models):
    print('getting predictions from model {}...'.format(index+1))
    y_onehot = classifier.predict(X_test, batch_size=1024)
    y_pred = np.argmax(y_onehot, axis=1)
    y_preds.append(y_pred)

print('taking average and rounding...')
y_pred = np.rint(np.mean(y_preds, axis=0)) + 1
y_pred = y_pred.astype(int)
sub = pd.DataFrame()
sub['Id'] = ID_test
sub['Cover_Type'] = y_pred
sub.to_csv('my_submission.csv', index=False)
print('good luck!')