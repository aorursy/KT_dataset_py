# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import scipy as sc

        

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.pipeline import Pipeline

from keras.layers import Dense, Input, Dropout

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import FunctionTransformer



from sklearn.impute import SimpleImputer



from keras.optimizers import RMSprop, Adam, SGD

from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile

from sklearn.preprocessing import LabelEncoder

import keras.backend as K

from sklearn import metrics

import tensorflow as tf



import seaborn as sns

# Any results you write to the current directory are saved as output.
train_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")

train_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")



test_identity = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")

test_transaction = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
train_transaction.head()
train_identity.head()
print("train_transaction (Nbr Samples/Nbr Columns): ", train_transaction.shape) 

print("train_identity (Nbr Samples/Nbr Columns): ", train_identity.shape)
nans = train_transaction.isnull().mean(axis = 0).sort_values(ascending=False)*100

nans.reset_index().rename({"index": "column", 0: "NaNs rate"}, axis=1)
nans = train_identity.isnull().mean(axis = 0).sort_values(ascending=False)*100

nans.reset_index().rename({"index": "column", 0: "NaNs rate"}, axis=1)
ax = sns.countplot(x="isFraud", data=train_transaction)
categorical_features = [

    'ProductCD',

    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',

    'addr1', 'addr2',

    'P_emaildomain',

    'R_emaildomain',

    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'

]



data = train_transaction.merge(train_identity, on="TransactionID", how="left")
data = data.loc[:, data.isnull().mean() <= .5]

columns_to_keep = data.columns

categorical_cols = [c for c in categorical_features if c in data.columns]
# Class count

count_class_0, count_class_1 = data.isFraud.value_counts()



# Divide by class

df_class_0 = data[data['isFraud'] == 0]

df_class_1 = data[data['isFraud'] == 1]



df_class_0_under = df_class_0.sample(int(count_class_1 / 2))

balenced_df = pd.concat([df_class_0_under, df_class_1], axis=0)



balenced_df = balenced_df.reset_index(drop=True)
y = balenced_df.isFraud

X = balenced_df.drop(["TransactionID", "isFraud"],axis=1)
def auc(y_true, y_pred):

    return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)
def batch_generator(X, y, batch_size=16, shuffle=True):

    '''

    Return a random sample from X, y

    '''

    y = np.array(y)

    list_of_index_0 = np.where(y == 0)[0]

    list_of_index_1 = np.where(y == 1)[0]

    batch_0 = int(batch_size / 2)

    batch_1 = batch_size - batch_0

    

    while True:

        idx_0 = np.random.choice(list_of_index_0, size=batch_0, replace=False,)

        idx_1 = np.random.choice(list_of_index_1, size=batch_1, replace=False,)

        idx = np.concatenate((idx_0, idx_1), axis=None)



        if sc.sparse.issparse(X[idx]): 

            sample = X[idx].toarray()

        else:

            sample = X[idx]

        label = y[idx]

        

        yield sample, label
def create_model(optimizer="adam", dim=100):

    model = Sequential()

    

    model.add(Dense(50, activation='sigmoid', input_shape=(dim,)))

    model.add(Dropout(0.5), )

    

    model.add(Dense(20, activation='relu'))

    model.add(Dropout(0.2), )

 

    model.add(Dense(20, activation='relu'))

    model.add(Dropout(0.2), )

    

    model.add(Dense(1, activation='sigmoid', kernel_initializer='uniform'))

    

    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=[auc])

    return model
kfold_splits = 5

batch_size = 512

epochs = 100

optimizer = "NAdam"

imputing_strategy = "mean"
results = { "cv_val": [], "cv_train": [],}



# Instantiate the cross validator

skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)



# Loop through the indices the split() method returns

for index, (train_indices, val_indices) in enumerate(skf.split(X, y)):

    print ("Training on fold " + str(index+1) + "/" + str(kfold_splits) + "...")

  

    #Split 

    xtrain, xval = X.iloc[train_indices], X.iloc[val_indices]

    ytrain, yval = y.iloc[train_indices], y.iloc[val_indices]

    

    #LabelEncoding categorical columns

    label_encoders = {c: LabelEncoder() for c in categorical_cols}

    for c in categorical_cols:

        xtrain.loc[:,c], xval.loc[:,c] = xtrain[c].map(str), xval[c].map(str)

        #Handling Unknown Labels

        label_encoders[c].fit(np.concatenate((xtrain[c].values, np.array(["other"])), axis=None))

        xval.loc[:,c] = xval[c].map(lambda s: 'other' if s not in label_encoders[c].classes_ else s)

        #LabelEncoding

        xtrain.loc[:,c]  = label_encoders[c].transform(xtrain[c].values)

        xval.loc[:,c]  = label_encoders[c].transform(xval[c].values)

 

    #Imputing Missing Values

    imp = SimpleImputer(missing_values=np.nan, strategy=imputing_strategy).fit(xtrain)

    xtrain = imp.transform(xtrain)

    xval = imp.transform(xval) 

    

    #Normalize

    dim = xtrain.shape[1]

    scaler = StandardScaler(with_mean=False).fit(xtrain)

    xtrain = scaler.transform(xtrain)

    xval = scaler.transform(xval)

  

    # Create generators for fit_generator method

    train_gen = batch_generator(xtrain, ytrain, batch_size=batch_size)

    valid_gen = batch_generator(xval, yval, batch_size=batch_size)

    

    model = create_model(optimizer=optimizer, dim=dim)

   

    history = model.fit_generator(

            generator=train_gen,

            epochs=epochs,

            verbose=1,

            steps_per_epoch=xtrain.shape[0] // batch_size, #xtrain.shape[0] // batch_size

            validation_data=valid_gen,

            validation_steps=xval.shape[0] // batch_size, #xval.shape[0] // batch_size

        )

    #Evaluate Model

    val_score, train_score = metrics.roc_auc_score(yval, [x[0] for x in model.predict(xval,verbose=0)]), metrics.roc_auc_score(ytrain, [x[0] for x in model.predict(xtrain,verbose=0)])

    print("RoC AuC:   train %f val %f " % (train_score, val_score))

    results["cv_val"].append(val_score)

    results["cv_train"].append(train_score)



print("Final Score: train %f (+/- %f) val %f (+/- %f)" % (np.mean(results["cv_train"]), np.std(results["cv_train"]), np.mean(results["cv_val"]), np.std(results["cv_val"])))    