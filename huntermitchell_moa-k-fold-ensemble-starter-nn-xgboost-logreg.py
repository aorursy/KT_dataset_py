### Importing necessary libraries



import sys

sys.path.append('/kaggle/input/iterativestratification') # Multilabel Stratified K-Fold package



from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



import numpy as np 

import pandas as pd 



import tensorflow as tf

import tensorflow.keras.layers as L



from xgboost import XGBClassifier



from sklearn.linear_model import LogisticRegression

from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import log_loss



import time



import category_encoders as ce



%matplotlib inline



import matplotlib

import matplotlib.pyplot as plt
### Settings



SEED = 2020 

FOLDS = 4

EPOCHS = 25

BATCH_SIZE = 128





# Which models to use

include_xgboost = False

include_neural_net = True

include_logreg = False





# Whether to implement PCA

include_pca = True





lr_start=0.0001

lr_max=0.0003

lr_min=0.00001

lr_rampup_epochs=5

lr_sustain_epochs=2

lr_exp_decay=.7
### Creating dataframes



TEST_FEATURES_PATH = "/kaggle/input/lish-moa/test_features.csv"

TRAIN_FEATURES_PATH = "/kaggle/input/lish-moa/train_features.csv"

TRAIN_TARGETS_PATH = "/kaggle/input/lish-moa/train_targets_scored.csv"

TRAIN_TARGETS_NONSCORED_PATH = "/kaggle/input/lish-moa/train_targets_nonscored.csv"

SAMPLE_SUB_PATH = "/kaggle/input/lish-moa/sample_submission.csv"



test_features_df = pd.read_csv(TEST_FEATURES_PATH).sort_values(by='sig_id')

train_features_df = pd.read_csv(TRAIN_FEATURES_PATH).sort_values(by='sig_id')

train_targets_df = pd.read_csv(TRAIN_TARGETS_PATH).sort_values(by='sig_id')

train_targets_nonscored_df = pd.read_csv(TRAIN_TARGETS_NONSCORED_PATH)

sample_sub_df = pd.read_csv(SAMPLE_SUB_PATH).sort_values(by='sig_id')
### Features

#print(train_features_df.head()) 

#print(train_features_df.describe())





### Labels

#print(train_targets_df.head())

#print(train_targets_df.describe())





### Submission

#print(sample_sub_df.head())
### Check how many positive labels are in each class



value_counts_arr = np.sort([train_targets_df[col].value_counts()[1] for col in train_targets_df.columns])



print(value_counts_arr)
### Plot histogram of 1s counts in classes 



matplotlib.rcParams['figure.figsize'] = [10, 5]



plt.hist(value_counts_arr, 50, facecolor='g', alpha=0.75)

plt.xlabel('Number of 1\'s')

plt.ylabel('Number of classes')

plt.title('Value Counts of 1\'s in classes')

plt.show()
### Rename dataframes and drop 'id' columns



X = train_features_df.drop(columns=['sig_id'])

X_test = test_features_df.drop(columns=['sig_id'])

y = train_targets_df.drop(columns=['sig_id'])
# Encode categorical features

X_type = X['cp_type'].apply(lambda x: 1 if x == 'trt_cp' else 0)

X_dose = X['cp_dose'].apply(lambda x: 1 if x == 'D2' else 0)



# Encode categorical test features

X_type_test = X_test['cp_type'].apply(lambda x: 1 if x == 'trt_cp' else 0)

X_dose_test = X_test['cp_dose'].apply(lambda x: 1 if x == 'D2' else 0)
# Put encoded features back in



X = pd.concat([X_type,X_dose,X.drop(columns=['cp_type','cp_dose'])], axis=1)

X_test = pd.concat([X_type_test,X_dose_test,X_test.drop(columns=['cp_type','cp_dose'])], axis=1)
### Verify



#print(X.head())

#print(X_test.head())



#print(X.describe())

#print(X_test.describe())
### Initialize stratified k-fold object



skf = MultilabelStratifiedKFold(n_splits = FOLDS,random_state=SEED,shuffle=True)
def get_tf_model():

    model = tf.keras.Sequential([

        L.Flatten(input_shape=(1,X.shape[1])),

        L.BatchNormalization(),

        L.Dense(2000, activation='relu'),

        L.BatchNormalization(),

        L.Dropout(.4),

        L.Dense(1000, activation='relu'),

        L.BatchNormalization(),

        L.Dropout(.4),

        L.Dense(1000, activation='relu'),

        L.BatchNormalization(),

        L.Dropout(.4),

        L.Dense(206, activation='sigmoid')

    ])



    model.compile(

        optimizer='adam',

        loss = 'binary_crossentropy',

        metrics=['accuracy']

    )

    

    model.summary()

    

    return model
### learning rate schedule



def lrfn(epoch):

    

    if epoch < lr_rampup_epochs:

        lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

    elif epoch < lr_rampup_epochs + lr_sustain_epochs:

        lr = lr_max

    else:

        lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

    return lr
def plot_lr():

    rng = [i for i in range(EPOCHS)]

    y = [lrfn(x) for x in rng]

    plt.plot(rng, y)

    print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
### Fit model



def fit_model(model,X_train,X_valid,y_train,y_valid):



    start = time.time()

    

    print('Beginning to fit ',type(model))



    if 'tensorflow' in str(type(model)): # Fit neural net model

    

        model.fit(

            X_train,

            y_train,

            epochs=EPOCHS,

            verbose=1,

            batch_size=BATCH_SIZE,

            callbacks=[lr_schedule],

            validation_data=(X_valid,y_valid)

        )

    

    else: # Fit other type of model

    

        model.fit(X_train,y_train)



    print('Total time taken to fit model: ', time.time() - start, ' seconds')
### Make Predictions



def get_preds(model,X_valid,final=False):



    if 'tensorflow' in str(type(model)): # Neural Network model predictions 

        

        if final==True:

            preds = np.array(model.predict(X_test).astype("float64"))

        else:

            preds = np.array(model.predict(X_valid).astype("float64"))

            

    else:    # Other model predictions          

        

        if final==True:

            preds = np.array(model.predict_proba(X_test))

        else:

            preds = np.array(model.predict_proba(X_valid))

        

        preds = preds[:,:,1].T

    

    return preds
### Calculate validation score



def calc_loss(vals,preds):



    score = log_loss(np.ravel(vals),np.ravel(preds))

    

    cv_scores.append(score)



    print('Validation log loss score: {}'.format(score))
def run_model(model,X_train,X_valid,y_train,y_valid):



    ### fit the model

    fit_model(model,X_train,X_valid,y_train,y_valid)



    print('Getting validation predictions...')

    

    ### get the predictions

    temp_val_preds = get_preds(model,X_valid,final=False)

    

    ### calculate log loss

    calc_loss(y_valid,temp_val_preds)

    

    print('Calculating final predictions...')



    ### final preds

    final_preds.append(get_preds(model,X_valid,final=True))

    

    print('Done')
### XGBoost Model



if include_xgboost == True:

    

    model_1 = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))



    # The MultiOutputClassifier wrapper creates one model for each class (i.e. 206 different models total)



    # Using parameters from https://www.kaggle.com/fchmiel/xgboost-baseline-multilabel-classification

    params = {'estimator__colsample_bytree': 0.6522,

          'estimator__gamma': 3.6975,

          'estimator__learning_rate': 0.0503,

          'estimator__max_delta_step': 2.0706,

          'estimator__max_depth': 10,

          'estimator__min_child_weight': 31.5800,

          'estimator__n_estimators': 166,

          'estimator__subsample': 0.8639

         }



    model_1.set_params(**params)
### Neural Network Model



if include_neural_net == True:

    

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

    

    plot_lr()

    

    model_2 = get_tf_model()
### Logistic Regression model



if include_logreg == True:

    

    model_3 = MultiOutputClassifier(LogisticRegression(max_iter=10000, tol=0.1, C = 0.5,verbose=0,random_state = SEED))
cv_scores = []

final_preds = []
### Stratified K-Fold loop 



for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):

    

    print('Beginning fold',fold+1)

    print("TRAIN INDEX:", train_index, "VALID INDEX:", valid_index)

    

    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    if include_xgboost == True:

        

        run_model(model_1,X_train,X_valid,y_train,y_valid) # takes ~4 minutes

        

    if include_neural_net == True:

        

        run_model(model_2,X_train,X_valid,y_train,y_valid) # takes ~10 seconds with GPU

        

    if include_logreg == True:

        

        run_model(model_3,X_train,X_valid,y_train,y_valid) # takes ~8 min

        

### Show all CV scores



print('Cross Validation scores: ',cv_scores)
### Ensemble final predictions



print('Ensembling final predictions')

final_predictions = np.mean(np.array(final_preds),axis=0)



print('Done')
### Output final predictions



sample_sub_df.iloc[:,1:] = final_predictions

sample_sub_df.to_csv('submission.csv',index=False)
### Insight into final predictions



sample_sub_df.describe()