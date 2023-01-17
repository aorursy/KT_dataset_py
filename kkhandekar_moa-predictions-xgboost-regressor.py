'''Libraries'''



import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error,accuracy_score

from xgboost import XGBRegressor



import warnings

warnings.filterwarnings('ignore')
'''Data'''



#Sample

sample = pd.read_csv("../input/lish-moa/sample_submission.csv")



#Test

test_features = pd.read_csv("../input/lish-moa/test_features.csv",index_col='sig_id')



#Train

train_features = pd.read_csv("../input/lish-moa/train_features.csv",index_col='sig_id')

train_nonscore = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv",index_col='sig_id')

train_score = pd.read_csv("../input/lish-moa/train_targets_scored.csv",index_col='sig_id')
g_features = [feature for feature in train_features.columns if feature.startswith('g-')]

c_features = [feature for feature in train_features.columns if feature.startswith('c-')]

other_features = [feature for feature in train_features.columns if feature not in g_features and feature not in c_features]

                                                            



print(f'Number of g- Features: {len(g_features)}')

print(f'Number of c- Features: {len(c_features)}')

print(f'Number of Other Features: {len(other_features)} ({other_features})')
cols = train_score.columns

submission = pd.DataFrame({'sig_id': test_features.index})

total_loss = 0



SEED = 42
'''Build Model & Traning'''



for c, column in enumerate(cols,1):

    

    y = train_score[column]

    

    # Split

    X_train_full, X_valid_full, y_train, y_valid = train_test_split(train_features, y, train_size=0.9, test_size=0.1, random_state=SEED)

    X_train = X_train_full.copy()

    X_valid = X_valid_full.copy()

    X_test = test_features.copy()



    # One-hot encode the data (to shorten the code, we use pandas)

    X_train = pd.get_dummies(X_train)

    X_test = pd.get_dummies(X_test)

    X_valid = pd.get_dummies(X_valid)

    

    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

    

    

    # Define Regressor Model

    model = XGBRegressor(

                         tree_method = 'gpu_hist',

                         min_child_weight = 31.580,

                         learning_rate = 0.055,

                         colsample_bytree = 0.655,

                         gamma = 3.705,

                         max_delta_step = 2.080,

                         max_depth = 25,

                         n_estimators = 170,

                         #subsample =  0.864, 

                         subsample =  0.910,

                         booster='dart',

                         validate_parameters = True,

                         grow_policy = 'depthwise',

                         predictor = 'gpu_predictor'

                              

                        )

                        

    # Train Model

    model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    

    # Loss

    mae = mean_absolute_error(y_valid,pred)

    mdae = median_absolute_error(y_valid,pred)

    mse = mean_squared_error(y_valid,pred)

    

    total_loss += mae

    

    # Prediction

    predictions = model.predict(X_test)

    submission[column] = predictions

    

    print("Regressing through col-"+str(c)+", Mean Abs Error: "+str(mae)+", Median Abs Error: "+str(mdae)+", Mean Sqrd Error: "+str(mse))





print("Loss: ", total_loss/206)
# Saving the submission

submission.to_csv('submission.csv', index=False)