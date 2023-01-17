import numpy as np



# Set the random seed for reproducability

np.random.seed(42)
import pandas as pd
# Reads in the csv-files and creates a dataframe using pandas



# base_set = pd.read_csv('data/housing_data.csv')

# benchmark = pd.read_csv('data/housing_test_data.csv')

# sampleSubmission = pd.read_csv('data/sample_submission.csv')
base_set = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

benchmark = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')

sample_submission = pd.read_csv('../input/porto-seguro-safe-driver-prediction/sample_submission.csv')
%matplotlib inline



import matplotlib.pyplot as plt

import seaborn as sns
base_set.head()
benchmark.head()
base_set.info()
benchmark.info()
base_set.describe()
correlations = base_set.corr()

correlations["target"]
base_set.hist(bins=50, figsize=(15,15))

plt.show()
base_set_id = base_set['id']

benchmark_id = benchmark['id']



base_set = base_set.drop(columns=['id'])

benchmark = benchmark.drop(columns=['id'])
base_non_calc_cols = [c for c in base_set.columns if (not c.startswith('ps_calc_'))]

benchmark_non_calc_cols = [c for c in benchmark.columns if (not c.startswith('ps_calc_'))]



base_set = base_set[base_non_calc_cols]

benchmark = benchmark[benchmark_non_calc_cols]
from keras.utils import to_categorical



# Not sure how to do this yet
base_set = base_set.replace(-1, np.NaN)

benchmark = benchmark.replace(-1, np.NaN)



base_set = base_set.fillna(base_set.median())

benchmark = benchmark.fillna(benchmark.median())
base_set.isnull().any()
benchmark.isnull().any()
labels_column = 'target'



X = base_set.drop(columns=[labels_column])

Y = pd.DataFrame(base_set[labels_column], columns=[labels_column])
X.head()
Y.head()
benchmark.head()
from sklearn.model_selection import train_test_split



train_to_valtest_ratio = .5

validate_to_test_ratio = .5



# First split our main set

(X_train,

 X_validation_and_test,

 Y_train,

 Y_validation_and_test) = train_test_split(X, Y, test_size=train_to_valtest_ratio)



# Then split our second set into validation and test

(X_validation,

 X_test,

 Y_validation,

 Y_test) = train_test_split(X_validation_and_test, Y_validation_and_test, test_size=validate_to_test_ratio)
def gini(y_true, y_pred):

    # check and get number of samples

    assert y_true.shape == y_pred.shape

    n_samples = y_true.shape[0]

    

    # sort rows on prediction column 

    # (from largest to smallest)

    arr = np.array([y_true, y_pred]).transpose()

    true_order = arr[arr[:,0].argsort()][::-1,0]

    pred_order = arr[arr[:,1].argsort()][::-1,0]

    

    # get Lorenz curves

    L_true = np.cumsum(true_order) / np.sum(true_order)

    L_pred = np.cumsum(pred_order) / np.sum(pred_order)

    L_ones = np.linspace(1/n_samples, 1, n_samples)

    

    # get Gini coefficients (area between curves)

    G_true = np.sum(L_ones - L_true)

    G_pred = np.sum(L_ones - L_pred)

    

    # normalize to true Gini coefficient

    return G_pred/G_true
from keras.models import Sequential

from keras.layers import Dense, Dropout



model = Sequential([

    Dense(64, activation='relu', input_dim=X_train.shape[1]),

    Dropout(.30),

    Dense(64, activation='relu'),

    Dropout(.15),

    Dense(32, activation='relu'),

    Dropout(.15),

    Dense(16, activation='relu'),

    Dense(1),

])



model.summary()
import keras.backend as K



model.compile(optimizer='adam', # adam, sgd, adadelta

              loss='binary_crossentropy',

              metrics=['binary_crossentropy'])
from keras.callbacks import EarlyStopping



early_stopper = EarlyStopping(patience=3)



training_result = model.fit(X_train, Y_train,

                            batch_size=4096,

                            epochs=256,

                            validation_data=(X_validation, Y_validation),

                            callbacks=[early_stopper])
print(training_result.history)



# Plot model accuracy over epoch

plt.plot(training_result.history['binary_crossentropy'])

plt.plot(training_result.history['val_binary_crossentropy'])

plt.title('Model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()



# Plot model loss over epoch

plt.plot(training_result.history['loss'])

plt.plot(training_result.history['val_loss'])

plt.title('Model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
validate_result = model.test_on_batch(X_validation, Y_validation)

validate_result
test_result = model.test_on_batch(X_test, Y_test)

test_result
from sklearn.ensemble import RandomForestRegressor



rfr_model = RandomForestRegressor()

rfr_model.fit(X_train, Y_train)



rfr_predictions = rfr_model.predict(X_test)
rfr_error =  gini(Y_test['target'], rfr_predictions)

rfr_error
import re



regex = re.compile(r"[|]|<", re.IGNORECASE)



# XGBoost does not support some of the column names



X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]

X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]



from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV



import scipy.stats as st



one_to_left = st.beta(10, 1)  

from_zero_positive = st.expon(0, 50)



xgb_reg = XGBRegressor(nthreads=-1)



xgb_gs_params = {  

    "n_estimators": st.randint(3, 40),

    "max_depth": st.randint(3, 40),

    "learning_rate": st.uniform(0.05, 0.4),

    "colsample_bytree": one_to_left,

    "subsample": one_to_left,

    "gamma": st.uniform(0, 10),

    'reg_alpha': from_zero_positive,

    "min_child_weight": from_zero_positive,

}



xgb_gs = RandomizedSearchCV(xgb_reg, xgb_gs_params, n_jobs=1)  

xgb_gs.fit(X_train.values, Y_train)  



xgb_model = xgb_gs.best_estimator_ 



xgb_predictions = xgb_model.predict(X_test.values)
xgb_error =  gini(Y_test['target'], xgb_predictions)

xgb_error
print(f'NN:                                 {test_result[0]}')

print(f'RandomForestRegressor Gini:         {rfr_error}')

print(f'XGBRegressor Gini:                  {xgb_error}')
benchmark.head()
X.head()
target = xgb_model.predict(benchmark.values)
len(target)
target
submission = pd.DataFrame({

    'id': benchmark_id,

    'target': target.flatten()

})
submission.head()
# Stores a csv file to submit to the kaggle competition

submission.to_csv('submission.csv', index=False)