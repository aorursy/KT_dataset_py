import numpy as np

import pandas as pd

import os

import sklearn

import scipy

import matplotlib.pyplot as plt

from copy import copy, deepcopy

import seaborn as sns

from sklearn.linear_model import Ridge, Lasso, LinearRegression, ElasticNet



print(os.listdir("../input"))
# Logger is a list where I will store everything we'll need to reimplement our predictions on a test set

LOGGER = []



# seed for reproducibility

np.random.seed(101)
df = pd.read_csv('../input/train.csv', index_col = "Id")

print(df.shape)

df.head()
df.hist(bins=50, figsize=(20,15))

plt.tight_layout(pad=0.4)
def plot_distribution(df, col):

    ser = df[col]

    i = 0

    while pd.isnull(ser.iloc[i]):

        i += 1

    if isinstance(ser.iloc[i], str):

        plt.figure(figsize = (6, 4))

        plt.title(col)

        counts = ser.value_counts(normalize = True)

        plt.pie(counts, labels = counts.index)

        plt.figtext(0.1, 0.9, "Null: {}%".format(str(100*np.sum(pd.isnull(ser))/ser.size)[:4]))

    

for col in df:

    plot_distribution(df, col)
corrmat = df.corr()

plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1.0, square=True, cmap="Blues");
Ids = np.arange(df.shape[0]) + 1

np.random.shuffle(Ids)

df = df.reindex(Ids)

shuffled_df = deepcopy(df)

df.head()
def class_reg_y(df, mode = "train"): 

    X_df, y_df = df.iloc[:, :-1], df.iloc[:, -1]

    classes_df = X_df.select_dtypes(include='object')

    

    classes_dict = {}

    data = []

    if mode == "train": 

        not_mode = "test"

    elif mode == "test":

        not_mode = "train" 

    else: 

        raise ValueError("mode must be either 'train' or 'test'")

    

    for cls in classes_df:

        series = classes_df[cls]

        

        combined_series = pd.concat(

            [series, pd.read_csv("../input/{}.csv".format(not_mode), index_col = "Id")[cls]]

        )

        

        uniq = combined_series.unique()

        

        classes_dict[cls] = reverse(uniq)

        

        data.append(

            [classes_dict[cls][string] for string in series]

        )

    

    classes_df = pd.DataFrame(

        data = np.array(data).T, 

        columns = classes_df.columns, 

        index = classes_df.index

    )

        

    regressors_df = X_df.select_dtypes(exclude='object')

    return classes_dict, classes_df.values, regressors_df.values, y_df.values



def class_reg_y_test(df, classes_dict): 

    X_df, y_df = df.iloc[:, :-1], df.iloc[:, -1]

    classes_df = X_df.select_dtypes(include='object')

    data = []

    for cls in classes_df:

        series = classes_df[cls]

        data.append(

            [classes_dict[cls][string] if string in classes_dict[cls].keys() else 0 for string in series]

        )

    classes_df = pd.DataFrame(

        data = np.array(data).T, 

        columns = classes_df.columns, 

        index = classes_df.index

    )

    regressors_df = X_df.select_dtypes(exclude='object')

    return classes_df.values, regressors_df.values, y_df.values



def split(arr):

    train, test = arr[:CUTOFF_VALUE], arr[CUTOFF_VALUE:]

    print(train.shape, test.shape)

    return train, test 



def reverse(arr): 

    dic = {}

    for index in range(arr.size):

        dic[arr[index]] = index

    return dic
class_dict, classes, regressors, y = class_reg_y(df)



LOGGER.append(class_dict)
# now try replacing nan with mean

def replace_nan_with_mean(arr):

    if np.sum(np.isnan(arr)) > 0:

        array = deepcopy(arr)

        col_mean = np.nanmean(array, axis = 0)

        inds = np.where(np.isnan(array))

        array[inds] = np.take(col_mean, inds[1])

        return array

    else:

        return arr



def validate_model(X, y, model_class = LinearRegression, k_folds = 50, **kwargs):

    train_scores = []

    val_scores = []

    

    Xcopy = replace_nan_with_mean(X)

    

    fold_size = (y.size // k_folds)

    for fold in range(k_folds):

        startIx = fold_size * fold

        endIx = startIx + fold_size 

        

        train_X = np.concatenate([Xcopy[:startIx, :], Xcopy[endIx:, :]], axis = 0)

        train_y = np.concatenate([y[:startIx], y[endIx:]])

        

        val_X = Xcopy[startIx:endIx, :]

        val_y = y[startIx:endIx]

        

        model = model_class(**kwargs)

        model.fit(train_X, train_y)

        train_scores.append( model.score(train_X, train_y) )

        val_scores.append( model.score(val_X, val_y) )

        

    print("Average Training R_sq: ", np.mean(train_scores))

    print("Average Validation R_sq: ", np.mean(val_scores))
from scipy.stats import mode



def remove_features(array):

    # array.shape : (n_examples, n_features)

    removed_features =  []

    output_array = []

    for feature in range(array.shape[1]):

        arr = array[:, feature]

        if (np.sum(np.isnan(arr))/arr.shape[0] > NAN_THRESH) or (np.sum(arr == mode(arr))/arr.shape[0] > MODE_THRESH):

            removed_features.append(feature)

        else:

            output_array.append(arr)

    print("{} features removed".format(len(removed_features)))

    output_array = np.stack(output_array, axis = -1)

    return removed_features, output_array



def remove_by_index(array, removed_features): 

    output_array = [array[:, i] for i in range(array.shape[1]) if i not in removed_features]

    return np.stack(output_array, axis = -1)
print("Before feature removal: ")



validate_model(regressors, y)



print("After feature removal: ")



NAN_THRESH = 0.9

MODE_THRESH = 0.9



validate_model(remove_features(regressors)[1], y)
removed_regressors, regressors = remove_features(regressors)

removed_classes, classes = remove_features(classes)



LOGGER.append(removed_regressors)

LOGGER.append(removed_classes)
print(classes.shape, regressors.shape)
def plot_distributions(array):

    # (n_samples, n_features)

    fig = plt.figure(figsize = (20, 20))

    n_features = array.shape[1]

    gridshape_x = int(np.sqrt(n_features))

    gridshape_y = int(n_features//gridshape_x + 1)

    

    feature_ix = 0

    for i in range(gridshape_x): 

        for j in range(gridshape_y):

            ax = plt.subplot2grid((gridshape_x, gridshape_y), (i, j))

            ax.hist(array[:, feature_ix], bins = 80)

            ax.set_title("Feature {}".format(feature_ix))

            feature_ix += 1

            if feature_ix >= array.shape[1]:

                return

plt.close('all')
plot_distributions(regressors)
def split_distributions(classes, regressors, regressor_indexes):

    new_regressors = deepcopy(regressors)

    new_classes = np.zeros((regressors.shape[0], regressors.shape[1] + len(regressor_indexes)))

    

    new_classes[:, :classes.shape[1]] = classes

    class_ix = classes.shape[1]

    for ix in regressor_indexes:

        zeros = regressors[:, ix] == 0

        temp = new_regressors[:, ix]

        gmean_array = regressors[:, ix][~np.isnan(regressors[:, ix]) & ~zeros].flatten()

        temp[zeros] = scipy.stats.mstats.gmean(gmean_array)

        

        new_regressors[:, ix] = temp

        assert np.nansum(new_regressors[:, ix] == 0) == 0

        

        new_classes[:, class_ix] = zeros.astype(new_classes.dtype)

        class_ix += 1

    

    

    return new_classes, new_regressors



bernoulli_transformed_regressors = [7, 8, 10, 11, 13, 23, 24, 25, 26]



classes, regressors = split_distributions(classes, regressors, bernoulli_transformed_regressors)



LOGGER.append(bernoulli_transformed_regressors)
def log_transform(arr):

    logged_features = []

    box_coxed_features = []

    array = deepcopy(arr)

    for i in range(array.shape[1]): 

        regressor = array[:, i]

        isnan = np.isnan(regressor)

        skew = scipy.stats.skew(regressor[~isnan])

        if skew > 0.8 and (np.sum(regressor <= 0) == 0):

                array[:, i] = np.log(regressor)

                logged_features.append(i)

        else:

            pass

    return array, logged_features



print("Before Log Transform: ")

validate_model(regressors, y)



print("After Log Transform: ")

validate_model(log_transform(regressors)[0], y)
plt.hist(y, bins = 100)

plt.title('Sale Prices') 

None
print("Trying to predict log(y)...")



print("Before Log Transform of X: ")

validate_model(regressors, np.log(y))



print("After Log Transform of X: ")

validate_model(log_transform(regressors)[0], np.log(y))
log_y = np.log(y)

regressors, logged_features = log_transform(regressors)



print("Logged features: ", logged_features) 

LOGGER.append(logged_features)
regressors = replace_nan_with_mean(regressors)
# observe multicollinearity

corrmat = pd.DataFrame(regressors).corr()

plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1.0, square=True, cmap="Blues");
plt.figure(figsize = (20, 12))

plt.hist(corrmat.values.flatten(), bins = 100)

plt.title('Training data R density before PCA')
from sklearn import decomposition



CUTOFF_VALUE = 1200



PCA = decomposition.PCA()



train_PCA_features = deepcopy(regressors[:CUTOFF_VALUE])



means = np.mean(train_PCA_features, axis = 0)

stdev = np.std(train_PCA_features, axis = 0)



PCA_features = (train_PCA_features - means)/stdev                          # assign everything to its z-score



PCA.fit(train_PCA_features)



train_PCA_features =  train_PCA_features @ PCA.components_.T               # components.shape = (_components, n_features)



corrmat = pd.DataFrame(train_PCA_features).corr()

plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1.0, square=True, cmap="Blues");
# try validation to see if PCA correlations are 0 for validation data



val_PCA_features = deepcopy(regressors[CUTOFF_VALUE:])



val_PCA_features = (val_PCA_features - means)/stdev



val_PCA_features =  val_PCA_features @ PCA.components_.T



corrmat = pd.DataFrame(val_PCA_features).corr()

plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1.0, square=True, cmap="Blues");
plt.figure(figsize = (20, 12))

plt.hist(corrmat.values.flatten(), bins = 100)

plt.title('Validation Data R-density after PCA')
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from scipy.stats import linregress
dummy_classes = pd.get_dummies(

        pd.concat(

            [

                pd.DataFrame(classes), pd.DataFrame( class_reg_y_test(pd.read_csv('../input/test.csv', index_col = "Id"), class_dict)[0])

            ], 

            axis = 0

        ).astype('str')

).values



dummy_classes = dummy_classes[:classes.shape[0], :]

test_dummy_classes = dummy_classes[classes.shape[0]:]



validate_model(dummy_classes, log_y)
include_dummies = []

r = []

for varIx in range(dummy_classes.shape[1]):

    r.append(linregress(dummy_classes[:, varIx].flatten(), log_y)[2])

    if abs(r[-1]) > 0.15:

        include_dummies.append(varIx)

    

plt.figure(figsize = (12, 9))

plt.hist(np.array(r)**2, bins = 100)

plt.title('Distribution of R_sq of dummy variables')
print("dummies included ", len(include_dummies))
def Random_Forest_Optimizer(X_data):

    for n_estimators in range(45, 130, 20):

        print("N: ", n_estimators)

        validate_model(X_data, y, RandomForestRegressor, n_estimators = n_estimators, max_depth = 9)

        # max_depth should be just under the sqrt of the number of classes
Random_Forest_Optimizer(X_data = dummy_classes[:, include_dummies])
# adaboost on linear regression on dummy classes

def AdaBoost_Optimizer(X_data):

    for n_estimators in range(45, 130, 20):

        print("N: ", n_estimators)

        validate_model(X_data, y, AdaBoostRegressor, n_estimators = n_estimators, base_estimator = DecisionTreeRegressor(max_depth = 9))
AdaBoost_Optimizer(X_data = dummy_classes[:, include_dummies])
def plot_r_hist(X, y): 

    rs = []

    for i in range(X.shape[1]): 

        r = linregress(X[:, i], y)[2]

        rs.append(r)

    print("std: ", np.std(rs))

    plt.hist(rs, bins = 20)
plot_r_hist(regressors, log_y)
plot_r_hist(PCA_features, log_y[:PCA_features.shape[0]])
n_estimators = 65

max_depth = 10



RFRegressor = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth)

RFRegressor.fit(regressors, log_y)

y_RF_regress =RFRegressor.predict(regressors)



validate_model(regressors, y, RandomForestRegressor, n_estimators = n_estimators, max_depth = max_depth, k_folds = 20)
linregressor = LinearRegression()

linregressor.fit(regressors, log_y)

y_linregress = linregressor.predict(regressors)



validate_model(regressors, y, RandomForestRegressor, n_estimators = n_estimators, max_depth = max_depth, k_folds = 20)
RFClasses = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth)

RFClasses.fit(dummy_classes, log_y) 

y_RF_classes = RFClasses.predict(dummy_classes)



validate_model(dummy_classes, y, RandomForestRegressor, n_estimators = n_estimators, max_depth = max_depth, k_folds = 20)
plt.plot(y_RF_regress, y_linregress, 'r.')

linregress(y_RF_regress, y_linregress)[2]**2
plt.plot(y_RF_regress, y_RF_classes, 'r.')

linregress(y_RF_regress, y_RF_classes)[2]**2
plt.plot(y_linregress, y_RF_classes, 'r.')

linregress(y_linregress, y_RF_classes)[2]**2
bagged_prediction = 0.6*linregressor.predict(regressors) + 0.2*RFRegressor.predict(regressors) + 0.2*RFClasses.predict(dummy_classes)



print("Combined Model r_sq: ", linregress(bagged_prediction, log_y)[2]**2) # this is a training score
bagged_prediction = linregressor.predict(regressors)



print("Combined Model r_sq: ", linregress(bagged_prediction, log_y)[2]**2) # this is a training score
def class_reg_y_test(df, classes_dict): 

    X_df, y_df = df.iloc[:, :-1], df.iloc[:, -1]

    classes_df = X_df.select_dtypes(include='object')

    data = []

    for cls in classes_df:

        series = classes_df[cls]

        data.append(

            [classes_dict[cls][string] if string in classes_dict[cls].keys() else 0 for string in series]

        )

    classes_df = pd.DataFrame(

        data = np.array(data).T, 

        columns = classes_df.columns, 

        index = classes_df.index

    )

    regressors_df = X_df.select_dtypes(exclude='object')

    return classes_df.values, regressors_df.values, y_df.values



def remove_features_test(array, features_list): # features list is the list of removed features

    output_array = []

    for i in range(array.shape[1]): 

        if i not in features_list:

            output_array.append(array[:, i])

    return np.stack(output_array, axis = -1)





def log_transform_test(arr, logged_features):

    array = deepcopy(arr)

    for i in range(array.shape[1]): 

        regressor = array[:, i]

        isnan = np.isnan(regressor)

        if i in logged_features:

            array[:, i] = np.log(regressor)

        else:

            pass

    return array
mock_df = pd.read_csv('../input/train.csv', index_col = "Id")

def process(mock_df, mode = "test"):

    # class regressor separator

    mock_classes, mock_regressors, mock_y = class_reg_y_test(mock_df, LOGGER[0])



    # remove features

    mock_regressors = remove_features_test(mock_regressors, LOGGER[1])

    mock_classes = remove_features_test(mock_classes, LOGGER[2])



    # log transform

    bernoulli_transformed_regressors = LOGGER[3]

    mock_classes, mock_regressors = split_distributions(mock_classes, mock_regressors, bernoulli_transformed_regressors) # same for training and test

    mock_regressors = log_transform_test(mock_regressors, LOGGER[4])



    mock_regressors = replace_nan_with_mean(mock_regressors)

    

    # make dummy variables

    if mode == 'train':

        not_mode = 'test'

    elif mode == 'test':

        not_mode = 'train'

    else:

        raise ValueError('mode must be "train" or "test".')



    mock_dummy_classes = pd.get_dummies(

        pd.concat(

            [

                pd.DataFrame(classes), pd.DataFrame( class_reg_y_test(pd.read_csv('../input/test.csv', index_col = "Id"), class_dict)[0])

            ], 

            axis = 0

        ).astype('str')

    ).values

    

    if mode == 'train':

        mock_dummy_classes = mock_dummy_classes[:classes.shape[0], :]

    elif mode == 'test':

        mock_dummy_classes = mock_dummy_classes[classes.shape[0]:, :]



    # make predictions with trained models

    mock_predictions = linregressor.predict(mock_regressors) # + 0.2*RFRegressor.predict(mock_regressors) + 0.2*RFClasses.predict(mock_dummy_classes)

    return mock_predictions

plt.figure(figsize = (8, 6))

plt.title("Original prediction distribution") 

plt.hist(bagged_prediction, bins = 100)



plt.figure(figsize = (8, 6))

plt.title("Re-implemented prediction distribution")

plt.hist(process(shuffled_df, mode = "train"), bins = 100)

None
test_df = pd.read_csv('../input/test.csv', index_col = "Id")



test_bagged_predictions = process(test_df, mode = "test")

plt.figure(figsize = (8, 6))

plt.title("Test prediction distribution")

plt.hist(test_bagged_predictions, bins = 100)

None
sub = pd.DataFrame()

sub['Id'] = test_df.index.values

sub['SalePrice'] = np.e**test_bagged_predictions # remember to reverse the logarithm because we were predicting log_y

sub.to_csv('submission.csv',index=False)