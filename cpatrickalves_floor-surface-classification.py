# If you will use tqdm

#!pip install ipywidgets 

#!jupyter nbextension enable --py widgetsnbextension

#!pip install -r requirements.txt
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from tqdm import tqdm_notebook as tqdm

%matplotlib inline
# Folder with datasets

data_folder = "data/"



# Running on kaggle?

kaggle = True



if kaggle:

    data_folder = "/kaggle/input/competicao-dsa-machine-learning-sep-2019/"



# Load the data for training ML models

xtrain = pd.read_csv(data_folder + "X_treino.csv")

ytrain = pd.read_csv(data_folder + "y_treino.csv") # Target

train_data = pd.merge(xtrain, ytrain, how = "left", on = "series_id")



#Load the Test dataset to predict the results (used for submission)

xtest = pd.read_csv(data_folder + "X_teste.csv")

test_data = xtest



# Submission data

submission = pd.read_csv(data_folder + "sample_submission.csv")



# Showing the number of samples and columns for each dataset

print(train_data.shape)

print(test_data.shape)
train_data.head()
test_data.head()
# Check unique values

train_count_series = len(train_data.series_id.unique())

test_count_series = len(test_data.series_id.unique())

train_freq_distribution_surfaces = train_data.surface.value_counts()



print(f"Number of time series in train dataset: {train_count_series}")

print(f"Number of time series in test dataset: {test_count_series}\n")



print(f"Surfaces frequency distribution in train dataset:\n{train_freq_distribution_surfaces}")

train_freq_distribution_surfaces.plot(kind="barh", figsize=(10,5))

plt.title("Sample distribution by class")

plt.ylabel("Number of time series")

plt.show()
plt.subplots_adjust(top=0.8)

for i, col in enumerate(xtrain.columns[3:]):

    g = sns.FacetGrid(train_data, col="surface", col_wrap=5, height=3, aspect=1.1)

    g = g.map(sns.distplot, col)    

    g.fig.suptitle(col, y=1.09, fontsize=23)
# Function that performs all data transformation and pre-processing

def data_preprocessing(df, labeled=False):

    

    # New dataframe that will saves the tranformed data

    X = pd.DataFrame()



    # This list will save the type of surface for each series ID

    Y = []



    # The selected attributes used in training

    selected_attributes = ['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W', 

                           'angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z', 'linear_acceleration_X',

                           'linear_acceleration_Y', 'linear_acceleration_Z']



    # The total number of series in training data

    total_test_series = len(df.series_id.unique())        



    for series in tqdm(range(total_test_series)):

    #for series in range(total_test_series):

        

        # Filter the series id in the DataFrame

        _filter = (df.series_id == series)



        # If data with labels

        if labeled:

            # Saves the type of surface (label) for each series ID        

            Y.append((df.loc[_filter, 'surface']).values[0])



        # Compute new values for each attribute

        for attr in selected_attributes:

            

            # Compute a new attribute for each series and save in the X DataFrame

            X.loc[series, attr + '_mean'] = df.loc[_filter, attr].mean()

            X.loc[series, attr + '_std'] = df.loc[_filter, attr].std() 

            X.loc[series, attr + '_min'] = df.loc[_filter, attr].min()

            X.loc[series, attr + '_max'] = df.loc[_filter, attr].max()

            X.loc[series, attr + '_kur'] = df.loc[_filter, attr].kurtosis()

            X.loc[series, attr + '_skew'] = df.loc[_filter,attr].skew()          

            

    

    return X,Y

# Apply the Pre-Processing to train data

X_train, Y_train = data_preprocessing(train_data, labeled=True)



# Here is the result DataFrame

X_train.head()
# Transform the Y list in an array

Y_train=np.array(Y_train)



# Print the size

X_train.shape, Y_train.shape
# Apply the Pre-Processing to test data

X_test, _ = data_preprocessing(test_data, labeled=False)



X_test.head()
print(X_test.shape)
# Importing packages

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
# Get the labels (concrete, tiled, wood, etc.)

unique_labels=list(train_data.surface.unique())



# Encode the train labels with value between 0 and n_classes-1 to use in Random Forest Classifier.

le = LabelEncoder()

Y_train_encoded = le.fit_transform(Y_train)

Y_train_encoded
# Function to perform all training steps for LGBM

def train_lgbm_model(X_train, Y_train, X_test):

    

    # Variables that save the probabilities of each class 

    predicted = np.zeros((X_test.shape[0],9))

    measured= np.zeros((X_train.shape[0],9))

    

    # Create a dictionary that saves the model create in each fold

    models = {}

    

    # Used to compute model accuracy

    all_scores = 0

    

    # Use Stratified ShuffleSplit cross-validator

    # Provides train/test indices to split data in train/test sets.

    n_folds = 5

    sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.30, random_state=10)



    # Control the number of folds in cross-validation (5 folds)

    k=1



    # From the generator object gets index for series to use in train and validation

    for train_index, valid_index in sss.split(X_train, Y_train):



        # Saves the split train/validation combinations for each Cross-Validation fold

        X_train_cv, X_validation_cv = X_train.loc[train_index,:], X_train.loc[valid_index,:]

        Y_train_cv, Y_validation_cv = Y_train[train_index], Y_train[valid_index]

        

        # Create the model

        lgbm = lgb.LGBMClassifier(objective='multiclass', is_unbalance=True, max_depth=10,

                               learning_rate=0.05, n_estimators=500, num_leaves=30)



        # Training the model

        # eval gets the tuple pairs to use as validation sets

        lgbm.fit(X_train_cv, Y_train_cv,

            eval_set=[(X_train_cv, Y_train_cv), (X_validation_cv, Y_validation_cv)],

            early_stopping_rounds=60, # stops if 60 consequent rounds without decrease of error

            verbose=False, eval_metric='multi_error')        

       

        # Get the class probabilities of the input samples

        # Save the probabilities for submission

        y_pred = lgbm.predict_proba(X_test)

        predicted += y_pred

        

        # Save the probabilities of validation

        measured[valid_index] = lgbm.predict_proba(X_validation_cv)

        

        # Cumulative sum of the score

        score = lgbm.score(X_validation_cv,Y_validation_cv)        

        all_scores += score

                

        print("Fold: {} - LGBM Score: {}".format(k, score)) 

        

        # Saving the model

        models[k] = lgbm

        k += 1  

               

    # Compute the mean probability

    predicted /= n_folds

    # Save the mean score value

    mean_score = all_scores/n_folds

    # Save the first trained model

    trained_model = models[1]

    

            

    return measured, predicted, mean_score, trained_model
# Models is a dict that saves the model create in each fold in cross-validation

measured_lgb, predicted_lgb, accuracy_lgb, model_lgb = train_lgbm_model(X_train, Y_train_encoded, X_test)

print(f"\nMean accuracy for LGBM: {accuracy_lgb}")
# Plot the Feature Importance for the first model created

plt.figure(figsize=(15,30))

ax=plt.axes()

lgb.plot_importance(model_lgb, height=0.5, ax=ax)

plt.show()
# Removing features with a importance score bellow 400

# The 400 values was chosen from several tests

features_to_remove = []

feat_imp_threshold = 400



# A list of features and importance scores

feat_imp = []

for i in range(len(X_train.columns)):

    feat_imp.append((X_train.columns[i], model_lgb.feature_importances_[i]))



for fi in feat_imp:

    if fi[1] < feat_imp_threshold:

        features_to_remove.append(fi[0])



print(f"Number of feature to be remove: {len(features_to_remove)}\n")    

print(features_to_remove)
# Removing features

X_train_v2 = X_train.copy()

X_test_v2 = X_test.copy()



for f in features_to_remove:

    del X_train_v2[f]

    del X_test_v2[f]

    

X_train_v2.shape, X_test_v2.shape
# Train a new set of models

measured_lgb, predicted_lgb, accuracy_lgb, lgbm_model = train_lgbm_model(X_train_v2, Y_train_encoded, X_test_v2)

print(f"\nMean accuracy for LGBM: {accuracy_lgb}")
# Function to perform all training steps

def train_rfc(X_train, Y_train, X_test):

    

    # Create a dictionary that saves the model create in each fold

    models = {}

            

    # Variables that save the probabilities of each class 

    predicted = np.zeros((X_test.shape[0],9))

    measured = np.zeros((X_train.shape[0],9))

    

    # Use Stratified ShuffleSplit cross-validator

    # Provides train/test indices to split data in train/test sets.

    n_folds = 5

    sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.30, random_state=10)



    # Control the number of folds in cross-validation (5 folds)

    k=1

    

    # Used to compute model accuracy

    all_scores = 0    

 

    # From the generator object gets index for series to use in train and validation

    for train_index, valid_index in sss.split(X_train, Y_train):



        # Saves the split train/validation combinations for each Cross-Validation fold

        X_train_cv, X_validation_cv = X_train.loc[train_index,:], X_train.loc[valid_index,:]

        Y_train_cv, Y_validation_cv = Y_train[train_index], Y_train[valid_index]

        

        # Training the model

        rfc = RandomForestClassifier(n_estimators=500, min_samples_leaf = 1, max_depth= None, n_jobs=-1, random_state=30)

        rfc.fit(X_train_cv,Y_train_cv)        

        

        # Get the class probabilities of the input samples

        # Save the probabilities for submission

        y_pred = rfc.predict_proba(X_test)

        predicted += y_pred

        

        # Save the probabilities of validation

        measured[valid_index] = rfc.predict_proba(X_validation_cv)

        

        # Cumulative sum of the score

        score = rfc.score(X_validation_cv,Y_validation_cv)        

        all_scores += score

                

        print("Fold: {} - RF Score: {}".format(k, score)) 

        

        # Saving the model

        models[k] = rfc

        k += 1  

               

    # Compute the mean probability

    predicted /= n_folds

    # Save the mean score value

    mean_score = all_scores/n_folds

    # Save the first trained model

    trained_model = models[1]

    

            

    return measured, predicted, mean_score, trained_model
measured_rf, predicted_rf, accuracy_rf, model_rf = train_rfc(X_train_v2, Y_train, X_test_v2)

print(f"\nMean accuracy for RF: {accuracy_rf}")
# Function to perform all training steps

def train_etc(X_train, Y_train, X_test):

    

    # Create a dictionary that saves the model create in each fold

    models = {}

            

    # Variables that save the probabilities of each class 

    predicted = np.zeros((X_test.shape[0],9))

    measured = np.zeros((X_train.shape[0],9))

    

    # Use Stratified ShuffleSplit cross-validator

    # Provides train/test indices to split data in train/test sets.

    n_folds = 5

    sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.30, random_state=10)



    # Control the number of folds in cross-validation (5 folds)

    k=1

    all_scores = 0

 

    # From the generator object gets index for series to use in train and validation

    for train_index, valid_index in sss.split(X_train, Y_train):



        # Saves the split train/validation combinations for each Cross-Validation fold

        X_train_cv, X_validation_cv = X_train.loc[train_index,:], X_train.loc[valid_index,:]

        Y_train_cv, Y_validation_cv = Y_train[train_index], Y_train[valid_index]

                

        # Training the model

        etc = ExtraTreesClassifier(n_estimators=400, max_depth=10, min_samples_leaf=2, n_jobs=-1, random_state=30)

        etc.fit(X_train_cv,Y_train_cv)        

        

        # Get the class probabilities of the input samples

        # Save the probabilities for submission

        y_pred = etc.predict_proba(X_test)

        predicted += y_pred

        

        # Save the probabilities of validation

        measured[valid_index] = etc.predict_proba(X_validation_cv)

        

        # Cumulative sum of the score

        score = etc.score(X_validation_cv,Y_validation_cv)        

        all_scores += score

                

        print("Fold: {} - ET Score: {}".format(k, score)) 

        

        # Saving the model

        models[k] = etc

        k += 1  

               

    # Compute the mean probability

    predicted /= n_folds

    # Save the mean score value

    mean_score = all_scores/n_folds

    # Save the first trained model

    trained_model = models[1]

    

            

    return measured, predicted, mean_score, trained_model
measured_et, predicted_et, accuracy_et, model_et = train_rfc(X_train_v2, Y_train, X_test_v2)

print(f"\nMean accuracy for ET: {accuracy_et}")
print(f"LGBM accuracy: {accuracy_lgb}")

print(f"RF accuracy: {accuracy_rf}")

print(f"ET accuracy: {accuracy_et}")
# Creatin train and test datasets

x_train = np.concatenate((measured_et, measured_rf, measured_lgb), axis=1)

x_test = np.concatenate((predicted_et, predicted_rf, predicted_lgb), axis=1)



print(x_train.shape, x_test.shape)
# Training the model

from sklearn.linear_model import LogisticRegression

stacker = LogisticRegression(solver="lbfgs", multi_class="auto")

stacker.fit(x_train,Y_train)



# Perform predictions

stacker_pred = stacker.predict_proba(x_test)
# Creating submission file

submission['surface'] =  le.inverse_transform(stacker_pred.argmax(1))

submission.to_csv('submission_stack.csv', index=False)

submission.head()