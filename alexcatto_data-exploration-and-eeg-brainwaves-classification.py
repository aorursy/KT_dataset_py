%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
df_samples = pd.read_csv("data/EEG_data.csv")

df_people = pd.read_csv("data/datasets_106_24522_demographic_info.csv")
print(f"EGG Brainwave samples shape: {df_samples.shape}")

print(f"People info: {df_people.shape}")
df_samples.head(5)
df_people.head(5)
df_samples.info()
df_people.info()
#Dataframes Inner Join

EGG_dataset = pd.merge(left=df_samples, right=df_people, left_on='SubjectID', right_on='subject ID')

#duplicated column drop

EGG_dataset.drop(columns='SubjectID', inplace=True)
print(f"Complete dataset shape: {EGG_dataset.shape}")
EGG_dataset.head(5)
EGG_dataset.isnull().sum()
from sklearn.impute import KNNImputer



def impute_dataset(df):

    imputer = KNNImputer(missing_values=np.nan)

    ds_idxs = df.index

    ds_cols = df.columns 

    df = pd.DataFrame(imputer.fit_transform(df), index=ds_idxs, columns=ds_cols)

    return df
import numbers



def encode_categorical_features(df):

    '''

    This function encodes features with non numerical values.

    Features with two values are incoded into 0 an 1 (binaries).

    Features with more than two non numerical values are one-hot encoded with dummies

    '''

    to_binaries = []

    to_encode = []

    

    for feature in df.columns:

        values = df[feature].unique()

        values = [x for x in values if not pd.isnull(x)]

        if not all(isinstance(value, numbers.Number) for value in values):

            if len(values) == 2:

                to_binaries.append(feature)

            else:

                to_encode.append(feature)



    for binary in to_binaries:

        values = df[binary].unique()

        values = [x for x in values if not pd.isnull(x)]

        df[binary] = df[binary].map(lambda x: 0 if x == values[0] else 1 if x == values[1] else np.nan)



    df = pd.get_dummies(df, columns=to_encode)

    

    return df
encoded_df = encode_categorical_features(EGG_dataset)
encoded_df.head(5)
selected_df = encoded_df.drop(columns=['VideoID', 'predefinedlabel'])
selected_df.head(5)
from sklearn.preprocessing import StandardScaler



def scale_dataset(df, scaler=None):

    ds_idxs = df.index

    ds_cols = df.columns

    

    if scaler == None:

        scaler = StandardScaler()

        scaler = scaler.fit(df.values)

        

    df = pd.DataFrame(scaler.transform(df.values), index=ds_idxs, columns=ds_cols)

    return df, scaler
scaled_df, scaler = scale_dataset(selected_df)
scaled_df.head(5)
from sklearn.decomposition import PCA



def do_pca(df, n_components = None, pca=None):

    if pca == None:

        if n_components == None: 

            pca = PCA()

        else:

            pca = PCA(n_components=n_components)

            

    df_reduced = pca.fit_transform(df)

    return pca, df_reduced
def pca_variance_plot(variance): #function inspired by the one used in an excercise of the lessons.

    n_components = len(variance)

    idxs = np.arange(n_components)

 

    plt.figure(figsize=(20, 10))

    ax = plt.subplot(111)

    cumvals = np.cumsum(variance)

    ax.bar(idxs, variance)

    ax.plot(idxs, cumvals)

 

    ax.xaxis.set_tick_params(width=2)

    ax.yaxis.set_tick_params(width=5, length=20)

 

    ax.set_xlabel("Principal Component")

    ax.set_ylabel("Variance Explained")

    plt.title('Explained Variance Per Principal Component')
pca, dataset_reduct = do_pca(scaled_df)
pca_variance_plot(pca.explained_variance_ratio_)
def pca_results(full_dataset, pca): #This function has taken from an excercise of the PCA lessons (in helper_functions.py)

    '''

    Create a DataFrame of the PCA results

    Includes dimension feature weights and explained variance

    Visualizes the PCA results

    '''

    # Dimension indexing

    dimensions = dimensions = ['{}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components

    components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())

    components.index = dimensions

    # PCA explained variance

    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)

    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])

    variance_ratios.index = dimensions

    # Create a bar plot visualization

    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components

    components.plot(ax = ax, kind = 'bar');

    ax.set_ylabel("Feature Weights")

    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios

    for i, ev in enumerate(pca.explained_variance_ratio_):

        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Expl Var\n          %.4f"%(ev))

    # Return a concatenated DataFrame

    return pd.concat([variance_ratios, components], axis = 1)
pca_res = pca_results(scaled_df, pca)
pca_res = pd.DataFrame(pca_res)

display(pca_res)
reduced_df = scaled_df.drop(columns=[' age', ' gender', ' ethnicity_Bengali',' ethnicity_English', ' ethnicity_Han Chinese'])
data_subset = {}

for v in reduced_df['user-definedlabeln'].unique():

    data_subset[v] = reduced_df[reduced_df['user-definedlabeln'] == v]
print(data_subset.keys())
data_subset[0.975097266665175].groupby(['subject ID']).agg('mean')
data_subset[-1.0255387171989436].groupby(['subject ID']).agg('mean')
reduced_df['subject ID'].unique()
reduced_df = reduced_df[reduced_df['subject ID'] != -0.8681212082604366]
reduced_df = reduced_df[reduced_df['subject ID'] != 0.5279122818574891]
reduced_df['subject ID'].unique()
reduced_df.drop(columns=['subject ID'], inplace=True)
reduced_df.head(5)
from sklearn.impute import KNNImputer

import numbers



def impute_dataset(df):

    imputer = KNNImputer(missing_values=np.nan)

    ds_idxs = df.index

    ds_cols = df.columns 

    df = pd.DataFrame(imputer.fit_transform(df), index=ds_idxs, columns=ds_cols)

    return df



def remove_outliers(EGG_data_df):

    EGG_data_df = EGG_data_df[EGG_data_df['SubjectID'] != 2] #remove outlier student 3

    EGG_data_df = EGG_data_df[EGG_data_df['SubjectID'] != 6] #remove outlier student 7

    return EGG_data_df



def preprocess_data(EGG_data_df):

    

    EGG_data_df = impute_dataset(EGG_data_df)

    EGG_data_df = remove_outliers(EGG_data_df)

    EGG_data_df.drop(columns=['VideoID', 'predefinedlabel', 'SubjectID'], inplace=True)

    return EGG_data_df

    
dataset = pd.read_csv("data/EEG_data.csv")

dataset = preprocess_data(dataset)
dataset.head(5)
# Import train_test_split

from sklearn.model_selection import train_test_split



y = dataset['user-definedlabeln']

X = dataset.drop(columns=['user-definedlabeln'])



# Split the 'features' and 'income' data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, 

                                                    y, 

                                                    test_size = 0.2, 

                                                    random_state = 0)



# Show the results of the split

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer

from sklearn.model_selection import GridSearchCV





def get_best_estimator(learner, hyperparameters_combinations, X_train, y_train):

    '''

    This function takes a classifier and a combination of parameters.

    It returns a model tuned by the hyperparameters combination.

    '''

    #Get a scorer for Grid Search

    scorer = make_scorer(fbeta_score, beta=0.5)

    #Perform grid search on the classifier using 'scorer' as the scoring method 

    grid_obj = GridSearchCV(learner, hyperparameters_combinations, scoring=scorer)

    #Fit the grid search object to the training data and find the optimal parameters

    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the nest estimator

    learner = grid_fit.best_estimator_

    print(f"Best params: {grid_fit.best_params_}")

    #return

    return learner
from time import time



def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 

    '''

    inputs:

       - learner: the learning algorithm to be trained and predicted on

       - hyperparameters_combinations: the dictionary containing hyperparameters possible values, for GridSearch Tuning

       - sample_size: the size of samples (number) to be drawn from training set

       - X_train: features training set

       - y_train: income training set

       - X_test: features testing set

       - y_test: income testing set

    '''

    results = {}

    

    #Fit the learner to the training data again in order to record the training time

    start = time() # Get start time

    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])

    end = time() # Get end time

    

    #Calculate the training time

    results['train_time'] = end - start

        

    # Get the predictions on the test set(X_test),

    #then get predictions on the first 300 training samples(X_train)

    start = time() # Get start time

    predictions_test = learner.predict(X_test)

    predictions_train = learner.predict(X_train[:300])

    if learner.__class__.__name__ == 'XGBClassifier':

        predictions_test = [round(value) for value in predictions_test]

        predictions_train = [round(value) for value in predictions_train]

    end = time() # Get end time

    

    #Calculate the total prediction time

    results['pred_time'] = end - start

   

    # Compute accuracy on the first 300 training samples which is y_train[:300]

    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

   

    #Compute accuracy on test set using accuracy_score()

    results['acc_test'] = accuracy_score(y_test, predictions_test)

    

    #Compute F-score on the the first 300 training samples using fbeta_score()

    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)

    

    #Compute F-score on the test set which is y_test

    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)

        

    # Success

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    print('Classifier {} Accuracy {} f_score {}'.format(learner.__class__.__name__, results['acc_test'], results['f_test']))

          

    # Return the results

    return results
#Import the models from sklearn

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

import xgboost as xgb



#Initialize classifiers

clf_A = DecisionTreeClassifier(random_state = 0)

parameters_A = {'max_depth': list(range(4, 20, 2)),

                'min_samples_split': list(range(2, 15, 2)),

                'min_samples_leaf': list(range(1,  20, 2))

               }

clf_B = AdaBoostClassifier(random_state = 0)

parameters_B = {'algorithm':['SAMME','SAMME.R'],

                'n_estimators':[10, 40, 60, 100, 120, 130, 140]

               }

clf_C = SVC(random_state = 0)

parameters_C = {'kernel': ['rbf'],

                'C': list(np.arange(0.5, 1.5, 0.1)),

                'gamma': ['scale', 'auto']

               }

clf_D = xgb.XGBClassifier(seed=0)

parameters_D = {'base_score': list(np.arange(0.2, 0.5, 0.1)),

                'n_estimators': [10, 40, 60, 100, 120, 130, 140],

                'objective': ['binary:logistic']}





#Collect results on the learners

best_estimators = {}

for clf, params in [(clf_A, parameters_A), (clf_B, parameters_B), (clf_C, parameters_C), (clf_D, parameters_D)]:

    print(f'searching for best estimator: classifier {clf.__class__.__name__}')

    best_estimators[clf.__class__.__name__] = get_best_estimator(clf, params, X_train, y_train)
#Calculate the number of samples for 1%, 25%, 50%, and 100% of the training data

samples_100 = len(y_train)

samples_50 = int((len(y_train)/100)*50)

samples_25 = int((len(y_train)/100)*25)

samples_1 = int((len(y_train)/100)*1)



#Collect results on the learners

results = {}

for clf, params in [(best_estimators['DecisionTreeClassifier'], parameters_A),

                    (best_estimators['AdaBoostClassifier'], parameters_B),

                    (best_estimators['SVC'], parameters_C),

                    (best_estimators['XGBClassifier'], parameters_D)

                   ]:

    clf_name = clf.__class__.__name__

    results[clf_name] = {}

    for i, samples in enumerate([samples_1, samples_25, samples_50, samples_100]):

        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)
import matplotlib.pyplot as pl

import matplotlib.patches as mpatches

import numpy as np

import pandas as pd

from time import time

from sklearn.metrics import f1_score, accuracy_score



#taken from a notebook used in the lessons

def evaluate(results):

    """

    Visualization code to display results of various learners.

    

    inputs:

      - learners: a list of supervised learners

      - stats: a list of dictionaries of the statistic results from 'train_predict()'

    """

  

    # Create figure

    fig, ax = pl.subplots(2, 3, figsize = (19,10))



    # Constants

    bar_width = 0.22

    colors = ['#A00000','#00A0A0','#00A000', '#F5B041']

    

    # Super loop to plot four panels of data

    for k, learner in enumerate(results.keys()):

        for j, metric in enumerate(['train_time', 'acc_train', 'f_train', 'pred_time', 'acc_test', 'f_test']):

            for i in np.arange(4):

                

                # Creative plot code

                ax[j//3, j%3].bar(i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])

                ax[j//3, j%3].set_xticks([0.30, 1.30, 2.30, 3.30])

                ax[j//3, j%3].set_xticklabels(["1%", "25%", "50%", "100%"])

                ax[j//3, j%3].set_xlabel("Training Set Size")

                ax[j//3, j%3].set_xlim((-0.1, 4))

    

    # Add unique y-labels

    ax[0, 0].set_ylabel("Time (in seconds)")

    ax[0, 1].set_ylabel("Accuracy Score")

    ax[0, 2].set_ylabel("F-score")

    ax[1, 0].set_ylabel("Time (in seconds)")

    ax[1, 1].set_ylabel("Accuracy Score")

    ax[1, 2].set_ylabel("F-score")

    

    # Add titles

    ax[0, 0].set_title("Model Training")

    ax[0, 1].set_title("Accuracy Score on Training Subset")

    ax[0, 2].set_title("F-score on Training Subset")

    ax[1, 0].set_title("Model Predicting")

    ax[1, 1].set_title("Accuracy Score on Testing Set")

    ax[1, 2].set_title("F-score on Testing Set")

    

    

    # Set y-limits for score panels

    ax[0, 1].set_ylim((0, 1))

    ax[0, 2].set_ylim((0, 1))

    ax[1, 1].set_ylim((0, 1))

    ax[1, 2].set_ylim((0, 1))



    # Create patches for the legend

    patches = []

    for i, learner in enumerate(results.keys()):

        patches.append(mpatches.Patch(color = colors[i], label = learner))

    pl.legend(handles = patches, bbox_to_anchor = (-.80, 2.53), \

               loc = 'upper center', borderaxespad = 0., ncol = 4, fontsize = 'x-large')

    

    # Aesthetics

    pl.suptitle("Performance Metrics for Three Supervised Learning Models", fontsize = 16, x = 0.63, y = 1.05)

    # Tune the subplot layout

    # Refer - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots_adjust.html for more details on the arguments

    pl.subplots_adjust(left = 0.125, right = 1.2, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.3) 

    pl.tight_layout()

    pl.show()
evaluate(results)
selected_model = best_estimators['XGBClassifier']
def feature_plot(importances, X_train, y_train):

    

    # Display the five most important features

    indices = np.argsort(importances)[::-1]

    columns = X_train.columns.values[indices[:5]]

    values = importances[indices][:5]



    # Creat the plot

    fig = pl.figure(figsize = (9,5))

    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)

    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \

          label = "Feature Weight")

    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \

          label = "Cumulative Feature Weight")

    pl.xticks(np.arange(5), columns)

    pl.xlim((-0.5, 4.5))

    pl.ylabel("Weight", fontsize = 12)

    pl.xlabel("Feature", fontsize = 12)

    

    pl.legend(loc = 'upper center')

    pl.tight_layout()

    pl.show()  
feature_plot(best_estimators['AdaBoostClassifier'].feature_importances_, X_train, y_train)