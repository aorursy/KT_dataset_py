# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import operator

import statistics



from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from mlxtend.classifier import StackingCVClassifier



from sklearn.metrics import accuracy_score

from scipy.stats import norm, skew,skewtest #for some statistics

import warnings



#import utility scripts

import classify_forest_utility_script as utils



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/learn-together/train.csv", index_col='Id')

test_df  = pd.read_csv("/kaggle/input/learn-together/test.csv", index_col='Id')
train_df.info()
train_df.describe()
test_df.describe()
train_df.head()
sns.countplot(x='Cover_Type', data=train_df)
training_df_cols = train_df.columns.tolist()

training_df_cols = training_df_cols[-1:] + training_df_cols[:-1]

train_df = train_df[training_df_cols]

corr = train_df.corr()
# Mask (for the upper triangle)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15,15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask andcorrect aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

           square=True, linewidths=.5, cbar_kws={"shrink": .5})
soil_type_cols = train_df.columns[15:]



utils.barplot_onehot_encoded_features(train_df, soil_type_cols, ylabel="Total amount of positive Soil Types", title="Soil Types = 1 in train data frame")
for idx in [7, 8, 15, 25]:

    soil_type = soil_type_cols[idx - 1]

    print("Number of datapoints in Soil Type %d: %d" % (idx, train_df[soil_type].sum()))
utils.barplot_onehot_encoded_features(test_df, soil_type_cols, ylabel="Total amount of positive Soil Types", title="Soil Types = 1 in test data frame")
for idx in [7, 8, 15, 25]:

    soil_type = soil_type_cols[idx - 1]

    print("Number of datapoints in Soil Type %d: %d" % (idx, test_df[soil_type].sum()))
soil_type_extract_traindf = train_df[soil_type_cols].copy()

soil_type_extract_traindf = soil_type_extract_traindf.join(train_df.Cover_Type)



utils.stackplotbar_target_over_feature(soil_type_extract_traindf, 'Cover_Type')
wilderness_area_cols = train_df.columns[11:15]



utils.barplot_onehot_encoded_features(train_df, wilderness_area_cols, ylabel="Total amount of Wilderness Area Types", title="Wilderness Area = 1 in train data frame")
utils.barplot_onehot_encoded_features(test_df, wilderness_area_cols, ylabel="Total amount of Wilderness Area Types", title="Wilderness Area = 1 in train data frame")
wilderness_area_extract_traindf = train_df[wilderness_area_cols].copy()

wilderness_area_extract_traindf = wilderness_area_extract_traindf.join(train_df.Cover_Type)



utils.stackplotbar_target_over_feature(wilderness_area_extract_traindf, 'Cover_Type')
onehot_encoded_features = np.concatenate([wilderness_area_cols,soil_type_cols])

numerical_features_train_df = train_df.drop(onehot_encoded_features, axis=1)



f, ax = plt.subplots(figsize=(8,6))

sns.heatmap(numerical_features_train_df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
hillshades = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']



# I need to see some statistics on the Hillshades

numerical_features_train_df[hillshades].describe()
# additionally we need to know the median and mode of each Hillshade feature

for feature in hillshades:

    print("Feature %s" % feature)

    print("     Mean:   %d" % (statistics.mean(numerical_features_train_df[feature])))

    print("     Median: %d" % (statistics.median(numerical_features_train_df[feature])))

    print("     Mode:   %d" % (statistics.mode(numerical_features_train_df[feature])))
# Produce a scatter matrix for each pair of features in the data

pd.plotting.scatter_matrix(numerical_features_train_df[hillshades], alpha = 0.3, figsize = (15,12), diagonal = 'kde')



f, ax = plt.subplots(figsize=(8,6))

sns.heatmap(numerical_features_train_df[hillshades].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
sns.distplot(a=numerical_features_train_df.Hillshade_9am, label="Hillshade 9am")

sns.distplot(a=numerical_features_train_df.Hillshade_Noon, label="Hillshade_Noon")

sns.distplot(a=numerical_features_train_df.Hillshade_3pm, label="Hillshade_3pm")

# Add title

plt.title("Histogram of Hillshades")

# Force legend to appear

plt.legend()
# I need to see some statistics on the Hillshades of the test data frame

test_df[hillshades].describe()
# additionally we need to know the median and mode of each Hillshade feature on the test data frame

for feature in hillshades:

    print("Feature %s" % feature)

    print("     Mean:   %d" % (statistics.mean(test_df[feature])))

    print("     Median: %d" % (statistics.median(test_df[feature])))

    print("     Mode:   %d" % (statistics.mode(test_df[feature])))
# Produce a scatter matrix for each pair of features in the data

pd.plotting.scatter_matrix(test_df[hillshades], alpha = 0.3, figsize = (15,12), diagonal = 'kde')



f, ax = plt.subplots(figsize=(8,6))

sns.heatmap(test_df[hillshades].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
sns.distplot(a=test_df.Hillshade_9am, label="Hillshade 9am")

sns.distplot(a=test_df.Hillshade_Noon, label="Hillshade_Noon")

sns.distplot(a=test_df.Hillshade_3pm, label="Hillshade_3pm")

# Add title

plt.title("Histogram of Hillshades")

# Force legend to appear

plt.legend()
skewed_hillshades = ['Hillshade_9am', 'Hillshade_Noon']
features_log_transformed_train_df = numerical_features_train_df.copy()

features_log_transformed_train_df[skewed_hillshades] = features_log_transformed_train_df[skewed_hillshades].apply(lambda x: np.power(x, 5))

features_log_transformed_train_df[skewed_hillshades].describe()
# Produce a scatter matrix for each pair of features in the data

pd.plotting.scatter_matrix(features_log_transformed_train_df[hillshades], alpha = 0.3, figsize = (15,12), diagonal = 'kde')



f, ax = plt.subplots(figsize=(8,6))

sns.heatmap(features_log_transformed_train_df[hillshades].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
# let's make a copy o

numerical_veatures_train_df_transformed_hillshades = numerical_features_train_df.drop(hillshades, axis=1).copy()

numerical_veatures_train_df_transformed_hillshades = numerical_veatures_train_df_transformed_hillshades.join(features_log_transformed_train_df[hillshades],

                                                                                                             on='Horizontal_Distance_To_Roadways')





f, ax = plt.subplots(figsize=(8,6))

sns.heatmap(numerical_features_train_df.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()



f, ax = plt.subplots(figsize=(8,6))

sns.heatmap(numerical_veatures_train_df_transformed_hillshades.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
# let get now the ouliers we can find in the hillshades

hillshade_outliers = utils.detect_outliers(numerical_veatures_train_df_transformed_hillshades, min_num_outliers=2, features=hillshades)

hillshade_outliers
# Remove Soil Types that are not needed.

def Remove_SoilTypes(df):

    return df.drop(['Soil_Type7', 'Soil_Type8', 'Soil_Type15', 'Soil_Type25'], axis=1).copy()



def Transform_Hillshades(df):

    df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']] = df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']].apply(lambda x: np.power(x, 5))

    return df.copy()



def Preprocess_data_noHllshadeTrans(df):

    return Remove_SoilTypes(df)



def Preprocess_data(df):

    return Transform_Hillshades(Remove_SoilTypes(df))
# we must remove Soil Types 7, 8, 15, 25

prep_data_noHllshadeTrans = Preprocess_data_noHllshadeTrans(train_df.drop('Cover_Type', axis=1).copy())

prep_data = Preprocess_data(train_df.drop('Cover_Type', axis=1).copy())

#outlier_indexes = utils.detect_outliers(prep_data, min_num_outliers=2, features=None)

#prep_data.drop(outlier_indexes, axis=0, inplace=True)

#len(outlier_indexes)
import warnings

warnings.filterwarnings("ignore")

classifier_rf = RandomForestClassifier(random_state=81)

params_grid_rf = {'n_estimators' : [560, 561, 562],

                  #n_estimators' : [560<-, 561, 562],

                  #'n_estimators' : [560<-, 563, 565],

                  #'n_estimators' : [560<-, 565, 570],

                  #'n_estimators' : [550, 560<-, 570, 600],

                  #'n_estimators' : [550<-, 600],

                  #'n_estimators' : [500<-, 600],

                  #'n_estimators' : [450, 500<-, 719, 800],

                  'min_samples_leaf': [1,2,3,4,5],

                  'min_samples_split': [2,3,4,5],

                  'oob_score': [False, True]}



#grid =GridSearchCV(estimator = classifier_rf, cv=5, param_grid=params_grid_rf,

#                   scoring='accuracy', verbose=1, n_jobs=-1, refit=True)

#grid.fit(prep_data_noHllshadeTrans, train_df.Cover_Type)

#print("Best Score: " + str(grid.best_score_))

#print("Best Parameters: " + str(grid.best_params_))



#best_parameters = grid.best_params_

#Best Score: 0.7880952380952381

#Best Parameters: {'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 560, 'oob_score': False}
# split data into training and validation data

train_X, test_X, train_y, test_y = train_test_split(prep_data_noHllshadeTrans, train_df.Cover_Type, test_size=0.2, random_state=81)



clf_rf = RandomForestClassifier(random_state=81, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 560, oob_score = False)

clf_rf.fit(train_X, train_y)

predict = clf_rf.predict(test_X)



acc = accuracy_score(test_y, predict)

print("Accuracy on data with no transformation on Hillshades: ", acc)
# split data into training and validation data

train_X, test_X, train_y, test_y = train_test_split(prep_data, train_df.Cover_Type, test_size=0.2, random_state=81)



clf_rf = RandomForestClassifier(random_state=81, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 560, oob_score = False)

clf_rf.fit(train_X, train_y)

predict = clf_rf.predict(test_X)



acc = accuracy_score(test_y, predict)

print("Accuracy on data with transformation on Hillshades: ", acc)
outliers = utils.detect_outliers(numerical_veatures_train_df_transformed_hillshades, min_num_outliers=2)

outliers
prep_data_noOutliers = prep_data.drop(outliers, axis=0)

prep_data_noOutliers.iloc[9610:9614,:]
y_transHillshades_noOutliers = train_df.drop(outliers, axis=0)['Cover_Type']

# split data into training and validation data

train_X, test_X, train_y, test_y = train_test_split(prep_data_noOutliers, y_transHillshades_noOutliers, test_size=0.2, random_state=81)



clf_rf = RandomForestClassifier(random_state=81, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 560, oob_score = False)

clf_rf.fit(train_X, train_y)

predict = clf_rf.predict(test_X)



acc = accuracy_score(test_y, predict)

print("Accuracy on data with transformation on Hillshades and without outliers (2) : ", acc)
# StandardScaling the dataset before splitting it and fitting the algorithms

scaler = StandardScaler()

scaled_data = scaler.fit_transform(prep_data_noOutliers)
# split data into training and validation data

train_X, test_X, train_y, test_y = train_test_split(scaled_data, y_transHillshades_noOutliers, test_size=0.2, random_state=81)



clf_rf = RandomForestClassifier(random_state=81, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 560, oob_score = False)

clf_rf.fit(train_X, train_y)

predict = clf_rf.predict(test_X)



acc = accuracy_score(test_y, predict)

print("Accuracy on data with transformation on Hillshades and without outliers but with data set scaled (2) : ", acc)
classifier_rf = RandomForestClassifier(n_estimators = 560,

                                       max_features = 0.3,

                                       max_depth = 464,

                                       min_samples_split = 2,

                                       min_samples_leaf = 1,

                                       bootstrap = False,

                                       random_state=81)

classifier_rf.fit(train_X, train_y)

predict = classifier_rf.predict(test_X)

acc = accuracy_score(test_y, predict)

print("Accuracy on data with transformation on Hillshades and without outliers (2) : ", acc)
### define the classifiers

### Parameters from :https://www.kaggle.com/joshofg/pure-random-forest-hyperparameter-tuning



classifier_rf = RandomForestClassifier(n_estimators = 560,

                                       max_features = 0.3,

                                       max_depth = 464,

                                       min_samples_split = 2,

                                       min_samples_leaf = 1,

                                       bootstrap = False,

                                       random_state=81)

classifier_xgb = OneVsRestClassifier(XGBClassifier(n_estimators = 560,

                                                   max_depth = 464,

                                                   random_state=81))

classifier_et = ExtraTreesClassifier(random_state=81)



classifier_adb = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=81), random_state=81)

classifier_bg = BaggingClassifier(random_state=81)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

sclf = StackingCVClassifier(classifiers=[classifier_rf,

                                         classifier_xgb,

                                         classifier_et,

                                         classifier_adb,

                                         classifier_bg],

                            use_probas=True,

                            meta_classifier=classifier_rf)







labels = ['Random Forest', 'XGBoost', 'ExtraTrees', 'AdaBoost', 'Bagging', 'MetaClassifier']









for clf, label in zip([classifier_rf, classifier_xgb, classifier_et, classifier_adb, classifier_bg, sclf], labels):

    scores = cross_val_score(clf, train_X, train_y,

                             cv=5,

                             scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
sclf.fit(train_X, train_y)

prediction = sclf.predict(test_X)

accuracy = accuracy_score(test_y, prediction)

print("Accuracy obtained by stacking classifiers: ", accuracy)
# prepare the final test and train data

# remove the SoilTypes we didn't train

test_processed_data = Preprocess_data(test_df.copy())

train_processed_data = Preprocess_data(train_df.drop('Cover_Type', axis=1).copy())

target_training = train_df.Cover_Type



# StandardScaling the test dataset

train_scaler = StandardScaler()

test_scaler = StandardScaler()

test_scaled_data = test_scaler.fit_transform(test_processed_data)

train_scaled_data = train_scaler.fit_transform(train_processed_data)
sclf_final = StackingCVClassifier(classifiers=[classifier_rf,

                                         classifier_xgb,

                                         classifier_et,

                                         classifier_adb,

                                         classifier_bg],

                                  use_probas=True,

                                  meta_classifier=classifier_rf)



sclf_final.fit(train_scaled_data, target_training)



# prepare submission

test_ids = pd.read_csv("/kaggle/input/learn-together/test.csv")["Id"]

final_prediction = sclf_final.predict(test_scaled_data)



# save file with predictions

submission = pd.DataFrame({'Id' : test_ids,

                           'Cover_Type' : final_prediction})

submission.to_csv('submission.csv', index=False)