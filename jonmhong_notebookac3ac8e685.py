import numpy as np

import pandas as pd

import matplotlib.pyplot as pl

%matplotlib inline

import time
# Let's preview dataset

filename = "../input/2016 Stack Overflow Survey Responses.csv"

df = pd.read_csv(filename)



df.head()
# These are all the columns I'm going to use for the dataset

select_columns = ['age_range', 'experience_range', 'programming_ability', 'agree_loveboss',

                  'important_variety', 'important_control', 'important_promotion',

                  'important_newtech', 'important_companymission', 'important_buildexisting',

                  'important_buildnew', 'important_wfh', 'developer_challenges',

                  'new_job_value', 'education', 'occupation', 'open_to_new_job']



reduced_df = df[select_columns]



reduced_df.head()
# Analyzing each row's number of NaN entries

nan_count = {}



for row, data in reduced_df.iterrows():

    no_of_nan = data.isnull().sum()

    

    if no_of_nan in nan_count:

        nan_count[no_of_nan] += 1

    else:

        nan_count[no_of_nan] = 1

        

for i, j in nan_count.items():

    print("{} rows have {} NaN values".format(j, i))
# 1 Drop the rows with a lot of NaN values



# drop the rows that have more than 3 NaN entries

reduced_df = reduced_df.dropna(thresh=(min(*reduced_df.shape) - 3))



# after dropping rows, this updates the indices from [1, 3, 5, 7] to [1, 2, 3, 4, 5]

reduced_df.index = range(1, max(*reduced_df.shape)+1)

reduced_df.head()
# Let's look at the number of NaN values again

nan_count = {}



for row, data in reduced_df.iterrows():

    no_of_nan = data.isnull().sum()

    

    if no_of_nan in nan_count:

        nan_count[no_of_nan] += 1

    else:

        nan_count[no_of_nan] = 1

        

for i, j in nan_count.items():

    print("{} rows now have {} NaN values".format(j, i))
# 2 Convert the remaining NaN values to -1



fillna_df = reduced_df.fillna(value=-1)



fillna_df.head()
# split the features and label columns

fillna_features = fillna_df[fillna_df.columns[:-1]]

fillna_label = fillna_df[fillna_df.columns[-1]]



fillna_features.head()
# 3 Vectorize the categorical/text data into numerical data

# Here are example columns of each of the data types



# numerical entries

numerical = fillna_features["programming_ability"].head()

# categorical entries have only one survey answer

categorical = fillna_features["occupation"].head()

# these have multiple survey answers

text = fillna_features["education"].head()



pd.concat([numerical, categorical, text], axis=1)
# retains NaN values as -1, while vectorizing categorical data

def vectorize_data(df, col):

    # first create the mapper

    mapper = {}

    counter = 0

    

    for idx, entry in df[col].iteritems():

        if entry == -1 and entry not in mapper:

            mapper[entry] = entry

        elif entry not in mapper:

            mapper[entry] = counter

            counter += 1

            

    # need to shift data, so the first index is 1, not 0

    return pd.DataFrame(df[col].map(mapper), index=range(1, max(*df.shape)+1))



vectorize_data(fillna_features, "occupation").head()
# this function splits the multiple answers into all columns and inputs binary entries

def convert_to_binary_columns(df, column_name):

    a = df[column_name].str.get_dummies(sep='; ')

    if "-1" in a.columns:

        del a["-1"]

    return a



convert_to_binary_columns(fillna_features, "education").head()
# this main function uses the converters above to transform all the features

def main_transform_columns(df):

    outX = pd.DataFrame(index=df.index)

    binary = ["developer_challenges", "new_job_value", "education"]

    

    for col in df.columns:

        if fillna_features[col].dtype == float:

            outX = outX.join(df[col].fillna(-1))

        elif col in binary:

            expanded_col = convert_to_binary_columns(df, col)

            outX = outX.join(expanded_col)

        else:

            vec_col = vectorize_data(df, col)

            outX = outX.join(vec_col)

            

    return outX

    



# fillna for features

vectorized_features = main_transform_columns(fillna_features)

vectorized_features.head()
# vectorize label column

mapper = {"I am not interested in new job opportunities": 0,

          "I'm not actively looking, but I am open to new opportunities": 1,

          "I am actively looking for a new job": 1}



vectorized_label = pd.DataFrame(fillna_label.map(mapper))



vectorized_label = vectorized_label.fillna(value=-1)



# concatenate the feature and label columns

vectorized_df = pd.concat([vectorized_features, vectorized_label], axis=1)



vectorized_df.head()
# 4 Convert the -1 values into the mode value of its 5 nearest neighbors

from sklearn.neighbors import NearestNeighbors

from scipy import stats

from collections import Counter

import warnings

warnings.filterwarnings("ignore")



t1 = time.time()



def convert_nan(df):

    # train NearestNeighbors

    five_neighbors = NearestNeighbors(n_neighbors=5)

    five_neighbors.fit(df)

    

    # iterate through each row

    for row, data in df.iterrows():

        # counts the number of the row's -1 values

        no_of_nan = df.iloc[row - 1][df.iloc[row - 1] == -1].count()

        counter = 0

        

        # if a -1 value exists

        if no_of_nan > 0:

            # get the indices of the five nearest neighbors

            dist, n_indices = five_neighbors.kneighbors(df.iloc[row])

            

            while counter < no_of_nan:

                # for each entry in the row

                for col_name, v in data.iteritems():

                    if int(v) == -1:

                        counter += 1

                        # get the five neighbors' mode

                        neighbors_mode = stats.mode(df[col_name].iloc[n_indices[0]])[0][0]

                        # set the entry to the mode

                        df.set_value(row, col_name, neighbors_mode)

                        

        # used for observing running speed here

        # if row % 10 == 0: print row

                    

    return df





processed_df = convert_nan(vectorized_df)

run_time = time.time() - t1 # ~60 sec
processed_df.head()
run_time
%matplotlib inline

# I visualized some of the text data



# these are the three columns that have multiple answers

expanded_columns = ["education", "developer_challenges", "new_job_value"]



def class_name(objct):

    return objct.__class__.__name__





def plot_binary_df(fillna_features, column_name):

    if class_name(df) == 'DataFrame':

        binary_df_sum = convert_to_binary_columns(df, column_name).sum()

    elif class_name(df) == 'Series':

        binary_df_sum = df

        

    binary_df_sum.sort_values(inplace=True, 

                              ascending=False,

                              na_position='first')

    x = binary_df_sum.index

    y = binary_df_sum

    

    ind = np.arange(len(y))



    pl.figure(figsize=(10,8))

    pl.title("{}".format(column_name.upper()))

    pl.bar(np.arange(len(x)),

           y.values,

           color='r',

           tick_label=list(x))

    pl.xticks(np.arange(len(x)), list(x), rotation='vertical', fontsize=13, horizontalalignment='left')

    pl.subplots_adjust(bottom=0.30)

    pl.show()



# plot bar graphs of each of the listed columns, using the functions above

for column in expanded_columns:

    plot_binary_df(fillna_features, column)
# use PCA to reduce the data into lower dimensions

from sklearn.decomposition import PCA



def feature_components_evr(feature_columns):

    pca = PCA(n_components=min(*feature_columns.shape))

    pca.fit(feature_columns)

    

    return pca.components_, pca.explained_variance_ratio_

    

# used to determine the number of principal components for PCA

def scree_plot(evr):

    evr = np.concatenate([[0], evr])

    cum_evr = np.cumsum(evr)

    

    pl.figure()

    pl.title("Scree Plot")

    pl.plot(cum_evr, lw=2, label="Cumulative Explained Variance Ratio")

    pl.xlabel("Number of Principal Components")

    pl.ylabel("Percent of Variance Covered")

    pl.show()



fce = feature_components_evr(vectorized_features)



print(scree_plot(fce[1]))

print(np.cumsum(fce[1])[:10])
from sklearn.preprocessing import StandardScaler



first_pc = feature_components_evr(vectorized_features)[0][0]

second_pc = feature_components_evr(vectorized_features)[0][1] 

third_pc = feature_components_evr(vectorized_features)[0][2]



vectorized_features = vectorized_df[vectorized_df.columns[:-1]]



# scale to a distribution with mean = 0 and sd = 1

# this bad with data that isn't normally distributed, can be a problem with categorized data that turned into vectors

# might have to change the scaling method later

scaled_data = StandardScaler().fit_transform(vectorized_features)

# compress data into 3 dimensions, using 3 principal components

reduced_data = PCA(n_components=3).fit_transform(vectorized_features)

# reduced and scaled

reduced_scaled_data = StandardScaler().fit_transform(reduced_data)



print("Original data: {} rows, {} columns".format(*vectorized_features.shape))

print("Scaled data: {} rows, {} columns".format(*scaled_data.shape))

print("Reduced data: {} rows, {} columns".format(*reduced_data.shape))

print("Reduced and scaled data: {} rows, {} columns".format(*reduced_scaled_data.shape))

print("First PC: {}".format(first_pc[:5]))

print("Second PC: {}".format(second_pc[:5]))

print("Third PC: {}".format(third_pc[:5]))
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.font_manager import FontProperties



scaled_pca = PCA(n_components=3).fit(scaled_data)

first_scaled_pc = [x*10 for x in scaled_pca.components_[0]]

second_scaled_pc = [y*10 for y in scaled_pca.components_[1]]

third_scaled_pc = [z*10 for z in scaled_pca.components_[2]]



scaled_pcs = [first_scaled_pc, second_scaled_pc, third_scaled_pc]



# plot in 3d

def plot_pc_data(features, scaled_pcs):

    x = features[:, 0]

    y = features[:, 1]

    z = features[:, 2]

    

    fig = pl.figure(figsize=(10, 8))

    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("Principal Components Plot", size=20)

    

    # plotting data points

    ax.scatter(x, y, z, 'k.')

    # plotting pc vectors

    for i, j in enumerate(vectorized_features.columns):

        ax.plot(xs=[0, scaled_pcs[0][i]], ys=[0, scaled_pcs[1][i]], zs=[0, scaled_pcs[2][i] ], lw=2, label=j)

#         ax.annotate(range(0, len(original.columns)), 

#                     (scaled_pcs[0][i], scaled_pcs[1][i], scaled_pcs[2][i]), 

#                     label=j, markersize=2)

        

    ax.set_xlabel("First Principal Component")

    ax.set_ylabel("Second Principal Component")

    ax.set_zlabel("Third Principal Component")

    

    ax.auto_scale_xyz(x, y, z)

    pl.show()



# use data, compressed down to three dimensions

plot_pc_data(reduced_scaled_data, scaled_pcs)
from sklearn.cross_validation import train_test_split

from sklearn.cross_validation import KFold

from sklearn.cross_validation import cross_val_score

from sklearn.metrics import f1_score

from sklearn.metrics import make_scorer

from sklearn.ensemble import RandomForestClassifier



X = processed_df[processed_df.columns[:-1]]

y = processed_df[processed_df.columns[-1]]



# split into training and testing, test size is 20% of the data, random state is arbitrary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)



# create a scorer for the cross_val_score() function

scorer = make_scorer(f1_score, pos_label=1)

# use KFold() for the cross_val_score() function

cv_algo = KFold(n=max(*vectorized_df.shape), n_folds=10, shuffle=True, random_state=12)



cvs = cross_val_score(estimator=RandomForestClassifier(),

                      X=X.as_matrix(),

                      y=y,

                      scoring=scorer,

                      cv=cv_algo)





print("Using this score as a benchmark: {}".format(cvs.mean()))
# SVM handles large amounts of features well, so long as the

# number of features isn't higher than the number of data points

from sklearn.svm import SVC



clf = SVC()



t1 = time.time()

clf.fit(X_train, y_train)

t2 = time.time()



pred = clf.predict(X_test)

t3 = time.time()



print("Training time: {}".format(t2 - t1))

print("Prediction time: {}".format(t3 - t2))

print("F1 Score: {}".format(f1_score(y_test, pred)))
# Trying out random forests and extra trees classifier

# They 

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier



rfc = RandomForestClassifier()

xrfc = ExtraTreesClassifier()



t1 = time.time()

rfc.fit(X_train, y_train)

t2 = time.time()



rfc_pred = rfc.predict(X_test)

t3 = time.time()



xrfc.fit(X_train, y_train)

t4 = time.time()



xrfc_pred = xrfc.predict(X_test)

t5 = time.time()



rfc_f1 = f1_score(y_test, rfc_pred)

xrfc_f1 = f1_score(y_test, xrfc_pred)



print("Random Forest")

print("Training time: {}".format(t2 - t1))

print("Prediction time: {}".format(t3 - t2))

print("F1 Score: {}\n".format(rfc_f1))

print("Extra Trees")

print("Training time: {}".format(t4 - t3))

print("Predictino time: {}".format(t5 - t4))

print("F1 Score: {}".format(xrfc_f1))
# from sklearn.grid_search import GridSearchCV



# # param ranges were narrowed down from original, but still within range of the best estimator,

# # in order to reduce gridsearch algorithm running time

# a = time.time()

# params = {'n_estimators': range(1, 21),

#           'criterion': ['gini', 'entropy'],

#           'min_samples_split': range(1, 3),

#           'min_samples_leaf': range(1, 21)}



# scorer = make_scorer(f1_score, pos_label=1)

# est = RandomForestClassifier()

# est.fit(X_train, y_train)



# gs = GridSearchCV(est, param_grid=params, scoring=scorer)

# gs.fit(X, y)



# print gs.best_estimator_

# print gs.best_score_

# print gs.best_params_

# print "Running time: {}".format(time.time() - a)
# input the optimal parameters into the classifier

# I chose RandomForest since it

hrfc = RandomForestClassifier(min_samples_split=2, 

                             n_estimators=7, 

                             criterion='entropy', 

                             min_samples_leaf=13)



t1 = time.time()

hrfc.fit(X_train, y_train)

t2 = time.time()

hrfc_pred = hrfc.predict(X_test)

t3 = time.time()



hrfc_f1 = f1_score(y_test, hrfc_pred, pos_label=1)

print("Training time for {}: {}".format(class_name(hrfc), t3 - t2))

print("Prediction time: {}".format(t2 - t1))

print("F1 Score: {}".format(hrfc_f1))



# analyze the most important features that help make the best predictions

hrfc_fi_ = pd.Series(hrfc.feature_importances_, index=X.columns)

hrfc_fi_ = hrfc_fi_.sort_values(ascending=False)
from sklearn.metrics import confusion_matrix



def plot_confusion_matrix(classifier, y_test,cmap=pl.cm.Blues):

    cm = confusion_matrix(y_test, classifier.predict(X_test), labels=[1, 0])

    

    pl.figure(figsize=(10, 10))

    pl.imshow(cm, interpolation='nearest', cmap=cmap)

    pl.title("{} Confusion Matrix".format(classifier.__class__.__name__), size=20)

    pl.colorbar()

    for x in range(0,2):

        for y in range(0, 2):

            pl.annotate(str(cm[x][y]), xy = (y, x),

                        horizontalalignment='center', 

                        verticalalignment='center',

                        size=25)





for classifier in [clf, rfc, hrfc]:

    plot_confusion_matrix(classifier, y_test)
from sklearn.metrics import classification_report



print("Random Forest:")

print(classification_report(y_test, rfc_pred, labels=[1, 0]))

print("\nTuned Random Forest:")

print(classification_report(y_test, hrfc_pred, labels=[1, 0]))

print("\nSupport Vector Machine:")

print(classification_report(y_test, pred, labels=[1, 0]))
print("Feature importances:\n\n{}".format(hrfc_fi_[:10]))
# plotting feature importances in descending order

# The first 6, especially the first feature, are heavily influential

pl.figure(figsize=(14, 6))

pl.title("Feature Importances", size=20)

pl.bar(np.arange(len(hrfc_fi_.index)),

       hrfc_fi_.values,

       color=('red', 'blue', 'green'),

       tick_label=list(hrfc_fi_.index))

pl.xticks(np.arange(len(hrfc_fi_.index)), list(hrfc_fi_.index), rotation='vertical', fontsize=12)

pl.yticks(size=14)

pl.subplots_adjust(bottom=.15)