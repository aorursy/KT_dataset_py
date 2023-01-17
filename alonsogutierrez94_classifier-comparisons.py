from matplotlib import style

from sklearn import model_selection

from sklearn import preprocessing

from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn.model_selection import StratifiedShuffleSplit



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 

import pprint as pp
labels = '../input/actual.csv'

test_path = '../input/data_set_ALL_AML_independent.csv'

train_path = '../input/data_set_ALL_AML_train.csv'



labels_df = pd.read_csv(labels, index_col = 'patient')

test_df = pd.read_csv(test_path)

train_df = pd.read_csv(train_path)
train_df.head()
# Remove "call" columns from training a test dataframes

ds = [col for col in test_df.columns if 'call' in col]

test_df.drop(ds, axis = 1, inplace = True)



ds = [col for col in train_df.columns if 'call' in col]

train_df.drop(ds, axis = 1, inplace = True)

train_df.head()
# Transpose the columns and rows so that genes become features and rows become observations

test_df = test_df.T

train_df = train_df.T

train_df.head()
# Clean up the column names for training and testing data

test_df.columns = test_df.iloc[1]

train_df.columns = train_df.iloc[1]



test_df = test_df.drop(['Gene Description', 'Gene Accession Number']).apply(pd.to_numeric)

train_df = train_df.drop(['Gene Description', 'Gene Accession Number']).apply(pd.to_numeric)



train_df.head()
# Translate label types from strings to respective int values for ml processing.

dic = {'ALL':0,'AML':1}

labels_df.replace(dic,inplace=True)



# Translate index types from strings to respective numeric type values for sorting. 

test_df.index = pd.to_numeric(test_df.index)

test_df.sort_index(inplace = True)



train_df.index = pd.to_numeric(train_df.index)

train_df.sort_index(inplace = True)



labels_df.index = pd.to_numeric(labels_df.index)

labels_df.sort_index(inplace = True)



# Rename indexes for technical purposes 

test_df.index.name = 'patient'

train_df.index.name = 'patient'



# Subset the first 38 patient's cancer types and concatinate with their respective

# rows in the test_df

labels_test = labels_df[labels_df.index > 38]

test_df = pd.concat([labels_test,test_df], axis = 1)



# Subset the last 34 patient's cancer types and concatinate with their respective

# rows in the  train_df

labels_train = labels_df[labels_df.index <= 38]

train_df = pd.concat([labels_train,train_df], axis = 1)



# Replace all infinite values with NaNs as a precaution

test_df.replace(np.inf, np.nan, inplace = True)

train_df.replace(np.inf, np.nan, inplace = True)



# Fill all NaNs with the respective means of each dataframe 

test_df.fillna(value = test_df.values.mean(), inplace = True)

train_df.fillna(value = train_df.values.mean(), inplace = True)



# Append test_df to the train_df 

train_df = train_df.append(test_df)

sample = train_df.iloc[:,2:].sample(n=100, axis=1)

sample["cancer"] = train_df.cancer

sample.describe().round()
from sklearn import preprocessing
sample = sample.drop("cancer", axis=1)

sample.plot(kind="hist", legend=None, bins=20, color='k')

sample.plot(kind="kde", legend=None)
sample_scaled = pd.DataFrame(preprocessing.scale(sample))

sample_scaled.plot(kind="hist", normed=True, legend=None, bins=10, color='k')

sample_scaled.plot(kind="kde", legend=None)
# Dimensionality reduction before or after scaling?
def features_pca(df):

    

    df1 = df.copy()

    labels = df1.cancer.values

    features = preprocessing.scale(df1.drop("cancer", axis = 1).values) 

    features_names = np.array(df1.drop("cancer", axis = 1).columns.tolist())



    sss = StratifiedShuffleSplit(n_splits = 9, test_size = 0.1)



    a = []



    for train_index, test_index in sss.split(features, labels): 

        X_train, X_test = features[train_index], features[test_index]

        y_train, y_test = labels[train_index], labels[test_index]



        s = SelectPercentile(f_classif, percentile = 1)



        s.fit(X_train, y_train)

    

        important_features = s.get_support()



        scores = s.scores_

        scores = scores[important_features].tolist() 



        pca_features = features_names[important_features].tolist()



        scores_report = {pca_features[i]:scores[i] for i in range(len(pca_features))}



        if len(a) == 0:



            a = set(sorted(scores_report, key = scores_report.get))



        else:



            a = set(a).intersection(set(sorted(scores_report, key = scores_report.get)))

    

    a = list(a)

    a.append('cancer')



    return a



train_df = train_df.loc[:,features_pca(train_df)]
def visualize_data(df):



    column_labels = df.columns

    row_labels = df.index

    df1 = preprocessing.scale(df)

    fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)



    heatmap1 = ax1.pcolor(df1, cmap=plt.cm.bone)

    fig1.colorbar(heatmap1)



    ax1.set_xticks(np.arange(df1.shape[1]) + 0.5, minor=False)

    ax1.set_yticks(np.arange(df1.shape[0]) + 0.5, minor=False)

    ax1.invert_yaxis()

    ax1.xaxis.tick_top()

    ax1.set_xticklabels(column_labels)

    ax1.set_yticklabels(row_labels)

    plt.xticks(rotation=90)

    heatmap1.set_clim(-1,1)

#     plt.tight_layout()

    plt.savefig("correlations.png", dpi = (300))

    plt.show()



visualize_data(train_df[train_df.index < 36])

visualize_data(train_df[train_df.index >= 36])
