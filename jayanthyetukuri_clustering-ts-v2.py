import functools, operator, numpy as np, pandas as pd, os, sys

from numpy import int64

from keras.models import Sequential, load_model

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score, silhouette_score

from sklearn.neighbors import KNeighborsClassifier

from statistics import mean 

    

import warnings; warnings.simplefilter('ignore')



def read_file_to_XY_dfs(file):

    # arguments: file path including the name, ex: /users/xyz/docs/temp_text.tsv

    # reads the file into a dataframe and renames the y column name to 'label_df'

    # returns a dataframe with the labels and another dataframe with only the labels

    df = pd.read_csv(file, header=None, sep="\t")

    labels_df = df.loc[:, df.columns == 0]

    labels_df.columns = ['label']

    df = df.drop(0, axis=1)

    return df, labels_df

    

def data_preprocess(df, labels_df):

    classes_num = len(set(list(labels_df['label'])))

    labels_df = np.array(labels_df['label']-1, dtype=int64)

    labels_df = np.eye(classes_num)[labels_df]

    return df, labels_df



def corr1D(x1, x2):

    # arguments: two numpy arrays

    # calculates the full correlation between the arrays

    # returns the full correlation(for each shift value) between the passed arguments

    return np.correlate(x1, x2, 'full')



def corr2D(a, b):

    m = len(a)

    CC_complete = []

    row = []

    for w in np.arange(1, 2*len(a)):

        k = w - m

        if k >= 0:

            row = []

            for l in np.arange(m - k):

                row.append(a[l+k]*b[l])

        if k < 0:

            row = []

            for l in np.arange(m + k):

                row.append(b[l-k]*a[l])

        CC_complete.append(row)

    CC_complete = functools.reduce(operator.iconcat, CC_complete, [])

    return CC_complete



def cal_xx_df(df):

    corr_list = []

    for index_outer, i in df.iterrows():

        temp = np.array(i.iloc[:-1])

        corr_list.append(np.multiply(temp, temp))

    corr_df = pd.DataFrame(corr_list)

    

    def roll_smooth(data):

        return (data.rolling(window=normalizing_window, win_type='triang', min_periods=1, axis=0).mean())

    corr_df = corr_df.apply(roll_smooth, axis=1)

    

    return corr_df



def cal_xxx_df(df):

    corr_list = []

    for index_outer, i in df.iterrows():

        temp = np.array(i.iloc[:-1])

        corr_list.append(np.multiply(np.multiply(temp, temp), temp))

    corr_df = pd.DataFrame(corr_list)

    

    def roll_smooth(data):

        return (data.rolling(window=normalizing_window, win_type='triang', min_periods=1, axis=0).mean())

    corr_df = corr_df.apply(roll_smooth, axis=1)

    

    return corr_df



def calc_corr_df(df, corr_2d):

    corr_list = []

    for index_outer, i in df.iterrows():

        if corr_2d == False:

            corr_list.append(corr1D(np.array(i.iloc[:-1]), np.array(i.iloc[:-1])))

        elif corr_2d == True:

            corr_list.append(corr2D(np.array(i.iloc[:-1]), np.array(i.iloc[:-1])))

    corr_df = pd.DataFrame(corr_list)

    

    def roll_smooth(data):

        return (data.rolling(window=normalizing_window, win_type='triang', min_periods=1, axis=0).mean())

    corr_df = corr_df.apply(roll_smooth, axis=1)

    

    return corr_df



def calc_corr_all_df(df, corr_2d):

    corr_list = []

    labels_list = []

    for index_outer, i in df.iterrows():

        for outer_outer, j in df.iterrows():

            label_1 = int(i.iloc[-1:])

            label_2 = int(j.iloc[-1:])

            if label_1 == label_2 and outer_outer > index_outer:

                if corr_2d == False:

                    corr_list.append(corr1D(np.array(i.iloc[:-1]), np.array(j.iloc[:-1])))

                elif corr_2d == True:

                    corr_list.append(corr2D(np.array(i.iloc[:-1]), np.array(j.iloc[:-1])))

                labels_list.append(str(100 + label_1) + str(100 + label_2))

    corr_df = pd.DataFrame(corr_list)

    labels_df = pd.DataFrame(labels_list)

    def roll_smooth(data):

        return (data.rolling(window=normalizing_window, win_type='triang', min_periods=1, axis=0).mean())

    corr_df = corr_df.apply(roll_smooth, axis=1)

    

    return corr_df, labels_df



def data_split(df, labels_df, split_ratio):

    msk = np.random.rand(len(df)) < split_ratio

    x_train = df[msk]

    x_test = df[~msk]

    y_train = labels_df[msk]

    y_test = labels_df[~msk]

    

    x_train = x_train.reset_index(drop=True)

    x_test = x_test.reset_index(drop=True)

    y_train = y_train.reset_index(drop=True)

    y_test = y_test.reset_index(drop=True)



    return x_train, y_train, x_test, y_test



def fit_knn(x_train, y_train, n_neighbors):

    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean') #minkowski

    classifier.fit(x_train, y_train)

    return classifier

    

def classifier_accuracy(classifier, x_test, y_test):

    y_pred = classifier.predict(x_test)

    return accuracy_score(y_test, y_pred)



print('data_set|corr1D accr|corr2D accr|raw accr')

print('------------------------------------------------------------------')



normalizing_window = 5

n_neighbors = 1

split_ratio = 0.75

folds = 1



data_dir = '/Users/jay/Downloads/UCRArchive_2018/'

data_set_list = os.listdir(data_dir)



from scipy import stats



for data_name in ['ECG200']: #['Herring', SmoothSubspace', 'Worms', 'MiddlePhalanxTW', 'DistalPhalanxTW']

    train_file = data_dir + data_name + '/' + data_name + '_TRAIN.tsv'

    test_file = data_dir + data_name + '/' + data_name + '_TEST.tsv'

    df_train, labels_df_train = read_file_to_XY_dfs(train_file)

    df_test, labels_df_test = read_file_to_XY_dfs(test_file)



    df = df_train.append(df_test)

    df = df.reset_index(drop=True)

    labels_df = labels_df_train.append(labels_df_test)

    labels_df = labels_df.reset_index(drop=True)

  

    del df_train, df_test, labels_df_train, labels_df_test





#     x_train, y_train = read_file_to_XY_dfs(train_file)

#     x_test, y_test = read_file_to_XY_dfs(test_file)

#         print(x_train.head(2))

#         print(y_train.head(2))



#         corr_1d_df = calc_corr_df(df, corr_2d = False)

#         corr_2d_df = calc_corr_df(df, corr_2d = True)

#     print(df.head(2))

#     print(labels_df.head(2))

    df['label']=labels_df

#     print(df.head(2))

    

    new_corr_df, new_labels_df = calc_corr_all_df(df, corr_2d = False)

    x_train, y_train, x_test, y_test = data_split(new_corr_df, new_labels_df, split_ratio)      

    classifier = fit_knn(x_train, y_train, n_neighbors)

    result_1d = round(classifier_accuracy(classifier, x_test, y_test), 3)



    new_corr_df, new_labels_df = calc_corr_all_df(df, corr_2d = True)

    x_train, y_train, x_test, y_test = data_split(new_corr_df, new_labels_df, split_ratio)      

    classifier = fit_knn(x_train, y_train, n_neighbors)

    result_2d = round(classifier_accuracy(classifier, x_test, y_test), 3)



    df = df.drop('label', axis=1)

    x_train, y_train, x_test, y_test = data_split(df, labels_df, split_ratio)      

    classifier = fit_knn(x_train, y_train, n_neighbors)

    result = round(classifier_accuracy(classifier, x_test, y_test), 3)

    print(data_name, result_1d, result_2d, result)



    

 

'''

data_set|raw accr|corr1D accr|corr2D accr|corrXX accr|corrXXX accr

------------------------------------------------------------------

MiddlePhalanxTW  0.925  0.993  0.56

Herring  0.854  0.999  0.462

ECG200 0.974 0.998 0.878



'''