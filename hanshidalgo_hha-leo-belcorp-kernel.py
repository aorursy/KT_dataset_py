import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pyplot import figure

figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')



from sklearn import preprocessing
import numpy as np

import math

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import StratifiedKFold

from numpy.random import normal

from sklearn.preprocessing import OneHotEncoder
def freq_encod(data, columns):

    '''Returns a DataFrame with encoded columns'''

    encoded_cols = []

    nsamples = data.shape[0]

    for col in columns:    

        freqs_cat = data.groupby(col)[col].count()/nsamples

        encoded_col = data[col].map(freqs_cat)

        encoded_col[encoded_col.isnull()] = 0

        data['freq_'+col]=encoded_col

    return data
def one_hot_encode(train_data, test_data, columns):

    '''Returns a DataFrame with encoded columns'''

    conc = pd.concat([train_data, test_data], axis=0)

    encoded_cols = []

    for col in columns:

        encoded_cols.append(pd.get_dummies(conc[col], prefix='one_hot_'+col, 

                                      drop_first=True))

    all_encoded = pd.concat(encoded_cols, axis=1)

    return (all_encoded.iloc[:train_data.shape[0],:], 

            all_encoded.iloc[train_data.shape[0]:,:])





def one_hot_encode(train_data, test_data, columns):

    conc = pd.concat([train_data, test_data], axis=0)

    encoded = pd.get_dummies(conc.loc[:, columns], drop_first=True,

                             sparse=True) 

    return (encoded.iloc[:train_data.shape[0],:], 

            encoded.iloc[train_data.shape[0]:,:])





def label_encode(train_data, test_data, columns):

    'Returns a DataFrame with encoded columns'

    encoded_cols = []

    for col in columns:

        factorised = pd.factorize(train_data[col])[1]

        labels = pd.Series(range(len(factorised)), index=factorised)

        encoded_col_train = train_data[col].map(labels) 

        encoded_col_test = test_data[col].map(labels)

        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)

        encoded_col[encoded_col.isnull()] = -1

        encoded_cols.append(pd.DataFrame({'label_'+col:encoded_col}))

    all_encoded = pd.concat(encoded_cols, axis=1)

    return (all_encoded.loc[train_data.index,:], 

            all_encoded.loc[test_data.index,:])



def freq_encode(train_data, test_data, columns):

    '''Returns a DataFrame with encoded columns'''

    encoded_cols = []

    nsamples = train_data.shape[0]

    for col in columns:    

        freqs_cat = train_data.groupby(col)[col].count()/nsamples

        encoded_col_train = train_data[col].map(freqs_cat)

        encoded_col_test = test_data[col].map(freqs_cat)

        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)

        encoded_col[encoded_col.isnull()] = 0

        encoded_cols.append(pd.DataFrame({'freq_'+col:encoded_col}))

    all_encoded = pd.concat(encoded_cols, axis=1)

    return (all_encoded.loc[train_data.index,:], 

            all_encoded.loc[test_data.index,:])



def mean_encode(train_data, test_data, columns, target_col, reg_method=None,

                alpha=0, add_random=False, rmean=0, rstd=0.1, folds=1):

    '''Returns a DataFrame with encoded columns'''

    encoded_cols = []

    target_mean_global = train_data[target_col].mean()

    for col in columns:

        # Getting means for test data

        nrows_cat = train_data.groupby(col)[target_col].count()

        target_means_cats = train_data.groupby(col)[target_col].mean()

        target_means_cats_adj = (target_means_cats*nrows_cat + 

                                 target_mean_global*alpha)/(nrows_cat+alpha)

        # Mapping means to test data

        encoded_col_test = test_data[col].map(target_means_cats_adj)

        # Getting a train encodings

        if reg_method == 'expanding_mean':

            train_data_shuffled = train_data.sample(frac=1, random_state=1)

            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]

            cumcnt = train_data_shuffled.groupby(col).cumcount()

            encoded_col_train = cumsum/(cumcnt)

            encoded_col_train.fillna(target_mean_global, inplace=True)

            if add_random:

                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 

                                                               size=(encoded_col_train.shape[0]))

        elif (reg_method == 'k_fold') and (folds > 1):

            kfold = StratifiedKFold(train_data[target_col].values, folds, shuffle=True, random_state=1)

            parts = []

            for tr_in, val_ind in kfold:

                # divide data

                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]

                # getting means on data for estimation (all folds except estimated)

                nrows_cat = df_for_estimation.groupby(col)[target_col].count()

                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()

                target_means_cats_adj = (target_means_cats*nrows_cat + 

                                         target_mean_global*alpha)/(nrows_cat+alpha)

                # Mapping means to estimated fold

                encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)

                if add_random:

                    encoded_col_train_part = encoded_col_train_part + normal(loc=rmean, scale=rstd, 

                                                                             size=(encoded_col_train_part.shape[0]))

                # Saving estimated encodings for a fold

                parts.append(encoded_col_train_part)

            encoded_col_train = pd.concat(parts, axis=0)

            encoded_col_train.fillna(target_mean_global, inplace=True)

        else:

            encoded_col_train = train_data[col].map(target_means_cats_adj)

            if add_random:

                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 

                                                               size=(encoded_col_train.shape[0]))



        # Saving the column with means

        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)

        encoded_col[encoded_col.isnull()] = target_mean_global

        encoded_cols.append(pd.DataFrame({'mean_'+target_col+'_'+col:encoded_col}))

    all_encoded = pd.concat(encoded_cols, axis=1)

    return (all_encoded.loc[train_data.index,:], 

            all_encoded.loc[test_data.index,:])



def test_clf(X_train, y_train, X_test, y_test, iterations):

    train_scores = []

    val_scores = []

    for i in iterations:

        model = GradientBoostingRegressor(n_estimators=i, learning_rate=1, max_depth=3, 

                                           min_samples_leaf=3, random_state=0)

        model.fit(X_train, y_train)

        y_train_pred_scores = model.predict(X_train)

        y_test_pred_scores = model.predict(X_test)

        train_scores.append(mean_absolute_error(y_train, y_train_pred_scores))

        val_scores.append(mean_absolute_error(y_test, y_test_pred_scores))

    return train_scores, val_scores



def test_reg(X_train, y_train, X_test, y_test, iterations):

    train_scores = []

    val_scores = []

    for i in n_estimators_list:   

        model = GradientBoostingClassifier(n_estimators=i, learning_rate=1, max_depth=3, 

                                           min_samples_leaf=3, random_state=0, max_features=max_features)

        model.fit(X_train, y_train)

        y_train_pred_scores = model.predict_proba(X_clf_train)[:,1]

        y_test_pred_scores = model.predict_proba(X_clf_test)[:,1]

        train_scores.append(roc_auc_score(y_clf_train, y_train_pred_scores))

        val_scores.append(roc_auc_score(y_clf_test, y_test_pred_scores))

    return train_scores, val_scores



def scoring_gbr_sklern(X_train, y_train, X_test, y_test, n_estimators=100, 

                       learning_rate=1, max_depth=3, random_state=0, max_features=None,

                       min_samples_leaf=1, verbose=False):

    scores_train = []

    scores_test = []

    iterations = []

    log_iters = list(set((np.logspace(math.log(1, 8), math.log(400, 8), 

                                      num=50, endpoint=True, base=8, 

                                      dtype=np.int))))

    log_iters.sort()

    for i in log_iters:

        model = GradientBoostingRegressor(n_estimators=i, learning_rate=learning_rate, 

                                          max_depth=max_depth, random_state=random_state,

                                          min_samples_leaf=min_samples_leaf, max_features=max_features)

        model.fit(X_train, y_train)

        y_train_pred_scores = model.predict(X_train)

        y_test_pred_scores = model.predict(X_test)

        scores_train.append(mean_squared_error(y_train, y_train_pred_scores))

        scores_test.append(mean_squared_error(y_test, y_test_pred_scores))

        iterations.append(i)

        if verbose:

            print(i, scores_train[-1], scores_test[-1])

    best_score = min(scores_test)

    best_iter = iterations[scores_test.index(best_score)]

    if verbose:

        print('Best score: {}\nBest iter: {}'.format(best_score, best_iter))

    return scores_train, scores_test, iterations, model



def scoring_gbc_sklern(X_train, y_train, X_test, y_test, n_estimators=100, 

                       learning_rate=1, max_depth=3, random_state=0, max_features=None,

                       min_samples_leaf=1, verbose=False):

    scores_train = []

    scores_test = []

    iterations = []

    weight_0 = 1

    weight_1 = (len(y_train) - y_train.sum())/y_train.sum()

    sample_weights = [weight_1 if i else weight_0 for i in y_train]

    log_iters = list(set((np.logspace(math.log(1, 8), math.log(500, 8), 

                                      num=50, endpoint=True, base=8, 

                                      dtype=np.int))))

    log_iters.sort()

    for i in log_iters:

        model = GradientBoostingClassifier(n_estimators=i, learning_rate=learning_rate, 

                                          max_depth=max_depth, random_state=random_state,

                                          min_samples_leaf=min_samples_leaf, max_features=max_features)

        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_train_pred_scores = model.predict_proba(X_train)

        y_test_pred_scores = model.predict_proba(X_test)

        scores_train.append(roc_auc_score(y_train, y_train_pred_scores[:,1]))

        scores_test.append(roc_auc_score(y_test, y_test_pred_scores[:,1]))

        iterations.append(i)

        if verbose:

            print(iterations[-1], scores_train[-1], scores_test[-1])

    best_score = max(scores_test)

    best_iter = iterations[scores_test.index(best_score)]

    if verbose:

        print('Best score: {}\nBest iter: {}'.format(best_score, best_iter))

    return scores_train, scores_test, iterations, model



def test_encoding(train_data, test_data, cols_to_encode, target_col, encoding_funcs, 

                  scoring_func, scoring_func_params={}, other_cols_to_use=None,

                  alpha=0):

    y_train = train_data[target_col]

    y_test = test_data[target_col]

    X_train_cols = []

    X_test_cols = []

    for encoding_func in encoding_funcs:  

        if (encoding_func==mean_encode) or (encoding_func==mean_and_freq_encode):

            encoded_features = encoding_func(train_data, test_data, cols_to_encode, 

                                             target_col=target_col, alpha=alpha)

        else:

            encoded_features = encoding_func(train_data, test_data, cols_to_encode)

        X_train_cols.append(encoded_features[0]), 

        X_test_cols.append(encoded_features[1])

    X_train = pd.concat(X_train_cols, axis=1)

    X_test = pd.concat(X_test_cols, axis=1)

    if other_cols_to_use:

        X_train = pd.concat([X_train, train_data.loc[:, other_cols_to_use]], axis=1)

        X_test = pd.concat([X_test, test_data.loc[:, other_cols_to_use]], axis=1)

    return scoring_func(X_train, y_train, X_test, y_test, **scoring_func_params)



def describe_dataset(data, target_col):

    ncats = []

    ncats10 = []

    ncats100 = []

    nsamples_median = []

    X_col_names = list(data.columns)

    X_col_names.remove(target_col)

    print('Number of samples: ', data.shape[0])

    for col in X_col_names:

        counts = data.groupby([col])[col].count()

        ncats.append(len(counts))

        ncats10.append(len(counts[counts<10]))

        ncats100.append(len(counts[counts<100]))

        nsamples_median.append(counts.median())

    data_review_df = pd.DataFrame({'Column':X_col_names, 'Number of categories':ncats, 

                                   'Categories with < 10 samples':ncats10,

                                   'Categories with < 100 samples':ncats100,

                                   'Median samples in category':nsamples_median})

    data_review_df = data_review_df.loc[:, ['Column', 'Number of categories',

                                             'Median samples in category',

                                             'Categories with < 10 samples',

                                             'Categories with < 100 samples']]

    return data_review_df.sort_values(by=['Number of categories'], ascending=False)



def make_vgsales():

    vgsales = pd.read_csv('../input/vgsales1.csv')

    vgsales = vgsales.loc[(vgsales['Year'].notnull()) & (vgsales['Publisher'].notnull()), 

                         ['Platform', 'Genre', 'Publisher', 'Year', 'Global_Sales']]

    vgsales['Year'] = vgsales.loc[:,['Year']].astype('str')

    vgsales['Platform x Genre'] = vgsales['Platform'] + '_' + vgsales['Genre']

    vgsales['Platform x Year'] = vgsales['Platform'] + '_' + vgsales['Year']

    vgsales['Genre x Year'] = vgsales['Genre'] + '_' + vgsales['Year']

    return vgsales



def make_poverty():

    poverty = pd.read_csv('../input/A_indiv_train.csv')

    poverty_cols_to_use = ['HeUgMnzF', 'gtnNTNam', 'XONDGWjH', 'hOamrctW', 'XacGrSou', 

                           'ukWqmeSS', 'SGeOiUlZ', 'RXcLsVAQ', 'poor']

    poverty['poor'] = poverty['poor'].astype(int)

    poverty = poverty.loc[:, poverty_cols_to_use]

    return poverty



def make_poverty_interaction():

    poverty = pd.read_csv('../input/A_indiv_train.csv')

    poverty_cols_to_use = ['HeUgMnzF', 'gtnNTNam', 'XONDGWjH', 'hOamrctW', 'XacGrSou', 

                            'ukWqmeSS', 'SGeOiUlZ', 'RXcLsVAQ', 'poor']

    poverty = poverty.loc[:, poverty_cols_to_use]

    poverty.loc[:, poverty_cols_to_use[:-1]] = poverty.loc[:, poverty_cols_to_use[:-1]].astype(str)

    poverty['poor'] = poverty['poor'].astype(int)

    poverty['interaction_1'] = poverty['HeUgMnzF'] + poverty['XONDGWjH']

    poverty['interaction_2'] = poverty['gtnNTNam'] + poverty['hOamrctW']

    poverty['interaction_3'] = poverty['XONDGWjH'] + poverty['XacGrSou']

    poverty['interaction_4'] = poverty['hOamrctW'] + poverty['ukWqmeSS']

    poverty['interaction_5'] = poverty['XacGrSou'] + poverty['SGeOiUlZ']

    poverty['interaction_6'] = poverty['ukWqmeSS'] + poverty['RXcLsVAQ']

    poverty['interaction_7'] = poverty['SGeOiUlZ'] + poverty['RXcLsVAQ']

    poverty['interaction_8'] = poverty['HeUgMnzF'] + poverty['gtnNTNam']

    poverty['interaction_9'] = poverty['ukWqmeSS'] + poverty['hOamrctW']

    poverty['interaction_10'] = poverty['XONDGWjH'] + poverty['RXcLsVAQ']

    return poverty



def make_poverty_interaction_only():

    poverty = pd.read_csv('../input/A_indiv_train.csv')

    poverty_cols_to_use = ['HeUgMnzF', 'gtnNTNam', 'XONDGWjH', 'hOamrctW', 'XacGrSou', 

                            'ukWqmeSS', 'SGeOiUlZ', 'RXcLsVAQ', 'poor']

    poverty = poverty.loc[:, poverty_cols_to_use]

    poverty.loc[:, poverty_cols_to_use[:-1]] = poverty.loc[:, poverty_cols_to_use[:-1]].astype(str)

    poverty['poor'] = poverty['poor'].astype(int)

    poverty_interactions = poverty.loc[:,['poor']]

    poverty_interactions['interaction_1'] = poverty['HeUgMnzF'] + poverty['XONDGWjH']

    poverty_interactions['interaction_2'] = poverty['gtnNTNam'] + poverty['hOamrctW']

    poverty_interactions['interaction_3'] = poverty['XONDGWjH'] + poverty['XacGrSou']

    poverty_interactions['interaction_4'] = poverty['hOamrctW'] + poverty['ukWqmeSS']

    poverty_interactions['interaction_5'] = poverty['XacGrSou'] + poverty['SGeOiUlZ']

    poverty_interactions['interaction_6'] = poverty['ukWqmeSS'] + poverty['RXcLsVAQ']

    poverty_interactions['interaction_7'] = poverty['SGeOiUlZ'] + poverty['RXcLsVAQ']

    poverty_interactions['interaction_8'] = poverty['HeUgMnzF'] + poverty['gtnNTNam']

    poverty_interactions['interaction_9'] = poverty['ukWqmeSS'] + poverty['hOamrctW']

    poverty_interactions['interaction_10'] = poverty['XONDGWjH'] + poverty['RXcLsVAQ']

    return poverty_interactions



def make_ctr():

    ctr = pd.read_csv('../input/ctr_data.csv', nrows=100000)

    ctr = ctr.astype('str')

    ctr['click'] = ctr['click'].astype('int')

    ctr['interaction_1'] = (ctr['site_category'] + ctr['C15'] + ctr['C16'] + ctr['C20'] + ctr['C17'])

    ctr['interaction_2'] = (ctr['site_category'] + ctr['C18'] + ctr['C19'] + ctr['device_model'])

    ctr_cols_to_use = ['site_id', 'app_id', 'device_id',

                      'device_model', 'C14', 'interaction_1', 'interaction_2', 'click']

    ctr = ctr.loc[:, ctr_cols_to_use]

    return ctr



def make_movie():

    movie = pd.read_csv('../input/IMDB-Movie-Data1.csv').loc[:, ['Genre', 'Year', 'Rating', 

                                                       'Revenue (Millions)']]

    movie = movie.loc[movie['Revenue (Millions)'].notnull(), :]

    movie['Year x Rating'] = movie['Year'] + movie['Rating']

    return movie



def make_credit():

    credit = pd.read_csv('../input/credit1.csv')

    cols = list(credit.columns)

    cols.remove('Unnamed: 0')

    return credit.loc[:, cols]



def encoding_stats(train_data, test_data, X_train, X_test, target_col, encoding_function,

                  feature_cols_to_use):

    if encoding_function.__name__ == 'one_hot_encode':

        return np.nan, np.nan, np.nan, np.nan

    if encoding_function.__name__ == 'mean_encode':

        enc_suffix = 'mean_'+target_col+'_'

    if encoding_function.__name__ == 'freq_encode':    

        enc_suffix = 'freq_'

    if encoding_function.__name__ == 'label_encode':

        enc_suffix = 'label_'

    cols_to_encoded_mapping = {}

    for col in feature_cols_to_use:

        for col_enc in X_train.columns:

            if col == col_enc[len(enc_suffix):]:

                cols_to_encoded_mapping[col] = col_enc

    train_conc = pd.concat([train_data, X_train], axis=1)

    test_conc = pd.concat([test_data, X_test], axis=1)

    mean_stds_train = []

    std_means_train = []

    mean_stds_test = []

    std_means_test = []

    for key in cols_to_encoded_mapping.keys():

        #how much randomisation added

        mean_stds_train.append(train_conc.groupby(key)[cols_to_encoded_mapping[key]].std().mean())

        mean_stds_test.append(test_conc.groupby(key)[cols_to_encoded_mapping[key]].std().mean())

        # how distinguishable are categories with that encoding

        std_means_train.append(train_conc.groupby(key)[cols_to_encoded_mapping[key]].mean().std())

        std_means_test.append(test_conc.groupby(key)[cols_to_encoded_mapping[key]].mean().std())

    

    encoding_stats = (np.mean(mean_stds_train), np.mean(std_means_train),

                      np.mean(mean_stds_test), np.mean(std_means_test))

    return encoding_stats



def test_all_encodings(train_data, test_data, target_col, testing_params, 

                       test_one_hot=False, regression=False, skip_first_iters_graph=0,

                      max_features_one_hot=0.01):

    encoding_settings = [[label_encode, {}, 'Label encoding', '#960000'],

                         [freq_encode, {}, 'Frequency encoding', '#FF2F02'],

                         [mean_encode, {'alpha':0, 'folds':None, 'reg_method':None, 

                                        'add_random':False, 'rmean':0, 'rstd':0.0,

                                        'target_col':target_col},

                         'Mean encoding, alpha=0', '#A4C400'],

                         [mean_encode, {'alpha':2, 'folds':None, 'reg_method':None, 

                                        'add_random':False, 'rmean':0, 'rstd':0.0,

                                        'target_col':target_col}, 

                         'Mean encoding, alpha=2', '#73B100'],

                         [mean_encode, {'alpha':5, 'folds':None, 'reg_method':None, 

                                        'add_random':False, 'rmean':0, 'rstd':0.0,

                                        'target_col':target_col}, 

                         'Mean encoding, alpha=5', '#2B8E00'],

                         [mean_encode, {'alpha':5, 'folds':3, 'reg_method':'k_fold', 

                                        'add_random':False, 'rmean':0, 'rstd':0.0,

                                        'target_col':target_col}, 

                         'Mean encoding, alpha=5, 4 folds', '#00F5F2'],

                         [mean_encode, {'alpha':5, 'folds':5, 'reg_method':'k_fold', 

                                        'add_random':False, 'rmean':0, 'rstd':0.0,

                                        'target_col':target_col}, 

                         'Mean encoding, alpha=5, 7 folds', '#00BAD3'],

                         [mean_encode, {'alpha':5, 'folds':None, 'reg_method':'expanding_mean', 

                                        'add_random':False, 'rmean':0, 'rstd':0.0,

                                        'target_col':target_col}, 

                         'Mean encoding, alpha=5, expanding mean', '#B22BFA']]

    review_rows = []

    if test_one_hot:

        oh_settings = [[one_hot_encode, {}, 'One hot encoding', '#E7E005']]

        encoding_settings = oh_settings + encoding_settings

    feature_cols_to_use = list(train_data.columns)

    feature_cols_to_use.remove(target_col)

    if regression:

        scoring_function = scoring_gbr_sklern

        best_score_function = min

    else:

        scoring_function = scoring_gbc_sklern

        best_score_function = max     

    plt.figure(figsize=(10,7))

    for encoding_function, encoding_params, str_name, color in encoding_settings:

        if encoding_function.__name__ == 'one_hot_encode':

            testing_params['max_features'] = max_features_one_hot

        else:

            testing_params['max_features'] = None

        X_train, X_test = encoding_function(train_data, test_data, feature_cols_to_use,

                                            **encoding_params)

        scores = scoring_function(X_train, train_data[target_col], X_test, 

                                    test_data[target_col], 

                                    min_samples_leaf=1, max_depth=3, **testing_params)

        skip_it = int(skip_first_iters_graph)

        train_scores, test_scores, iters, model_ = scores

        plt.plot(iters[skip_it:], 

                 test_scores[skip_it:], 

                 label='Test, ' + str_name, linewidth=1.5, color=color)

        best_score_test = best_score_function(test_scores)

        best_iter_test = iters[test_scores.index(best_score_test)]

        best_score_train = best_score_function(train_scores[:best_iter_test])

        print('Best score for {}: is {}, on iteration {}'.format(str_name, 

                                                                 best_score_test, 

                                                                 best_iter_test,

                                                                 best_score_train))

        enc_stats = encoding_stats(train_data, test_data, X_train, X_test, 

                                   target_col, encoding_function, feature_cols_to_use)

        review_rows.append([str_name, best_score_train, best_score_test, best_iter_test] + list(enc_stats))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if regression:

        columns=['Encoding', 'Train RMSE score on best iteration', 

             'Best RMSE score (test)', 'Best iteration (test)',

             'EV (train)', 'ED (train)', 'EV (test)', 'ED (test)']

    else:

        columns=['Encoding', 'Train AUC score on best iteration', 

             'Best AUC score (test)', 'Best iteration (test)',

             'EV (train)', 'ED (train)', 'EV (test)', 'ED (test)']

    return pd.DataFrame(review_rows, columns=columns)
def missing_values_table(df):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        return mis_val_table_ren_columns
def agregar_sufijo(lista,sufijo):

    lista_final = []

    for elemento in lista:

        lista_final.append(str(elemento)+'_'+sufijo)

    return lista_final
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#201907

predict_submission = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/predict_submission.csv')
predict_submission.head()
predict_submission.shape
predict_submission.idconsultora.nunique()
predict_submission['campana'] = 201907
predict_submission.columns = ['IdConsultora','Flagpasopedido','campana']
campana_consultora = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/campana_consultora.csv')
campana_consultora.head()
campana_consultora.info()
missing_values_table(campana_consultora)
campana_consultora.drop(['Unnamed: 0','codigocanalorigen'],axis=1,inplace=True)
campana_consultora.IdConsultora.nunique()
plt.figure(figsize=(15,5))

sns.countplot(campana_consultora['campana'])
campana_consultora['IdConsultora'].nunique()
campana_consultora['Flagpasopedido'].value_counts()
table=pd.crosstab(campana_consultora.campana,campana_consultora.Flagpasopedido)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,figsize=(15,5))

plt.title('Campana vs target')

plt.xlabel('Campana')

plt.ylabel('Proporción de target')
campana_consultora['flagactiva'].value_counts()
table=pd.crosstab(campana_consultora.campana,campana_consultora.flagactiva)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,figsize=(15,5))

plt.title('Campana vs flagactiva')

plt.xlabel('Campana')

plt.ylabel('Proporción de flagactiva')
table=pd.crosstab(campana_consultora.campana,campana_consultora.flagpasopedidotratamientocorporal)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,figsize=(15,4))

plt.title('Campana vs flagpasopedidotratamientocorporal')

plt.xlabel('Campana')

plt.ylabel('Proporción de flagpasopedidotratamientocorporal')
table=pd.crosstab(campana_consultora.campana,campana_consultora.flagpedidoanulado)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,figsize=(15,4))

plt.title('Campana vs flagpedidoanulado')

plt.xlabel('Campana')

plt.ylabel('Proporción de flagpedidoanulado')
campana_consultora.flagpedidoanulado.value_counts()
plt.figure(figsize=(15,5))

sns.countplot(campana_consultora['codigofactura'].fillna('NOID'))
table=pd.crosstab(campana_consultora.campana,campana_consultora['codigofactura'].fillna('NOID'))

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,figsize=(15,4))

plt.title('Campana vs codigofactura')

plt.xlabel('Campana')

plt.ylabel('Proporción de codigofactura')
plt.figure(figsize=(15,5))

sns.countplot(campana_consultora['evaluacion_nuevas'].fillna('NOID'))
table=pd.crosstab(campana_consultora.campana,campana_consultora['evaluacion_nuevas'].fillna('NOID'))

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,figsize=(15,4))

plt.title('Campana vs evaluacion_nuevas')

plt.xlabel('Campana')

plt.ylabel('Proporción de evaluacion_nuevas')
campana_consultora['segmentacion'].head()
plt.figure(figsize=(15,5))

sns.countplot(campana_consultora['segmentacion'].fillna('NOID'))
table=pd.crosstab(campana_consultora.campana,campana_consultora['segmentacion'].fillna('NOID'))

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,figsize=(15,4))

plt.title('Campana vs segmentacion')

plt.xlabel('Campana')

plt.ylabel('Proporción de segmentacion')
plt.figure(figsize=(15,5))

campana_consultora['cantidadlogueos'].dropna().hist()
table=pd.crosstab(campana_consultora.campana,campana_consultora['flagdigital'].fillna('NOID'))

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,figsize=(15,4))

plt.title('Campana vs flagdigital')

plt.xlabel('Campana')

plt.ylabel('Proporción de flagdigital')
campana_consultora['flagdigital'].value_counts()
plt.figure(figsize=(15,5))

sns.countplot(campana_consultora['geografia'].fillna('NOID'))
table=pd.crosstab(campana_consultora.campana,campana_consultora['geografia'].fillna('NOID'))

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True,figsize=(15,4))

plt.title('Campana vs geografia')

plt.xlabel('Campana')

plt.ylabel('Proporción de geografia')
campana_consultora.drop(['flagpedidoanulado','flagdigital'],axis=1,inplace=True)
campana_consultora['codigofactura'] = campana_consultora['codigofactura'].fillna('NOID')

campana_consultora['evaluacion_nuevas'] = campana_consultora['evaluacion_nuevas'].fillna('NOID_EN')

campana_consultora['segmentacion'] = campana_consultora['segmentacion'].fillna('NOID')

campana_consultora['geografia'] = campana_consultora['geografia'].fillna('NOID')
target_col = 'Flagpasopedido'

columnas_categoricas=['codigofactura','evaluacion_nuevas','segmentacion','geografia','Flagpasopedido']

describe_dataset(campana_consultora[columnas_categoricas]  , target_col)
columnas_categoricas=['codigofactura','evaluacion_nuevas','segmentacion','geografia']
%%time

campana_consultora = freq_encod(campana_consultora,columnas_categoricas)
def usa_somos_Belcorp(codigofactura):

    if codigofactura in ['WEB','MIX','WMX','AWM']:

        return 1

    else:

        return 0



def usa_APP(codigofactura):

    if codigofactura in ['APP','APW','AWM','APM']:

        return 1

    else:

        return 0



def usa_medios_NoDigitales(codigofactura):

    if codigofactura in ['MIX','WMX','AWM','APM']:

        return 1

    else:

        return 0

    

def numero_canales(codigofactura):

    if codigofactura in ['WEB','APP','DD']:

        return 1

    elif codigofactura in ['MIX','WMX','APW','APM']:

        return 2

    elif codigofactura in ['AWM']:

        return 3

    else:

        return 999
campana_consultora['flg_SomoBelcorp'] = campana_consultora['codigofactura'].apply(lambda x: usa_somos_Belcorp(x))

campana_consultora['flg_APP'] = campana_consultora['codigofactura'].apply(lambda x: usa_APP(x))

campana_consultora['flg_no_digital'] = campana_consultora['codigofactura'].apply(lambda x: usa_medios_NoDigitales(x))

campana_consultora['numero_canales'] = campana_consultora['codigofactura'].apply(lambda x: numero_canales(x))
def codigofactura_funcion(x):

    if x in ['WEB','APP','APW','NOID']:

        return x

    else:

        return 'OTRO'
campana_consultora['codigofactura'] = campana_consultora['codigofactura'].apply(lambda x: codigofactura_funcion(x))
plt.figure(figsize=(15,5))

sns.countplot(campana_consultora['codigofactura'])
plt.figure(figsize=(20,5))

sns.countplot(campana_consultora['evaluacion_nuevas'])
def cantidad_camapannas_primer_pedido(evaluacion_nueva):

    if evaluacion_nueva=='Est':

        return 10

    elif evaluacion_nueva=='NOID_EN':

        return -1

    else:

        return int(evaluacion_nueva[4:5])

    

def cantidad_campanas_pedido(evaluacion_nueva):

    if evaluacion_nueva=='Est':

        return 10

    elif evaluacion_nueva=='NOID_EN':

        return 0

    else:

        return int(evaluacion_nueva[2:3])

    

def todas_campanas_pedidos(evaluacion_nueva):

    if evaluacion_nueva[0:1]=='C':

        return 1

    else:

        return 0
%%time

campana_consultora['cant_cam_primer_pedido'] = campana_consultora['evaluacion_nuevas'].apply(lambda x: cantidad_camapannas_primer_pedido(x))

campana_consultora['cant_cam_con_pedido'] = campana_consultora['evaluacion_nuevas'].apply(lambda x: cantidad_campanas_pedido(x))

campana_consultora['flg_todas_campanas'] = campana_consultora['evaluacion_nuevas'].apply(lambda x: todas_campanas_pedidos(x))
campana_consultora['ratio_cant_cam_con_pedido'] = campana_consultora['cant_cam_con_pedido']/campana_consultora['cant_cam_primer_pedido']
campana_consultora.head()
def evaluacion_nuevas_funcion(x):

    if x in ['Est','C_1d1','C_2d2','C_3d3','C_4d4','C_5d5','C_6d6']:

        return x

    else:

        return 'OTRO_EN'
campana_consultora['evaluacion_nuevas'] = campana_consultora['evaluacion_nuevas'].apply(lambda x: evaluacion_nuevas_funcion(x))
def label_segmentacion(segmentacion):

    if segmentacion=='Tops':

        return 0

    elif segmentacion=='Nivel5':

        return 5

    elif segmentacion=='Nivel2':

        return 2

    elif segmentacion=='Nuevas':

        return 999

    elif segmentacion=='Nivel4':

        return 4

    elif segmentacion=='Nivel3':

        return 3

    elif segmentacion=='Nivel7':

        return 7

    elif segmentacion=='Nivel1':

        return 1
campana_consultora['label_segmentacion'] = campana_consultora['segmentacion'].apply(lambda x: label_segmentacion(x))
plt.figure(figsize=(15,5))

sns.countplot(campana_consultora['segmentacion'].fillna('NOID'))
def segmentacion_funcion(x):

    if x in ['Tops','Nivel1','Nivel2']:

        return 'Nivel_1'

    elif x in ['Nivel3','Nivel4']:

        return 'Nivel_2'

    elif x in ['Nivel5','Nivel7']:

        return 'Nivel_3'

    else:

        return x
campana_consultora['segmentacion'] = campana_consultora['segmentacion'].apply(lambda x: segmentacion_funcion(x))
plt.figure(figsize=(15,5))

sns.countplot(campana_consultora['segmentacion'].fillna('NOID'))
plt.figure(figsize=(15,5))

sns.countplot(campana_consultora['campana'])
le = preprocessing.LabelEncoder()
le.fit(campana_consultora['campana'])
campana_consultora['campana_label'] = le.transform(campana_consultora['campana']) 
campana_consultora.groupby(['campana','campana_label']).size()
maestro_consultora = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/maestro_consultora.csv')

maestro_consultora.drop(['Unnamed: 0'],axis=1,inplace=True)

maestro_consultora.head()
maestro_consultora.info()
missing_values_table(maestro_consultora)
maestro_consultora.shape
maestro_consultora.IdConsultora.nunique()
print(maestro_consultora.campanaingreso.min())

print(maestro_consultora.campanaingreso.max())
print(maestro_consultora.campanaultimopedido.min())

print(maestro_consultora.campanaultimopedido.max())
plt.figure(figsize=(15,5))

sns.countplot(maestro_consultora.estadocivil.fillna('NOID'))
maestro_consultora.estadocivil.value_counts()
plt.figure(figsize=(15,5))

sns.countplot(maestro_consultora.flagsupervisor.fillna('NOID'))
maestro_consultora.flagsupervisor.value_counts()
print(maestro_consultora.campanaprimerpedido.min())

print(maestro_consultora.campanaprimerpedido.max())
plt.figure(figsize=(15,5))

sns.countplot(maestro_consultora.flagconsultoradigital.fillna('NOID'))
maestro_consultora.flagconsultoradigital.value_counts()
plt.figure(figsize=(15,5))

sns.countplot(maestro_consultora.flagcorreovalidad.fillna('NOID'))
plt.figure(figsize=(15,5))

sns.countplot(maestro_consultora.flagcelularvalidado.fillna('NOID'))
plt.figure(figsize=(15,5))

maestro_consultora['edad'].hist()
maestro_consultora['estadocivil'] = maestro_consultora['estadocivil'].fillna('Soltero(a)')

maestro_consultora['flagcorreovalidad'] = maestro_consultora['flagcorreovalidad'].fillna('NOID_CORREO')
maestro_consultora.drop('flagsupervisor',axis=1,inplace=True)

maestro_consultora.drop('flagconsultoradigital',axis=1,inplace=True)
catalogo_campannas = pd.DataFrame(maestro_consultora.campanaingreso.value_counts()).reset_index()

catalogo_campannas.columns = ['campana_id','Registros']

catalogo_campannas.head()
le = preprocessing.LabelEncoder()

catalogo_campannas['campana_id_label'] = le.fit_transform(catalogo_campannas['campana_id'])
catalogo_campannas.drop('Registros',axis=1,inplace=True)
detalle_ventas = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/dtt_fvta_cl.csv',low_memory=False)
detalle_ventas.head()
detalle_ventas.columns = ['campana','IdConsultora','codigotipooferta','descuento','ahorro','canalingresoproducto','idproducto','codigopalancapersonalizacion','palancapersonalizacion','preciocatalogo','grupooferta','realanulmnneto','realdevmnneto','realuuanuladas','realuudevueltas','realuufaltantes','realuuvendidas','realvtamnfaltneto','realvtamnneto','realvtamncatalogo','realvtamnfaltcatalogo']
missing_values_table(detalle_ventas)
detalle_ventas.info()
plt.figure(figsize=(15,5))

sns.countplot(detalle_ventas['campana'])
detalle_ventas['IdConsultora'].nunique()
detalle_ventas['codigotipooferta'].value_counts()*100.0/len(detalle_ventas)
detalle_ventas['codigotipooferta'].nunique()
plt.figure(figsize=(15,5))

detalle_ventas['codigotipooferta'].hist()
plt.figure(figsize=(15,5))

detalle_ventas['descuento'].hist()
plt.figure(figsize=(15,5))

detalle_ventas[detalle_ventas['ahorro']<500000]['ahorro'].hist()
plt.figure(figsize=(15,5))

detalle_ventas[detalle_ventas['ahorro']>200000]['ahorro'].hist()
plt.figure(figsize=(15,5))

sns.countplot(detalle_ventas['canalingresoproducto'].fillna('NOID'))
detalle_ventas['canalingresoproducto'].fillna('NOID').value_counts()*100.0/len(detalle_ventas)
plt.figure(figsize=(15,5))

detalle_ventas['preciocatalogo'].hist()
plt.figure(figsize=(15,5))

sns.countplot(detalle_ventas['grupooferta'].fillna('NOID'))
detalle_ventas['grupooferta'].fillna('NOID').value_counts()*100.0/len(detalle_ventas)
plt.figure(figsize=(15,5))

detalle_ventas['realuudevueltas'].hist()
listado_codigotipooferta = [13, 15, 11, 116, 30, 106, 49, 4, 3, 48, 123, 18, 1, 33, 51, 225, 36, 29, 212, 114, 202, 7, 34]
def codigotipooferta_funcion(x):

    if x in listado_codigotipooferta:

        return x

    else:

        return 999
detalle_ventas['codigotipooferta'] = detalle_ventas['codigotipooferta'].apply(lambda x: codigotipooferta_funcion(x))
detalle_ventas['codigotipooferta'].value_counts()*100.0/len(detalle_ventas)
detalle_ventas['canalingresoproducto'] = detalle_ventas['canalingresoproducto'].fillna('NOID')
def canalingresoproducto_funcion(x):

    if x in ['WEB','APP','NOID']:

        return x

    else:

        return 'OTRO'
detalle_ventas['canalingresoproducto'] = detalle_ventas['canalingresoproducto'].apply(lambda x: canalingresoproducto_funcion(x))
detalle_ventas['canalingresoproducto'].value_counts()*100.0/len(detalle_ventas)
detalle_ventas['grupooferta'] = detalle_ventas['grupooferta'].fillna('NOID')
detalle_ventas['grupooferta'].value_counts()*100.0/len(detalle_ventas)
detalle_ventas['codigopalancapersonalizacion'] = detalle_ventas['codigopalancapersonalizacion'].fillna(99999999)
detalle_ventas['codigopalancapersonalizacion'] = detalle_ventas['codigopalancapersonalizacion'].apply(lambda x: int(x))
valores_codigopalanca = [40210, 42, 12, 22, 10210, 99999999, 20210]
def codigopalancapersonalizacion_funcion(x):

    if x in valores_codigopalanca:

        return x

    else:

        return 55555555
detalle_ventas['codigopalancapersonalizacion'] = detalle_ventas['codigopalancapersonalizacion'].apply(lambda x: codigopalancapersonalizacion_funcion(x))
detalle_ventas['codigopalancapersonalizacion'].value_counts()*100.0/len(detalle_ventas)
detalle_ventas['palancapersonalizacion'] = detalle_ventas['palancapersonalizacion'].fillna('NOID')
palanca_desc = ['NOID', 'App Consultora Pedido Digitado', 'Desktop Pedido Digitado', 'Mobile Pedido Digitado', 'Ofertas Para ti', 'Showroom']
def palancapersonalizacion_funcion(x):

    if x in palanca_desc:

        return x

    else:

        return 'OTRO'
detalle_ventas['palancapersonalizacion'] = detalle_ventas['palancapersonalizacion'].apply(lambda x: palancapersonalizacion_funcion(x))
detalle_ventas['palancapersonalizacion'].value_counts()*100.0/len(detalle_ventas)
detalle_ventas[['realuuanuladas','realuudevueltas','realuuvendidas','realvtamncatalogo','realvtamnfaltcatalogo','preciocatalogo']].head(10)
le = preprocessing.LabelEncoder()

le.fit(detalle_ventas['campana'])

detalle_ventas['campana_label'] = le.transform(detalle_ventas['campana'])
maestro_producto = pd.read_csv('/kaggle/input/datathon-belcorp-prueba/maestro_producto.csv')

maestro_producto.drop('Unnamed: 0',inplace=True,axis=1)

maestro_producto.head()
missing_values_table(maestro_producto)
plt.figure(figsize=(15,5))

sns.countplot(maestro_producto['codigounidadnegocio'])
plt.figure(figsize=(15,5))

sns.countplot(maestro_producto['unidadnegocio'])
maestro_producto.groupby(['codigounidadnegocio','unidadnegocio']).size()
plt.figure(figsize=(15,5))

sns.countplot(maestro_producto['marca'])
maestro_producto.groupby(['codigomarca','marca']).size()
maestro_producto.groupby(['codigocategoria','categoria']).size()
maestro_producto.groupby(['codigotipo','tipo']).size()
maestro_producto[['idproducto','codigounidadnegocio','codigomarca','codigocategoria','codigotipo']].head()
codigo_tipos = ['001','027','026','086','383','025','013','230','999','041','366','035','032','503','512']
def codigotipo_funcion(x):

    if x in codigo_tipos:

        return x

    else:

        return 'OTRO'
# Función que construye la matriz
def creacion_variables(mes_campana,meses_historia):

    ## MES Y TARGET

    if mes_campana==18:

        data_temp = predict_submission[['campana','IdConsultora','Flagpasopedido']].copy()

    else:

        data_temp = campana_consultora[campana_consultora['campana_label']==mes_campana][['campana','IdConsultora','Flagpasopedido']].copy()

        data_temp = pd.merge(predict_submission[['IdConsultora']],data_temp,how='left',on='IdConsultora')

        data_temp['campana'] = data_temp['campana'].fillna(max(data_temp['campana']))

        data_temp['Flagpasopedido'] = data_temp['Flagpasopedido'].fillna(0)

        data_temp['campana'] = data_temp['campana'].apply(lambda x: int(x))

        data_temp['Flagpasopedido'] = data_temp['Flagpasopedido'].apply(lambda x: int(x))

        data_temp = data_temp[['campana','IdConsultora','Flagpasopedido']]



    # INFORMACIÓN ÚLTIMO MES (M-1)

    data_temp_M1 = campana_consultora[campana_consultora['campana_label']==mes_campana-1].copy()

    columnas_M1 = data_temp_M1.columns



    columnas_M1_nuevo = []

    for columna in columnas_M1:

        if columna=='IdConsultora':

            columnas_M1_nuevo.append(columna)

        else:

            columnas_M1_nuevo.append(columna+'_M1')

    data_temp_M1.columns = columnas_M1_nuevo

    data_temp_M1.drop('campana_M1',axis=1,inplace=True)



    ## UNIR

    data_temp = pd.merge(data_temp,data_temp_M1,how='left',on=['IdConsultora'])



    del data_temp_M1



    ## HISTÓRICO

    data_temp_Meses = campana_consultora[(campana_consultora['campana_label']<=mes_campana-1)&(campana_consultora['campana_label']>=mes_campana-meses_historia)].copy()



    ## VARIABLES CON FLAGS

    columnas_flags = ['Flagpasopedido','flagactiva','flagpasopedidocuidadopersonal','flagpasopedidomaquillaje','flagpasopedidotratamientocorporal','flagpasopedidotratamientofacial','flagpasopedidofragancias','flagpasopedidoweb','flagdispositivo','flagofertadigital','flagsuscripcion','flg_SomoBelcorp','flg_APP','flg_no_digital','flg_todas_campanas']



    ## PROMEDIO

    data_temp_Meses_flag = data_temp_Meses[columnas_flags+['IdConsultora']].groupby('IdConsultora',as_index=False).sum()

    data_temp_Meses_flag[columnas_flags] = data_temp_Meses_flag[columnas_flags]*100/meses_historia

    data_temp_Meses_flag.columns = ['IdConsultora']+list(agregar_sufijo(columnas_flags,'PROM'))



    cantidad_campannas_list = [3,6,9]

    for campana_ in cantidad_campannas_list:

        if campana_<meses_historia:

            data_temp_Meses_2 = campana_consultora[(campana_consultora['campana_label']<=mes_campana-1)&(campana_consultora['campana_label']>=mes_campana-campana_)].copy()

            data_temp_Meses_2 = data_temp_Meses_2[columnas_flags+['IdConsultora']].groupby('IdConsultora',as_index=False).sum()

            data_temp_Meses_2[columnas_flags] = data_temp_Meses_2[columnas_flags]*100/campana_

            data_temp_Meses_2.columns = ['IdConsultora']+list(agregar_sufijo(columnas_flags,str(campana_)+'_PROM'))

            

            data_temp_Meses_flag = pd.merge(data_temp_Meses_flag,data_temp_Meses_2,how='left',on='IdConsultora')



        print(campana_)

        

    data_temp_Meses_flag.fillna(0,inplace=True)



    ## MAX

    data_temp_Meses_flag_MAX = data_temp_Meses[columnas_flags+['IdConsultora']].groupby('IdConsultora',as_index=False).max()

    data_temp_Meses_flag_MAX.columns = ['IdConsultora']+list(agregar_sufijo(columnas_flags,'MAX'))



    ## MIN

    data_temp_Meses_flag_MIN = data_temp_Meses[columnas_flags+['IdConsultora']].groupby('IdConsultora',as_index=False).min()

    data_temp_Meses_flag_MIN.columns = ['IdConsultora']+list(agregar_sufijo(columnas_flags,'MIN'))



    ## UNIR

    data_temp = pd.merge(data_temp,data_temp_Meses_flag,how='left',on=['IdConsultora'])

    data_temp = pd.merge(data_temp,data_temp_Meses_flag_MAX,how='left',on=['IdConsultora'])

    data_temp = pd.merge(data_temp,data_temp_Meses_flag_MIN,how='left',on=['IdConsultora'])



    ## DIFERENCIA_ULTIMO Y PRIMER MES

    for columna in columnas_flags:

        # ÚLTIMO

        data_temp_diferencia = data_temp_Meses[data_temp_Meses[columna]==1].groupby('IdConsultora',as_index=False)[['campana_label']].max()

        data_temp_diferencia.columns = ['IdConsultora',columna+'_ultimo']

        data_temp_diferencia[columna+'_ultimo'] = mes_campana - data_temp_diferencia[columna+'_ultimo']

        

        data_temp = pd.merge(data_temp,data_temp_diferencia,how='left',on=['IdConsultora'])

        

        # PRIMERO

        data_temp_diferencia_2 = data_temp_Meses[data_temp_Meses[columna]==1].groupby('IdConsultora',as_index=False)[['campana_label']].min()

        data_temp_diferencia_2.columns = ['IdConsultora',columna+'_primero']

        data_temp_diferencia_2[columna+'_primero'] = mes_campana - data_temp_diferencia_2[columna+'_primero']

        data_temp = pd.merge(data_temp,data_temp_diferencia_2,how='left',on=['IdConsultora'])

        #print(columna)



    ## CANTIDADLOGUEOS

    # SUMA

    data_temp_Meses_logueos = data_temp_Meses.groupby('IdConsultora',as_index=False)['cantidadlogueos'].sum()

    data_temp_Meses_logueos.columns = ['IdConsultora','cantidadlogueos_sum']

    data_temp = pd.merge(data_temp,data_temp_Meses_logueos,how='left',on=['IdConsultora'])



    # AVG

    data_temp_Meses_logueos = data_temp_Meses.groupby('IdConsultora',as_index=False)['cantidadlogueos'].mean()

    data_temp_Meses_logueos.columns = ['IdConsultora','cantidadlogueos_mean']

    data_temp = pd.merge(data_temp,data_temp_Meses_logueos,how='left',on=['IdConsultora'])



    # median

    data_temp_Meses_logueos = data_temp_Meses.groupby('IdConsultora',as_index=False)['cantidadlogueos'].median()

    data_temp_Meses_logueos.columns = ['IdConsultora','cantidadlogueos_median']

    data_temp = pd.merge(data_temp,data_temp_Meses_logueos,how='left',on=['IdConsultora'])



    # MIN

    data_temp_Meses_logueos = data_temp_Meses.groupby('IdConsultora',as_index=False)['cantidadlogueos'].min()

    data_temp_Meses_logueos.columns = ['IdConsultora','cantidadlogueos_min']

    data_temp = pd.merge(data_temp,data_temp_Meses_logueos,how='left',on=['IdConsultora'])



    # MAX

    data_temp_Meses_logueos = data_temp_Meses.groupby('IdConsultora',as_index=False)['cantidadlogueos'].max()

    data_temp_Meses_logueos.columns = ['IdConsultora','cantidadlogueos_max']

    data_temp = pd.merge(data_temp,data_temp_Meses_logueos,how='left',on=['IdConsultora'])



    # std

    #data_temp_Meses_logueos = data_temp_Meses.groupby('IdConsultora',as_index=False)['cantidadlogueos'].std()

    #data_temp_Meses_logueos.columns = ['IdConsultora','cantidadlogueos_std']

    #data_temp_temp = pd.merge(data_temp_temp,data_temp_Meses_logueos,how='left',on=['IdConsultora'])



    # VAR

    data_temp_Meses_logueos = data_temp_Meses.groupby('IdConsultora',as_index=False)['cantidadlogueos'].var()

    data_temp_Meses_logueos.columns = ['IdConsultora','cantidadlogueos_var']

    data_temp = pd.merge(data_temp,data_temp_Meses_logueos,how='left',on=['IdConsultora'])



    ## CATEGÓRICAS

    columnas_categoricas = list(data_temp.select_dtypes(include=['object']).columns)

    df_columnas_categoricas = pd.get_dummies(data_temp[columnas_categoricas])

    data_temp = pd.concat([data_temp,df_columnas_categoricas],axis=1)

    data_temp.drop(columnas_categoricas,axis=1,inplace=True)



    #############################################

    ####   CRUCE CON MAESTRO CONSULTORA

    #############################################



    data_temp = pd.merge(data_temp,maestro_consultora,how='left',on=['IdConsultora'])



    ## Diferencia entre camapaña actual y campaña de ingreso

    data_temp['campana_diferencia_ingreso'] = round(data_temp['campana']/100) - round(data_temp['campanaingreso']/100)



    catalogo_campannas.columns = ['campana_id','campana_label_target']

    data_temp = pd.merge(data_temp,catalogo_campannas, how='left', left_on='campana', right_on='campana_id')

    data_temp.drop('campana_id',axis=1,inplace=True)

    # La última campaña con información disponible

    data_temp['campana_label_target'] = data_temp['campana_label_target']-1



    catalogo_campannas.columns = ['campana_id','campana_label_ingreso']

    data_temp = pd.merge(data_temp,catalogo_campannas, how='left', left_on='campanaingreso', right_on='campana_id')

    data_temp.drop('campana_id',axis=1,inplace=True)



    data_temp['campana_diff_ingreso_campana'] = data_temp['campana_label_target'] - data_temp['campana_label_ingreso']



    ## Diferencia entre camapaña actual y campaña de ingreso

    data_temp['campana_diferencia_ult_pedido'] = round(data_temp['campana']/100) - round(data_temp['campanaultimopedido']/100)

    data_temp['campana_diferencia_ult_pedido'] = data_temp['campana_diferencia_ult_pedido'].apply(lambda x: 0 if x<0 else x)



    catalogo_campannas.columns = ['campana_id','campana_label_pedido']

    data_temp = pd.merge(data_temp,catalogo_campannas, how='left', left_on='campanaultimopedido', right_on='campana_id')

    data_temp.drop('campana_id',axis=1,inplace=True)



    data_temp['campana_diff_ult_pedido_campana'] = data_temp['campana_label_target'] - data_temp['campana_label_pedido']



    ## Diferencia entre camapaña actual y campaña de primer pedido



    data_temp['campanaprimerpedido'] = data_temp['campanaprimerpedido'].fillna(data_temp['campana'].max())

    data_temp['campanaprimerpedido'] = data_temp['campanaprimerpedido'].apply(lambda x: int(x))

    data_temp['campana_diferencia_pri_pedido'] = round(data_temp['campana']/100) - round(data_temp['campanaprimerpedido']/100)



    catalogo_campannas.columns = ['campana_id','campana_label_pri_pedido']

    data_temp = pd.merge(data_temp,catalogo_campannas, how='left', left_on='campanaprimerpedido', right_on='campana_id')

    data_temp.drop('campana_id',axis=1,inplace=True)

    data_temp['campana_diff_pri_pedido_campana'] = data_temp['campana_label_target'] - data_temp['campana_label_pri_pedido']



    ## Dropear labels y campos de campañas



    data_temp.drop(['campana_label_target','campana_label_ingreso','campana_label_pedido','campana_label_pri_pedido'],axis=1,inplace=True)

    data_temp.drop(['campanaingreso','campanaultimopedido','campanaprimerpedido'],axis=1,inplace=True)

    df_dummies = pd.get_dummies(data_temp[['estadocivil','flagcorreovalidad']])

    data_temp = pd.concat([data_temp,df_dummies],axis=1)

    data_temp.drop(['estadocivil','flagcorreovalidad'],axis=1,inplace=True)

    

    #############################################

    ####   CRUCE CON DETALLE VENTA

    #############################################

    print('Detalle venta')

    detalle_ventas_temp = detalle_ventas[(detalle_ventas['campana_label']<=mes_campana-1)&(detalle_ventas['campana_label']>=mes_campana-int(meses_historia/2))].copy()

    

    columnas_numericas = ['descuento','ahorro','realanulmnneto','realdevmnneto','realuuanuladas','realuudevueltas','realuufaltantes','realuuvendidas','realvtamnfaltneto','realvtamnneto','realvtamncatalogo','realvtamnfaltcatalogo']

    

    # Log Columnas

    columnas_log = []

    for columna in columnas_numericas:

        if columna!='realvtamnfaltcatalogo':

            detalle_ventas_temp[columna+'_log'] = np.log(1+detalle_ventas_temp[columna])

            columnas_log.append(columna+'_log')

        else:

            detalle_ventas_temp[columna+'_log'] = np.log(10486050+detalle_ventas_temp[columna])

            columnas_log.append(columna+'_log')

    

    ## SUMA

    detalle_ventas_temp_sum = detalle_ventas_temp[columnas_numericas+columnas_log+['IdConsultora']].groupby('IdConsultora',as_index=False).sum()#.add_suffix('_SUM')

    detalle_ventas_temp_sum[columnas_numericas+columnas_log] = detalle_ventas_temp_sum[columnas_numericas+columnas_log]/int(meses_historia/2)

    detalle_ventas_temp_sum.columns = ['IdConsultora']+list(agregar_sufijo(columnas_numericas+columnas_log,'SUM'))

    

    ## MAX

    detalle_ventas_temp_max = detalle_ventas_temp[columnas_numericas+columnas_log+['IdConsultora']].groupby('IdConsultora',as_index=False).max()#.add_suffix('_MAX')

    detalle_ventas_temp_max.columns = ['IdConsultora']+list(agregar_sufijo(columnas_numericas+columnas_log,'MAX'))

    

    ## MIN

    detalle_ventas_temp_min = detalle_ventas_temp[columnas_numericas+columnas_log+['IdConsultora']].groupby('IdConsultora',as_index=False).min()#.add_suffix('_MIN')

    detalle_ventas_temp_min.columns = ['IdConsultora']+list(agregar_sufijo(columnas_numericas+columnas_log,'MIN'))

    

    ## STD

    detalle_ventas_temp_std = detalle_ventas_temp[columnas_numericas+columnas_log+['IdConsultora']].groupby('IdConsultora').std()

    detalle_ventas_temp_std.reset_index(inplace=True)

    detalle_ventas_temp_std.columns = ['IdConsultora']+list(agregar_sufijo(columnas_numericas+columnas_log,'STD'))

    

    ## AVG

    detalle_ventas_temp_avg = detalle_ventas_temp[columnas_numericas+columnas_log+['IdConsultora']].groupby('IdConsultora',as_index=False).mean()#.add_suffix('_AVG')

    detalle_ventas_temp_avg.columns = ['IdConsultora']+list(agregar_sufijo(columnas_numericas+columnas_log,'AVG'))

    

    ## UNIR

    data_temp = pd.merge(data_temp,detalle_ventas_temp_sum,how='left',on=['IdConsultora'])

    data_temp = pd.merge(data_temp,detalle_ventas_temp_max,how='left',on=['IdConsultora'])

    data_temp = pd.merge(data_temp,detalle_ventas_temp_min,how='left',on=['IdConsultora'])

    data_temp = pd.merge(data_temp,detalle_ventas_temp_std,how='left',on=['IdConsultora'])

    data_temp = pd.merge(data_temp,detalle_ventas_temp_avg,how='left',on=['IdConsultora'])

    

    del detalle_ventas_temp_sum,detalle_ventas_temp_max,detalle_ventas_temp_min,detalle_ventas_temp_std,detalle_ventas_temp_avg

    

    ###   CRUCE CON CATÁLOGO PRODUCTO

    

    detalle_ventas_temp = pd.merge(detalle_ventas_temp,maestro_producto[['idproducto','codigounidadnegocio','codigomarca','codigocategoria','codigotipo']],how='left',on='idproducto')

    detalle_ventas_temp['codigotipo'] = detalle_ventas_temp['codigotipo'].apply(lambda x: codigotipo_funcion(x))

    

    columnas_categoricas = ['codigomarca']

    columna_numerica = 'realvtamncatalogo'

    for columna_categorica in columnas_categoricas:

        ###    SUMA    ###

        detalle_ventas_cat_num = detalle_ventas_temp.pivot_table(index='IdConsultora', columns=columna_categorica, values=columna_numerica, aggfunc=np.sum, fill_value = 0)

        nuevas_columns = list(detalle_ventas_cat_num.columns)

        detalle_ventas_cat_num.reset_index(inplace=True)

        detalle_ventas_cat_num[nuevas_columns] = detalle_ventas_cat_num[nuevas_columns]/int(meses_historia/2)

        detalle_ventas_cat_num.columns = ['IdConsultora']+list(agregar_sufijo(nuevas_columns,columna_categorica+'_'+columna_numerica+'_SUM_ING'))

        data_temp = pd.merge(data_temp,detalle_ventas_cat_num,how='left',on=['IdConsultora'])

        

        ###   AVG

        detalle_ventas_cat_num = detalle_ventas_temp.pivot_table(index='IdConsultora', columns=columna_categorica, values=columna_numerica, aggfunc=np.mean, fill_value = 0)

        nuevas_columns = list(detalle_ventas_cat_num.columns)

        detalle_ventas_cat_num.reset_index(inplace=True)

        detalle_ventas_cat_num.columns = ['IdConsultora']+list(agregar_sufijo(nuevas_columns,columna_categorica+'_'+columna_numerica+'_AVG_ING'))

        data_temp = pd.merge(data_temp,detalle_ventas_cat_num,how='left',on=['IdConsultora'])



        ###  MAX

        detalle_ventas_cat_num = detalle_ventas_temp.pivot_table(index='IdConsultora', columns=columna_categorica, values=columna_numerica, aggfunc=np.max, fill_value = 0)

        nuevas_columns = list(detalle_ventas_cat_num.columns)

        detalle_ventas_cat_num.reset_index(inplace=True)

        detalle_ventas_cat_num.columns = ['IdConsultora']+list(agregar_sufijo(nuevas_columns,columna_categorica+'_'+columna_numerica+'_MAX_ING'))

        data_temp = pd.merge(data_temp,detalle_ventas_cat_num,how='left',on=['IdConsultora'])



        ###  MIN

        detalle_ventas_cat_num = detalle_ventas_temp.pivot_table(index='IdConsultora', columns=columna_categorica, values=columna_numerica, aggfunc=np.min, fill_value = 0)

        nuevas_columns = list(detalle_ventas_cat_num.columns)

        detalle_ventas_cat_num.reset_index(inplace=True)

        detalle_ventas_cat_num.columns = ['IdConsultora']+list(agregar_sufijo(nuevas_columns,columna_categorica+'_'+columna_numerica+'_MIN_ING'))

        data_temp = pd.merge(data_temp,detalle_ventas_cat_num,how='left',on=['IdConsultora'])

        

        print(columna_categorica,columna_numerica)

    return data_temp
## PARÁMETROS

mes_campana = 17

meses_historia = 12 ## cantidad de historia de campanas
matriz_modelo = creacion_variables(18,meses_historia) ## matriz de test
matriz_modelo.head()
# LA EJECUCIÓN LO REALICÉ EN MI LAPTOP, EN KAGGLE FALLA POR FALTA DE RAM, DEBERÍA USARSE ESTA CELDA

# Agregar campanas para el train



#for mes_campana in range(17,11,-1):

#    print(mes_campana,mes_campana-meses_historia)

#    data_temp = creacion_variables(mes_campana,meses_historia)

#    matriz_modelo = pd.concat([matriz_modelo,data_temp],axis=0,ignore_index=True)
# Solo usare una campaña para que se pueda ejecutar en Kaggle

for mes_campana in range(17,16,-1):

    print(mes_campana,mes_campana-meses_historia)

    data_temp = creacion_variables(mes_campana,meses_historia)

    matriz_modelo = pd.concat([matriz_modelo,data_temp],axis=0,ignore_index=True)
del data_temp
matriz_modelo.head()
matriz_modelo.shape
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import xgboost as xgb

import operator
X = matriz_modelo[matriz_modelo['campana']<201907].drop(['campana','IdConsultora','Flagpasopedido'],axis=1)

y = matriz_modelo[matriz_modelo['campana']<201907]['Flagpasopedido']
print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=11) # 65
import lightgbm

from sklearn.metrics import (roc_curve, auc, accuracy_score)

from sklearn.model_selection import GridSearchCV
train_data = lightgbm.Dataset(X_train, label=y_train)

test_data = lightgbm.Dataset(X_test, label=y_test)
parameters = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    #'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 100,#31,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.9,

    'bagging_freq': 20,

    'learning_rate': 0.003,

    'verbose': 0

}
model = lightgbm.train(parameters,

                       train_data,

                       valid_sets=test_data,

                       num_boost_round=500,

                       early_stopping_rounds=100)
feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(),X_train.columns)), columns=['Importancia','Variable'])

feature_imp.sort_values(by='Importancia', ascending=False, inplace=True)
variables_importantes_lgbm = list(feature_imp.head(50).Variable)
train_data = lightgbm.Dataset(X_train[variables_importantes_lgbm] , label=y_train)

test_data = lightgbm.Dataset(X_test[variables_importantes_lgbm], label=y_test)
model = lightgbm.train(parameters,

                       train_data,

                       valid_sets=test_data,

                       num_boost_round=2000,

                       early_stopping_rounds=100)
predict_lgthgbm = model.predict(matriz_modelo[matriz_modelo['campana']==201907][variables_importantes_lgbm],num_iteration=model.best_iteration)
submit = matriz_modelo[matriz_modelo['campana']==201907][['IdConsultora']]

submit['flagpasopedido'] = predict_lgthgbm

submit.columns = ['idconsultora','flagpasopedido']
submit.head()
submit.to_csv('Submit_Final.csv',index=False)