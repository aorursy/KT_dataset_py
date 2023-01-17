# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', delimiter = ',' )

test = pd.read_csv('../input/test.csv', delimiter = ',' )



df_sub = pd.DataFrame( columns = ['ID', 'y'] )



df_sub['ID'] = test.pop('ID')



train_labels = np.log1p( train.pop('y') )

train.drop('ID', axis = 1, inplace = True)
#Normalize numeric features

def normdf( train, test ):



    from scipy.stats import skew



    numeric_feats = train.dtypes[train.dtypes != "object"].index



    for col in numeric_feats:

        cardinality_train = len(np.unique(train[col]))

        cardinality_test = len(np.unique(test[col]))



        if cardinality_train == 1 | cardinality_test == 1:

            train.drop(col, axis = 1, inplace = True)

            test.drop(col, axis = 1, inplace = True)



    numeric_feats = train.dtypes[train.dtypes != "object"].index



    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

    left_skewed_feats = skewed_feats[skewed_feats > 0.75].index

    right_skewed_feats = skewed_feats[skewed_feats < -0.75].index



    train[left_skewed_feats] = np.log1p(train[left_skewed_feats])

    test[left_skewed_feats] = np.log1p(test[left_skewed_feats])



    train[right_skewed_feats] = np.expm1(train[right_skewed_feats])

    test[right_skewed_feats] = np.expm1(test[right_skewed_feats])



    return train, test
def dataPrep( df ):



    cat_cols = df.select_dtypes(['object']).columns



    n = len(cat_cols)



    print ('\nConcat string columns')



    #This concat string columns together, first in twos and then in threes

    for i in range(n):



        col1 = cat_cols[i]



        for j in range(i+1,n):



            col2 = cat_cols[j]



            new_col = col1 + '_' + col2



            df[new_col] = df[col1].str.cat(df[col2], sep = '_')



    cat_columns = df.select_dtypes(['object']).columns



    df_cat = df[cat_columns]



    df_cat = pd.get_dummies( df_cat )



    #Remove low frequency cat columns

    sums = df_cat.sum(axis = 0)



    l_bound = 0.2*df.shape[0]

    u_bound = 0.8*df.shape[0]



    to_remove = sums[sums < l_bound].index.values

    df_cat.drop(to_remove, axis = 1, inplace = True)



    to_remove = sums[sums > u_bound].index.values

    df_cat.drop(to_remove, axis = 1, inplace = True)

    

    df.drop(cat_columns, axis=1, inplace = True)



    df = pd.concat( [df, df_cat], axis = 1 )



    return df
train, test = normdf( train.copy(), test.copy() )



print ('Create training dataset')



train = dataPrep( train )



print ('\nCreate testing dataset')



test = dataPrep( test )



cols = list( set(train.columns) & set(test.columns) )



train = train[cols]

test = test[cols]
from sklearn.decomposition import PCA, FastICA

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import TruncatedSVD



n_comp = 12



    # tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)

tsvd_results_train = tsvd.fit_transform(train)

tsvd_results_test = tsvd.transform(test)



    # PCA

pca = PCA(n_components=n_comp, random_state=420)

pca2_results_train = pca.fit_transform(train)

pca2_results_test = pca.transform(test)



    # ICA

ica = FastICA(n_components=n_comp, random_state=420)

ica2_results_train = ica.fit_transform(train)

ica2_results_test = ica.transform(test)



    # GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

grp_results_train = grp.fit_transform(train)

grp_results_test = grp.transform(test)



    # SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

srp_results_train = srp.fit_transform(train)

srp_results_test = srp.transform(test)



for i in range(1, n_comp + 1):

    train['pca_' + str(i)] = pca2_results_train[:, i - 1]

    test['pca_' + str(i)] = pca2_results_test[:, i - 1]



    train['ica_' + str(i)] = ica2_results_train[:, i - 1]

    test['ica_' + str(i)] = ica2_results_test[:, i - 1]



    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]

    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]



    train['grp_' + str(i)] = grp_results_train[:, i - 1]

    test['grp_' + str(i)] = grp_results_test[:, i - 1]



    train['srp_' + str(i)] = srp_results_train[:, i - 1]

    test['srp_' + str(i)] = srp_results_test[:, i - 1]





print ('Feature Space Before: ' + str( train.shape[1] ) )



from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV



feat_sel = SelectFromModel( LassoCV( cv = 5, fit_intercept = False ) )



feat_sel.fit( train, train_labels )



train = feat_sel.transform( train )



test = feat_sel.transform( test )



print ('Feature Space After: ' + str( train.shape[1] ) )
def optGBM(train, test, train_labels):

    

    from sklearn.model_selection import GridSearchCV

    from sklearn.ensemble import GradientBoostingRegressor

    

    param_grid = {

        'max_depth': range(4,7),

        'min_samples_split' : range(3,7),

        'min_samples_leaf' : range(2,6)

    }

    

    

    from sklearn.linear_model import ElasticNetCV

    

    l1_list = np.arange(0.15, 1.0, 0.15)

    

    reg = ElasticNetCV( l1_ratio = l1_list, cv = 5, normalize = True )

    

    gbm = GradientBoostingRegressor()

    

    gbm_cv = GridSearchCV( gbm, param_grid, cv = 5, 

                          scoring = 'r2', n_jobs = -1, verbose = 2)

    

    gbm_cv.fit(train, train_labels)

    

    train_pred = np.zeros( ( train.shape[0], 100) )

    test_pred = np.zeros( ( test.shape[0], 100) )

    

    for i in range( 100 ):

        est = gbm_cv.best_estimator_.estimators_[i, 0]

        

        train_pred[:, i] = est.predict( train )

        test_pred[:, i] = est.predict( test )

        

    reg.fit(train_pred, train_labels)

    

    return reg.predict( train_pred ), reg.predict( test_pred )
def createStackPred( train, test, train_labels) :



    #Base Learners

    from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor

    from sklearn.linear_model import ElasticNetCV, LassoLarsCV, LassoLarsIC, LassoCV

    from sklearn.svm import LinearSVR, SVR, NuSVR



    

    l1_list = np.arange(0.15,1.0,0.15)



    print ("\n\tFitting our training model")



    #Lists of regressors



    line_list = [

                ExtraTreesRegressor( n_estimators = 100, n_jobs = -1 ),

                AdaBoostRegressor( loss = 'linear'),

                AdaBoostRegressor( loss = 'square'),

                LassoLarsIC( criterion = 'bic', fit_intercept = False ),

                LassoLarsIC( criterion = 'aic', fit_intercept = False ),

                LinearSVR( loss = 'epsilon_insensitive', fit_intercept = False),

                LinearSVR( loss = 'squared_epsilon_insensitive', fit_intercept = False),

                SVR(),

                NuSVR()

                ]



    line_n = len( line_list )



    n = line_n



    #Sets up the arrays to store the predictions



    test_pred = np.zeros( ( test.shape[0], n) )

    train_pred = np.zeros( ( train.shape[0], n) )

    

    for i in range(line_n):

       

        print ("\n\t\tAt regression model: " + str(i + 1) )

        est = line_list[i]

        

        est.fit( train, train_labels )

        

        train_pred[:,i] = est.predict( train )

        test_pred[:,i] = est.predict( test )



    print ("\n\tFitting... Done")



    return train_pred, test_pred
#This makes a linear relationship between the initial predictions of the train labels with the actual

#, and projections that onto the testing labels

def stacker(train_df, test_df, train_labels ) :



    import matplotlib.pyplot as plt



    print ("\nCreating our stacks of predictions")



    train_pred, test_pred = createStackPred( train_df, test_df, train_labels)



    plt.figure( figsize = (10, 10) )

    

    colormap = plt.cm.gist_ncar

    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, train_pred.shape[1] + 1)])



    print ("\nCreating our averaging systems")



    for i in range(train_pred.shape[1]):

        plt.scatter( train_pred[:,i], train_labels, label = 'Regression Model_' + str(i+1) )



    train_pred, test_pred = optGBM(train_pred, test_pred, train_labels)



    plt.scatter( train_pred, train_labels, label = 'Stacked' )



    plt.legend()



    plt.show()



    return np.expm1( test_pred )
df_sub['y'] = stacker( train, test, train_labels )
df_sub.to_csv('m_bendz.csv', index = False)
df_sub['y']