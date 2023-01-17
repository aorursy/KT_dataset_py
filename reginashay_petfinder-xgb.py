import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import re  # regular expressions
from pandas.api.types import is_string_dtype, is_numeric_dtype





def train_cats(df):

    """

    Change any columns of strings in a pandas' dataframe to a column of

    categorical values. This applies the changes inplace.

    """

    for n,c in df.items():

        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()





def fix_missing(df, col, name, na_dict):

    """

    Fill missing data in a column of df with the median, and add a {name}_na column

    which specifies if the data was missing.

    """

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict





def numericalize(df, col, name, max_n_cat):

    """

    Changes the column col from a categorical type to it's integer codes.

    """

    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):

        df[name] = col.cat.codes+1





def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,

            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

    """

    Takes a data frame df and splits off the response variable, and

    changes the df into an entirely numeric dataframe. For each column of df 

    which is not in skip_flds nor in ignore_flds, na values are replaced by the

    median value of the column.

    """

    if not ignore_flds: ignore_flds=[]

    if not skip_flds: skip_flds=[]

    if subset: df = get_sample(df,subset)

    else: df = df.copy()

    ignored_flds = df.loc[:, ignore_flds]

    df.drop(ignore_flds, axis=1, inplace=True)

    if preproc_fn: preproc_fn(df)

    if y_fld is None: y = None

    else:

        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes

        y = df[y_fld].values

        skip_flds += [y_fld]

    df.drop(skip_flds, axis=1, inplace=True)



    if na_dict is None: na_dict = {}

    else: na_dict = na_dict.copy()

    na_dict_initial = na_dict.copy()

    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)

    if len(na_dict_initial.keys()) > 0:

        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

    if do_scale: mapper = scale_vars(df, mapper)

    for n,c in df.items(): numericalize(df, c, n, max_n_cat)

    df = pd.get_dummies(df, dummy_na=True)

    df = pd.concat([ignored_flds, df], axis=1)

    res = [df, y, na_dict]

    if do_scale: res = res + [mapper]

    return res
train = pd.read_csv('../input/train/train.csv')

#to_drop = ['Name', 'Description', 'RescuerID']   # optional

#to_drop = ['Description', 'RescuerID']           # optional

to_drop = ['PetID', 'Description', 'RescuerID']  # optional: categorical data with too many unique values

to_drop.append('Name')                           # optional (we try to make the best out of XGB)

train.drop(to_drop, axis=1, inplace=True)        # optional

train.head()
train.shape
#train.isna().sum()

#train.Description = train.Description.fillna('None')



# replace nan values with a string:

#train.Name = train.Name.fillna('NoNameSpecified')

##train.Name = train.Name.fillna('Baby')



# clean: remove all characters except for alphanumeric:

# (NB: removed characters are replaced with a zero length string)

##train.Name = train.Name.map(lambda name: re.sub('[^A-Za-z0-9]+', '', name))



# replacing zero length strings that appeared after cleaning (above):

##train.loc[train.Name == '', 'Name'] = 'NoNameLostIt'



#train.isna().sum()
# count unique Names:

##train.Name.nunique()
##train.Name.value_counts()



#train[train.Name.apply(lambda x: len(str(x))) < 2].Name.value_counts()

#train[train.Name.str.startswith('No')].Name.value_counts()
train.shape
train.dtypes
#plt.figure(figsize=(20,16))

train.hist(figsize=(20,16));  # info about features
# feature correlation

#plt.figure(figsize=(20,16))

plt.figure(figsize=(10,8))

corr = train.corr()

sns.heatmap(corr, cmap='YlGnBu',  #annot=True,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
# drop features that are badly correlated with others

#feat_drop = ['Breed2', 'MaturitySize', 'VideoAmt', 'State']



# drop features that are too well correlated with each other (gives better results in classification)

feat_drop = ['Dewormed', 'Vaccinated', 'Sterilized']



train.drop(feat_drop, axis=1, inplace=True)

train.shape
##train_cats(train)

##train.Name.cat.set_categories(list(train.Name.cat.categories), ordered=True, inplace=True)

##train.Name = train.Name.cat.codes



#train.PetID.cat.set_categories(list(train.PetID.cat.categories), ordered=True, inplace=True)

#train.PetID = train.PetID.cat.codes
##train.head()
##train.dtypes  # now the dataset is numeric entirely
test = pd.read_csv('../input/test/test.csv')

test.drop(to_drop, axis=1, inplace=True)  # optional

test.head()
#test.isna().sum()

#test.Description = test.Description.fillna('None')



# replace nan values with a string:

#test.Name = test.Name.fillna('NoNameSpecified')

##test.Name = test.Name.fillna('Baby')



# clean: remove all characters except for alphanumeric:

# (NB: removed characters are replaced with a zero length string)

##test.Name = test.Name.map(lambda name: re.sub('[^A-Za-z0-9]+', '', name))



# replacing zero length strings that appeared after cleaning (above):

##test.loc[test.Name == '', 'Name'] = 'NoNameLostIt'



#test.isna().sum()
##test.Name.value_counts()
#test.isna().sum()

test.shape
test.drop(feat_drop, axis=1, inplace=True)

test.shape
##train_cats(test)

##test.Name.cat.set_categories(list(test.Name.cat.categories), ordered=True, inplace=True)

##test.Name = test.Name.cat.codes



#test.PetID.cat.set_categories(list(test.PetID.cat.categories), ordered=True, inplace=True)

#test.PetID = test.PetID.cat.codes
test.head()
#breeds = pd.read_csv('../input/breed_labels.csv')

#breeds.head()
#breeds.shape
#colors = pd.read_csv('../input/color_labels.csv')

#colors
#states = pd.read_csv('../input/state_labels.csv')

#states
from sklearn.model_selection import GridSearchCV, StratifiedKFold  #, train_test_split

from sklearn.metrics import make_scorer



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
train.AdoptionSpeed.unique()  #hist(figsize=(6,4));
# The following 3 functions have been taken from Ben Hamner's github repository

# https://github.com/benhamner/Metrics



def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):

    """

    Returns the confusion matrix between rater's ratings

    """

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(rater_a + rater_b)

    if max_rating is None:

        max_rating = max(rater_a + rater_b)

    num_ratings = int(max_rating - min_rating + 1)

    conf_mat = [[0 for i in range(num_ratings)]

                for j in range(num_ratings)]

    for a, b in zip(rater_a, rater_b):

        conf_mat[a - min_rating][b - min_rating] += 1

    return conf_mat





def histogram(ratings, min_rating=None, max_rating=None):

    """

    Returns the counts of each type of rating that a rater made

    """

    if min_rating is None:

        min_rating = min(ratings)

    if max_rating is None:

        max_rating = max(ratings)

    num_ratings = int(max_rating - min_rating + 1)

    hist_ratings = [0 for x in range(num_ratings)]

    for r in ratings:

        hist_ratings[r - min_rating] += 1

    return hist_ratings





def quadratic_weighted_kappa(y, y_pred):

    """

    Calculates the quadratic weighted kappa

    axquadratic_weighted_kappa calculates the quadratic weighted kappa

    value, which is a measure of inter-rater agreement between two raters

    that provide discrete numeric ratings.  Potential values range from -1

    (representing complete disagreement) to 1 (representing complete

    agreement).  A kappa value of 0 is expected if all agreement is due to

    chance.

    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b

    each correspond to a list of integer ratings.  These lists must have the

    same length.

    The ratings should be integers, and it is assumed that they contain

    the complete range of possible ratings.

    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating

    is the minimum possible rating, and max_rating is the maximum possible

    rating

    """

    rater_a = y

    rater_b = y_pred

    min_rating=None

    max_rating=None

    rater_a = np.array(rater_a, dtype=int)

    rater_b = np.array(rater_b, dtype=int)

    assert(len(rater_a) == len(rater_b))

    if min_rating is None:

        min_rating = min(min(rater_a), min(rater_b))

    if max_rating is None:

        max_rating = max(max(rater_a), max(rater_b))

    conf_mat = Cmatrix(rater_a, rater_b,

                                min_rating, max_rating)

    num_ratings = len(conf_mat)

    num_scored_items = float(len(rater_a))



    hist_rater_a = histogram(rater_a, min_rating, max_rating)

    hist_rater_b = histogram(rater_b, min_rating, max_rating)



    numerator = 0.0

    denominator = 0.0



    for i in range(num_ratings):

        for j in range(num_ratings):

            expected_count = (hist_rater_a[i] * hist_rater_b[j]

                              / num_scored_items)

            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)

            numerator += d * conf_mat[i][j] / num_scored_items

            denominator += d * expected_count / num_scored_items



    return (1.0 - numerator / denominator)
#kappa_scorer = make_scorer(cohen_kappa_score(weights='quadratic'), greater_is_better=True)  # wrong

kappa_scorer = make_scorer(quadratic_weighted_kappa, greater_is_better=True)
X, y, nan = proc_df(train, 'AdoptionSpeed')



#X = train.drop(['AdoptionSpeed'], axis=1)

#y = train.AdoptionSpeed

#X_train, X_test, y_train, y_test = train_test_split(X, y)
#tree = DecisionTreeClassifier()

#tree_params = {

#    'criterion' : ['gini', 'entropy'],

#    'max_depth' : list(range(2, 11))

#}

#tree_grid = GridSearchCV(tree, tree_params, n_jobs=-1, cv=5, verbose=True, scoring=kappa_scorer)

#tree_grid.fit(X, y);
#tree_grid.best_params_, tree_grid.best_score_
#max_depth_values = range(2, 16)

#max_features_values = range(2, 16)

#n_estimators = [60, 70]  #[40, 50, 60]

#forest_params = {'max_depth': max_depth_values,

#                 'max_features': max_features_values,

#                 'n_estimators': n_estimators}
#skf = StratifiedKFold(n_splits=5, shuffle=True)
#%%time

#forest = RandomForestClassifier()

#rf_grid = GridSearchCV(forest, forest_params, n_jobs=-1, scoring=kappa_scorer, cv=skf)

#rf_grid.fit(X, y);
#rf_grid.best_params_, rf_grid.best_score_
skf = StratifiedKFold(n_splits=5, shuffle=True)
max_depth_values = range(3, 9)

min_child_weight = range(1, 3)

n_estimators = [80]

xgb_params = {

    'max_depth' : max_depth_values,

    'min_child_weight' : min_child_weight,

    'n_estimators' : n_estimators,

}
%%time

xgb = XGBClassifier()  #subsample=0.7)

xgb_grid = GridSearchCV(xgb, xgb_params, n_jobs=-1, scoring=kappa_scorer, cv=skf)  #5)

xgb_grid.fit(X, y);
xgb_grid.best_params_, xgb_grid.best_score_
xgb_grid.best_estimator_.predict(test)
sub = pd.read_csv('../input/test/sample_submission.csv')



#pred = tree_grid.predict(test)

#pred = rf_grid.predict(test)

pred = xgb_grid.predict(test)



sub['AdoptionSpeed'] = pd.Series(pred)

sub.to_csv('submission.csv', index=False)