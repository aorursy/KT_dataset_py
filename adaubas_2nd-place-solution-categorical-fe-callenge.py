__seed = 0

__n_folds = 5

__nrows = None



import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')



from tqdm import tqdm_notebook



import numpy as np

import pandas as pd

pd.set_option('max_colwidth', 500)

pd.set_option('max_columns', 500)

pd.set_option('max_rows', 500)

from scipy.stats import chi2_contingency, kruskal, ks_2samp



from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.preprocessing import OneHotEncoder, StandardScaler



from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import StratifiedKFold, cross_validate



from string import ascii_lowercase

import random



# To avoid target leakage

folds1 = StratifiedKFold(n_splits=__n_folds, shuffle=True, random_state=__seed)

folds2 = StratifiedKFold(n_splits=__n_folds, shuffle=True, random_state=__seed+2)

folds3 = StratifiedKFold(n_splits=__n_folds, shuffle=True, random_state=__seed+4)
train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col = "id", nrows = __nrows)

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col = "id", nrows = __nrows)

train.head()
def coef_vcramer(contingency_df):

    '''

    A partir de la table de contingence de 2 variables, calcule le V de Cramer

    

    Paramètres :

    ------------

        - contingency_df : dataframe pandas

            Table de contingence

            Peut-ête fabriqué à partir d'un pd.crosstab(v1, v2)

    Résultats:

    ----------

        - v de Cramer

    

    Exemple : (G.Saporta, Probabilités, annalyse des données et Statistiques, 3ème édition p.150

    --------------------------------------------------------------------------------------------

    Tableau de contingence, lieu de vacances des français selon leur profession (Saporta p.147) :

    

    data={"Agri":[41, 47, 13, 59, 17, 26, 4, 9, 19],

      "Arti":[220, 260, 71, 299, 120, 42, 64, 35, 29],

      "Cadres":[685, 775, 450, 1242, 706, 139, 122, 100, 130],

      "profinter":[485, 639, 292, 1250, 398, 189, 273, 68, 193],

      "Employés":[190, 352, 67, 813, 163, 92, 161, 49, 72],

      "Ouvriers":[224, 591, 147, 1204, 181, 227, 306, 74, 114],

      "Retraités":[754, 393, 692, 1158, 223, 25, 195, 47, 115],

      "autres":[31, 34, 2, 225, 42, 33, 5, 6, 14]}

      

    a=pd.DataFrame(data, index=["Hotel", "Loc", "rs", "rpp", "rspa", "tente", "carav", "aj", "village"])

    print(vcramer(a), " = 0.12 ? Oui, alors conforme Saporta p.150")

    '''

    chi2 = chi2_contingency(contingency_df)[0]

    n = contingency_df.sum().sum()

    r, k = contingency_df.shape

    return np.sqrt(chi2 / (n * min((r-1), (k-1))))
def fit_describe_infos(train, test, __featToExcl = [], target_for_vcramer = None):

    '''Describe data and difference between train and test datasets.'''

    

    stats = []

    __featToAnalyze = [v for v in list(train.columns) if v not in __featToExcl]

    

    for col in tqdm_notebook(__featToAnalyze):

            

        dtrain = dict(train[col].value_counts())

        dtest = dict(test[col].value_counts())



        set_train_not_in_test = set(dtest.keys()) - set(dtrain.keys())

        set_test_not_in_train = set(dtrain.keys()) - set(dtest.keys())

        

        dict_train_not_in_test = {key:value for key, value in dtest.items() if key in set_train_not_in_test}

        dict_test_not_in_train = {key:value for key, value in dtrain.items() if key in set_test_not_in_train}

            

        nb_moda_test, nb_var_test = len(dtest), pd.Series(dtest).sum()

        nb_moda_abs, nb_var_abs = len(dict_train_not_in_test), pd.Series(dict_train_not_in_test).sum()

        nb_moda_train, nb_var_train = len(dtrain), pd.Series(dtrain).sum()

        nb_moda_abs_2, nb_var_abs_2 = len(dict_test_not_in_train), pd.Series(dict_test_not_in_train).sum()

        

        if not target_for_vcramer is None:

            vc = coef_vcramer(pd.crosstab(train[target_for_vcramer], train[col].fillna(-1)))       

        else:

            vc = 0

            

        stats.append((col, round(vc, 3), train[col].nunique()

            , str(nb_moda_abs) + '   (' + str(round(100 * nb_moda_abs / nb_moda_test, 1))+'%)'

            , str(nb_moda_abs_2) +'   (' + str(round(100 * nb_moda_abs_2 / nb_moda_train, 1))+'%)'

            , str(train[col].isnull().sum()) +'   (' + str(round(100 * train[col].isnull().sum() / train.shape[0], 1))+'%)'

            , str(test[col].isnull().sum()) +'   (' + str(round(100 * test[col].isnull().sum() / test.shape[0], 1))+'%)'

            , str(round(100 * train[col].value_counts(normalize = True, dropna = False).values[0], 1))

            , train[col].dtype))

            

    df_stats = pd.DataFrame(stats, columns=['Feature', "Target Cramer's V"

        , 'Unique values (train)', "Unique values in test not in train (and %)"

        , "Unique values in train not in test (and %)"

        , 'NaN in train (and %)', 'NaN in test (and %)', '% in the biggest cat. (train)'

        , 'dtype'])

    

    if target_for_vcramer is None:

        df_stats.drop("Target Cramer's V", axis=1, inplace=True)

            

    return df_stats, dict_train_not_in_test, dict_test_not_in_train
dfi, _, _ = fit_describe_infos(train, test, __featToExcl=['target'], target_for_vcramer='target')

dfi
def color_and_top(nb_mod, feature, typ, top_n=None):

    

    if top_n is None:

        resu = ["g", nb_mod]

    elif nb_mod > 2*top_n:

        resu = ["r", top_n]

    elif nb_mod > top_n:

        resu =["orange", top_n]

    else: 

        resu = ["g", nb_mod]

    

    title = feature[:20]+" ("+typ[:3]+"-{})".format(nb_mod)

    resu.append(title)

    

    return resu





def plot_multiple_categorical(df, features, col_target=None, top_n=None

                              , nb_subplots_per_row = 4, hspace = 1.3, wspace = 0.5

                              , figheight=15, m_figwidth=4.2, landmark = .01):

    

#    sns.set_style('whitegrid')

    

    if not (col_target is None):

        ref = df[col_target].mean() # Reference

    

    plt.figure()

    if len(features) % nb_subplots_per_row >0:

        nb_rows = int(np.floor(len(features) / nb_subplots_per_row)+1)

    else:

        nb_rows = int(np.floor(len(features) / nb_subplots_per_row))

    fig, ax = plt.subplots(nb_rows, nb_subplots_per_row, figsize=(figheight, m_figwidth * nb_rows))

    plt.subplots_adjust(hspace = hspace, wspace = wspace)



    i = 0; n_row=0; n_col=0

    for feature in features:

        

        i += 1

        plt.subplot(nb_rows, nb_subplots_per_row, i)



        dff = df[[feature, col_target]].copy() # I don't want transform data, only study them

        

        # Missing values

        if dff[feature].dtype.name in ["float16", "float32", "float64"]:

            dff[feature].fillna(-997, inplace=True)

            

        if dff[feature].dtype.name in ["object"]:

            dff[feature].fillna("_NaN", inplace=True)

            

        if dff[feature].dtype.name == "category" and dff[feature].isnull().sum() > 0:

            dff[feature] = dff[feature].astype(str).replace('', '_NaN', regex=False).astype("category")

            

        # Colors, title

        bar_colr, top_nf, title = color_and_top(dff[feature].nunique(), feature, str(dff[feature].dtype), top_n)

        

        # stats

        tdf = dff.groupby([feature]).agg({col_target: ['count', 'mean']})

        tdf = tdf.sort_values((col_target, 'count'), ascending=False).head(top_nf).sort_index()

        

        tdf.index = tdf.index.map(str)

        tdf = tdf.rename(index={'-997.0':'NaN'}) # Missing values

        if not (top_n is None):

            tdf.index = tdf.index.map(lambda x: x[:top_n]) # tronque les libellés des modalités en abcisse

        

        tdf["ref"] = ref

        tdf["ref-"] = ref-landmark

        tdf["ref+"] = ref+landmark

        

        # First Y axis, on the left

        plt.bar(tdf.index, tdf[col_target]['count'].values, color=bar_colr) # Count of each category

        

        plt.title(title, fontsize=11)

        plt.xticks(rotation=90)

        

        # Second Y axis, on the right

        xx = plt.xlim()

        if nb_subplots_per_row == 1:

            ax2 = fig.add_subplot(nb_rows, nb_subplots_per_row, i, sharex = ax[n_row], frameon = False)

        else:

            ax2 = fig.add_subplot(nb_rows, nb_subplots_per_row, i, sharex = ax[n_row, n_col], frameon = False)

        if not (col_target is None):

            ax2.plot(tdf[col_target]['mean'].values, marker = 'x', color = 'b', linestyle = "solid") # Mean of each Category

            ax2.plot(tdf["ref"].values, marker = '_', color = 'black', linestyle = "solid", linewidth=4.0) # Reference

            ax2.plot(tdf["ref-"].values, marker = '_', color = 'black', linestyle = "solid", linewidth=1.0) # Reference

            ax2.plot(tdf["ref+"].values, marker = '_', color = 'black', linestyle = "solid", linewidth=1.0) # Reference

        ax2.yaxis.tick_right()

        ax2.axes.get_xaxis().set_visible(False)

        plt.xlim(xx)



        n_col += 1

        if n_col == nb_subplots_per_row:

            n_col = 0

            n_row += 1

            

    plt.show();
plot_multiple_categorical(train, [v for v in list(train.columns) if v not in ["target"]], "target", top_n=17) 
def cross_val_and_print(pipe, X=train, y=train["target"], cv=folds1, scoring="roc_auc"

                        , best_score = 0, comment1="", comment2=""):

    ''' 

    Cross validate score and print result and print previous result

    And show the score and the previous best score.

    '''

    scores = cross_validate(pipe, X, y, cv = cv, scoring = scoring, return_train_score = True)

    cv_score = scores["test_score"].mean()

    

    if cv == folds1:

        precision = 1

    elif cv == folds2: 

        precision = 2

    else: 

        precision = 3

    

    if comment1 == "":

        print("CV{} score on valid : {:.7f}  - Previous best valid score : {:.7f} - Train mean score : {:6f}".\

          format(precision, cv_score, best_score, scores["train_score"].mean()))

    elif comment2 == "":

        print("CV{} score on valid for {} : {:.7f}  - Previous best valid score : {:.7f} - Train mean score : {:6f}".\

          format(precision, comment1, cv_score, best_score, scores["train_score"].mean()))

    else:

        print("CV{} score on valid for {}={} : {:.7f}  - Previous best valid score : {:.7f} - Train mean score : {:6f}".\

          format(precision, comment1, comment2, cv_score, best_score, scores["train_score"].mean()))

    

    if cv_score > best_score:

        best_score = cv_score



    return cv_score, best_score
# Logistic Regression parameters

lr_params = {'penalty': 'l2', 'solver': 'lbfgs', 'C': 0.123456789, 'max_iter':500}



ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"),

        [f for f in train.columns if not f in ["target", "bin_0"]])

    

# Pipeline ; I use make column transformer, because I will use several encoder

# and beacause it's a way to drop bin_0

pipe = make_pipeline(make_column_transformer(ohe1), LogisticRegression(**lr_params))
_, best_score1 = cross_val_and_print(pipe)

_, best_score2 = cross_val_and_print(pipe, cv=folds2)

_, best_score3 = cross_val_and_print(pipe, cv=folds3)
ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"),

        [f for f in train.columns if not f in ["target"]])

pipe = make_pipeline(make_column_transformer(ohe1), LogisticRegression(**lr_params))
_, best_score1 = cross_val_and_print(pipe, best_score=best_score1)

_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2)
for df in [train, test]:

    df["bin_0_bin_3"] = df["bin_3"].astype(str) + df["bin_0"].astype(str)
ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"),

        [f for f in train.columns if not f in ["target", "bin_0", "bin_3"]])

pipe = make_pipeline(make_column_transformer(ohe1), LogisticRegression(**lr_params))
_, best_score1 = cross_val_and_print(pipe, best_score=best_score1)

_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2)

_, best_score3 = cross_val_and_print(pipe, cv=folds3, best_score=best_score3)
plot_multiple_categorical(train, ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

                          , "target", nb_subplots_per_row=2) 
def transf_ordinal_features(serie):

    

    dtransf = {"ord_1":{'Novice':0, 'Contributor':1, 'Expert':2, 'Master':3, 'Grandmaster':4}

        , "ord_2":{'Freezing':0, 'Cold':1, 'Warm':2, 'Hot':3, 'Boiling Hot':4, 'Lava Hot':5}

        , "nom_0":{"Blue":1, "Green":2, "Red":3}

        , "nom_1":{"Circle":1, "Trapezoid":2, "Star":3, "Polygon":4, "Square":5, "Triangle":6}

        , "nom_2":{"Dog":1, "Lion":2, "Snake":3, "Axolotl":4, "Cat":5, "Hamster":6}

        , "nom_3":{"Finland":1, "Russia":2, "China":3, "Costa Rica":4, "Canada":5, "India":6}

        , "nom_4":{"Bassoon":1, "Piano":2, "Oboe":3, "Theremin":4}

        , "bin_0_bin_3_bis":{"T0":0, "F1":2, "F0":1}

              }



    if serie.name == "ord_0":

        new_serie = serie - 1

    elif serie.name == "ord_5":

        lm = serie.unique()

        new_serie = serie.map({l:i for i, l in enumerate(list(np.sort(lm)))})

    elif serie.name in ["ord_3", "ord_4"]:

        new_serie = serie.str.lower().map({l:i for i, l in enumerate(list(ascii_lowercase))})

    else:

        new_serie = serie.map(dtransf[serie.name])

        

    return new_serie
df = train[["target"]].copy()

for f in ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5", "nom_0", "nom_1", "nom_2", "nom_3"

          , "nom_4"]:

    df[f] = transf_ordinal_features(train[f])

    

plot_multiple_categorical(df, ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"

                              , "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"]

                          , "target", nb_subplots_per_row=2) 



del df
class MyFeaturesEngineering(BaseEstimator, TransformerMixin):

    

    def __init__(self, list_ordinal_features =[]):

        

        self.list_ordinal_features = list_ordinal_features

        



    def fit(self, x, y=None):

        

        return self

    



    def transform(self, x, y=None):

        

        df = x.copy()

        

        # Ordinal features

        for v in self.list_ordinal_features:

            df[v] = transf_ordinal_features(df[v])

        

        return df
ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day", "month"]

ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]



for feat in ordinal_features:

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"),

        list(set(ohe1_feats+ordinal_features)-{feat}))

    

    # To transform ordinal features to integer sorted by there target mean

    MyFeE = MyFeaturesEngineering(list_ordinal_features = [feat])

    

    # Ordinal features : relation with target is linear, we need to normalize before regression

    StdScalE = (StandardScaler(copy=False), [feat])

    

    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, StdScalE), LogisticRegression(**lr_params))

    _, _ = cross_val_and_print(pipe, best_score=best_score1, comment1=feat)

#    _, _ = cross_val_and_print(pipe, cv=folds2, best_score=best_score2, comment1=feat)
# How to be sure that ordinal encoding is better for ord_0 ?

# Let's see on the 2 others combinations of 5 folds.



ordinal_features = ["ord_0"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4", "nom_5"

        , "nom_6", "nom_7", "nom_8", "nom_9", "day", "month"] + ["ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]



ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"),

        ohe1_feats)

    

# To transform ordinal features to integer sorted by there target mean

MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features)

    

# Ordinal features : relation with target is linear, we need to normalize before regression

StdScalE = (StandardScaler(copy=False), ordinal_features)

    

pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, StdScalE), LogisticRegression(**lr_params))

_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2)

_, best_score3 = cross_val_and_print(pipe, cv=folds3, best_score=best_score3)
ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day", "month"]



ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"),

        ohe1_feats)

    

# To transform ordinal features to integer sorted by there target mean

MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features)

    

# Ordinal features : relation with target is linear, we need to normalize before regression

StdScalE = (StandardScaler(copy=False), ordinal_features)

    

pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, StdScalE), LogisticRegression(**lr_params))
_, best_score1 = cross_val_and_print(pipe, best_score=best_score1)

_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2)

_, best_score3 = cross_val_and_print(pipe, cv=folds3, best_score=best_score3)
plot_multiple_categorical(train, ["day", "bin_0_bin_3", "month"]

                          , "target", nb_subplots_per_row=2) 
for df in [train, test]:

    df["day_bis"] = df["day"].map({1:3, 2:2, 3:1, 4:0, 5:1, 6:2, 7:3})

    df["bin_0_bin_3_bis"] = df["bin_0_bin_3"].map({"F0":1, "F1":2, "T0":0, "T1":0})

    

plot_multiple_categorical(train, ["day", "day_bis", "bin_0_bin_3", "bin_0_bin_3_bis"]

                          , "target", nb_subplots_per_row=2) 
ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day_bis", "month"]

ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"),

        ohe1_feats)

    

# To transform ordinal features to integer sorted by there target mean

MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features)

    

# Ordinal features : relation with target is linear, we need to normalize before regression

StdScalE = (StandardScaler(copy=False), ordinal_features)

    

pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, StdScalE), LogisticRegression(**lr_params))



_, best_score1 = cross_val_and_print(pipe, best_score=best_score1)

_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2)

_, best_score3 = cross_val_and_print(pipe, cv=folds3, best_score=best_score3)
class MyBinsEncoder(BaseEstimator, TransformerMixin):

    

    def __init__(self, nbins=100, nmin=20):

        

        self.nbins = nbins

        self.nmin = nmin

        

    def fit(self, x, y=None):

        

        temp = pd.concat([x, y], axis=1)

        

        # Compute target mean for each value

        averages = temp.groupby(by=x.name)[y.name].mean()

        means = dict(zip(averages.index.values, averages.values))

        

        # binning in self.nbins bins

        bins = np.linspace(averages.min(), averages.max(), self.nbins)

        self.map_ = dict(zip(averages.index.values, np.digitize(averages.values, bins=bins)))



        # But if there are more than self.nmin observations in a original value, keep the original value

        # instead of bins.

        count = temp.groupby(by=x.name)[y.name].count()

        nobs = dict(zip(averages.index.values, count))

        

        for key, value in nobs.items():

            if value > self.nmin:

                self.map_[key] = key

        

        return self

    

    def transform(self, x, y=None):

        

        temp = x.map(self.map_)

        # Especially for nom_7, nom_8 and nom_9

        temp.fillna(random.choice(list(self.map_.values())), inplace=True)

        temp = temp.astype(str)

        

        return temp

    

#essai = MyBinsEncoder(nbins=20, nmin=3)

#essai.fit(train["nom_8"], train["target"])

#print(essai.map_)

#av = essai.transform(train["nom_8"])



essai = MyBinsEncoder(nbins=4, nmin=100000)

essai.fit(train["day"], train["target"])

print(essai.map_)

av = essai.transform(train["day"])

print(av.value_counts(dropna=False))

print(av.nunique())
class MyFeaturesEngineering(BaseEstimator, TransformerMixin):

    

    def __init__(self, list_ordinal_features =[], feat_to_bins_encode = {}):

        

        self.list_ordinal_features = list_ordinal_features

        

        self.feat_to_bins_encode = feat_to_bins_encode

        self.BinsEncoder={}

        



    def fit(self, x, y=None):

        

        # bins encoders

        for feat, value in self.feat_to_bins_encode.items():

            self.BinsEncoder[feat] = MyBinsEncoder(nbins=value[0], nmin=value[1])

            self.BinsEncoder[feat].fit(x[feat], y)



        return self

    



    def transform(self, x, y=None):

        

        df = x.copy()

        

        for v in self.feat_to_bins_encode.keys():

            df[v] = self.BinsEncoder[v].transform(df[v])

            

        # Ordinal features

        for v in self.list_ordinal_features:

            df[v] = transf_ordinal_features(df[v])

        

        return df
ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day_bis", "month"]



feat = "nom_5"

for nbins in [[20, 35]]: # combination selected by CV

    

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

            , ohe1_feats)



    MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                 , feat_to_bins_encode={feat:nbins})



    StdScalE = (StandardScaler(copy=False), ordinal_features)

    

    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, StdScalE), LogisticRegression(**lr_params))

    _, best_score1 = cross_val_and_print(pipe, best_score=best_score1, comment1=feat, comment2=nbins)
_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2, comment1=feat, comment2=nbins)

_, best_score3 = cross_val_and_print(pipe, cv=folds3, best_score=best_score3, comment1=feat, comment2=nbins)
ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day_bis", "month"]



feat = "nom_6"

for nbins in [[20, 5]]: # combination selected by CV

    

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

            , ohe1_feats)



    MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                 , feat_to_bins_encode={feat:nbins, "nom_5":[20, 35]})



    StdScalE = (StandardScaler(copy=False), ordinal_features)

    

    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, StdScalE), LogisticRegression(**lr_params))

    _, best_score1 = cross_val_and_print(pipe, best_score=best_score1, comment1=feat, comment2=nbins)
_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2, comment1=feat, comment2=nbins)

_, best_score3 = cross_val_and_print(pipe, cv=folds3, best_score=best_score3, comment1=feat, comment2=nbins)
ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day_bis", "month"]



feat = "nom_8"

for nbins in [[15, 3]]: # choose by CV

    

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

            , ohe1_feats)



    MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                 , feat_to_bins_encode={feat:nbins, "nom_5":[20, 35]})



    StdScalE = (StandardScaler(copy=False), ordinal_features)

    

    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, StdScalE), LogisticRegression(**lr_params))

    _, best_score1 = cross_val_and_print(pipe, best_score=best_score1, comment1=feat, comment2=nbins)
_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2, comment1=feat, comment2=nbins)

_, best_score3 = cross_val_and_print(pipe, cv=folds3, best_score=best_score3, comment1=feat, comment2=nbins)
ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day_bis", "month"]



for feat in ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_8", "day_bis", "month"]:

    

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

            , list(set(ohe1_feats)-{feat}))



    ohe2 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', drop="first")

            , [feat])



    MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                 , feat_to_bins_encode={"nom_5":[20, 35], "nom_8":[15, 3]})



    StdScalE = (StandardScaler(copy=False), ordinal_features)

    

    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, ohe2, StdScalE)

                         , LogisticRegression(**lr_params))

    _, _ = cross_val_and_print(pipe, best_score=best_score1, comment1=feat)
ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day_bis", "month"]



for feat in ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_8", "day_bis", "month"]:

    

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

            , list(set(ohe1_feats)-{feat}))



    lval_to_drop = [train[feat].value_counts().index[0]] # the most frequent

    ohe2 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', drop=lval_to_drop)

            , [feat])



    MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                 , feat_to_bins_encode={"nom_5":[20, 35], "nom_8":[15, 3]})



    StdScalE = (StandardScaler(copy=False), ordinal_features)

    

    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, ohe2, StdScalE)

                         , LogisticRegression(**lr_params))

    _, _ = cross_val_and_print(pipe, best_score=best_score1, comment1=feat)
ohe1_feats = ["bin_0_bin_3", "bin_1", "nom_5", "nom_7", "nom_9", "month"]

ohe2_feats = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_6', "day_bis"]

ohe3_feats = ['bin_2', 'bin_4', 'nom_8']

ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]



MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                , feat_to_bins_encode={"nom_5":[20, 35], "nom_8":[15, 3]})



StdScalE = (StandardScaler(copy=False), ordinal_features)

    

ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

        , ohe1_feats)



lval_to_drop = [train[v].value_counts().index[0] for v in ohe2_feats] 

ohe2 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', drop=lval_to_drop)

            , ohe2_feats)



ohe3 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', drop='first'), ohe3_feats)





pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, ohe2, ohe3, StdScalE)

                        , LogisticRegression(**lr_params))

_, best_score1 = cross_val_and_print(pipe, best_score=best_score1)

_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2)

_, best_score3 = cross_val_and_print(pipe, cv=folds3, best_score=best_score3)
ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day_bis", "month"]



for feat in ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_6', "day_bis"]:

    

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

            , list(set(ohe1_feats)-{feat}))



    lval_to_drop = [train[feat].value_counts().index[0]] 

    ohe2 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', drop=lval_to_drop)

            , [feat])



    MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                 , feat_to_bins_encode={"nom_5":[20, 35], "nom_8":[15, 3]})



    StdScalE = (StandardScaler(copy=False), ordinal_features)

    

    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, ohe2, StdScalE)

                         , LogisticRegression(**lr_params))

    _, _ = cross_val_and_print(pipe, cv=folds2, best_score=best_score2, comment1=feat)    

#    _, _ = cross_val_and_print(pipe, cv=folds3, best_score=best_score3, comment1=feat)
ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day_bis", "month"]



for feat in ['nom_1', 'nom_3', "day_bis"]:

    

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

            , list(set(ohe1_feats)-{feat}))



    lval_to_drop = [train[feat].value_counts().index[0]] 

    ohe2 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', drop=lval_to_drop)

            , [feat])



    MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                 , feat_to_bins_encode={"nom_5":[20, 35], "nom_8":[15, 3]})



    StdScalE = (StandardScaler(copy=False), ordinal_features)

    

    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, ohe2, StdScalE)

                         , LogisticRegression(**lr_params))

#    _, _ = cross_val_and_print(pipe, cv=folds2, best_score=best_score2, comment1=feat)    

    _, _ = cross_val_and_print(pipe, cv=folds3, best_score=best_score3, comment1=feat)
ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]

ohe1_feats = ["bin_0_bin_3", "bin_1", "bin_2", "bin_4", "nom_0", "nom_1", "nom_2", "nom_3", "nom_4"

             , "nom_5", "nom_6", "nom_7", "nom_8", "nom_9", "day_bis", "month"]



for feat in ['bin_2', 'bin_4', 'nom_8']:

    

    ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

            , list(set(ohe1_feats)-{feat}))



    lval_to_drop = [train[feat].value_counts().index[0]] 

    ohe2 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', drop='first')

            , [feat])



    MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                 , feat_to_bins_encode={"nom_5":[20, 35], "nom_8":[15, 3]})



    StdScalE = (StandardScaler(copy=False), ordinal_features)

    

    pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, ohe2, StdScalE)

                         , LogisticRegression(**lr_params))

    

    _, _ = cross_val_and_print(pipe, cv=folds2, best_score=best_score2, comment1=feat)    

#    _, _ = cross_val_and_print(pipe, cv=folds3, best_score=best_score3, comment1=feat)

ohe1_feats = ["bin_0_bin_3", "bin_1", 'bin_2', 'bin_4', 'nom_0', 'nom_2', 'nom_4', "nom_5", 'nom_6'

              , "nom_7", "nom_9", "month"]

ohe2_feats = ['nom_1', 'nom_3', "day_bis"]

ohe3_feats = ['nom_8']

ordinal_features = ["ord_0", "ord_1", "ord_2", "ord_3", "ord_4", "ord_5"]



MyFeE = MyFeaturesEngineering(list_ordinal_features = ordinal_features

                                , feat_to_bins_encode={"nom_5":[20, 35], "nom_8":[15, 3]})



StdScalE = (StandardScaler(copy=False), ordinal_features)

    

ohe1 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore")

        , ohe1_feats)



lval_to_drop = [train[v].value_counts().index[0] for v in ohe2_feats] 

ohe2 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', drop=lval_to_drop)

            , ohe2_feats)



ohe3 = (OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', drop='first'), ohe3_feats)





pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, ohe2, ohe3, StdScalE)

                        , LogisticRegression(**lr_params))

_, best_score1 = cross_val_and_print(pipe, best_score=best_score1)

_, best_score2 = cross_val_and_print(pipe, cv=folds2, best_score=best_score2)

_, best_score3 = cross_val_and_print(pipe, cv=folds3, best_score=best_score3)
lr_params = {'penalty': 'l2', 'solver': 'lbfgs', 'C': 0.123456789, 'max_iter':500}



pipe = make_pipeline(MyFeE, make_column_transformer(ohe1, ohe2, ohe3, StdScalE)

                     , LogisticRegression(**lr_params))



#score, _ = cross_val_and_print(pipe, best_score=best_score1)



# Submission

pred = pipe.fit(train.drop(["target"], axis = 1), train["target"]).predict_proba(test)[:, 1]

pd.DataFrame({"id": test.index, "target": pred}).to_csv("submission.csv", index=False)