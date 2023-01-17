import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))
X_train = pd.read_csv('../input/career-con-2019/X_train.csv')

y_train = pd.read_csv('../input/career-con-2019/y_train.csv')



X_test = pd.read_csv('../input/career-con-2019/X_test.csv')

y_sample = pd.read_csv('../input/career-con-2019/sample_submission.csv')



X_train.set_index('series_id', inplace=True)

y_train.set_index('series_id', inplace=True)



X_test.set_index('series_id', inplace=True)



new_col_names = ['oX', 'oY', 'oZ', 'oW', 'avX', 'avY', 'avZ', 'laX', 'laY', 'laZ']

columns_to_drop = ['row_id', 'measurement_number']



X_train = X_train.drop(columns_to_drop, axis=1)

y_train = y_train.drop(['group_id'], axis=1)



X_test = X_test.drop(columns_to_drop, axis=1)

#y_sample = y_sample.drop(['group_id'], axis=1)



X_train.columns = new_col_names

X_test.columns = new_col_names



X_train = X_train.groupby('series_id').rolling(10).mean()

X_test = X_test.groupby('series_id').rolling(10).mean()



def add_derivatives(df):

    for col in df.columns:

        series = df[col].values

        diff = np.diff(np.array(series))

        df[col+'_der'] = np.append(0, diff)

    return df



X_train = add_derivatives(X_train).dropna()

X_train.index = X_train.index.droplevel()



X_test = add_derivatives(X_test).dropna()

X_test.index = X_test.index.droplevel()
pd.DataFrame(y_train['surface'].value_counts()/y_train.shape[0]*100)
X_train.head()
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats



class ZScoreScaler(BaseException, TransformerMixin):

    

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        arr = X.groupby('series_id').apply(stats.zscore).values

        arr = np.concatenate(arr, axis=0)

        return pd.DataFrame(arr, columns=X.columns, index=X.index)
zscore = ZScoreScaler().fit_transform(X_train.loc[:100])
zscore.loc[20, 'oX':'avZ'].reset_index().drop(columns=['series_id']).plot()
zscore.loc[20, 'oX_der':'avZ_der'].reset_index().drop(columns=['series_id']).plot()
def rolling_window_2d(arr, size=2):

    shape = arr.shape

    strides = arr.strides

    arr = np.lib.stride_tricks.as_strided(arr,

                                         shape=(shape[1], shape[0]+1-size, size),

                                         strides=(strides[1],strides[0],strides[0]))

    return arr



class RollingWindowFourier(BaseEstimator, TransformerMixin):



    def __init__(self, window_length):

        self.window_length = window_length

        

    def fit(self, X, y=None):

        return self

        

    def transform(self, X):

        index = []

        arr_list = []

        for i in X.index.unique():

            new_one = rolling_window_2d(X.loc[i].values, self.window_length)

            f_transformed = np.fft.fft(new_one, 12)

            concated = np.concatenate((f_transformed.real, f_transformed.imag), axis=2)

            two_dims = concated.transpose(1,0,2).reshape(concated.shape[1],-1)

            arr_list.append(two_dims)

            index += [i for ind in range(new_one.shape[1])]

        df = pd.DataFrame(np.concatenate(arr_list, axis=0), index=index)

        del arr_list

        return df
fourier = RollingWindowFourier(10).fit_transform(zscore)
fourier.head()
fourier.shape
from sklearn.feature_selection import SelectKBest, f_classif



class ANOVA_ColumnSelector(BaseEstimator, TransformerMixin):



    def __init__(self, k_best):

        self.k = k_best

        

    def fit(self, X, y):

        y = X.merge(y, left_on=X.index, right_on=y.index)['surface']

        col_list = []

        for sensor in range(20):

            col_indexes = np.linspace(sensor*24, sensor*24+23, 24).astype('int16')

            skb = SelectKBest(f_classif, k=self.k).fit(X[col_indexes], y)

            col_list += list(np.argsort(skb.pvalues_)[:self.k] + sensor*24)

        self.columns = col_list

        return self

    

    def transform(self, X):

        df = X[self.columns]

        del X

        return df
import warnings

warnings.filterwarnings("ignore")
anova = ANOVA_ColumnSelector(4).fit_transform(fourier, y_train)
anova.merge(y_train, left_on=anova.index, right_on=y_train.index).boxplot(2, by='surface')
anova.merge(y_train, left_on=anova.index, right_on=y_train.index).boxplot(178, by='surface')
anova.merge(y_train, left_on=anova.index, right_on=y_train.index).boxplot(347, by='surface')
from sklearn import tree



class Binner(BaseEstimator, TransformerMixin):

    

    def __init__(self):

        pass

    

    def fit(self, X, y):

        y = X.merge(y, left_on=X.index, right_on=y.index)['surface']

        column_limits = []

        for col in X.columns:

            clf = tree.DecisionTreeClassifier(max_depth=2, max_leaf_nodes=4).fit(X[col].values.reshape(-1,1), y)

            threshold = clf.tree_.threshold[:3]

            limits = np.sort(np.insert(threshold,0,[np.NINF, np.inf]))

            column_limits.append(limits)

        self.column_limits = column_limits

        return self

    

    def transform(self, X):

        for idx, col in enumerate(X.columns):

            X.loc[:,col] = pd.cut(X[col], bins=self.column_limits[idx], labels=[1,2,3,4])

        return X.astype('int16')
binned = Binner().fit_transform(anova, y_train)
binned.head()
class CreateWords(BaseEstimator, TransformerMixin):

    

    def __init__(self, window_length):

        self.window_length = window_length

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        cols = X.columns

        ind = X.index

        col_names = ['oX', 'oY', 'oZ', 'oW', 'avX', 'avY', 'avZ', 'laX', 'laY', 'laZ',

                     'oX_der', 'oY_der', 'oZ_der', 'oW_der', 'avX_der', 'avY_der', 'avZ_der',

                     'laX_der', 'laY_der', 'laZ_der']

        df_dict = {}

        for i in range(20):

            x = X[cols[i*4:i*4+4]].apply(lambda x: int(''.join(map(str,x))), axis=1)

            df_dict[str(self.window_length)+col_names[i]] = x

        df = pd.DataFrame(df_dict, index=ind)

        del x, df_dict, X

        return df
words = CreateWords(10).fit_transform(binned)
words.head()
class GetBigrams(BaseEstimator, TransformerMixin):

    

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        arr_list = []

        index = []

        for i in X.index.unique():

            unigrams2d = rolling_window_2d(X.loc[i].values, 2)

            bigrams = np.apply_along_axis(lambda x: int(''.join(map(str,x))), 2, unigrams2d).T

            stacked = np.vstack((X.loc[i], bigrams))

            arr_list.append(stacked)

            index += [i for ind in range(stacked.shape[0])]

        df = pd.DataFrame(np.concatenate(arr_list, axis=0), columns=X.columns, index=index)

        del X, arr_list

        return df
bigrams = GetBigrams().fit_transform(words)
bigrams.tail(5)
class TextTransformer(BaseEstimator, TransformerMixin):

    

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        for col in X.columns:

            X[col] = col+X[col].astype(str)

        df = X.groupby(X.index).apply(lambda x: ' '.join(x.values.flatten()))

        del X

        return df
text = TextTransformer().fit_transform(bigrams)
from sklearn.feature_extraction.text import CountVectorizer



class Dummify(BaseEstimator, TransformerMixin):

    

    def __init__(self):

        pass

    

    def fit(self, X, y=None):

        self.CV = CountVectorizer().fit(X)

        self.columns = self.CV.get_feature_names()

        return self

    

    def transform(self, X):

        counts = self.CV.transform(X)

        index = X.index

        del X

        return pd.DataFrame(counts.toarray(), index=index, columns=self.columns)
dummies = Dummify().fit_transform(text)
dummies.head()
from sklearn.pipeline import make_pipeline, make_union

from sklearn.feature_selection import SelectKBest, chi2



union = make_union(

    make_pipeline(

        ZScoreScaler(),

        RollingWindowFourier(5),

        ANOVA_ColumnSelector(4),

        Binner(),

        CreateWords(5),

        GetBigrams(),

        TextTransformer(),

        Dummify(),

        SelectKBest(chi2, k=350)

    ),

    make_pipeline(

        ZScoreScaler(),

        RollingWindowFourier(10),

        ANOVA_ColumnSelector(4),

        Binner(),

        CreateWords(10),

        GetBigrams(),

        TextTransformer(),

        Dummify(),

        SelectKBest(chi2, k=350)

    ),

    make_pipeline(

        ZScoreScaler(),

        RollingWindowFourier(14),

        ANOVA_ColumnSelector(4),

        Binner(),

        CreateWords(14),

        GetBigrams(),

        TextTransformer(),

        Dummify(),

        SelectKBest(chi2, k=350)

    ),

    make_pipeline(

        ZScoreScaler(),

        RollingWindowFourier(23),

        ANOVA_ColumnSelector(4),

        Binner(),

        CreateWords(23),

        GetBigrams(),

        TextTransformer(),

        Dummify(),

        SelectKBest(chi2, k=350)

    ),

    make_pipeline(

        ZScoreScaler(),

        RollingWindowFourier(32),

        ANOVA_ColumnSelector(4),

        Binner(),

        CreateWords(32),

        GetBigrams(),

        TextTransformer(),

        Dummify(),

        SelectKBest(chi2, k=350)

    ),

    make_pipeline(

        ZScoreScaler(),

        RollingWindowFourier(41),

        ANOVA_ColumnSelector(4),

        Binner(),

        CreateWords(41),

        GetBigrams(),

        TextTransformer(),

        Dummify(),

        SelectKBest(chi2, k=350)

    ),

    make_pipeline(

        ZScoreScaler(),

        RollingWindowFourier(50),

        ANOVA_ColumnSelector(4),

        Binner(),

        CreateWords(50),

        GetBigrams(),

        TextTransformer(),

        Dummify(),

        SelectKBest(chi2, k=350)

    ),

    make_pipeline(

        ZScoreScaler(),

        RollingWindowFourier(59),

        ANOVA_ColumnSelector(4),

        Binner(),

        CreateWords(59),

        GetBigrams(),

        TextTransformer(),

        Dummify(),

        SelectKBest(chi2, k=350)

    ),

    make_pipeline(

        ZScoreScaler(),

        RollingWindowFourier(68),

        ANOVA_ColumnSelector(4),

        Binner(),

        CreateWords(68),

        GetBigrams(),

        TextTransformer(),

        Dummify(),

        SelectKBest(chi2, k=350)

    )    

)
"""union.fit(X_train, y_train)

X_train_transformed = union.transform(X_train)

X_test_transformed = union.transform(X_test)"""
"""X_train_transformed = pd.DataFrame(X_train_transformed)

X_test_transformed = pd.DataFrame(X_test_transformed)"""
"""X_train_transformed.to_csv('train_transformed.csv',index=False)

X_test_transformed.to_csv('test_transformed.csv',index=False)"""
"""train = pd.read_csv('../input/weaselmuse-robots/train_transformed.csv')

test = pd.read_csv('../input/weaselmuse-robots/test_transformed.csv')"""
"""from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, make_scorer





scorer = make_scorer(accuracy_score)



GBC_params = dict(n_estimators=(100,250,500),

                  learning_rate=(0.1,1,10),

                  min_samples_split=(2,3,4))



RFC_params = dict(n_estimators=(10,50,100,200),

                 oob_score=(False, True),

                 class_weight=['balanced'])



GBC_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), GBC_params, cv=5, scoring=scorer)

RFC_grid = GridSearchCV(RandomForestClassifier(random_state=42), RFC_params, cv=5, scoring=scorer)"""
#GBC = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, min_samples_split=3).fit(train, y_train.values.ravel())