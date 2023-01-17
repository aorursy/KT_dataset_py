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

train_X = pd.read_table('../input/train.tsv', engine='c')

train_y = train_X["price"]

train_X.drop(['price'], axis=1,inplace=True)

test_X = pd.read_table('../input/test.tsv', engine='c')
print(train_X.head())
from sklearn.base import TransformerMixin, BaseEstimator, clone

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

from sklearn.preprocessing import FunctionTransformer, LabelEncoder, Imputer, LabelBinarizer

from sklearn.decomposition import TruncatedSVD

from sklearn_pandas import DataFrameMapper

from IPython.display import display

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge
def cat_split(row):

    try:

        text = row

        txt1, txt2, txt3 = text.split('/')

        return txt1, txt2, txt3

    except:

        return "none", "none", "none"
class SelectColumnsTransfomer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=[]):

        self.columns = columns



    def transform(self, X, y=None, **transform_params):

        print("working")

        return X[self.columns].values



    def fit(self, X, y=None, **fit_params):

        print("working")

        return self



class To1DArrayTransfomer(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass



    def transform(self, X, y=None, **transform_params):

        return X.ravel() #ravel makes the shape (n,) we dont want (n,1)



    def fit(self, X, y=None, **fit_params):

        return self

    

class CategorySplitingTransformer(BaseEstimator, TransformerMixin):

    def transform(self, X, y=None, **transform_params):

        X = pd.DataFrame(X)

        X["cat_1"], X["cat_2"], X["cat_3"] = zip(*X[0].apply(lambda val: cat_split(val)))

        ans = []

        ans.append(X["cat_1"])

        ans.append(X["cat_2"])

        ans.append(X["cat_3"])

        return np.array(ans)

    

    def fit(self, X, y=None, **fit_params):

        return self



class removeNull(BaseEstimator, TransformerMixin):

    def __init__(self, val="nan"):

        self.val = val



    def transform(self, X, y=None, **transform_params):

        X_df = pd.DataFrame(X)

        X_df = X_df.fillna(self.val)

        X = X_df.values

        return X.ravel()



    def fit(self,X, y=None, **fit_params):

        return self

    

class parseToDense(BaseEstimator, TransformerMixin):

    def __init__(self):

        pass

    

    def transform(self, X, y=None, **transform_params):

        return X.toarray()



    def fit(self, X, y=None, **fit_params):

        return self



class LabelBinarizer_new(TransformerMixin, BaseEstimator):

    def fit(self, X, y = 0):

        self.encoder = None

        return self

    def transform(self, X, y = 0):

        if(self.encoder is None):

            print("Initializing encoder")

            self.encoder = LabelBinarizer();

            result = self.encoder.fit_transform(X)

        else:

            result = self.encoder.transform(X)

        return result;



class select_col(TransformerMixin, BaseEstimator):

    def __init__(self,n=0):

        self.n = n

    def fit(self, X, y = 0):

        return self

    def transform(self, X, y = 0):

        return X[self.n]
#settings, bump up the numbers for better accuracy

n_comp_name = 5

n_comp_description = 10

#pipelines

id_pipeline = Pipeline([

    ('id_select',SelectColumnsTransfomer(['train_id']))

])

name_pipeline = Pipeline([

    ('name_select',SelectColumnsTransfomer(['name'])),

    ('To1DArrayTransfomer',To1DArrayTransfomer()),

    ('TfidfVectorizer',TfidfVectorizer(stop_words='english')),

    ('TruncatedSVD',TruncatedSVD(n_components=n_comp_name, algorithm='arpack'))

])

item_description = Pipeline([

    ('item_descriptione_select',SelectColumnsTransfomer(['item_description'])),

    ('rvm_nans',removeNull("No decription")),

    ('To1DArrayTransfomer',To1DArrayTransfomer()),

    ('TfidfVectorizer',TfidfVectorizer(stop_words='english')),

    ('TruncatedSVD',TruncatedSVD(n_components=n_comp_description, algorithm='arpack'))

])

category_spliting_pipeline = Pipeline([

    ('category_name_select',SelectColumnsTransfomer(['category_name'])),

    ('category_spliting',CategorySplitingTransformer()),

    #('to_labels',LabelBinarizer_new())

])

cat_0_pipeline = Pipeline([

    ('category_spliting_pipeline',category_spliting_pipeline),

    ('select_col',select_col(n=0)),

    ('to_labels',LabelBinarizer_new())

])

cat_1_pipeline = Pipeline([

    ('category_spliting_pipeline',category_spliting_pipeline),

    ('select_col',select_col(n=1)),

    ('to_labels',LabelBinarizer_new())

])

cat_2_pipeline = Pipeline([

    ('category_spliting_pipeline',category_spliting_pipeline),

    ('select_col',select_col(n=2)),

    ('to_labels',LabelBinarizer_new())

])

item_condition_id_pipeline = Pipeline([

    ('item_condition_id_select',SelectColumnsTransfomer(['item_condition_id'])),

    ('to_labels',LabelBinarizer_new())

])

brand_name_pipeline = Pipeline([

    ('brand_name_select',SelectColumnsTransfomer(['brand_name'])),

    ('rvm_nans',removeNull("No brand")),

    ('to_labels',LabelBinarizer_new())

])

shipping_pipeline = Pipeline([

    ('shipping_select',SelectColumnsTransfomer(['shipping']))

])





all_feature_pipeline = FeatureUnion([

    #id_select', id_pipeline),

    ('name_select', name_pipeline),

    #('item_descriptione_select', item_description),

    #('item_condition_id_select', item_condition_id_pipeline),

    #('cat_0_select', cat_0_pipeline),

    #('cat_1_select', cat_1_pipeline),

    #('cat_2_select', cat_2_pipeline),

    #('brand_name_select', brand_name_pipeline),

    #('shipping_select', shipping_pipeline),

    

])

final_pipeline = Pipeline([

    ('all_feature_pipeline',all_feature_pipeline),

    ('RandomForestRegressor', RandomForestRegressor(n_jobs=-1,min_samples_leaf=3,n_estimators=200))

])
#all_feature_pipeline.fit(train_X,train_y)

#a1 = all_feature_pipeline.transform(train_X,train_y)

# print("start fit")

# final_pipeline.fit(train_X,train_y)

# print("end fit, start predict")

# a1 = final_pipeline.predict(test_X)

print("1")

data_X = all_feature_pipeline.fit_transform(train_X)

print("2")

data_test_X = all_feature_pipeline.transform(test_X)

print("3")

modl = Ridge(solver = "lsqr", fit_intercept=False)

print("4")

modl.fit(data_X,train_y)

print("5")

ans = modl.predict(data_test_X)

print("6")
print(ans)
ans1 = pd.DataFrame(ans)
ans1.columns = ['price']

ans1["test_id"] = test_X["test_id"]

ans1["price"] = ans1["price"]+1


ans1.to_csv("output.csv",index=False)