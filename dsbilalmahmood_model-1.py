# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.externals import joblib # load trained model

import scipy

from sklearn.feature_extraction.text import  TfidfVectorizer

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
## Reading test data

df_test = pd.read_csv("../input/mercari-price-suggestion-challenge/test.tsv",sep = "\t")



## Dealing with missing values

df_test = df_test.fillna('unavailable')

class CustomVectorizer(BaseEstimator, TransformerMixin):

    """ This is a custom transformer that joins text(product name + description) 

        with non text variables(product category, brand, condition & shipment)

        and is used in pipeline to tune and make predictions.

        

        name + description ---> DTM 

        category_name --->  OneHotEncoding

        brand name --->  OneHotEncoding

        item_condition --> Number

        shipping ---> Number

       

        

    Parameters

    ----------

    min_df = look at sklearn's tfidf vectorizer for its meaning,

    ngram_range = look at sklearn's tfidf vectorizer for its meaning

    stop_words = list of english words that do not convey vital information

    

    """

    def __init__(self, min_df = 10, ngram_range = (1,1), stop_words = None):

        ## Hyper parameters for the text part 

        self.min_df = min_df

        self.ngram_range = ngram_range

        self.stop_words = stop_words

        

        

    def fit(self, df, y=None):

        """Fits separate transformers on text and categorical data.

        Parameters

        ----------

        df : dataframe with text and non text variables

        y : None

            There is no need of a target in a transformer, yet the pipeline API

            requires this parameter, thus it is there

        Returns

        -------

        self : object

            Returns self.

        """



        ## Fitting Text

        self.text_vect_  = TfidfVectorizer(stop_words = self.stop_words, 

                                          ngram_range = self.ngram_range,

                                          min_df = self.min_df)

        text = (df["name"] + " | " + df["item_description"]).values        

        self.text_vect_.fit(text)

        

        ## Fitting Categories

        self.label_cat_enc_  = LabelEncoder()

        self.hot_cat_enc_  = OneHotEncoder()

        categories = df["category_name"].values

        self.hot_cat_enc_.fit(self.label_cat_enc_.fit_transform(categories).reshape(-1, 1))

        

        self.cat_classes_dict = {i:0 for i in self.label_cat_enc_.classes_}

        print("Number of categories {}".format(len(self.cat_classes_dict)))

        

        ## Fitting Brand names

        self.label_brand_enc_  = LabelEncoder()

        self.hot_brand_enc_  = OneHotEncoder()

        brands = df["brand_name"].values

        self.hot_brand_enc_.fit(self.label_brand_enc_.fit_transform(brands).reshape(-1, 1))

        

        self.brand_classes_dict = {i:0 for i in  self.label_brand_enc_.classes_}

        print("Number of brands {}".format(len(self.brand_classes_dict)))

        

        # Return the transformer

        return self



    def transform(self, df, y = None):

        """ Using the fitted individual transformers, transform the data and 

        concatenate them into a special DTM

        

        Parameters

        ----------

        df : dataframe containing the required column names

            The input samples.

        Returns

        -------

        X_transformed : Custom Matrix with DTM + encoded non text data

        """

        

        ## Transforming text 

        text = (df["name"] + " | " + df["item_description"]).values         

        text_dtm = self.text_vect_.transform(text)

        

        ## Transforming brands

        """

        new_brands = df["brand_name"].values

        new_brands[np.isin(new_brands, self.label_brand_enc_.classes_, 

                           invert = True)] = "unavailable"

        """

        new_brands = df["brand_name"]

        new_brands[~new_brands.isin(self.brand_classes_dict)] = "unavailable"

        

    

        trans_brands = self.hot_brand_enc_.transform(self.label_brand_enc_.transform(new_brands).reshape(-1, 1))

        

        ## Transforming categories

        """

        new_categories = df["category_name"].values

        new_categories[np.isin(new_categories, self.label_cat_enc_.classes_, 

                           invert = True)] = "unavailable"

        """

        new_categories = df["category_name"]

        new_categories[~new_categories.isin(self.cat_classes_dict)] = "unavailable"

        

        

        trans_categories = self.hot_cat_enc_.transform(self.label_cat_enc_.transform(new_categories).reshape(-1, 1))

         

        

        ## Item Condition and Shipping

        trans_item_condition = df["item_condition_id"].values.reshape(-1, 1)

        trans_shipping = df["shipping"].values.reshape(-1, 1)

        

        

        ## Sparse Vectors

        sparse_trans_categories =  scipy.sparse.csr.csr_matrix(trans_categories)   

        sparse_trans_brands =  scipy.sparse.csr.csr_matrix(trans_brands)   

        sparse_trans_item_condition = scipy.sparse.csr.csr_matrix(trans_item_condition)

        sparse_trans_shipping = scipy.sparse.csr.csr_matrix(trans_shipping)

        

        ## Stacked Sparse dataframe

        return scipy.sparse.hstack([sparse_trans_categories, sparse_trans_brands, 

                                    sparse_trans_item_condition, sparse_trans_shipping, text_dtm])

        

       







## Loading Model

trans  = joblib.load('../input/mercari-model/trained_transformer.pkl') 

reg_model  = joblib.load('../input/mercari-model/NN_100_relu_linear_output.pkl') 
## Transforming the features

X_test =  df_test[["category_name","brand_name","item_condition_id","shipping","name","item_description"]]

X_test_vector = trans.transform(X_test)



## Making predictions using NN

y_pred = reg_model.predict(X_test_vector)



## Predicted dataframe

df_prediction = df_test[["test_id"]]

df_prediction["price"] = y_pred 



## Saving the predictions

df_prediction.to_csv("sample_submission.csv",

                     index = False)
