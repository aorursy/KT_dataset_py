#feature extraction

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



#our pipeline

from sklearn.pipeline import Pipeline, FeatureUnion



#this is needed for our class below

from sklearn.base import BaseEstimator, TransformerMixin



#this returns a column

class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key, is_ordinal = False):

        self.key = key

        self.is_ordinal = is_ordinal

    def fit(self, X, y = None):

        return self

    

    def transform(self, X, y = None):

        

        if X.loc[:, self.key].dtype == object:

            return X.loc[:, self.key].values

        else:

            if self.is_ordinal:

                return X.loc[:, self.key].apply( lambda x : self.key + '_' + str(x) ).values

            else:

                return X.loc[:, self.key].values.reshape(-1, 1)

            

#this helps collects all of our features    

feats_pipe = FeatureUnion( [

    ('name', Pipeline( [

        ('col_select', ColumnSelector('name') ),

        ('tf_idf', TfidfVectorizer(stop_words = 'english', 

                                   sublinear_tf = True ) )

    ]

    )

    ),

    ('item_description', Pipeline( [

        ('col_select', ColumnSelector('item_description') ),

        ('tf_idf', TfidfVectorizer( stop_words = 'english', 

                                   sublinear_tf = True ) )

    ]

    )

    ),

    #this removes puncation, and helps divide our categories to the sub categories, and 

    #then weights them

    ('category_name', Pipeline( [

        ('col_select', ColumnSelector('category_name') ),

        ('tf_idf', TfidfVectorizer(stop_words = 'english', sublinear_tf = True  ) )

    ]

    )

    ),

    ('brand_name', Pipeline( [

        ('col_select', ColumnSelector('brand_name') ),

        ('tf_idf', TfidfVectorizer(stop_words = 'english', sublinear_tf = True  ) )

    ]

    )

    ),

    ('is_shipping', ColumnSelector('shipping') ),

    #this is essentially one-hot-encoding

    ('item_condition_id', Pipeline( [

        ('col_select', ColumnSelector('item_condition_id', is_ordinal = True) ),

        ('one_binary', CountVectorizer(lowercase = False, binary = True) ) 

    ]

    ) 

    )

]

)