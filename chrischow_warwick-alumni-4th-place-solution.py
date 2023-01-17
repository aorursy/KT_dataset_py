# Import required modules

import json

import numpy as np

import pandas as pd

import time

import warnings



from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline, FeatureUnion



%matplotlib inline

warnings.filterwarnings('ignore')
# MAP2

def MAP2(y_true, y_pred):

    

    # Ensure number of rows are the same

    if len(y_true) != y_pred.shape[0]:

        

        # Throw an error

        raise Exception("Length of ground truth vector and predictions differ.")

    

    # Compute Average Precision (AP)

    ap = ( (y_pred[:, 0] == y_true).astype(int) +

           (y_pred[:, 1] == y_true).astype(int) / 2 )

    

    # Compute mean of AP across all observations

    output = np.mean(ap)

    

    # Return

    return output
# Logistic regression that outputs 2 recommendations

class lr_map2(BaseEstimator, ClassifierMixin):

    

    def __init__(self, multi_class='ovr', solver='saga', max_iter=100, C=1.0, random_state=123, n_jobs=4, class_weight=None):

        

        self.multi_class=multi_class

        self.solver=solver

        self.C=C

        self.class_weight = class_weight

        self.max_iter=max_iter

        self.random_state=random_state

        self.n_jobs=n_jobs

    

    def fit(self, X, y=None):

        

        self.model = LogisticRegression(

            multi_class=self.multi_class,

            solver=self.solver,

            C=self.C,

            class_weight=self.class_weight,

            max_iter=self.max_iter,

            random_state=self.random_state,

            n_jobs=self.n_jobs

        )

        

        self.model.fit(X, y)

        

        return self

    

    def predict(self, X, y=None):

        

        # Predict probability of all classes

        pred_probs = self.model.predict_proba(X)

        

        # Extract top two classes

        pred = self.model.classes_[np.apply_along_axis(lambda x: x.argsort()[-2:][::-1], axis=1, arr=pred_probs)]

        

        return pred
# Transformer to fit TF-IDF vectorizer with target labels and transform titles

class LabelTransform(TransformerMixin):

    

    def __init__(self, labels_tgt, ngram_range, max_df, min_df):

        

        self.labels_tgt = labels_tgt

        self.ngram_range = ngram_range

        self.max_df = max_df

        self.min_df = min_df

    

    def set_params(self, labels_tgt=None, ngram_range=None, max_df= None, min_df=None):

        

        if labels_tgt:

            self.labels_tgt = labels_tgt

        

        if ngram_range:

            self.ngram_range = ngram_range

        

        if max_df:

            self.max_df = max_df

        

        if min_df:

            self.min_df = min_df

    

    def fit(self, X, y=None):

        

        # Initialise TF-IDF vectorizer for target labels

        self.vect_labels = TfidfVectorizer(

            

            # USE COUNTS

            use_idf=False, norm=False, binary=True,

            

            # ALLOW SINGLE ALPHANUMERICS

            token_pattern='(?u)\\b\\w+\\b',

            

            # TUNE THESE

            ngram_range=self.ngram_range,

            max_df=self.max_df,

            min_df=self.min_df

        )

        

        # Fit to target labels

        self.vect_labels.fit(self.labels_tgt)

        

        return self



    def transform(self, X, y=None):

        

        # Transform

        output = self.vect_labels.transform(X)

        

        # Return

        return output
# Function for extracting top 2 recommendations

def recommend_two(model, X_val):

    

    # Obtain predictions

    pred_probs = model.predict_proba(X_val)

    

    # Extract top two classes

    pred = pd.DataFrame(model.classes_[np.apply_along_axis(lambda x: x.argsort()[-2:][::-1], axis=1, arr=pred_probs)])

    

    return pred
# Function for preparing submissions

def submit_kaggle(itemid, var, pred):

    

    # Establish output

    output = pred.copy()

    

    # Set itemid

    output = pd.DataFrame(itemid.astype(str) + '_' + str(var))

    output.rename(columns = {'itemid': 'id'}, inplace=True)

    

    # Set tagging

    output['tagging'] = pred.iloc[:, 0].astype(int).astype(str) + ' ' + pred.iloc[:, 1].astype(int).astype(str)

    

    # Return

    return output
# INPUT ATTRIBUTE AND LABEL HERE:

VAR = 'Camera'

LABEL = VAR

DATASET = 'mobile'



# Read data

main = pd.read_csv('../input/mobile_data_info_train_competition.csv')

val = pd.read_csv('../input/mobile_data_info_val_competition.csv')



# Import codebook

with open('../input/mobile_profile_train.json', 'r') as f:

    main_cb = json.load(f)



# Configure stopwords

stop_words = set([

    'promo','diskon','baik','terbaik', 'murah',

    'termurah', 'harga', 'price', 'best', 'seller',

    'bestseller', 'ready', 'stock', 'stok', 'limited',

    'bagus', 'kualitas', 'berkualitas', 'hari', 'ini',

    'jadi', 'gratis'

])
# Set output csv name

filename = 'mobile_' + VAR.lower().replace(' ', '_') + '.csv'



# Get all titles - BIG MISTAKE

all_titles = main['title'].append(val['title'])



# Drop image

main.drop('image_path', axis = 1, inplace=True)



# Delete missing values

main = main[~main[VAR].isnull()]



# Translate words

main['title'] = main['title'].str.replace('tahun', 'year')

main['title'] = main['title'].str.replace('bulan', 'month')

main['title'] = main['title'].str.replace('hitam', 'black')

main['title'] = main['title'].str.replace('putih', 'white')

main['title'] = main['title'].str.replace('hijau', 'green')

main['title'] = main['title'].str.replace('merah', 'red')

main['title'] = main['title'].str.replace('ungu', 'purple')

main['title'] = main['title'].str.replace(' abu', 'gray')

main['title'] = main['title'].str.replace('perak', 'silver')

main['title'] = main['title'].str.replace('kuning', 'yellow')

main['title'] = main['title'].str.replace('coklat', 'brown')

main['title'] = main['title'].str.replace('emas', 'gold')

main['title'] = main['title'].str.replace('biru', 'blue')

main['title'] = main['title'].str.replace('tahan air', 'waterproof')

main['title'] = main['title'].str.replace('layar', 'touchscreen')



# Translate words

val['title'] = val['title'].str.replace('tahun', 'year')

val['title'] = val['title'].str.replace('bulan', 'month')

val['title'] = val['title'].str.replace('hitam', 'black')

val['title'] = val['title'].str.replace('putih', 'white')

val['title'] = val['title'].str.replace('hijau', 'green')

val['title'] = val['title'].str.replace('merah', 'red')

val['title'] = val['title'].str.replace('ungu', 'purple')

val['title'] = val['title'].str.replace(' abu', 'gray')

val['title'] = val['title'].str.replace('perak', 'silver')

val['title'] = val['title'].str.replace('kuning', 'yellow')

val['title'] = val['title'].str.replace('coklat', 'brown')

val['title'] = val['title'].str.replace('emas', 'gold')

val['title'] = val['title'].str.replace('biru', 'blue')

val['title'] = val['title'].str.replace('tahan air', 'waterproof')

val['title'] = val['title'].str.replace('layar', 'touchscreen')



# Configure target labels

labels_tgt = pd.Series(list(main_cb[LABEL].keys()))



# Rename data

X_data = main['title']

y_data = main[VAR]
# Set up feature union

opt_feats = FeatureUnion(

    [

        ('labels', LabelTransform(

            labels_tgt=labels_tgt,

            ngram_range=(1,1),

            max_df=0.2,

            min_df=1

        )),

        

        # Using LabelTransform to speed up code

        ('titles', LabelTransform(

            labels_tgt=all_titles,

            ngram_range=(1,3),

            max_df=0.3,

            min_df=1

        ))

    ]

)
# Create temporary Pipeline

temp_pipe = Pipeline([('test_feats', opt_feats)])



# Fit to data

temp_pipe.fit(X_data, y_data)



# Transform

temp_pipe_output = temp_pipe.transform(X_data)



# Select rows

pd.DataFrame(temp_pipe_output[:20, :10].toarray())
# Params

params_test = {

    'lr__C': [1]

}



# Test pipe

test_pipe = Pipeline(

    [

        ('tfidf', opt_feats),

        ('lr', lr_map2(multi_class='ovr', solver='saga', C=1))

    ]

)
# Initialise GridSearchCV

opt_test = GridSearchCV(

    estimator = test_pipe,

    param_grid = params_test,

    cv=5,

    scoring=make_scorer(MAP2, greater_is_better=True),

    iid=False,

    verbose=20,

    n_jobs=4,

)



# Start timer

start_time = time.time()



# Fit

opt_test.fit(X_data, y_data)



# Stop timer

end_time = time.time()

print('Time taken: %s mins.' % ('{:.2f}'.format((end_time-start_time)/60)))



# Extract parameters

cv_results = pd.DataFrame(opt_test.cv_results_['params'])



# Extract mean test score

cv_results['mean_test_score'] = pd.Series(opt_test.cv_results_['mean_test_score'])

cv_results['std_test_score'] = pd.Series(opt_test.cv_results_['std_test_score'])



# Display

cv_results.sort_values('mean_test_score', ascending=False)
# Set up feature union

opt_feats = FeatureUnion(

    [

        ('labels', LabelTransform(

            labels_tgt=labels_tgt,

            ngram_range=(1,1),

            max_df=0.2,

            min_df=1

        )),

        

        ('titles', LabelTransform(

            labels_tgt=all_titles,

            ngram_range=(1,3),

            max_df=0.3,

            min_df=1

        ))

    ]

)



# Set up Logistic Regression

opt_clf = LogisticRegression(

    multi_class='ovr',

    solver='saga',

    C=1,

    random_state=123,

    n_jobs=4

)



# Set up full pipeline

opt_pipe = Pipeline(

    [

        ('tfidf', opt_feats),

        ('lr', opt_clf)

    ]

)



# Train on dataset

opt_pipe.fit(X_data, y_data)



# Obtain predictions

pred = recommend_two(opt_pipe, val['title'])



# Prepare dataset

submission = submit_kaggle(val.itemid, VAR, pred)



# Output

submission.to_csv(filename, index=False)