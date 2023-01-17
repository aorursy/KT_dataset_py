import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotig graphs library

import statsmodels.api as sm # Used for classical LM and GLM modeling

from statsmodels.graphics.api import abline_plot # Diagnostic plots

from statsmodels.graphics.regressionplots import * # Diagnostic plots

from statsmodels.stats.outliers_influence import variance_inflation_factor # Outlier Diagnostic

from patsy import dmatrices # statsmodels input preparation

import datetime



import nltk.stem # Stemming tool for processing the transcript tokens

import xgboost as xgb # Gradient Boosting Trees library

import ast # Parser for python objects



from sklearn.decomposition import TruncatedSVD # SVD decomposition for performing LSA

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.cluster import MiniBatchKMeans # Efficient clustering algorithm

from sklearn.utils.validation import check_array # Input checker for metic calculation

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, StratifiedKFold, train_test_split

from sklearn.preprocessing import Normalizer # Normalizer with instance, useful as preprocessing before clustering



SEED = 12

N_CLUSTERS_TAGS = 6

# Precision slack for numerical stability

EPSILON = 1e-6



################# Inherit TfidfVectorizer to adapt it to stemming ###################

english_stemmer = nltk.stem.SnowballStemmer('english')



class StemmedTfidfVectorizer(TfidfVectorizer):

    """Tf-idf Vectorizer which is applying also stemming i.e. it boils down

    each word to its root. In the semantic analysis we are doing where we don't

    take into account the parts of speach stemming is a good choice to generalize

    better over the general meaning the text of interest has."""

    # Overriding the build_analyzer

    def build_analyzer(self):

        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()

        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

    

######################### Define RatingsVectorizer ##################################



class RatingsVectorizer():

    """Used to parse and convert to vector the ratings for each TED talk. It has

    to 'book keep' all of the rating terms and their corresponding indices."""



    def __init__(self):

        # It stores as key the rating name and as value

        # the corresponding index.

        self.vocabulary_ = dict()

        

    def get_feature_names(self):

        return list(self.vocabulary_.keys())

    

    def parse_ratings(self, json_collection):

        """It returns a frequency table where the rows are talks and columns - ratings"""



        # First pass.

        for json_string in json_collection:

            ratings = ast.literal_eval(json_string)

            

            for rating in ratings:

                if rating['name'] not in self.vocabulary_:

                    self.vocabulary_[rating['name']] = len(self.vocabulary_)

                    

        # Second pass.

        freq_table = np.zeros(shape = (len(json_collection), len(self.vocabulary_)))

        for i in range(len(json_collection)):

            ratings = ast.literal_eval(json_collection[i])

            

            for rating in ratings:

                freq_table[i,self.vocabulary_[rating['name']]] = rating['count']



        return freq_table



################################## Benchmark Methods ###############################



def cluster_stratification(k, ted_df, order_centroids, df_clust_name, terms):

    """Diagnostic plotting for clusters"""

    

    for i in range(k):

        print("Cluster %d:" % i, end='')

        for ind in order_centroids[i, :10]:

            print(' %s;' % terms[ind], end='')

        print()



    plt.figure(1, figsize=(21, 7))

    tran_count = ted_df.groupby([df_clust_name]).size()

    tran_count.plot.bar()

    plt.ticklabel_format(style='plain', axis='y')

    plt.xticks(rotation=0)

    plt.xlabel('Cluster indices')

    plt.ylabel('Count of Ted talks')

    plt.title('Frequency of the talks within the clusters')

    plt.show()



    plt.figure(1, figsize=(21, 7))

    tran_comments = ted_df.groupby([df_clust_name])['comments'].agg('mean')

    tran_comments.plot.bar()

    plt.ticklabel_format(style='plain', axis ='y')

    plt.xticks(rotation=0)

    plt.xlabel('Cluster indices')

    plt.ylabel('Average comments count')

    plt.title('Average comments count within the clusters')

    plt.show()



    plt.figure(1, figsize = (21, 7))

    tran_views = ted_df.groupby([df_clust_name])['views'].agg('mean')

    tran_views.plot.bar()

    plt.ticklabel_format(style ='plain', axis ='y')

    plt.xticks(rotation=0)

    plt.xlabel('Cluster indices')

    plt.ylabel('Average reviews count')

    plt.title('Average reviews count within the clusters')

    plt.show()



def poisson_ce(y_true, y_pred):

    """Poisson cross entropy metric used in the benchmark."""

    y_true = check_array(y_true)

    y_pred = check_array(y_pred)

    return np.mean(y_pred - y_true * np.log(y_pred + EPSILON))



def xgb_poisson_ce(y_preds, dtrain):

    """ Poisson Cross Entropy metric used in xgb validation."""

    

    y_true = dtrain.get_label()

    y_p_log = np.log(y_preds + EPSILON)





    loss = y_preds - y_true * y_p_log

    return 'wloss', loss

 

def delta(y_true, y_pred):

    """Delta Error for each observation"""

    y_true = check_array(y_true)

    y_pred = check_array(y_pred)

    return y_true - y_pred



def mde(d):

    """Mean Delta Error"""

    return np.mean(d)



def mape(d, y_true):

    """Mean Absolute Percentage Error"""

    y_true = check_array(y_true)

    return np.mean(np.abs(d / y_true)) * 100



def mae(d):

    """Mean Absolute Error"""

    return np.mean(np.abs(d))



def mse(d):

    """Mean Squared Error"""

    return np.mean(d * d)



def output_benchmark_results(pred_df, models = None):

    """Calculates and returns metrics for each TED talks semantic cluster."""

    

    def build_metrics(pred_df, models):    

        aggs = dict()

        for m in models:

            y_true = pred_df['comments'].values.reshape(-1, 1)

            y_hat = pred_df[m].values.reshape(-1, 1)

            d = delta(y_true, y_hat)

            

            aggs["mde_" + m] = mde(d)

            aggs["mape_" + m] = mape(d, y_true)

            aggs["mae_" + m] = mae(d)

            aggs["mse_" + m] = mse(d)

            aggs["po_" + m] = poisson_ce(y_true, y_hat)

            

        return pd.Series(aggs, index=aggs.keys())

    

    if models is None:       

        models = list(pred_df.columns.values)

        models = [m for m in models if m != 'semantic_labels' and m != 'comments']

        

    benchmark_df = pred_df.groupby('semantic_labels').apply(build_metrics, models = models)

    

    # Reordering the columns alphabetically so that it is easier to compare by model within metric

    return benchmark_df.reindex(sorted(benchmark_df.columns), axis=1)

    

def calculate_metrics(y_true, y_pred, df, model_name):

    d = delta(y_true, y_pred)

    i = df.shape[0]

    df.loc[i,'MAE'] = mae(d)

    df.loc[i,'MSE'] = mse(d)

    df.loc[i,'MDE'] = mde(d)

    df.loc[i,'MAPE'] = mape(d, y_true)

    df.loc[i,'PCE'] = poisson_ce(y_true, y_pred)

    df.loc[i,'Model'] = model_name

    

    return df



################################# XGB Cross Validation Model Ensambling ###############################



def train_regressors(params, full_train=None, y=None):

    """Training k-fold cross-validation of xgb where we use the validation

    part for assessing early stopping. Afterwards we return the models for

    each fold training."""



    folds = StratifiedKFold(n_splits=7,random_state=SEED)

    clfs = []

    importances = pd.DataFrame()

     

    oof_preds = np.zeros((len(full_train), ))

    regs = []

    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):

        trn_x, trn_y = full_train[trn_,:], y[trn_]

        val_x, val_y = full_train[val_,:], y[val_]

        reg = xgb.XGBRegressor(**xgb_params)

        reg.fit(

            trn_x, trn_y,

            eval_set=[(trn_x, trn_y), (val_x, val_y)],

            eval_metric = 'poisson-nloglik',

            verbose=100,

            early_stopping_rounds=5

        )

        oof_preds[val_] = reg.predict(val_x, ntree_limit=reg.best_ntree_limit)

        print("The poisson cross entropy in cv is " ,\

              poisson_ce(val_y.reshape(-1, 1), reg.predict(val_x, ntree_limit=reg.best_ntree_limit).reshape(-1, 1)))



        regs.append(reg)

        print("The poisson cross entropy is: ", poisson_ce(y_true=y.reshape(-1, 1), y_pred=oof_preds.reshape(-1, 1)))

        

    return regs



def predict_with_regressors(regs, test):

    """Ensambling by building the average from the predictions

    of multiple xgb Regressors."""

    predictions = None

    for reg in regs:

        if predictions is None:

            predictions = reg.predict(test) / len(regs)

        else:

            predictions += reg.predict(test) / len(regs)

    fig, ax = plt.subplots(figsize=(6,9))

    xgb.plot_importance(reg, max_num_features=10, height=0.8, ax=ax)

    plt.show()

    

    return predictions
# Inner join between the bot tables. We rop some rows, but it is ok.

ted_df = pd.merge(pd.read_csv('../input/transcripts.csv'), pd.read_csv('../input/ted_main.csv'), on='url')



ted_df['published_month'] = ted_df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%m'))



# Filter fill the NAs for 'speaker_occupation'

ted_df['speaker_occupation'].fillna('Not specified', inplace = True)



# The dataset has been published in 2017-09-09 ~ 1504915200(unix)

# Caluclate the approximate age in days

ted_df['talk_age'] = ted_df['published_date'].apply(lambda x: int((1504915200 - x)/(86_400))) # 86400s = 3600s * 24h



# Preparing the tags to be parsed by the CountVectorizer(We separate the tags by space and unite

# the words within a tag by '_'), expecting to be parsed as sentence and treated as sentence tokens.

ted_df['tags_text'] = ted_df['tags'].str.replace('(?<=[A-Za-z])(\s)(?=[A-Za-z])','_')

ted_df['tags_text'] = ted_df['tags_text'].str.replace('\[|\'|\]', ' ')



count_vectorizer = CountVectorizer()



# Very inefficient stroring of the word frequencies within the talk description,

# in the form of a dense matrix with row for each talk and a column for each word ever observed.

# Effectivelly we are using count vectorizer for quick and dirty transforming of the tags into

# One-hot Encoding, since we have max one tag occurrence in a talk

tags_one_hot = count_vectorizer.fit_transform(ted_df['tags_text'].values).toarray()



# We are using MiniBatchKMeans which is well suited in case of sparse examples as in our case with one-hot

# encoded tags. See the very good paper of the algorithm  http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

km = MiniBatchKMeans(n_clusters=N_CLUSTERS_TAGS, init='k-means++', n_init=1,

                         init_size=1000, batch_size=1000, random_state = SEED)



km.fit(tags_one_hot)



terms = count_vectorizer.get_feature_names()

cluster_tags_freq = np.empty((N_CLUSTERS_TAGS,tags_one_hot.shape[1]))

for i, c in enumerate(km.labels_):

    cluster_tags_freq[c,:] += tags_one_hot[i,:]

    

# Normalized by the total tag frequency

cluster_tags_freq /= cluster_tags_freq.sum(axis = 0, keepdims = True)

order_centroids = cluster_tags_freq.argsort()[:, ::-1]



ted_df['tag_clust'] = km.labels_



cluster_stratification(N_CLUSTERS_TAGS, ted_df, order_centroids, 'tag_clust', terms)



CLUSTER_LABELS = ['0:ecology_rare_uninteresting',

                  '1:medicine_rare_uninteresting',

                  '2:politic_usual_neutral',

                  '3:existential_usual_populare',

                  '4:science_usual_neutral',

                  '5:creativity_frequent_neutral']



# Map the semantic labels to a column

ted_df['semantic_labels'] = [CLUSTER_LABELS[x] for x in km.labels_] 
from scipy import stats # using Kruskal Wallis unparametric test

stats.kruskal(*[group["comments"].values for name, group in ted_df.groupby("event")])
ted_df = ted_df.sort_values(by= 'event', ascending=True)



# We sort the data chronologically in order to see with the linear regression if we have a time drift of the

# residuals.

ted_df = ted_df.sort_values(by= 'film_date', ascending=True)

ted_df.reset_index(inplace=True, drop = True)

ted_df['event'].tail(500)



TEST_OFFSET = ted_df[ted_df['event'] == 'TEDWomen 2015'].index.min()



ted_train = ted_df[0:TEST_OFFSET].reset_index()

ted_test = ted_df[TEST_OFFSET:].reset_index()



print("Train set size is {}".format(ted_train.shape[0]))

print("Test set size is {}".format(ted_test.shape[0]))



tags_one_hot_train = tags_one_hot[0:TEST_OFFSET]

tags_one_hot_test = tags_one_hot[TEST_OFFSET:]



train_semantic_freq = ted_train.groupby('semantic_labels').size()

train_semantic_freq.columns = ['semantic_labels', 'count']



plt.figure(1, figsize=(21, 7))

train_semantic_freq.plot.bar()

plt.xticks(rotation=280)

plt.xlabel('Semantic Labels')

plt.ylabel('Semantic Labels Frequency in Train Set')

plt.title('Count of the talks in Train Set')

plt.show()



test_semantic_freq = ted_test.groupby('semantic_labels').size()

test_semantic_freq.columns = ['semantic_labels', 'count']



plt.figure(1, figsize=(21, 7))

test_semantic_freq.plot.bar()

plt.xticks(rotation=280)

plt.xlabel('Semantic Labels')

plt.ylabel('Semantic Labels Frequency in Test Set')

plt.title('Count of the talks in Test Set')

plt.show()
formula = """comments ~ views + duration + talk_age + C(published_month)"""

response, predictors = dmatrices(formula, ted_train, return_type='dataframe')

test_response, test_predictors = dmatrices(formula, ted_test, return_type='dataframe')
lin_m = sm.OLS(response, predictors).fit()



# Print the linear regrgression summary

print(lin_m.summary())



# Plot the fitted values against the observed values to

# verify visualy what is the quality of the fitted model.

fig, ax = plt.subplots()

ax.scatter(lin_m.fittedvalues, response)

line_fit = sm.OLS(response, sm.add_constant(lin_m.fittedvalues, prepend=True)).fit()

abline_plot(model_results=line_fit, ax=ax)



ax.set_title('Model Fit Plot')

ax.set_ylabel('Observed values')

ax.set_xlabel('Fitted values')



influence_plot(lin_m)
# Calculate and plot lot the Cooks' distanes and the high leverage and residual points.

influence = lin_m.get_influence()

#c is thec_p distance and p is p-value

(c_dist, c_p) = influence.cooks_distance

plt.stem(np.arange(len(c_dist)), c_dist, markerfmt=",")

plt.show()
# Add the Cook's distance in the training data in order to use it

# for filtering if necesery 

ted_train['cooks_dist'] = c_dist

outier_cutoff = 0.0002
# Predict with the testing set

lin_out_prediction = lin_m.predict(test_predictors).clip(0, None)
# Define the prediction tabe with all model predictions

predictions_df = ted_test[['semantic_labels', 'comments']]

predictions_df['out_lin_model'] = lin_out_prediction



output_benchmark_results(predictions_df, models = ['out_lin_model'])
# Define overall score table with all the results for the models

overall_metrics = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'MDE', 'MAPE', 'PCE'])

calculate_metrics(ted_test['comments'].values.reshape(-1, 1), lin_out_prediction.values.reshape(-1, 1), overall_metrics,'out_lin_model')
# Removing the outliers

print("""We remove {} outliers from the training set.""".format(ted_train[ted_train['cooks_dist'] > outier_cutoff].shape[0]))

ted_train_no_outliers = ted_train[ted_train['cooks_dist'] <= outier_cutoff].drop('cooks_dist', axis = 1)



# Setting the new prediction formula

formula = """comments ~ views + I(views**2) + I(talk_age**2) + talk_age + duration

+ languages + duration * views + languages * talk_age"""



response, predictors = dmatrices(formula, ted_train_no_outliers, return_type='dataframe')

test_response, test_predictors = dmatrices(formula, ted_test, return_type='dataframe')



lin_m = sm.OLS(response, predictors).fit()

print(lin_m.summary())



fig, ax = plt.subplots()

ax.scatter(lin_m.fittedvalues, response)

line_fit = sm.OLS(response, sm.add_constant(lin_m.fittedvalues, prepend=True)).fit()

abline_plot(model_results=line_fit, ax=ax)



ax.set_title('Model Fit Plot')

ax.set_ylabel('Observed values')

ax.set_xlabel('Fitted values')



# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF Factor"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]

vif["features"] = predictors.columns

print(vif)



# Predict

lin_prediction = lin_m.predict(test_predictors).clip(0, None)



predictions_df['lin_model'] = lin_prediction

calculate_metrics(ted_test['comments'].values.reshape(-1, 1), lin_prediction.values.reshape(-1, 1), overall_metrics,'lin_model')
poisson_m = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()

print(poisson_m.summary())



fig, ax = plt.subplots()

ax.scatter(poisson_m.mu, response)

line_fit = sm.OLS(response, sm.add_constant(poisson_m.mu, prepend=True)).fit()

abline_plot(model_results=line_fit, ax=ax)



ax.set_title('Model Fit Plot')

ax.set_ylabel('Observed values')

ax.set_xlabel('Fitted values');



fig, ax = plt.subplots()



ax.scatter(poisson_m.mu, poisson_m.resid_pearson)

ax.set_title('Residual Dependence Plot')

ax.set_ylabel('Pearson Residuals')

ax.set_xlabel('Fitted values')
poiss_prediction = poisson_m.predict(test_predictors)



calculate_metrics(ted_test['comments'].values.reshape(-1, 1), poiss_prediction.values.reshape(-1, 1), overall_metrics,'poiss_model')
predictions_df['poiss_model'] = poiss_prediction

output_benchmark_results(predictions_df, models = ['lin_model', 'poiss_model'])
neg_bin_m = sm.GLM(response, predictors, family=sm.families.NegativeBinomial(sm.families.links.log)).fit()

print(neg_bin_m.summary())



fig, ax = plt.subplots()

ax.scatter(neg_bin_m.mu, response)

line_fit = sm.OLS(response, sm.add_constant(neg_bin_m.mu, prepend=True)).fit()

abline_plot(model_results=line_fit, ax=ax)



ax.set_title('Model Fit Plot')

ax.set_ylabel('Observed values')

ax.set_xlabel('Fitted values');



fig, ax = plt.subplots()



ax.scatter(neg_bin_m.mu, neg_bin_m.resid_pearson)

ax.set_title('Residual Dependence Plot')

ax.set_ylabel('Pearson Residuals')

ax.set_xlabel('Fitted values')
neg_bin_prediction = neg_bin_m.predict(test_predictors)



calculate_metrics(ted_test['comments'].values.reshape(-1, 1), neg_bin_prediction.values.reshape(-1, 1), overall_metrics,'neg_bin_model')
predictions_df['neg_bin_model'] = neg_bin_prediction

output_benchmark_results(predictions_df, models = ['lin_model', 'poiss_model', 'neg_bin_model'])
train_x_num = ted_train[['views', 'published_date', 'duration', 'languages', 'talk_age']].values

test_x_num = ted_test[['views', 'published_date', 'duration', 'languages', 'talk_age']].values

train_x = train_x_num

test_x = test_x_num



rf = RandomForestRegressor(bootstrap = True, max_depth = 60, max_features = 'auto', min_samples_leaf = 7, min_samples_split = 2, n_estimators = 206)



rf.fit(train_x, ted_train['comments'])



importances = rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(train_x.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(train_x.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(train_x.shape[1]), indices)

plt.xlim([-1, train_x.shape[1]])

plt.show()
rf_predictions = rf.predict(test_x)

predictions_df['rf_num_model'] = rf_predictions



calculate_metrics(ted_test['comments'].values.reshape(-1, 1), rf_predictions.reshape(-1, 1), overall_metrics,'rf_num_model')
output_benchmark_results(predictions_df, models = ['rf_num_model', 'poiss_model'])
rf = RandomForestRegressor(bootstrap = True, max_depth = 60, max_features = 'auto', min_samples_leaf = 7,\

                           min_samples_split = 5, n_estimators = 600)



rf.fit(tags_one_hot_train, ted_train['comments'])



importances = rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Plot the feature importances of the forest

plt.figure()

plt.title("Feature importances")

plt.bar(range(tags_one_hot_train.shape[1]), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(tags_one_hot_train.shape[1]), indices)

plt.xlim([-1, tags_one_hot_train.shape[1]])

plt.show()
rf_predictions = rf.predict(tags_one_hot_test)

predictions_df['rf_tags_model'] = rf_predictions



calculate_metrics(ted_test['comments'].values.reshape(-1, 1), rf_predictions.reshape(-1, 1), overall_metrics,'rf_tags_model')
output_benchmark_results(predictions_df, models = ['rf_num_model','rf_tags_model', 'poiss_model'])
N_COMPONENTS = 40

transcript_vectorizer = StemmedTfidfVectorizer(stop_words = 'english', max_df=0.3, min_df=0.01)



# Vectorizing for the whole dataset

svd = TruncatedSVD(n_components=N_COMPONENTS, n_iter=10, random_state=SEED)



tfidf = transcript_vectorizer.fit_transform(ted_train['transcript'])

transcript_train = svd.fit_transform(tfidf)



tfidf = transcript_vectorizer.transform(ted_test['transcript'])

transcript_test = svd.transform(tfidf)



index = np.arange(N_COMPONENTS)

plt.bar(index,svd.explained_variance_ratio_)

plt.title('Explained variance ratio chart')

plt.xlabel('Sungular value index')

plt.ylabel('Explained variance ratio')
ratings_vectorizer = RatingsVectorizer()



ratings = ratings_vectorizer.parse_ratings(ted_df['ratings'])

ratings_train = ratings[:TEST_OFFSET,:]

ratings_test = ratings[TEST_OFFSET:,]
train_x = np.concatenate([ratings_train, train_x_num, transcript_train, tags_one_hot_train], axis=1)

test_x = np.concatenate([ratings_test, test_x_num, transcript_test, tags_one_hot_test], axis=1)



xgb_params = {

         'learning_rate' : 0.001,

         'n_estimators': 2000,

         'max_depth': 10,

         'min_child_weight': 3,

         'gamma':1.5,                        

         'subsample':0.9,

         'objective': 'count:poisson',

         'seed': SEED,

         'eta':0.1

    }



regs = train_regressors(xgb_params, train_x, ted_train['comments'].values)

xgb_predictions = predict_with_regressors(regs, test_x)

        

calculate_metrics(ted_test['comments'].values.reshape(-1, 1), xgb_predictions.reshape(-1, 1), overall_metrics,'xgb_model')



print("\nModel Report")

print ("Poisson cross entropy loss: %f" % poisson_ce(ted_test['comments'].values.reshape(-1, 1), xgb_predictions.reshape(-1, 1)))

print(overall_metrics)



predictions_df['xgb_model'] = xgb_predictions

output_benchmark_results(predictions_df, models = ['xgb_model', 'rf_num_model', 'poiss_model'])
from keras.layers import Input, Dense, Dropout, concatenate, BatchNormalization

from keras.models import Model

from keras import backend as K, objectives

from keras.optimizers import Adam

from keras import regularizers

from keras.callbacks import EarlyStopping



LR = 0.005

DROPOUT_RATE = 0.9



# Input shape of the numerical variables

NUM_INPUT_SHAPE = (5,)



# The input shape of the tags

TAG_INPUT_SHAPE = (421,)



# The input shape of the tags

LSA_INPUT_SHAPE = (40,)



# Gelu activation function https://arxiv.org/pdf/1606.08415.pdf

# it performs well with high dropout rate

def gelu(x):

 """Gaussian Error Linear Unit.

 """

 return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))



num_input = Input(shape = NUM_INPUT_SHAPE)

tag_input = Input(shape = TAG_INPUT_SHAPE)

lsa_input = Input(shape = LSA_INPUT_SHAPE)



XT = Dense(1, activation=gelu, use_bias=False)(tag_input)

XN = Dense(1, activation=gelu, use_bias=False, kernel_regularizer=regularizers.l2(0.01))(num_input)

XLSA = Dense(1, activation=gelu, use_bias=False)(lsa_input)



XD = concatenate([XN, XT, XLSA], axis = 1)

XD = BatchNormalization()(XD)

XD = Dropout(DROPOUT_RATE)(XD)



XD = Dense(3, activation=gelu)(XD)

XD = BatchNormalization()(XD)

XD = Dropout(DROPOUT_RATE)(XD)



O = Dense(1, activation='softplus', init='glorot_normal')(XD)



model = Model([num_input, lsa_input, tag_input], O)



optimizer = Adam(lr = LR)

model.compile(optimizer = optimizer, loss = 'poisson')
train_x = np.concatenate([train_x_num, transcript_train, tags_one_hot_train], axis=1)

test_x = np.concatenate([test_x_num, transcript_test, tags_one_hot_test], axis=1)



tr_offset = train_x_num.shape[1]

tag_offset = tr_offset + transcript_train.shape[1]



nn_train_x, nn_val_x, nn_train_y, nn_val_y = train_test_split(train_x, ted_train['comments'], test_size=0.05, shuffle= True)



# Define simple early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)



model.fit([nn_train_x[:,:tr_offset], nn_train_x[:,tr_offset:tag_offset], nn_train_x[:,tag_offset:]], nn_train_y,

          validation_data=([nn_val_x[:,:tr_offset], nn_val_x[:,tr_offset:tag_offset], nn_val_x[:,tag_offset:]], nn_val_y),

          epochs=60,

          verbose=2,

          shuffle=True,

          batch_size=8,

          callbacks = [es])



nn_predictions = model.predict([test_x_num, transcript_test, tags_one_hot_test])



calculate_metrics(ted_test['comments'].values.reshape(-1, 1), nn_predictions.reshape(-1, 1), overall_metrics,'nn_model')

print(overall_metrics)



predictions_df['nn_model'] = nn_predictions

output_benchmark_results(predictions_df, models = ['xgb_model', 'nn_model', 'rf_num_model', 'poiss_model'])
####################################### Model Stacking ########################################

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

from sklearn.utils.metaestimators import _BaseComposition

from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline



def k_fold_train_stacking(model,train,y, n_fold = 5, seed = 42):

    """K-fold cross-training, so that we generate predictions

    with all model on all the data."""

    folds=StratifiedKFold(n_splits=n_fold,random_state=seed)

    train_pred = np.empty(shape=(train.shape[0]))

    iter = 0

    for train_indices,val_indices in folds.split(train,y):

        x_train,x_val=train.iloc[train_indices],train.iloc[val_indices]

        y_train,y_val=y.iloc[train_indices],y.iloc[val_indices]



        model.fit(X=x_train,y=y_train)

        train_pred[iter:(iter+x_val.shape[0])] = np.squeeze(model.predict(x_val))

        iter+=x_val.shape[0]

    return train_pred



def train_stacking(models,train,y):

    """Training the stacked models on the whole training set."""

    for model_name, model in models.items():

        model.fit(train, y)



def predict_stacking(models, test):

    m = len(models)

    preds = np.zeros((test.shape[0], m))

    iter = 0

    for model_name, model in models.items():

        preds[:,iter] = np.squeeze(model.predict(test))

        iter+=1

    return preds



def stack_models(models, train, y, n_fold = 5, seed = 42):

    """Generating the training set for the stacking model out of the predictions

    of the stacked models."""

    m = len(models)

    preds = np.zeros((train.shape[0], m))

    iter = 0

    for model_name, model in models.items():

        preds[:,iter] = k_fold_train_stacking(model,train, y, n_fold, seed)

        iter+=1

    

    return pd.DataFrame(preds)



class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        assert isinstance(X, pd.DataFrame)



        try:

            return X[self.columns]

        except KeyError:

            cols_error = list(set(self.columns) - set(X.columns))

            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

            

class GLMTransformer(_BaseComposition, TransformerMixin):



    def __init__(self, formula, dmatrices):

        self.formula = formula

        self.dmatrices = dmatrices

        

    def fit(self, X, y = None):

        return self



    def transform(self, X):

        d = dict()

        response, predictors = self.dmatrices(self.formula, X, return_type='dataframe')

        d['response'] = response

        d['predictors'] = predictors

        return d



class GLMRegressor(BaseEstimator, RegressorMixin):

    """ A sklearn-style wrapper for formula-based statsmodels regressors """

    def __init__(self, model_class, exp_family):

        self.model_class = model_class

        self.exp_family = exp_family

        

    def fit(self, X, y=None):

        self.model_ = self.model_class(X['response'], X['predictors'], family=self.exp_family)

        self.results_ = self.model_.fit()

        

    def predict(self, X):

        return self.results_.predict(X['predictors'])

    

class NNTransformer(_BaseComposition, TransformerMixin):

    """The same use as FeatureUnion. It consolidates the input needed

    for NN."""

    def __init__(self, transformers):

        self.transformers = transformers



    def fit(self, X, y = None):

        for n, t in self.transformers:

            t.fit(X, y)



        return self



    def transform(self, X):

        return { n: t.transform(X) for n, t in self.transformers }



    def get_params(self, deep=True):

        return self._get_params('transformers', deep=deep)



    def set_params(self, **kwargs):

        self._set_params('transformers', **kwargs)

        return self

    

class NNSemanticModel(BaseEstimator):

    """This NN model takes as an input only the semantic features.

    It is resambling Generalized linear regression where the first

    layer has no biases and we add one layer to give some depth

    and capture non-linearities and interaction effects."""

        

    def __init__(self, optimizer = 'sgd'):

        self.optimizer = optimizer # an example of a tunable hyperparam



    def fit(self, X, y):

        

        DROPOUT_RATE = 0.9



        # Gelu activation function https://arxiv.org/pdf/1606.08415.pdf

        # it performs well with high dropout rate

        def gelu(x):

            """Gaussian Error Linear Unit."""

            return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))



        lsa_input = Input(name = 'lsa_t', shape = X['lsa_t'].shape[1:])

        tag_input = Input(name = 'tags_t', shape = X['tags_t'].shape[1:])



        XT = Dense(1, activation=gelu)(tag_input)

        XLSA = Dense(1, activation=gelu)(lsa_input)



        XD = concatenate([XT, XLSA], axis = 1)

        XD = BatchNormalization()(XD)

        XD = Dropout(DROPOUT_RATE)(XD)



        XD = Dense(3, activation=gelu)(XD)

        XD = BatchNormalization()(XD)

        XD = Dropout(DROPOUT_RATE)(XD)



        O = Dense(1, activation='softplus', init='glorot_normal')(XD)



        self.model = Model([lsa_input, tag_input], O)



        self.model.compile(self.optimizer, 'poisson')

              

        tag_offset = X['lsa_t'].shape[1]

        

        # The matrix 'tags_t' so far was sparse, but now we convert it to dense in order to use it as an

        # input for training.

        cx = np.concatenate([X['lsa_t'], X['tags_t'].toarray()], axis=1)



        nn_train_x, nn_val_x, nn_train_y, nn_val_y = train_test_split(cx, y, test_size=0.05, shuffle= False)



        # Define simple early stopping

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)



        self.model.fit([nn_train_x[:,:tag_offset], nn_train_x[:,tag_offset:]], nn_train_y,

                  validation_data=([nn_val_x[:,:tag_offset], nn_val_x[:,tag_offset:]], nn_val_y),

                  epochs=60,

                  verbose=0,

                  shuffle=True,

                  batch_size=8,

                  callbacks = [es])



        return self



    def predict(self, X):

        return self.model.predict([X['lsa_t'], X['tags_t']])

    

class NNNumericModel(BaseEstimator):

    """This NN model takes as an input only the numerical features

    It has a also L2 regularization since we've observed high multicolinearity

    among the predictors. It is resambling Generalized linear regression where

    the first layer has no biases and we add one layer to give some depth

    and capture non-linearities and interaction effects.

    """

    

    def __init__(self, optimizer = 'sgd'):

        self.optimizer = optimizer # an example of a tunable hyperparam



    def fit(self, X, y):

        

        DROPOUT_RATE = 0.9



        # Gelu activation function https://arxiv.org/pdf/1606.08415.pdf

        # it performs well with high dropout rate

        def gelu(x):

            """Gaussian Error Linear Unit."""

            return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))



        num_input = Input(name = 'num_cs', shape = X['num_cs'].shape[1:])



        XN = Dense(1, activation=gelu)(num_input)



        XD = BatchNormalization()(XN)

        XD = Dense(3, activation=gelu)(XD)

        XD = BatchNormalization()(XD)

        XD = Dropout(DROPOUT_RATE)(XD)



        O = Dense(1, activation='softplus', init='glorot_normal')(XD)



        self.model = Model(num_input, O)

        self.model.compile(self.optimizer, 'poisson')      



        nn_train_x, nn_val_x, nn_train_y, nn_val_y = train_test_split(X['num_cs'], y, test_size=0.05, shuffle= False)

        

        # Define simple early stopping

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)



        self.model.fit(nn_train_x, nn_train_y,

                  validation_data=(nn_val_x, nn_val_y),

                  epochs=60,

                  verbose=0,

                  shuffle=True,

                  batch_size=8,

                  callbacks = [es])



        return self



    def predict(self, X):

        return self.model.predict(X['num_cs'])



class XGBoostEarlyStop(BaseEstimator):

    """Wrapper for XGBoost which extends the the estimator with

    early stopping framework."""

    

    def __init__(self, early_stopping_rounds=5, val_size=0.1, 

                 eval_metric='mae', **estimator_params):

        self.early_stopping_rounds = early_stopping_rounds

        self.val_size = val_size

        self.eval_metric = eval_metric

        self.estimator = xgb.XGBRegressor()

        self.set_params(**estimator_params)



    def set_params(self, **params):

        return self.estimator.set_params(**params)



    def get_params(self, **params):

        return self.estimator.get_params()



    def fit(self, X, y):

        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size)



        self.estimator.fit(x_train, y_train, 

                           early_stopping_rounds=self.early_stopping_rounds, 

                           eval_metric=self.eval_metric, eval_set=[(x_val, y_val)],verbose=False)

        return self



    def predict(self, X):

        return self.estimator.predict(X)

    

class TranscriptLSA(BaseEstimator, TransformerMixin):

    """We use pre-trained TF-IDF vectorizer and SVD decomposer for

    transforming the transcript."""

    def __init__(self, tfidf_vectorizer, transcript_svd):

        self.tfidf_vectorizer = tfidf_vectorizer

        self.transcript_svd = transcript_svd

        

    def fit(self, X, y=None):

        return self



    def transform(self, X):

        tfidf = self.tfidf_vectorizer.transform(X)

        return self.transcript_svd.transform(tfidf)

    

class TagsOneHot(BaseEstimator, TransformerMixin):

    """We use pre-trained count vectorizer to convert the tags into

    one hot encoding."""

    def __init__(self, count_vectorizer):

        self.count_vectorizer = count_vectorizer



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        return self.count_vectorizer.transform(X)
############### GLM Models ################



formula ="""comments ~ views + talk_age + duration + languages + I(views**2) + I(talk_age**2) + duration * views + talk_age * views"""



poiss_glm = Pipeline([

    ('glm_t', GLMTransformer(formula, dmatrices)),

    ('glm_r', GLMRegressor(sm.GLM,sm.families.Poisson()))

])



neg_bin_glm = Pipeline([

    ('glm_t', GLMTransformer(formula, dmatrices)),

    ('glm_r', GLMRegressor(sm.GLM,sm.families.NegativeBinomial(sm.families.links.log)))

])



########### Random forest ###############



rm_f_num_params = {

'bootstrap': True,

'max_depth': 60,

'max_features':'auto',

'min_samples_leaf': 7,

'min_samples_split': 2,

'n_estimators': 206

}



rm_num_forest = Pipeline([

    ('rm_f_cs', ColumnSelector(columns=['views', 'published_date', 'duration', 'num_speaker', 'talk_age'])),

    ('rm_f_r', RandomForestRegressor(**rm_f_num_params))

])



rm_f_tag_params = {

'bootstrap': True,

'max_depth': 60,

'max_features':'auto',

'min_samples_leaf': 7,

'min_samples_split': 5,

'n_estimators': 600

}



rm_tag_forest = Pipeline([

    ('tags_cs', ColumnSelector(columns='tags_text')),

    ('one_hot_t', TagsOneHot(count_vectorizer)),

    ('rm_f_r', RandomForestRegressor(**rm_f_tag_params))

])



####################### XGB  #####################



xg_boost_tags = Pipeline([

            ('tags_cs', ColumnSelector(columns='tags_text')),

            ('one_hot_t', TagsOneHot(count_vectorizer)),

            ('xgb_r', XGBoostEarlyStop(eval_metric = 'poisson-nloglik', **xgb_params))

        ])



xg_boost_lsa = Pipeline([

            ('lsa_cs', ColumnSelector(columns='transcript')),

            ('tfidf_t', TranscriptLSA(transcript_vectorizer, svd)),

            ('xgb_r', XGBoostEarlyStop(eval_metric = 'poisson-nloglik', **xgb_params))

        ])



xg_boost_num = Pipeline([

            ('num_cs', ColumnSelector(columns=['views', 'languages', 'duration', 'talk_age'])),

            ('xgb_r', XGBoostEarlyStop(eval_metric = 'poisson-nloglik', **xgb_params))

        ])



################## Neural Network ###################





nn_r_num = Pipeline([

    ('fu_t', NNTransformer([

        ('num_cs', Pipeline([

            ('rm_f_cs', ColumnSelector(columns=['views', 'languages', 'duration', 'talk_age']))

        ])),

        ('lsa_t', Pipeline([

            ('lsa_cs', ColumnSelector(columns='transcript')),

            ('tfidf_t', TranscriptLSA(transcript_vectorizer, svd))

        ])),

        ('tags_t', Pipeline([

            ('tags_cs', ColumnSelector(columns='tags_text')),

            ('one_hot_t', TagsOneHot(count_vectorizer))

        ]))

    ])),

    ('nn_r_num', NNNumericModel(optimizer = Adam(lr = 0.005)))

])



nn_r_sem = Pipeline([

    ('fu_t', NNTransformer([

        ('num_cs', Pipeline([

            ('rm_f_cs', ColumnSelector(columns=['views', 'languages', 'duration', 'talk_age']))

        ])),

        ('lsa_t', Pipeline([

            ('lsa_cs', ColumnSelector(columns='transcript')),

            ('tfidf_t', TranscriptLSA(transcript_vectorizer, svd))

        ])),

        ('tags_t', Pipeline([

            ('tags_cs', ColumnSelector(columns='tags_text')),

            ('one_hot_t', TagsOneHot(count_vectorizer))

        ]))

    ])),

    ('nn_r_sem', NNSemanticModel(optimizer = Adam(lr = 0.005)))

])





models = {

    "nn_r_sem": nn_r_sem,

    "nn_r_num": nn_r_num,

    "poiss_glm": poiss_glm,

    "neg_bin_glm": neg_bin_glm,

    "rm_num_forest": rm_num_forest,

    "rm_tag_forest": rm_tag_forest,

    "xg_boost_num": xg_boost_num,

    "xg_boost_tags": xg_boost_tags,

    "xg_boost_lsa": xg_boost_lsa

}
# Generate the stacking data

stack_train = stack_models(models, ted_train, ted_train['comments'], 5, SEED)

stack_test = predict_stacking(models, ted_test)
stack_train.head()
regs = train_regressors(xgb_params, stack_train.values, ted_train['comments'].values)

xgb_stacked_predictions = predict_with_regressors(regs, stack_test)



calculate_metrics(ted_test['comments'].values.reshape(-1, 1), xgb_stacked_predictions.reshape(-1, 1), overall_metrics,'xgb_stacked')



print("\nModel Report")

print ("Poisson cross entropy loss: %f" % poisson_ce(ted_test['comments'].values.reshape(-1, 1), xgb_stacked_predictions.reshape(-1, 1)))

print(overall_metrics)



predictions_df['xgb_stacked_model'] = xgb_stacked_predictions

output_benchmark_results(predictions_df, models = ['xgb_model', 'xgb_stacked_model', 'rf_num_model', 'poiss_model', 'nn_model'])
x_train = np.concatenate([stack_train, ratings_train, train_x_num], axis=1)

x_test = np.concatenate([stack_test, ratings_test, test_x_num], axis=1)



regs = train_regressors(xgb_params, x_train, ted_train['comments'].values)

aug_xgb_stacked_predictions = predict_with_regressors(regs, x_test)



calculate_metrics(ted_test['comments'].values.reshape(-1, 1), aug_xgb_stacked_predictions.reshape(-1, 1), overall_metrics,'aug_xgb_stacked')



print("\nModel Report")

print ("Poisson cross entropy loss: %f" % poisson_ce(ted_test['comments'].values.reshape(-1, 1), aug_xgb_stacked_predictions.reshape(-1, 1)))

print(overall_metrics)



predictions_df['aug_xgb_stacked_model'] = xgb_stacked_predictions

output_benchmark_results(predictions_df, models = ['xgb_model', 'xgb_stacked_model', 'aug_xgb_stacked_model', 'rf_num_model', 'poiss_model', 'nn_model'])