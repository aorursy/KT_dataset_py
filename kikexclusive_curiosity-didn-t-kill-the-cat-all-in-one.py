import os
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


from IPython.display import display, HTML
# Table printing large
plt.rcParams['figure.figsize'] = (15, 7)
pd.set_option("display.max_columns", 400)
pd.options.display.max_colwidth = 250
pd.set_option("display.max_rows", 100)
# High defition plots
%config InlineBackend.figure_format = 'retina'
sns.set()
base_path_data = "../input/"
print(os.listdir("../input"))
df_train = pd.read_csv("../input/train/train.csv")
df_test = pd.read_csv("../input/test/test.csv")

print(f"train.csv shape is {df_train.shape}")
print(f"test.csv shape is {df_test.shape}")
print("Basic statistics of the train set")
display(df_train.describe(include="all").T)

print("Basic statistics of the test set")
display(df_test.describe(include="all").T)
df_train.sample(3)
df_breed_labels = pd.read_csv(os.path.join(base_path_data, "breed_labels.csv"))
df_color_labels = pd.read_csv(os.path.join(base_path_data, "color_labels.csv"))
df_state_labels = pd.read_csv(os.path.join(base_path_data, "state_labels.csv"))
print(f"breed_labels.csv shape is {df_breed_labels.shape}")
df_breed_labels.sample(5)
print(f"color_labels.csv shape is {df_color_labels.shape}")
df_color_labels.sample(5)
print(f"state_labels.csv shape is {df_state_labels.shape}")
df_state_labels.sample(5)
# sentiment
with open(os.path.join(base_path_data, "train_sentiment", "048cd8bc0.json")) as f:
    data = json.load(f)

pprint(data)
# metadata
with open(os.path.join(base_path_data, "train_metadata", "000fb9572-6.json")) as f:
    data = json.load(f)

pprint(data)
set(df_train.RescuerID.unique()).intersection(set(df_test.RescuerID.unique()))
common_names = list(set(df_train.Name.unique()).intersection(set(df_test.Name.unique())))
len(common_names)
common_names[:10]
fig, axes = plt.subplots(8,3, figsize=(15, 20))
images_train = os.listdir("../input/train_images/")
fig.suptitle("24 random pet images")
images_train = np.random.choice(images_train, 24)
for i, img in enumerate(images_train):
    image = Image.open("../input/train_images/" + img)
    pet_id = img.split("-")[0]
    axes[i//3, i%3].imshow(image)
    axes[i//3, i%3].grid(False)
    axes[i//3, i%3].set_axis_off()
    axes[i//3, i%3].set_title("Name: {}\nAdoptionSpeed: {}".format(*list(map(str, df_train[df_train.PetID==pet_id][["Name", "AdoptionSpeed"]].values.tolist()[0]))))
100 * df_train.isnull().sum() / len(df_train)
100 * df_test.isnull().sum() / len(df_test)
df_train[df_train.duplicated(keep=False)].shape
cols = [col for col in df_train.columns.ravel() if col not in ["PetID", "Name", "Description", "AdoptionSpeed"]]
dups = df_train[df_train[cols].duplicated(keep=False)].sort_values(by=cols)
print(f"Shape of matrix with duplicated rows not considering petid, name nor description {dups.shape}")
dups.head(2)
dups = df_test[df_test[cols].duplicated(keep=False)].sort_values(by=cols)
print(f"Shape of matrix with duplicated rows not considering petid, name nor description {dups.shape}")
dups.head(2)
def plot_correlation_matrix(df):
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(20, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
plot_correlation_matrix(df_train)
plot_correlation_matrix(df_test)
plot_correlation_matrix(df_train[df_train.Type==1].drop(columns="Type"))
plot_correlation_matrix(df_test[df_test.Type==1].drop(columns="Type"))
plot_correlation_matrix(df_test[df_test.Type==2].drop(columns="Type"))
plot_correlation_matrix(df_test[df_test.Type==2].drop(columns="Type"))
feats = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1',
       'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated',
       'Dewormed', 'Sterilized', 'Health', 'Quantity', 'Fee', 'State',
       'VideoAmt', 'PhotoAmt', 'AdoptionSpeed']
def plot_distribution(df, feat):
    fig, ax = plt.subplots(figsize=(17,5))
    sns.countplot(df_train[feat])
    ax.xaxis.set_label_text(feat,fontdict= {'size':14})
    ax.yaxis.set_label_text("Count",fontdict= {'size':14})
    plt.show()
    print(f"Total number of unique values for feature {feat} is {df[feat].nunique()}")
    print(100 * df_train[feat].value_counts(normalize=True, dropna=False))
plot_distribution(df_train, "AdoptionSpeed")
for feat in feats:
    print(f"Univariate distribution of feature {feat}")
    plot_distribution(df_train, feat)
for cat_c in feats:
    if cat_c == "AdoptionSpeed": continue
    nunique = df_train[cat_c].nunique()
    print(f'{cat_c}:')
    print(f'{nunique} unique values')
    if nunique < 50:
        print(f'\nValues:\n{100 * df_train[cat_c].value_counts(normalize=True, dropna=False)}')
      
        # Countplot
        fig, ax = plt.subplots(figsize=(12,4))
        sns.countplot(x=cat_c, hue="AdoptionSpeed", data=df_train, orient="h")
        #ax.text(5,5,"Boxplot After removing outliers", fontsize=18, color="r", ha="center", va="center")
        ax.xaxis.set_label_text(cat_c,fontdict= {'size':14})
        ax.yaxis.set_label_text("Count",fontdict= {'size':14})
        plt.xticks(rotation=90)
        plt.show()
    else:
        # Distplot to see the distribution after outliers have been removed
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12,4))
        for aspeed in range(5):
            sns.distplot(df_train[df_train.AdoptionSpeed == aspeed][cat_c].dropna(), hist=False, rug=False, label="AdoptionSpeed = {}".format(aspeed))
        ax.xaxis.set_label_text(cat_c,fontdict= {'size':14})
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()
c = df_train.corr().abs()
s = c.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
so = pd.DataFrame(so[20:]).reset_index()
so.columns = ["var1", "var2", "corr"]
so = so[so["corr"] > 0.3]
so

for k, v in so.iterrows():
    print(f'{v["var1"]} vs {v["var2"]} ({v["corr"]})')
    if (df_train[v["var1"]].nunique() < 50) and (df_train[v["var2"]].nunique() < 50):
        fig, ax = plt.subplots(figsize=(12,4))
        sns.countplot(x=v["var1"], hue=v["var2"], data=df_train, orient="h")
        #ax.text(5,5,"Boxplot After removing outliers", fontsize=18, color="r", ha="center", va="center")
        ax.xaxis.set_label_text(v["var1"], fontdict= {'size':14})
        ax.yaxis.set_label_text("Count",fontdict= {'size':14})
        plt.xticks(rotation=90)
        plt.show()
    else:
        # Distplot to see the distribution after outliers have been removed
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12,4))
        sns.scatterplot(x=v["var1"], y=v["var2"], data=df_train, hue="AdoptionSpeed")
        ax.xaxis.set_label_text(v["var1"], fontdict= {'size':14})
        #plt.xticks(rotation=90)
        plt.show()
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import scipy as sp
from sklearn import linear_model
from functools import partial
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.metrics import confusion_matrix as sk_cmatrix
from collections import Counter
import json
import lightgbm as lgb
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
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
    conf_mat = confusion_matrix(rater_a, rater_b,
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
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']
def rmse(actual, predicted):
    return mean_squared_error(actual, predicted)**0.5
train_desc = df_train.Description.fillna("none").values
test_desc = df_test.Description.fillna("none").values

tfv = TfidfVectorizer(min_df=3,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')
    
# Fit TFIDF
tfv.fit(list(train_desc) + list(test_desc))
X =  tfv.transform(train_desc)
X_test = tfv.transform(test_desc)


svd = TruncatedSVD(n_components=180)
svd.fit(X)
X = svd.transform(X)

X_test = svd.transform(X_test)

train_desc = df_train.Description.fillna("none").values
test_desc = df_test.Description.fillna("none").values

svd_n_components = 200

tfv = TfidfVectorizer(min_df=2,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        )
    
# Fit TFIDF
tfv.fit(list(train_desc))
X =  tfv.transform(train_desc)
X_test = tfv.transform(test_desc)

svd = TruncatedSVD(n_components=svd_n_components)
svd.fit(X)
print(svd.explained_variance_ratio_.sum())
print(svd.explained_variance_ratio_)
X = svd.transform(X)
X = pd.DataFrame(X, columns=['svd_{}'.format(i) for i in range(svd_n_components)])
df_train = pd.concat((df_train, X), axis=1)
X_test = svd.transform(X_test)
X_test = pd.DataFrame(X_test, columns=['svd_{}'.format(i) for i in range(svd_n_components)])
df_test = pd.concat((df_test, X_test), axis=1)



# Thanks to beloruk1

def readFile(fn):
    file = '../input/train_sentiment/'+fn['PetID']+'.json'
    if os.path.exists(file):
        with open(file) as data_file:    
            data = json.load(data_file)  

        df = json_normalize(data)
        mag = df['documentSentiment.magnitude'].values[0]
        score = df['documentSentiment.score'].values[0]
        return pd.Series([mag,score],index=['mag','score']) 
    else:
        return pd.Series([0,0],index=['mag','score'])
    
def readTestFile(fn):
    file = '../input/test_sentiment/' + fn['PetID'] + '.json'
    if os.path.exists(file):
        with open(file) as data_file:    
            data = json.load(data_file)  

        df = json_normalize(data)
        mag = df['documentSentiment.magnitude'].values[0]
        score = df['documentSentiment.score'].values[0]
        return pd.Series([mag,score],index=['mag','score']) 
    else:
        print(f'{file} does not exist')
        return pd.Series([0,0],index=['mag','score'])
    
df_train[['SentMagnitude', 'SentScore']] = df_train[['PetID']].apply(lambda x: readFile(x), axis=1)
df_test[['SentMagnitude', 'SentScore']] = df_test[['PetID']].apply(lambda x: readTestFile(x), axis=1)
# Not needed, as there's no overlap between RescuerID in train set and test set
#lbl_enc = LabelEncoder()
#lbl_enc.fit(df_train.RescuerID.values.tolist() + df_test.RescuerID.values.tolist())
#df_train.RescuerID = lbl_enc.transform(df_train.RescuerID.values)
#df_test.RescuerID = lbl_enc.transform(df_test.RescuerID.values)
y = df_train.AdoptionSpeed
train = np.hstack((df_train.drop(['Name', 'Description', 'PetID', 'AdoptionSpeed', 'RescuerID'], axis=1).values, X))
test = np.hstack((df_test.drop(['Name', 'Description', 'PetID', 'RescuerID'], axis=1).values, X_test))
df_train.columns.ravel()


target = df_train['AdoptionSpeed']
train_id = df_train['PetID']
test_id = df_test['PetID']
df_train.drop(['Name', 'Description', 'PetID', 'AdoptionSpeed', 'RescuerID'], axis=1, inplace=True, errors='ignore')
df_test.drop(['Name', 'Description', 'PetID', 'RescuerID'], axis=1, inplace=True, errors='ignore')
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    fold_splits = kf.split(train, target)
    cv_scores = []
    qwk_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0], 5))
    all_coefficients = np.zeros((5, 4))
    feature_importance_df = pd.DataFrame()
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/5')
        if isinstance(train, pd.DataFrame):
            dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        else:
            dev_X, val_X = train[dev_index], train[val_index]
            dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, importances, coefficients, qwk = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        all_coefficients[i-1, :] = coefficients
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            qwk_scores.append(qwk)
            print(label + ' cv score {}: RMSE {} QWK {}'.format(i, cv_score, qwk))
        fold_importance_df = pd.DataFrame()
        fold_importance_df['feature'] = train.columns.values
        fold_importance_df['importance'] = importances
        fold_importance_df['fold'] = i
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)        
        i += 1
    print('{} cv RMSE scores : {}'.format(label, cv_scores))
    print('{} cv mean RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std RMSE score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv QWK scores : {}'.format(label, qwk_scores))
    print('{} cv mean QWK score : {}'.format(label, np.mean(qwk_scores)))
    print('{} cv std QWK score : {}'.format(label, np.std(qwk_scores)))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label,
               'train': pred_train, 'test': pred_full_test,
                'cv': cv_scores, 'qwk': qwk_scores,
               'importance': feature_importance_df,
               'coefficients': all_coefficients}
    return results

params = {'application': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'num_leaves': 80,
          'max_depth': 9,
          'learning_rate': 0.01,
          'bagging_fraction': 0.85,
          'feature_fraction': 0.8,
          'min_split_gain': 0.01,
          'min_child_samples': 150,
          'min_child_weight': 0.1,
          'verbosity': -1,
          'data_random_seed': 3,
          'early_stop': 100,
          'verbose_eval': 100,
          'num_rounds': 10000}

def runLGB(train_X, train_y, test_X, test_y, test_X2, params):
    print('Prep LGB')
    d_train = lgb.Dataset(train_X, label=train_y)
    d_valid = lgb.Dataset(test_X, label=test_y)
    watchlist = [d_train, d_valid]
    print('Train LGB')
    num_rounds = params.pop('num_rounds')
    verbose_eval = params.pop('verbose_eval')
    early_stop = None
    if params.get('early_stop'):
        early_stop = params.pop('early_stop')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)
    print('Predict 1/2')
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    optR = OptimizedRounder()
    optR.fit(pred_test_y, test_y)
    coefficients = optR.coefficients()
    pred_test_y_k = optR.predict(pred_test_y, coefficients)
    print("Valid Counts = ", Counter(test_y))
    print("Predicted Counts = ", Counter(pred_test_y_k))
    print("Coefficients = ", coefficients)
    qwk = quadratic_weighted_kappa(test_y, pred_test_y_k)
    print("QWK = ", qwk)
    print('Predict 2/2')
    pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    return pred_test_y.reshape(-1, 1), pred_test_y2.reshape(-1, 1), model.feature_importance(), coefficients, qwk

results = run_cv_model(df_train, df_test, target, runLGB, params, rmse, 'lgb')

imports = results['importance'].groupby('feature')['feature', 'importance'].mean().reset_index()
imports.sort_values('importance', ascending=False)
optR = OptimizedRounder()
coefficients_ = np.mean(results['coefficients'], axis=0)
print(coefficients_)
train_predictions = [r[0] for r in results['train']]
train_predictions = optR.predict(train_predictions, coefficients_).astype(int)
Counter(train_predictions)
optR = OptimizedRounder()
test_predictions = [r[0] for r in results['test']]
test_predictions = optR.predict(test_predictions, coefficients_).astype(int)
Counter(test_predictions)
pd.DataFrame(sk_cmatrix(target, train_predictions), index=list(range(5)), columns=list(range(5)))
quadratic_weighted_kappa(target, train_predictions)

rmse(target, [r[0] for r in results['train']])

submission = pd.DataFrame({'PetID': test_id, 'AdoptionSpeed': test_predictions})
submission.head()
submission.to_csv('submission.csv', index=False)
