# Get essential packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns



%matplotlib inline



# Sci-kit Learn Essentials

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer, OneHotEncoder, LabelEncoder

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score

from sklearn.impute import SimpleImputer

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



# Models

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier



# Silence warnings

import warnings

warnings.filterwarnings('ignore')
# Need to drop RISK_MM due to data leakage

# Set date to index and sort

df = pd.read_csv('../input/weatherAUS.csv', parse_dates=['Date'], index_col=0).drop('RISK_MM', axis=1).sort_index()

df.head()
df.info()
df.index.min(), df.index.max()
df.index.max() - df.index.min()
df.index.value_counts()
df.loc['2013-12-22']
# Plot Min/Max Temps by Location

df.groupby('Location')['MinTemp'].plot(figsize=(17, 8))

plt.title('Min Temperature in Australia Colored by Region')

plt.xlabel('Date')

plt.ylabel('Temperature (Celsius)')

plt.show()
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(df.corr(),

            linewidths=0.1,

            vmax=1.0, 

            square=True, 

            cmap=colormap, 

            linecolor='white', 

            annot=True)

plt.show()
class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with mean of column.

        """

    

    def fit(self, X, y=None):

        """ Imputes categoricals with 'O' and numericals with the mean. """

        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        """ Executes transformation from fit. """

        return X.fillna(self.fill)



    

df_nl = df.drop('Location', axis=1)

cat_cols = list(df_nl.select_dtypes(include=object, exclude=None).columns)



pipe = make_pipeline(

    DataFrameImputer(),

    StandardScaler(),

    PCA(n_components=3)   

)



X_red = pipe.fit_transform(pd.get_dummies(df_nl))
X_red
# Plot 2D PCA projection

cmap = plt.get_cmap('jet', 20)

cmap.set_under('gray')

fig, ax = plt.subplots(figsize=(20, 20))

cax = ax.scatter(X_red[:, 0], X_red[:, 1], 

                 c=df.Location.astype('category').cat.codes, 

                 s=10, 

                 cmap=cmap, 

                 alpha=0.7)

fig.colorbar(cax, extend='min')

plt.title('2D PCA Projection of Data Colored by Location')

plt.show()
from mpl_toolkits.mplot3d import Axes3D



# Plot 3D PCA Projection

cmap = plt.get_cmap('jet', 20)

cmap.set_under('gray')

fig = plt.figure(figsize=(20, 20))

ax = fig.add_subplot(111, projection='3d')

cax = ax.scatter(X_red[:, 0], X_red[:, 1], X_red[:, 2], 

                 c=df.Location.astype('category').cat.codes, 

                 s=10, 

                 cmap=cmap, 

                 alpha=0.7)

fig.colorbar(cax, extend='min')

plt.title('3D PCA Projection of Data Colored by Location')

plt.show()
# Get X, y

X, y = df.drop('RainTomorrow', axis=1), df.RainTomorrow



# Recast y as int.  1: 'Yes', 0: 'No'

y = (y == 'Yes').astype('int64')
y.hist()

plt.show()
y.value_counts()/len(y)
# Add seasonality

X['Month'] = X.index.month_name()



# Get initial columns

cat_cols = list(X.select_dtypes(include=object).columns)

num_cols = [c for c in X.columns if c not in cat_cols]



# Get intraday columns

am_cols = sorted([c for c in num_cols if c.endswith('9am')])

pm_cols = sorted([c for c in num_cols if c.endswith('3pm')])



# Add intraday delta

for am_col, pm_col in zip(am_cols, pm_cols):

    X[am_col[:-3] + '_delta'] = X[pm_col] - X[am_col]

    

# Add max temp - min temp delta

X['MinMaxTemp_delta'] = X.MinTemp - X.MaxTemp

    

# Get train test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()
class DataFrameSelector(BaseEstimator, TransformerMixin):

    """ A DataFrame transformer that provides column selection. """

    

    def __init__(self, columns=[]):

        """ Get selected columns. """

        self.columns = columns

        

    def transform(self, X):

        """ Returns df with selected columns. """

        return X[self.columns].copy()

    

    def fit(self, X, y=None):

        """ Do nothing operation. """

        return self





# -- Get Pipelines --



# Get categoricals and numericals

cat_cols = list(X.select_dtypes(include=object).columns)

num_cols = [c for c in X.columns if c not in cat_cols]



# Fit numerical pipeline

num_pipeline = make_pipeline(

    DataFrameSelector(num_cols),

    SimpleImputer(strategy='median'),

    StandardScaler()

)



# Fit categorical pipeline

cat_pipeline = make_pipeline(

    DataFrameSelector(cat_cols),

    SimpleImputer(strategy='most_frequent'),

    OneHotEncoder(handle_unknown='ignore', sparse=False)

)



# Union pipelines

full_preproc = FeatureUnion(transformer_list=[

    ("cat_pipeline", cat_pipeline),

    ("num_pipeline", num_pipeline)

])



# (Optional) Reduce dimension to contain nearly all explained variance

# This will lower memory consumption and speed up training time

# at a small performance cost

red_preproc = make_pipeline(

    full_preproc,

    # Keep 95% of explained variance

    PCA(0.95)

)
X_train = full_preproc.fit_transform(X_train)

X_train.shape
X_test = full_preproc.transform(X_test)

X_test.shape
from pprint import pprint

models = {

    'LogisticRegression': LogisticRegression(),

    'RandomForestClassifier': RandomForestClassifier(),

    'AdaBoostClassifier': AdaBoostClassifier(),

    # Use log loss so we can get probability predictions

    'SGDClassifier': SGDClassifier(loss='log'),

    'GradientBoostingClassifier': GradientBoostingClassifier(),

    'GaussianNB': GaussianNB(),

}



pprint(models)
# Some useful parameters which will come in handy later on

ntrain = X_train.shape[0]

ntest = X_test.shape[0]

SEED = 0 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

kf = KFold(n_splits=NFOLDS, random_state=SEED)



def get_oof(clf, x_train=X_train, y_train=y_train, x_test=X_test):

    """ Get's out of fold predictions for a classifier. 

        Credit: https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python

        Slightly modified to handle sklearn 0.20.3

    """

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.fit(x_tr, y_tr)

        # Use predict_proba and get confidence of model predicting 1 = 'Yes'

        oof_train[test_index] = clf.predict_proba(x_te)[:, 1]

        # For test set, we use each of the 5 folds

        oof_test_skf[i, :] = clf.predict_proba(x_test)[:, 1]

    # and average predictions

    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train, oof_test
def build_pred_data(clf_dict=models):

    """ Get's oof predictions for every classifier in a dictionary of models. """

    X_train, X_test = pd.DataFrame(), pd.DataFrame()

    

    # Columns will be model predictions 

    # labeled according to model

    for key, clf in clf_dict.items():

        X_train[key], X_test[key] = get_oof(clf)

        

    # Lastly, MinMax Scale to avoid bias

    scaler = MinMaxScaler()

    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test



X_train_f, X_test_f = build_pred_data() 
# Examine train data 

X_train_f.head()
# Examine test data

X_test_f.head()
colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Model Probabilities', y=1.05, size=15)

sns.heatmap(X_train_f.corr(), 

            linewidths=0.1, 

            vmax=1.0, 

            square=True, 

            cmap=colormap, 

            linecolor='white', 

            annot=True)

plt.show()
def plot_2d_space(X_in=X_train_f, y=y_train, upsample=False):

    X = PCA(n_components=2).fit_transform(X_in)

    plt.figure(figsize=(15, 8))

    plt.scatter(X[(y == 1), 0], X[(y == 1), 1], 

                c='b', 

                label='Yes Rain Tomorrow',

                alpha=0.1,

                s=20)

    plt.scatter(X[(y == 0), 0], X[(y == 0), 1], 

                c='r', 

                label='No Rain Tomorrow',

                alpha=0.1,

                s=20)

    t = 'Model Probability PCA Projection 2D'

    if upsample: t += ' - Upsample'

    plt.title(t)

    plt.xlabel('PC1')

    plt.ylabel('PC2')

    plt.legend()

    plt.show()

plot_2d_space()
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_train_f_sm, y_train_sm = smote.fit_sample(X_train_f, y_train)

# Ensure columns are homogeneous - imblearn recasts as numpy array

X_train_f_sm = pd.DataFrame(X_train_f_sm, columns=X_train_f.columns)
plt.hist(y_train_sm)

plt.show()
plot_2d_space(X_train_f_sm, y_train_sm, upsample=True)
def get_cv_model(X, y):

    """ Given a dataset, performs cross-validation on an XGBoost model 

        and returns best one. Scoring: F1-Score. """

    # A parameter grid for XGBoost

    params = {

            'min_child_weight': [6, 8, 10, 12],

            'gamma': [0.75, 1, 1.25, 1.5],

            'subsample': [0.6, 0.8, 1.0],

            'colsample_bytree': [0.6, 0.8, 1.0],

            'max_depth': [3, 4, 5]

            }

    # Get model

    clf_xgb = XGBClassifier(learning_rate=0.1, 

                            n_estimators=500, 

                            objective='binary:logistic',

                            silent=True, 

                            nthread=1)

    # Get best hyperparameters

    random_search = RandomizedSearchCV(clf_xgb, 

                                       param_distributions=params, 

                                       n_iter=8, 

                                       scoring='f1', 

                                       n_jobs=-1, cv=5, 

                                       verbose=3, random_state=115)

    random_search.fit(X, y)

    print('\n Best estimator:')

    print(random_search.best_estimator_)

    print('\n Best F1 score')

    print(random_search.best_score_)

    print('\n Best hyperparameters:')

    print(random_search.best_params_)

    return random_search
# Get non-upsampled model

random_search = get_cv_model(X_train_f, y_train)
# Get final scores

def final_scores(model=random_search, X_test=X_test_f, y_test=y_test, upsample=False, model_name='Stacked XGBClassifier'):

    y_pred = model.predict(X_test)

    if upsample: print('-- Upsampled --')

    print(f'--- {model_name} Results ---')

    print('Test F1 Score: ', f1_score(y_test, y_pred)) 

    print('\n Classification Report: ')

    print(classification_report(y_test, y_pred))

    print('\n Confusion Matrix: ')

    print(confusion_matrix(y_test, y_pred))

    print('\n Test Accuracy Score: ', accuracy_score(y_test, y_pred))
final_scores()
from xgboost import plot_importance



# Plot feature importance

f = plot_importance(random_search.best_estimator_)
random_search_u = get_cv_model(X_train_f_sm, y_train_sm)
final_scores(model=random_search_u, upsample=True)
# Plot feature importance

f_up = plot_importance(random_search_u.best_estimator_)
# Compare to Base Learners

for key, model in models.items():

    model.fit(X_train, y_train)

    final_scores(model=model, X_test=X_test, model_name=key)

    print('\n\n')