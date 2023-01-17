import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import json



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



import os

import seaborn as sns



np.random.seed(42)

% matplotlib inline
train = pd.read_csv('../input/train/train.csv')

test = pd.read_csv('../input/test/test.csv')
def read_pet_sentiment(pet_id, dataset='train'):

    try:

        pet_sentiment_metadata = json.load(open(f'../input/{dataset}/sentiment/{pet_id}.json'))

        scores = [sentence['sentiment']['score'] for sentence in pet_sentiment_metadata['sentences']]

        return max(scores), min(scores)

    except FileNotFoundError as e:

        return 0, 0



sentiments = {}

for i, pet_id in enumerate(train['PetID']):

    sentiments[pet_id] = read_pet_sentiment(pet_id, dataset='train')

    print('Train pets read: {0}/{1}'.format(i, len(train['PetID'])), end='\r')



for i, pet_id in enumerate(test['PetID']):

    sentiments[pet_id] = read_pet_sentiment(pet_id, dataset='test')

    print('Test pets read: {0}/{1}'.format(i, len(test['PetID'])), end='\r')
def feature_engineering(animals_train, animals_test):

    train = animals_train.copy()

    test = animals_test.copy()

    train_labels = animals_train['AdoptionSpeed']



    drop_columns = ['RescuerID', 'Breed2']

    

    train.drop(['AdoptionSpeed'], axis=1, inplace=True)

    

    full_data = train.append(test)

    

    full_data = full_data.reset_index().drop(['index'], axis=1)

    

    full_data.drop(drop_columns, axis=1, inplace=True)

    

    # Breed

    breed_counts = full_data['Breed1'].value_counts()

    full_data['Breed1'] = full_data['Breed1'].fillna(307).replace(breed_counts[breed_counts < 100].index, 'other')

    

    # State

    state_counts = full_data['State'].value_counts()

    full_data['State'] = full_data['State'].fillna('other').replace(state_counts[state_counts < 100].index, 'other')

    

    # Name

    full_data['has_name'] = 1 - (full_data['Name'].isnull()).astype(int)

    full_data.drop(['Name'], axis=1, inplace=True)

    

    # VideoAmt

    full_data['VideoAmt'][full_data['VideoAmt'] > 0] = 'has_video'

    full_data['VideoAmt'][full_data['VideoAmt'] == 0] = 'none'

    full_data['VideoAmt'] = full_data['VideoAmt'].fillna('none')

    

    one_hot_columns = ['Type', 'Breed1', 'MaturitySize', 'Gender', 'Color1', 'Color2', 'Color3', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'State', 'VideoAmt']

    

    for column in one_hot_columns:    

        full_data = one_hot(full_data, column)

    

    train, test = full_data[:animals_train.shape[0]], full_data[animals_train.shape[0]:]

    

    return train, train_labels, test, animals_test['PetID']
def one_hot(df, column):

    return pd.concat(

        [df, pd.get_dummies(df[column]).rename({k: '{0}_{1}'.format(column, k) for k in df[column].unique()}, axis=1)],

        axis=1, 

        join_axes=[df.index]

    ).drop([column], axis=1)
def add_text_features(train, test):

    NUM_TEXT_FEATURES = 30

    

    full_data = train.append(test)

    

    descriptions = full_data['Description'].fillna('')

    vectorizer = TfidfVectorizer()

    description_bow = vectorizer.fit_transform(descriptions).toarray()

    

    pca = PCA(n_components=NUM_TEXT_FEATURES)

    text_features = pca.fit_transform(description_bow)

    

    text_df = pd.DataFrame(data = text_features, columns = ['text_{0}'.format(i) for i in range(text_features.shape[1])])

    

    sentiment_df = pd.DataFrame(data=sentiments).transpose().reset_index().rename(columns={'index': 'PetID', 0: 'max_sentiment', 1: 'min_sentiment'})

    full_data = pd.merge(full_data, sentiment_df, on='PetID', how='left')

    

    full_data.drop(['Description', 'PetID'], axis = 1, inplace=True)

    

    full_data = pd.concat([full_data, text_df], axis = 1)

    columns = full_data.columns

    

    whitened_centered = pd.DataFrame(data = StandardScaler().fit_transform(full_data), columns = columns)

    

    whitened_centered = whitened_centered.drop('Unnamed: 0', axis=1)

    

    train, test = whitened_centered[:train.shape[0]], whitened_centered[train.shape[0]:]

    

    return train, test
# animals_transformed = ... # DataFrame with numerical data where each row is each training point

# animals_labels = ... # Classification of each training point in the order in animals_transformed (Series object)



# animals_test_transformed = ... # DataFrame with the same features as animals_transformed but each row is each test point

# # IMPORTANT: make sure this has the same number of features (columns) as animals_transformed!



# animals_test_index = ... # the PetIDs associated with the test points in the same order as the rows in animals_test_transformed

# # IMPORTANT: make sure these are the PetIDs IN THE CORRECT ORDER associated with the test points. 

# # PetID is how we check if your prediction is correct, so if this is wrong, you will get bad scores on the leaderboard.



animals_transformed, animals_labels, animals_test_transformed, animals_test_index = feature_engineering(train, test)

animals_transformed, animals_test_transformed = add_text_features(animals_transformed, animals_test_transformed)
animals_transformed.head() # Sanity check to see if your data is formatted nicely
animals_transformed.shape
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(animals_transformed, animals_labels, test_size=0.20, random_state=42)
from sklearn.linear_model import LinearRegression



model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_valid)

rounded_pred = np.clip(np.round(y_pred), a_min=0.0, a_max=4.0)
cohen_kappa_score(y_valid, rounded_pred, weights='quadratic')
cm = confusion_matrix(y_valid, rounded_pred)

sns.heatmap(cm, annot=True)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_valid)
cohen_kappa_score(y_valid, y_pred, weights='quadratic')
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold



kf = KFold(n_splits=3)



ks = [1, 10, 20]



cv_scores = np.zeros(len(ks))



for ki, k in enumerate(ks):

    print('Training k =', k, end='\r')

    scores = np.zeros(3)

    for i, (train_index, test_index) in enumerate(kf.split(X_train)):

        X_tr, X_te = X_train.iloc[train_index, :], X_train.iloc[test_index, :]

        y_tr, y_te = y_train.iloc[train_index], y_train.iloc[test_index]

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_tr, y_tr)

        pred = knn.predict(X_te)

        scores[i] = cohen_kappa_score(y_te, pred, weights='quadratic')

    cv_scores[ki] = scores.mean()
best_k = ks[cv_scores.argmax()]

best_k
classifier = KNeighborsClassifier(n_neighbors=best_k)

classifier.fit(X_train, y_train)

valid_pred = classifier.predict(X_valid)

cohen_kappa_score(y_valid, valid_pred, weights='quadratic')
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_valid)
cohen_kappa_score(y_valid, y_pred, weights='quadratic')
cohen_kappa_score(classifier.predict(X_train), y_train, weights='quadratic')
decision_tree_classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 5)

decision_tree_classifier.fit(X_train, y_train)

y_pred = decision_tree_classifier.predict(X_valid)

cohen_kappa_score(y_valid, y_pred, weights='quadratic')
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(criterion = 'entropy', bootstrap = True)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_valid)

cohen_kappa_score(y_valid, y_pred, weights='quadratic')
params = {

    'n_splits': 10,

    'verbose_eval': 1000,

    'num_boost_rounds': 500,

    'early_stop': 50

}
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

import scipy

from functools import partial



def train_and_test(params, X_train, y_train, X_test, verbose=False):

    """Trains an XGBoost classifier on training data, reports accuracy

    and classifies test points.



    Parameters:

    - :param params: the hyperparameters of the XGBoost classifier

    - :param X_train: a DataFrame of the training data

    - :param y_train: a Series of the output labels of the training data

    - :param X_test: a DataFrame of the test data

    - :param verbose: set to True if you would like log of the training process



    """

    if verbose:

        print('Training the classifier')

    _, oof_train, oof_test = run_xgb(params, X_train, y_train, X_test, verbose=False)

    optR = OptimizedRounder()

    optR.fit(oof_train, y_train)

    coefficients = optR.coefficients()



    if verbose:

        print('Computing accuracy')

    valid_pred = optR.predict(oof_train, coefficients)



    if verbose:

        print('Predicting on test set')

    preds = optR.predict(oof_test.mean(axis=1), coefficients.copy()).astype(np.int8)

    return preds, optR, cohen_kappa_score(y_train, valid_pred, weights='quadratic')



class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _kappa_loss(self, coef, X, y):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return -cohen_kappa_score(y, preds, weights='quadratic')



    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X = X, y = y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = scipy.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return preds



    def coefficients(self):

        return self.coef_['x']



def run_xgb(params, X_train, y_train, X_test, verbose=False):



    xgb_params = {

        'eval_metric': 'rmse',

        'seed': 1337,

        'eta': 0.0123,

        'subsample': 0.8,

        'colsample_bytree': 0.85,

        'silent': 1 if verbose else 0,

    }



    kf = StratifiedKFold(n_splits=params['n_splits'], shuffle=True, random_state=1337)



    oof_train = np.zeros((X_train.shape[0]))

    oof_test = np.zeros((X_test.shape[0], params['n_splits']))



    i = 0



    for train_idx, valid_idx in kf.split(X_train, y_train):



        X_tr = X_train.iloc[train_idx, :]

        X_val = X_train.iloc[valid_idx, :]



        y_tr = y_train.iloc[train_idx]

        y_val = y_train.iloc[valid_idx]



        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)

        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)



        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(dtrain=d_train, num_boost_round=params['num_boost_rounds'], evals=watchlist,

                         early_stopping_rounds=params['early_stop'], verbose_eval=params['verbose_eval'], params=xgb_params)



        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)

        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)



        oof_train[valid_idx] = valid_pred

        oof_test[:, i] = test_pred



        i += 1

    return model, oof_train, oof_test
xgb_preds, model, accuracy = train_and_test(params, animals_transformed, animals_labels, animals_test_transformed, verbose=True)

print('Validation accuracy:', accuracy)
preds = pd.DataFrame({

    'PetID': animals_test_index,

    'AdoptionSpeed': xgb_preds # or choose another classifier's predictions, if you'd like

}).set_index('PetID')

preds.to_csv('submission.csv')
!head submission.csv