import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip')

print(data.shape)

data.head()
data_explore = data.copy()
data_explore = data_explore.drop(['id'], axis=1)
data_explore.info()
dac = np.vstack(data_explore.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)

data_explore['dac_year'] = dac[:,0]

data_explore['dac_month'] = dac[:,1]

data_explore['dac_day'] = dac[:,2]

data_explore = data_explore.drop(['date_account_created'], axis=1)
data_explore[data_explore['country_destination']!='NDF']['date_first_booking'].isna().sum()
data_explore.date_first_booking = data_explore.date_first_booking.fillna('2000-01-01')

first_booking = np.vstack(data_explore.date_first_booking.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)

data_explore['first_booking_year'] = first_booking[:,0]

data_explore['first_booking_month'] = first_booking[:,1]

data_explore['first_booking_day'] = first_booking[:,2]

data_explore = data_explore.drop(['date_first_booking'], axis=1)
data_explore.nunique()
data_explore.describe()
data_explore.isna().sum()
age_values = data_explore.age.values

data_explore['age'] = np.where(age_values>1000, np.random.randint(28, 43), age_values)

data_explore['age'] = data_explore['age'].fillna(np.random.randint(28, 43))



data_explore['first_affiliate_tracked'] = data_explore['first_affiliate_tracked'].fillna(data_explore['first_affiliate_tracked'].mode().values[0])
data_explore['language'].value_counts()[:10]
def plot_histogram(data):

    ax = plt.gca()

    counts, _, patches = ax.hist(data)

    for count, patch in zip(counts, patches):

        if count>0:

            ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()+5))

    if data.name:

        plt.xlabel(data.name)
plt.figure(figsize=(8, 5))

plot_histogram(data_explore['age'])

plt.xlim(15, 100)

plt.show()
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)

grp = data_explore[['gender', 'age']].groupby(by='gender').count()

plt.pie(grp.values, labels=list(grp.index), shadow=True, startangle=0,

        autopct='%1.1f%%', wedgeprops={'edgecolor':'black'})

plt.title('Gender')

plt.subplot(1, 3, 2)

grp = data_explore[['dac_year', 'age']].groupby(by='dac_year').count()

plt.pie(grp.values, labels=list(grp.index), shadow=True, startangle=0,

        autopct='%1.1f%%', wedgeprops={'edgecolor':'black'})

plt.title('Account Created: Year')

plt.subplot(1, 3, 3)

grp = data_explore[['dac_month', 'age']].groupby(by='dac_month').count()

plt.pie(grp.values, labels=list(grp.index), shadow=True, startangle=0,

        autopct='%1.1f%%', wedgeprops={'edgecolor':'black'})

plt.title('Account Created: Month')

plt.show()
ax = sns.countplot(x='affiliate_channel', data=data_explore)

for p in ax.patches:

    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.xticks(rotation=-45)

plt.show()
plt.figure(figsize=(16, 7))

plt.subplot(1, 2, 1)

grp = data_explore[['affiliate_channel', 'age']].groupby(by='affiliate_channel').count()

plt.pie(grp.values, labels=list(grp.index), shadow=True, startangle=0,

        autopct='%1.1f%%', wedgeprops={'edgecolor':'black'})

plt.title('Affiliate Channels')

plt.subplot(1, 2, 2)

ax = sns.countplot(x='affiliate_provider', data=data_explore)

for p in ax.patches:

    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.xticks(rotation=-45)

plt.xlim(-0.5, 10.5)

plt.title('Affiliate Providers')

plt.show()
plt.figure(figsize=(10, 6))

ax = sns.countplot(x='country_destination', data=data_explore)

for p in ax.patches:

    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))
data_explore_booked = data_explore[data_explore['country_destination']!='NDF']

data_explore.shape, data_explore_booked.shape
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plot_histogram(data_explore_booked['dac_year'])

plt.subplot(1, 2, 2)

plot_histogram(data_explore_booked['dac_month'])

plt.show()
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plot_histogram(data_explore_booked['first_booking_year'])

plt.subplot(1, 2, 2)

plot_histogram(data_explore_booked['first_booking_month'])

plt.show()
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plot_histogram(data_explore_booked[data_explore_booked['country_destination']=='US']['first_booking_year'])

plt.title('# of Booking in US')

plt.subplot(1, 2, 2)

plot_histogram(data_explore_booked[data_explore_booked['country_destination']=='US']['first_booking_month'])

plt.title('# of Booking in US')

plt.show()
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plot_histogram(data_explore_booked[data_explore_booked['country_destination']=='FR']['first_booking_year'])

plt.title('# of Booking in France')

plt.subplot(1, 2, 2)

plot_histogram(data_explore_booked[data_explore_booked['country_destination']=='FR']['first_booking_month'])

plt.title('# of Booking in France')

plt.show()
plt.figure(figsize=(12, 6))

sns.countplot(x='country_destination', hue='gender', data=data_explore_booked)

plt.title('Geneder distribution across destination countries')

plt.show()
plt.figure(figsize=(12, 6))

sns.countplot(x='first_booking_year', hue='gender', data=data_explore_booked[data_explore_booked['country_destination']=='US'])

plt.title('# of Travellers to USA')

plt.show()
plt.figure(figsize=(12, 6))

sns.countplot(x='first_booking_year', hue='gender', data=data_explore_booked[data_explore_booked['country_destination']=='FR'])

plt.title('# of Travellers to France')

plt.show()
plt.figure(figsize=(15, 6))

sns.boxplot(x='country_destination', y='age', hue='gender', data=data_explore_booked)

plt.ylim(15, 60)

plt.legend(loc='lower right')

plt.show()
plt.figure(figsize=(15, 6))

sns.boxplot(x='dac_year', y='age', hue='gender', data=data_explore_booked)

plt.ylim(15, 60)

plt.legend(loc='lower right')

plt.show()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
X = data.drop(columns=['country_destination'], axis=1).copy()

y = data['country_destination'].copy()



label_enc = LabelEncoder()

y = label_enc.fit_transform(y)

X.shape, y.shape
label_enc.classes_
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape
cat_attrs = ['gender', 'language', 'affiliate_channel', 'affiliate_provider']
pre_process = ColumnTransformer([('drop_cols', 'drop', ['id', 'date_first_booking', 'date_account_created', 'signup_method', 'timestamp_first_active', 

                                                        'signup_app', 'first_device_type', 'first_browser', 'first_affiliate_tracked', 'signup_flow']),

                                 ('num_imputer', SimpleImputer(strategy='median'), ['age']),

                                 ('cat_imputer', SimpleImputer(strategy='most_frequent'), cat_attrs)], remainder='passthrough')



X_train_transformed = pre_process.fit_transform(X_train)

X_test_transformed = pre_process.transform(X_test)

X_train_transformed.shape, X_test_transformed.shape
X_train_transformed = pd.DataFrame(X_train_transformed, columns=['age', 'gender', 'language', 'affiliate_channel', 'affiliate_provider'])

X_test_transformed = pd.DataFrame(X_test_transformed, columns=['age', 'gender', 'language', 'affiliate_channel', 'affiliate_provider'])

X_train_transformed.shape, X_test_transformed.shape
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import make_scorer, ndcg_score

ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)



def dcg_score(y_true, y_score, k=5):

    """Discounted cumulative gain (DCG) at rank K.



    Parameters

    ----------

    y_true : array, shape = [n_samples]

        Ground truth (true relevance labels).

    y_score : array, shape = [n_samples, n_classes]

        Predicted scores.

    k : int

        Rank.



    Returns

    -------

    score : float

    """

    order = np.argsort(y_score)[::-1]

    y_true = np.take(y_true, order[:k])



    gain = 2 ** y_true - 1



    discounts = np.log2(np.arange(len(y_true)) + 2)

    return np.sum(gain / discounts)





def ndcg_score(ground_truth, predictions, k=5):

    """Normalized discounted cumulative gain (NDCG) at rank K.



    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a

    recommendation system based on the graded relevance of the recommended

    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal

    ranking of the entities.



    Parameters

    ----------

    ground_truth : array, shape = [n_samples]

        Ground truth (true labels represended as integers).

    predictions : array, shape = [n_samples, n_classes]

        Predicted probabilities.

    k : int

        Rank.



    Returns

    -------

    score : float



    Example

    -------

    >>> ground_truth = [1, 0, 2]

    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]

    >>> score = ndcg_score(ground_truth, predictions, k=2)

    1.0

    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]

    >>> score = ndcg_score(ground_truth, predictions, k=2)

    0.6666666666

    """

    lb = LabelBinarizer()

    lb.fit(range(len(predictions) + 1))

    T = lb.transform(ground_truth)



    scores = []



    # Iterate over each y_true and compute the DCG score

    for y_true, y_score in zip(T, predictions):

        actual = dcg_score(y_true, y_score, k)

        best = dcg_score(y_true, y_true, k)

        score = float(actual) / float(best)

        scores.append(score)



    return np.mean(scores)





# NDCG Scorer function

ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)
def grid_search(model, grid_param):

    print("Obtaining Best Model for {}".format(model.__class__.__name__))

    grid_search = GridSearchCV(model, grid_param, cv=kf, scoring=ndcg_scorer, return_train_score=True, n_jobs=-1)

    grid_search.fit(X_train_transformed, y_train)

    

    print("Best Parameters: ", grid_search.best_params_)

    print("Best Score: ", grid_search.best_score_)

    

    cvres = grid_search.cv_results_

    print("Results for each run of {}...".format(model.__class__.__name__))

    for train_mean_score, test_mean_score, params in zip(cvres["mean_train_score"], cvres["mean_test_score"], cvres["params"]):

        print(train_mean_score, test_mean_score, params)

        

    return grid_search.best_estimator_
results = []

    

def performance_measures(model, store_results=True):

    train_ndcg = cross_val_score(model, X_train_transformed, y_train, scoring=ndcg_scorer, cv=kf, n_jobs=-1)

    test_ndcg = cross_val_score(model, X_test_transformed, y_test, scoring=ndcg_scorer, cv=kf, n_jobs=-1)

    print("Mean Train NDGC: {}\nMean Test NDGC: {}".format(train_ndcg.mean(), test_ndcg.mean()))
def plot_feature_importance(feature_columns, importance_values,top_n_features=0):

    feature_imp = [ col for col in zip(feature_columns, importance_values)]

    feature_imp.sort(key=lambda x:x[1], reverse=True)



    if top_n_features:

        imp = pd.DataFrame(feature_imp[0:top_n_features], columns=['feature', 'importance'])

    else:

        imp = pd.DataFrame(feature_imp, columns=['feature', 'importance'])

    plt.figure(figsize=(10, 8))

    sns.barplot(y='feature', x='importance', data=imp, orient='h')

    plt.title('Most Important Features', fontsize=16)

    plt.ylabel("Feature", fontsize=16)

    plt.xlabel("")

    plt.show()
from catboost import CatBoostClassifier





catboost_grid_params = [{'iterations':[500, 1000, 1500], 'depth':[4, 6, 8, 10],}]



catboost_clf = CatBoostClassifier(task_type="GPU", loss_function='MultiClass', bagging_temperature=0.3, 

                                  cat_features=[1, 2, 3, 4], random_state=42, verbose=0)



grid_search_results = catboost_clf.grid_search(catboost_grid_params,

            X_train_transformed,

            y_train,

            cv=5,

            partition_random_seed=42,

            calc_cv_statistics=True,

            search_by_train_test_split=True,

            refit=True,

            shuffle=True,

            stratified=None,

            train_size=0.8,

            verbose=0,

            plot=False)
grid_search_results['params']
catboost_clf.is_fitted()
catboost_clf.feature_importances_
plot_feature_importance(['age', 'gender', 'language', 'affiliate_channel', 'affiliate_provider'], catboost_clf.feature_importances_)
performance_measures(catboost_clf, store_results=False)
X_trasformed = pre_process.transform(X)

predicted_country = catboost_clf.predict(X_trasformed)

predicted_country = label_enc.inverse_transform(predicted_country)

data['predicted_country'] = predicted_country
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)

ax = sns.countplot(x='country_destination', data=data)

for p in ax.patches:

    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))

plt.subplot(2, 1, 2)

ax = sns.countplot(x='predicted_country', data=data)

for p in ax.patches:

    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+10))
final_model = Pipeline([('pre_process', pre_process),

                        ('catboost_clf', catboost_clf)])

final_model.fit(X_train, y_train)
test_data = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/test_users.csv.zip')

test_data.head()
test_data.info()
predictions = final_model.predict_proba(test_data)
#Taking the 5 classes with highest probabilities

id_test = list(test_data.id)

ids = []

countries = []

for i in range(len(id_test)):

    idx = id_test[i]

    ids += [idx] * 5

    countries += label_enc.inverse_transform(np.argsort(predictions[i])[::-1])[:5].tolist()
output = pd.DataFrame(np.column_stack((ids, countries)), columns=['id', 'country'])

output.head()
output.to_csv("./submission.csv", index=False)