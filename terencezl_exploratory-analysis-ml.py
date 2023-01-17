import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import mplleaflet

# sns.set_context("poster")

plt.rcdefaults()

plt.style.use('ggplot')

# np.set_printoptions(precision=2)

current_palette = sns.color_palette()

%matplotlib inline
train_df = pd.read_json("../input/train.json")

train_df.head()
# treat price

price_max = np.percentile(train_df.price, 99)

train_df.loc[train_df.price > price_max, 'price'] = price_max

train_df.price.hist(bins=50)
# treat latitude

latitude_max = np.percentile(train_df.latitude, 99)

latitude_min = np.percentile(train_df.latitude, 1)

train_df.loc[train_df.latitude > latitude_max, 'latitude'] = latitude_max

train_df.loc[train_df.latitude < latitude_min, 'latitude'] = latitude_min

train_df.latitude.hist(bins=50)
# treat longitude

longitude_max = np.percentile(train_df.longitude, 99)

longitude_min = np.percentile(train_df.longitude, 1)

train_df.loc[train_df.longitude > longitude_max, 'longitude'] = longitude_max

train_df.loc[train_df.longitude < longitude_min, 'longitude'] = longitude_min

train_df.longitude.hist(bins=50)
train_df['interest_level_coded'] = train_df.interest_level.map({'low': 0, 'medium': 1, 'high':2})

cmapping = train_df.interest_level_coded.map({i: current_palette[i] for i in [0, 1, 2]})

pd.scatter_matrix(train_df[['latitude', 'longitude', 'price']], figsize=[10, 9], s=6, c=cmapping, diagonal='kde')

f = plt.figure(figsize=(10, 10))

train_df_sampled = train_df.sample(100)

plt.scatter(train_df_sampled.longitude, train_df_sampled.latitude, s=50, c=cmapping)

mplleaflet.display(fig=f)
train_df["num_photos"] = train_df.photos.apply(len)

train_df["num_features"] = train_df.features.apply(len)

train_df["num_description_words"] = train_df.description.apply(lambda x: len(x.split(" ")))

train_df["created"] = pd.to_datetime(train_df.created)

train_df["created_year"] = train_df.created.dt.year

train_df["created_month"] = train_df.created.dt.month

train_df["created_day"] = train_df.created.dt.day
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.svm import LinearSVC, SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split

from sklearn.metrics import classification_report, confusion_matrix, log_loss



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    

    Stolen from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    import itertools

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')





X_vars = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', 'num_photos', 'num_features', 'num_description_words', 'created_year', 'created_month', 'created_day']

X_train, X_test, y_train, y_test = train_test_split(train_df[X_vars], train_df['interest_level'], stratify= train_df['interest_level'])
Cs = np.logspace(-3, 1, 10)

lrcv = LogisticRegressionCV(Cs=Cs, scoring='neg_log_loss').fit(X_train, y_train)
# scores correspond to three interest levels. low has the largest log loss

plt.semilogx(lrcv.Cs_, -lrcv.scores_['low'].mean(axis=0), '-o')

# plt.semilogx(lrcv.Cs_, -(lrcv.scores_['low'] + lrcv.scores_['medium'] + lrcv.scores_['high']).mean(axis=0)/3, '-o')

plt.xlabel(r'$\alpha$')

plt.ylabel('Log Loss')
y_pred = lrcv.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=['low', 'medium', 'high'], normalize=True)
lr = GridSearchCV(LogisticRegression(), {'C': Cs}, scoring='neg_log_loss').fit(X_train, y_train)
lr_cv_results = pd.DataFrame(lr.cv_results_)

plt.semilogx(lr_cv_results['param_C'], -lr_cv_results['mean_test_score'], '-o')

plt.xlabel(r'$\alpha$')

plt.ylabel('Log Loss')
print(lr.best_params_, lr.best_score_)
y_pred = lr.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=['low', 'medium', 'high'], normalize=True)
X_train_sample, y_train_sample = X_train.iloc[:5000], y_train.iloc[:5000]
svc_lin = GridSearchCV(LinearSVC(multi_class='ovr'),

                       {'C': np.logspace(-2, 10, 10)}).fit(X_train_sample, y_train_sample)
svc_lin_cv_results = pd.DataFrame(svc_lin.cv_results_)

plt.semilogx(svc_lin_cv_results['param_C'], svc_lin_cv_results['mean_test_score'], '-o')

plt.xlabel('C')

plt.ylabel('Accuracy')
print(svc_lin.best_params_, svc_lin.best_score_)
y_pred = svc_lin.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=['low', 'medium', 'high'])
svc_rbf = GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'), 

                       {'C': np.logspace(-2, 10, 3), 'gamma': np.logspace(-9, 3, 3)}

                      ).fit(X_train_sample, y_train_sample)
svc_rbf_cv_results = pd.DataFrame(svc_rbf.cv_results_)

plt.plot(svc_rbf_cv_results['mean_test_score'], '-o')

plt.xlabel('Num. of Trials')

plt.ylabel('Accuracy')
print(svc_rbf.best_params_, svc_rbf.best_score_)
y_pred = svc_rbf.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=['low', 'medium', 'high'], normalize=True)
n_estimators = np.logspace(1, 3, 5).astype(int)

scores = []

for i in n_estimators:

    rf = RandomForestClassifier(n_estimators=i, oob_score=True, n_jobs=-1).fit(X_train, y_train)

    scores.append(rf.oob_score_)
plt.plot(n_estimators, scores, '-o')

plt.xlabel('Num. of Estimators')

plt.ylabel('Accuracy') # this is not log loss!
rf = GridSearchCV(RandomForestClassifier(n_jobs=-1), {'n_estimators': n_estimators},

                 scoring='neg_log_loss').fit(X_train, y_train)
rf_cv_results = pd.DataFrame(rf.cv_results_)

plt.plot(rf_cv_results['param_n_estimators'], -rf_cv_results['mean_test_score'], '-o')

plt.xlabel('Num. of Estimators')

plt.ylabel('Log Loss')
print(rf.best_params_, rf.best_score_)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=['low', 'medium', 'high'], normalize=True)
params = {'learning_rate': [0.05, 0.1, 0.2], 'subsample': [1, 0.5]}

gbc = GridSearchCV(GradientBoostingClassifier(n_estimators=1000), 

                         params, scoring='neg_log_loss').fit(X_train, y_train)
gbc_cv_results = pd.DataFrame(gbc.cv_results_)

plt.plot(-gbc_cv_results['mean_test_score'], '-o')

plt.xlabel('Num. of Trials')

plt.ylabel('Log Loss')
print(gbc.best_params_, gbc.best_score_)
y_pred = gbc.predict(X_test)

print(classification_report(y_test, y_pred))

plot_confusion_matrix(confusion_matrix(y_test, y_pred), classes=['low', 'medium', 'high'], normalize=True)
test_df = pd.read_json("test.json")

test_df["num_photos"] = test_df.photos.apply(len)

test_df["num_features"] = test_df.features.apply(len)

test_df["num_description_words"] = test_df.description.apply(lambda x: len(x.split(" ")))

test_df["created"] = pd.to_datetime(test_df.created)

test_df["created_year"] = test_df.created.dt.year

test_df["created_month"] = test_df.created.dt.month

test_df["created_day"] = test_df.created.dt.day



y = gbc.predict_proba(test_df[X_vars])
