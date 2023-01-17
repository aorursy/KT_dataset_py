!pip install python-vivid
from sklearn.datasets import make_classification

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from vivid.utils import timer, get_logger



logger = get_logger(__name__)



def calc_score(y_true, y_pred):

    none_prob_functions = [

        accuracy_score,

        f1_score,

        precision_score,

        recall_score

    ]



    retval = {}

    for func in none_prob_functions:

        retval[func.__name__] = func(y_true, y_pred)



    return retval
X, y = make_classification(n_samples=int(2e5), n_features=10,)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5, shuffle=True)



logger.info('#train: {} - #features: {}'.format(*X_train.shape))
logger.info('#train: {} - #features: {}'.format(*X_train.shape))



# train simple svc

with timer(logger, prefix='fit simple svc '):

    svc = SVC()

    svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

score = calc_score(y_test, y_pred)

logger.info(score)
with timer(logger, prefix='fit bagging '):

    clf = BaggingClassifier(SVC(), n_estimators=10, max_samples=.1, n_jobs=1)

    clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

score = calc_score(y_test, y_pred)

logger.info(score)



with timer(logger, prefix='fit bagging (parallel) '):

    clf = BaggingClassifier(SVC(), n_estimators=10, max_samples=.1, n_jobs=-1)

    clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

score = calc_score(y_test, y_pred)

logger.info(score)



with timer(logger, prefix='fit bagging (n_samples x2) '):

    clf = BaggingClassifier(SVC(), n_estimators=10, max_samples=.2, n_jobs=-1)

    clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

score = calc_score(y_test, y_pred)

logger.info(score)
