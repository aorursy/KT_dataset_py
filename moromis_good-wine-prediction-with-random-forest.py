import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# check what the data we have is
import os
print(os.listdir("../input"))
# from https://gist.github.com/dmyersturnbull/035876942070ced4c565e4e96161be3e

from IPython.display import display, Markdown
import pandas as pd

def head(df: pd.DataFrame, n_rows:int=1) -> None:
    """Pretty-print the head of a Pandas table in a Jupyter notebook and show its dimensions."""
    display(Markdown("**whole table (below):** {} rows × {} columns".format(len(df), len(df.columns))))
    display(df.head(n_rows))
    
def tail(df: pd.DataFrame, n_rows:int=1) -> None:
    """Pretty-print the tail of a Pandas table in a Jupyter notebook and show its dimensions."""
    display(Markdown("**whole table (below):** {} rows × {} columns".format(len(df), len(df.columns))))
    display(df.tail(n_rows))
input = pd.read_csv('../input/winequality-red.csv')

# get X and y slices, do preprocessing
X = input.iloc[:, :10]

# https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
from sklearn import preprocessing

to_scale = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(to_scale)
X = pd.DataFrame(scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components=4)
pca.fit(X)

X = pd.DataFrame(pca.transform(X), columns=['PCA%i' % i for i in range(4)], index=X.index)

y = input.iloc[:, 11]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
def check_stats(model, model_name):
    sum = 0
    loss = 0
    total = X_test.shape[0]

    predictions = model.predict(X_test)

    index = 0
    for prediction in predictions:

        actual = y_test.iloc[index]

#         print('pred', prediction, 'actual: ', actual)

        loss += abs(actual - prediction)
        if prediction == actual:
            sum += 1

        index += 1

    accuracy = sum / total
    avg_loss = loss / total

    print('MODEL STATS: ' + model_name)
    print('loss: ', loss)
    print('avg loss: ', avg_loss)
    print('accuracy: ', round(accuracy * 100, 2), '%\n')
    
# https://www.kaggle.com/pranavcoder/random-forests-and-keras

from sklearn import ensemble, tree
from imblearn.pipeline import make_pipeline

cart = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
forest = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=1000, max_features=None, max_depth=None)
gboost = ensemble.GradientBoostingClassifier(max_depth=None)

cart.fit(X_train, y_train)
forest.fit(X_train, y_train)
gboost.fit(X_train, y_train)

check_stats(cart, 'Decision Tree')
check_stats(forest, 'Random Forest')
check_stats(gboost, 'GBoost')
# let's try a voting classifier
from sklearn.ensemble import VotingClassifier

cart = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
forest = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=1000, max_features=None, max_depth=None)
gboost = ensemble.GradientBoostingClassifier(max_depth=None)

vc = VotingClassifier(estimators=[('cart', cart), ('forest', forest), ('gboost', gboost)], voting='soft')

vc = vc.fit(X_train, y_train)

check_stats(vc, 'Voting Classifier')
    
from sklearn.model_selection import cross_val_score

#Now lets try to do some evaluation for random forest model using cross validation.
rfc_eval = cross_val_score(estimator = forest, X = X_train, y = y_train, cv = 8)
rfc_eval.mean()

