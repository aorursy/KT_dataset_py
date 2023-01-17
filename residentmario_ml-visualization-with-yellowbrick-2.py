import pandas as pd
import numpy as np

matches = pd.read_csv("../input/t_odds.csv")
matches.head()
book = matches.loc[:, [c for c in matches.columns if'odd' in c]].dropna()

bookies = set([c.split("_")[0] for c in matches.columns if'odd' in c])

def deltafy(srs):
    return pd.Series(
        {b: srs[b + '_player_1_odd'] - srs[b + '_player_2_odd'] for b in bookies}
    )

book = book.apply(deltafy, axis='columns')
book = book.assign(
    winner=(matches.player_1_score > matches.player_2_score).astype(int).loc[book.index]
)

X = book.iloc[:, :-1].loc[:, [c for c in X if c != 'betrally']]
y = book.loc[:, 'betrally']

book.head()
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = Ridge()
from yellowbrick.regressor import ResidualsPlot

vzr = ResidualsPlot(clf)
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()
from yellowbrick.regressor import PredictionError

vzr = PredictionError(clf)
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()
from yellowbrick.regressor import AlphaSelection
from sklearn.linear_model import RidgeCV

alphas = np.linspace(0.01, 10, 100)

clf = RidgeCV(alphas=alphas)
vzr = AlphaSelection(clf)

vzr.fit(X_train, y_train)
vzr.poof()
X = book.iloc[:, :-1]
y = book.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ConfusionMatrix

clf = GaussianNB()
vzr = ConfusionMatrix(clf)
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ClassificationReport

clf = GaussianNB()
vzr = ClassificationReport(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()
from yellowbrick.classifier import ROCAUC

clf = GaussianNB()
vzr = ROCAUC(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()
from yellowbrick.classifier import ClassBalance

clf = GaussianNB()
vzr = ClassBalance(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()
from yellowbrick.classifier import ClassPredictionError

clf = GaussianNB()
vzr = ClassPredictionError(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.score(X_test, y_test)
vzr.poof()
from yellowbrick.classifier import ThreshViz

clf = GaussianNB()
vzr = ThreshViz(clf, classes=['P1 Win', 'P2 Win'][::-1])
vzr.fit(X_train, y_train)
vzr.poof()