import numpy as np

from sklearn.linear_model import LinearRegression

clf = LinearRegression()



np.random.seed(42)

X = (np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.5))[:, 

                                                                                  np.newaxis]

y = (np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.25))[:, 

                                                                                   np.newaxis]



clf.fit(X, y)

y_pred = clf.predict(y)
import numpy as np



def r2_score(y, y_pred):

    rss_adj = np.sum((y - y_pred)**2)

    n = len(y)

    y_bar_adj = (1 / n) * np.sum(y)

    ess_adj = np.sum((y - y_bar_adj)**2)

    return 1 - rss_adj / ess_adj



r2_score(y, y_pred)
from sklearn.metrics import r2_score

r2_score(y, y_pred)
def rss_score(y, y_pred):

    return np.sum((y - y_pred)**2)
rss_score(y, y_pred)
def mean_squared_error(y, y_pred):

    return (1 / len(y)) * np.sum((y - y_pred)**2)



mean_squared_error(y, y_pred)
from sklearn.metrics import mean_squared_error



mean_squared_error(y, y_pred)
def mean_absolute_error(y, y_pred):

    return (1 / len(y)) * np.sum(np.abs(y - y_pred))

    

mean_absolute_error(y, y_pred)
from sklearn.metrics import mean_absolute_error

    

mean_absolute_error(y, y_pred)
def median_absolute_error(y, y_pred):

    return np.median(np.abs(y - y_pred))

    

mean_absolute_error(y, y_pred)
from sklearn.metrics import median_absolute_error



median_absolute_error(y, y_pred)
def explained_variance_score(y, y_pred):

    return 1 - (np.var(y - y_pred) / np.var(y))



explained_variance_score(y, y_pred)
from sklearn.metrics import explained_variance_score



explained_variance_score(y, y_pred)