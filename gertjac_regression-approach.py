import numpy as np

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KernelDensity



from matplotlib import pyplot as plt

%matplotlib inline
class RegressorConditional:

    def get_o_cat(self, o):

        return np.sum([o>pct for pct in self.percentiles], axis=0)

    def __init__(self, model=ExtraTreesRegressor(

        n_estimators=500, n_jobs=-1, bootstrap=True, oob_score=True)):

        self.model = model

    def fit(self, X, y):

        targ = np.where(y>=0, np.log(1+np.abs(y)), -np.log(1+np.abs(y)))

        self.model.fit(X, targ)

        o = self.model.oob_prediction_

        self.percentiles = np.percentile(o, list(range(10, 100, 10)))

        o_cat = self.get_o_cat(o)

        self.dist = {}

        for oc in range(len(self.percentiles) + 1):

            filt = [oi==oc for oi in o_cat]

            kde = KernelDensity(kernel='exponential', metric='manhattan', bandwidth=0.3)

            kde.fit(list(zip(y[filt])))

            self.dist[oc] = np.exp(kde.score_samples(list(zip(range(-99, 100)))))

            self.dist[oc] /= sum(self.dist[oc])

    def predict_proba(self, X):

        o = self.model.predict(X)

        o_cat = self.get_o_cat(o)

        return np.array([self.dist[oc] for oc in o_cat])
df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False).select_dtypes(include=np.number)

df_play = df[df.NflId==df.NflIdRusher].copy()



features = df_play.drop('Yards', axis=1).select_dtypes(include=np.number).columns.tolist()
model = RegressorConditional()

model.fit(df_play[features].fillna(-999), df_play.Yards)



plt.figure(figsize=(12, 4))

for oc in model.dist:

    plt.plot(model.dist[oc], label=oc)

plt.xticks(list(range(-1, 200, 25)), list(range(-100, 101, 25)))

plt.legend()

plt.show()
from kaggle.competitions import nflrush



names = dict(zip(range(199), ['Yards%d' % i for i in range(-99, 100)]))



env = nflrush.make_env()

for df_test, _ in env.iter_test():

    env.predict(pd.DataFrame([np.clip(np.cumsum(

        model.predict_proba(df_test[df_test.NflId==df_test.NflIdRusher][features].fillna(-999))

    ), 0, 1)]).rename(names, axis=1))

env.write_submission_file()