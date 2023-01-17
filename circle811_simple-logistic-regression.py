import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
season_result = pd.read_csv('../input/RegularSeasonCompactResults.csv')
tourney_result = pd.read_csv('../input/NCAATourneyCompactResults.csv')
sample_submission = pd.read_csv('../input/SampleSubmissionStage2.csv')
team_list = sorted(set(season_result.WTeamID) | set(season_result.LTeamID))
n_teams = len(team_list)
team_to_int = {t: i for i, t in enumerate(team_list)}

season_result['team_a'] = season_result.WTeamID.apply(team_to_int.get)
season_result['team_b'] = season_result.LTeamID.apply(team_to_int.get)
season_result['log_ratio'] = np.log(season_result.WScore / season_result.LScore)

tourney_result['team_a'] = tourney_result.WTeamID.apply(team_to_int.get)
tourney_result['team_b'] = tourney_result.LTeamID.apply(team_to_int.get)

sample_submission['team_a'] = sample_submission.ID.apply(lambda a: team_to_int[int(a.split('_')[1])])
sample_submission['team_b'] = sample_submission.ID.apply(lambda a: team_to_int[int(a.split('_')[2])])
def encode_team(d):
    n_samples = d.shape[0]
    a = np.arange(n_samples)
    x = sp.sparse.lil_matrix((n_samples, n_teams))
    x[a, d.team_a] = 1
    x[a, d.team_b] = -1
    return x
def cv_w_c(year_list, w_list, c_list):
    loss_array = np.zeros((len(year_list), len(w_list), len(c_list)))
    for i, year in enumerate(year_list):
        sr = season_result[season_result.Season == year]
        x_sr = encode_team(sr)
        y_sr = np.ones(x_sr.shape[0], dtype=np.int64)
        x_train = sp.sparse.vstack([x_sr, -x_sr])
        y_train = np.concatenate([y_sr, -y_sr])
        log_ratio = np.concatenate([sr.log_ratio, sr.log_ratio])

        tr = tourney_result[tourney_result.Season == year]
        x_tr = encode_team(tr)
        y_tr = np.ones(x_tr.shape[0], dtype=np.int64)
        x_test = sp.sparse.vstack([x_tr, -x_tr])
        y_test = np.concatenate([y_tr, -y_tr])

        for j, w in enumerate(w_list):
            if w is None:
                w_train = None
            else:
                w_train = w * log_ratio.min() + log_ratio
            for k, c in enumerate(c_list):
                cls = LogisticRegression(C=c)
                cls.fit(x_train, y_train, w_train)
                col = list(cls.classes_).index(1)
                y_pred = cls.predict_proba(x_test)[:, col]
                loss = log_loss(y_test, y_pred)
                loss_array[i, j, k] = loss

    return loss_array.mean(0)
def find_c(year_list, w, c_list):
    w_list = [w]
    loss_array = cv_w_c(year_list, w_list, c_list)
    a = divmod(loss_array.argmin(), loss_array.shape[-1])
    print('c_list={}'.format(c_list))
    print('w={} c={} loss={}'.format(w_list[a[0]], c_list[a[1]], loss_array[a]))
    plt.plot(c_list, loss_array[a[0], :])
    plt.show()
year_list = range(2010, 2018)
find_c(year_list, None, np.logspace(-3, 3, 21))
find_c(year_list, None, np.linspace(0.5, 2, 16))
def find_w(year_list, w_list, c_list):
    loss_array = cv_w_c(year_list, w_list, c_list)
    a = divmod(loss_array.argmin(), loss_array.shape[-1])
    print('w_list={}'.format(w_list))
    print('w={} c={} loss={}'.format(w_list[a[0]], c_list[a[1]], loss_array[a]))
    plt.plot(w_list, loss_array[:, a[1]])
    plt.show()
find_w(year_list,
       np.concatenate([np.zeros(1), np.logspace(-1, 1, 10)]),
       np.logspace(-3, 3, 21))
find_w(year_list,
       np.linspace(1, 4, 7),
       np.logspace(-3, 3, 21))
find_c(year_list, 2.5, np.linspace(2, 8, 31))
def predict(w, c, filename):
    year = 2018

    sr = season_result[season_result.Season == year]
    x_sr = encode_team(sr)
    y_sr = np.ones(x_sr.shape[0], dtype=np.int64)
    x_train = sp.sparse.vstack([x_sr, -x_sr])
    y_train = np.concatenate([y_sr, -y_sr])
    log_ratio = np.concatenate([sr.log_ratio, sr.log_ratio])

    x_test = encode_team(sample_submission)

    if w is None:
        w_train = None
    else:
        w_train = w * log_ratio.min() + log_ratio

    cls = LogisticRegression(C=c)
    cls.fit(x_train, y_train, w_train)
    col = list(cls.classes_).index(1)
    y_pred = cls.predict_proba(x_test)[:, col]

    submission = pd.DataFrame()
    submission['ID'] = sample_submission.ID
    submission['Pred'] = y_pred
    submission.to_csv(filename, index=False)

    return cls
lr = predict(w=None, c=1.1, filename='predict_with_logistic.csv')
wlr = predict(w=2.5, c=3.6, filename='predict_with_weighted_logistic.csv')
team_score = wlr.coef_[0]
tourney_teams = sorted(set(sample_submission.team_a) | set(sample_submission.team_b))
plt.hist(team_score)
plt.show()
plt.hist(team_score[tourney_teams])
plt.show()