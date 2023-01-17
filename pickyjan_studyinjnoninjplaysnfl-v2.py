import pandas as pd

from sqlalchemy import create_engine

import pickle

import numpy as np

import os
def get_dist_at_each_ts(xs, ys):

    dists = []

    for i in range(1, len(xs)):

        dists.append(math.sqrt((xs[i-1]-xs[i])**2 + (ys[i-1] - ys[i])**2))

    return dists



def get_dists_btw_ts(dists, s_ts, e_ts):

    new_dists = dists[s_ts:e_ts]

    return new_dists



def get_tot_dist_btw_ts(dists, s_ts, e_ts):

    new_dists = get_dists_btw_ts(dists, s_ts, e_ts)

    return sum(new_dists)



def get_mean_dist_btw_ts(dists, s_ts, e_ts):

    new_dists = get_dists_btw_ts(dists, s_ts, e_ts)

    return np.mean(new_dists)



def get_max_dist_btw_ts(dists, s_ts, e_ts):

    new_dists = get_dists_btw_ts(dist, s_ts, e_ts)

    return max(new_dists)



def get_std_dist_btw_ts(dists, s_ts, e_ts):

    new_dists = get_dists_btw_ts(dists, s_ts, e_ts)

    return np.std(new_dists)
def get_disp_btw_ts(xs, ys, s_ts, e_ts):

    try:

        return math.sqrt((xs[e_ts] - xs[s_ts])**2 + (ys[e_ts] - ys[s_ts])**2)

    except:

        return 0
def get_vel_at_each_ts(xs, ys, time):

    vels = []

    #compute a positive displacement using euclidean distance

    for i in range(1, len(xs)):

        disp = math.sqrt((xs[i-1]-xs[i])**2 + (ys[i-1] - ys[i])**2)

        diff_time = time[i] - time[i-1]

        vels.append(disp/diff_time)

    return vels 



def get_vels_btw_ts(vels, s_ts, e_ts):

    new_vels = vels[s_ts:e_ts]

    return new_vels



def get_mean_vel_btw_ts(vels, s_ts, e_ts):

    new_vels = get_vels_btw_ts(vels, s_ts, e_ts)

    return np.mean(new_vels)



def get_std_vel_btw_ts(vels, s_ts, e_ts):

    new_vels = get_vels_btw_ts(vels, s_ts, e_ts)

    return np.std(new_vels)



def get_max_vel_btw_ts(vels, s_ts, e_ts):

    new_vels = get_vels_btw_ts(vels, s_ts, e_ts)

    if len(new_vels) != 0:

        return max(new_vels)

    else:

        return 0



def get_diff_vel_btw_ts(vels, s_ts, e_ts):

    new_vels = get_vels_btw_ts(vels, s_ts, e_ts)

    if len(new_vels) == 0:

        return 0

    #could be negative

    return max(new_vels) - min(new_vels)



def get_time_used_to_reach_max_vel(vels):

    if len(vels) == 0:

        return 0

    return vels.index(max(vels))
def get_acc_at_each_ts(xs, ys, time):

    vel = get_vel_at_each_ts(xs, ys, time)

    accs = []

    for i in range(1, len(vel)):

        diff_v = vel[i] - vel[i-1]

        diff_time = time[i + 1] - time[i-1]

        accs.append(diff_v/diff_time)

    return accs



def get_accs_btw_ts(accs, s_ts, e_ts):

    new_accs = accs[s_ts:e_ts]

    return new_accs



def get_pos_accs_btw_ts(accs, s_ts, e_ts):

    new_accs = get_accs_btw_ts(accs, s_ts, e_ts)

    pos_accs = []

    for acc in new_accs:

        if acc > 0:

            pos_accs.append(acc)

    return pos_accs



#deceleration

def get_neg_accs_btw_ts(accs, s_ts, e_ts):

    new_accs = get_accs_btw_ts(accs, s_ts, e_ts)

    neg_accs = []

    for acc in new_accs:

        if acc < 0:

            neg_accs.append(acc)

    return neg_accs

        

def get_mean_acc_btw_ts(accs, s_ts, e_ts):

    new_accs = get_accs_btw_ts(accs, s_ts, e_ts)

    #could be negative

    return np.mean(new_accs)



def get_std_acc_btw_ts(accs, s_ts, e_ts):

    new_accs = get_accs_btw_ts(accs, s_ts, e_ts)

    #could be negative

    return np.std(new_accs)



def get_max_acc_btw_ts(accs, s_ts, e_ts):

    new_accs = get_accs_btw_ts(accs, s_ts, e_ts)

    if len(new_accs) == 0:

        return 0

    #could be negative

    return max(new_accs)



def get_min_acc_btw_ts(accs, s_ts, e_ts):

    new_accs = get_accs_btw_ts(accs, s_ts, e_ts)

    if len(new_accs) == 0:

        return 0

    return min(new_accs)



def get_diff_acc_btw_ts(accs, s_ts, e_ts):

    new_accs = get_accs_btw_ts(accs, s_ts, e_ts)

    if len(new_accs) == 0:

        return 0

    #could be negative

    return max(new_accs) - min(new_accs)
def get_chg_dir_at_each_ts(theta):

    chg_dirs = []

    for i in range(1, len(theta)):

        chg_dirs.append(theta[i] - theta[i-1])

    return chg_dirs



def get_chg_dirs_btw_ts(chg_dirs, s_ts, e_ts):

    new_chg_dirs = chg_dirs[s_ts:e_ts]

    return new_chg_dirs



def get_mean_chg_dir_btw_ts(chg_dirs, s_ts, e_ts):

    new_chg_dirs = get_chg_dirs_btw_ts(chg_dirs, s_ts, e_ts)

    #could be negative

    return np.mean(new_chg_dirs)



def get_std_chg_dir_btw_ts(chg_dirs, s_ts, e_ts):

    new_chg_dirs = get_chg_dirs_btw_ts(chg_dirs, s_ts, e_ts)

    #could be negative

    return np.std(new_chg_dirs)



def get_max_chg_dir_btw_ts(chg_dirs, s_ts, e_ts):

    new_chg_dirs = get_chg_dirs_btw_ts(chg_dirs, s_ts, e_ts)

    if len(new_chg_dirs) == 0:

        return 0

    #change to positive values

    abs_new_chg_dirs = np.abs(new_chg_dirs)

    return max(abs_new_chg_dirs)



def get_diff_chg_dir_btw_ts(chg_dirs, s_ts, e_ts):

    new_chg_dirs = get_chg_dirs_btw_ts(chg_dirs, s_ts, e_ts)

    if len(new_chg_dirs) == 0:

        return 0

    #could be negative

    return max(new_chg_dirs) - min(new_chg_dirs)



def num_maj_chg_dir_btw_ts(chg_dirs, s_ts, e_ts, m_dir=90):

    new_chg_dirs = get_chg_dirs_btw_ts(chg_dirs, s_ts, e_ts)

    if len(new_chg_dirs) == 0:

        return 0

    #change to positive values

    abs_new_chg_dirs = np.abs(new_chg_dirs)

    n = 0

    for cd in abs_new_chg_dirs:

        if cd > m_dir:

            n += 1

    return n



def get_vel_of_max_chg_dir_btw_ts(chg_dirs, s_ts, e_ts, vels):

    new_chg_dirs = get_chg_dirs_btw_ts(chg_dirs, s_ts, e_ts)

    if len(new_chg_dirs) == 0:

        return 0

    new_vels = get_vels_btw_ts(vels, s_ts, e_ts)

    abs_new_chg_dirs = list(np.abs(new_chg_dirs))

    max_idx = abs_new_chg_dirs.index(max(abs_new_chg_dirs))

    return new_vels[max_idx]



def get_vel_of_maj_chg_dir_btw_ts(chg_dirs, s_ts, e_ts, vels, m_dir=90):

    new_chg_dirs = get_chg_dirs_btw_ts(chg_dirs, s_ts, e_ts)

    if len(new_chg_dirs) == 0:

        return []

    new_vels = get_vels_btw_ts(vels, s_ts, e_ts)

    abs_new_chg_dirs = list(np.abs(new_chg_dirs))

    i = 0

    maj_chg_dirs_vels = []

    for i in range(len(abs_new_chg_dirs)):

        if abs_new_chg_dirs[i] > m_dir:

            maj_chg_dirs_vels.append(new_vels)

    return maj_chg_dirs_vels
def compute_all_metrics_of_motion(track_dat):

    xs = track_dat['x']

    ys = track_dat['y']

    time = track_dat['time']

    theta = track_dat['dir']

    

    metrics = {}

    dists = get_dist_at_each_ts(xs, ys)

    vels = get_vel_at_each_ts(xs, ys, time)

    accs = get_acc_at_each_ts(xs, ys, time)

    chg_dirs = get_chg_dir_at_each_ts(theta)

    

    #distance

    metrics['tot_dist'] = get_tot_dist_btw_ts(dists, 0, len(dists))

    metrics['mean_dist'] = get_mean_dist_btw_ts(dists, 0, len(dists))

    metrics['std_dist'] = get_std_dist_btw_ts(dists, 0, len(dists))

    time_reach_max = get_time_used_to_reach_max_vel(vels)

    metrics['time_to_max_vel'] = time_reach_max

    

    metrics['tot_dist_bf_max_vel'] = get_tot_dist_btw_ts(dists, 0, time_reach_max)

    metrics['mean_dist_bf_max_vel'] = get_mean_dist_btw_ts(dists, 0, time_reach_max)

    metrics['std_dist_bf_max_vel'] = get_std_dist_btw_ts(dists, 0, time_reach_max)

    

    metrics['tot_dist_af_max_vel'] = get_tot_dist_btw_ts(dists, time_reach_max + 1, len(dists))

    metrics['mean_dist_af_max_vel'] = get_mean_dist_btw_ts(dists, time_reach_max + 1, len(dists))

    metrics['std_dist_af_max_vel'] = get_std_dist_btw_ts(dists, time_reach_max + 1, len(dists))

    

    #displacement

    metrics['disp'] = get_disp_btw_ts(xs, ys, 0, len(dists))

    metrics['disp_bf_max_vel'] = get_disp_btw_ts(xs, ys, 0, time_reach_max)

    metrics['disp_af_max_vel'] = get_disp_btw_ts(xs, ys, time_reach_max + 1, len(dists))

    

    #velocity

    metrics['mean_vel'] = get_mean_vel_btw_ts(vels, 0, len(vels))

    metrics['std_vel'] = get_std_vel_btw_ts(vels, 0, len(vels))

    metrics['max_vel'] = get_max_vel_btw_ts(vels, 0, len(vels))

    metrics['diff_vel'] = abs(get_diff_vel_btw_ts(vels, 0, len(vels)))

    

    metrics['mean_vel_bf_max_vel'] = get_mean_vel_btw_ts(vels, 0, time_reach_max)

    metrics['std_vel_bf_max_vel'] = get_std_vel_btw_ts(vels, 0, time_reach_max)

    metrics['max_vel_bf_max_vel'] = get_max_vel_btw_ts(vels, 0, time_reach_max)

    metrics['diff_vel_bf_max_vel'] = abs(get_diff_vel_btw_ts(vels, 0, time_reach_max))

    

    metrics['mean_vel_af_max_vel'] = get_mean_vel_btw_ts(vels, time_reach_max + 1, len(vels))

    metrics['std_vel_af_max_vel'] = get_std_vel_btw_ts(vels, time_reach_max + 1, len(vels))

    metrics['max_vel_af_max_vel'] = get_max_vel_btw_ts(vels, time_reach_max + 1, len(vels))

    metrics['diff_vel_af_max_vel'] = get_diff_vel_btw_ts(vels, time_reach_max + 1, len(vels))

    

    #Acceleration

    metrics['mean_acc'] = abs(get_mean_acc_btw_ts(accs, 0, len(accs)))

    metrics['std_acc'] = abs(get_std_acc_btw_ts(accs, 0, len(accs)))

    metrics['max_acc'] = abs(get_max_acc_btw_ts(accs, 0, len(accs)))

    metrics['min_acc'] = abs(get_min_acc_btw_ts(accs, 0, len(accs)))

    metrics['diff_acc'] = abs(get_diff_acc_btw_ts(accs, 0, len(accs)))

    

    pos_accs = get_pos_accs_btw_ts(accs, 0, len(accs))

    metrics['mean_pos_acc'] = abs(get_mean_acc_btw_ts(pos_accs, 0, len(pos_accs)))

    metrics['std_pos_acc'] = abs(get_std_acc_btw_ts(pos_accs, 0, len(pos_accs)))

    

    neg_accs = get_neg_accs_btw_ts(accs, 0, len(accs))

    metrics['mean_neg_acc'] = abs(get_mean_acc_btw_ts(neg_accs, 0, len(neg_accs)))

    metrics['std_neg_acc'] = abs(get_std_acc_btw_ts(neg_accs, 0, len(neg_accs)))

    

    #Changing direction

    metrics['mean_chg_dir'] = abs(get_mean_chg_dir_btw_ts(chg_dirs, 0, len(chg_dirs)))

    metrics['std_chg_dir'] = abs(get_std_chg_dir_btw_ts(chg_dirs, 0, len(chg_dirs)))

    metrics['max_chg_dir'] = get_max_chg_dir_btw_ts(chg_dirs, 0, len(chg_dirs))

    metrics['diff_chg_dir'] = abs(get_diff_chg_dir_btw_ts(chg_dirs, 0, len(chg_dirs)))

    metrics['num_maj_chg_dir'] = num_maj_chg_dir_btw_ts(chg_dirs, 0, len(chg_dirs))

    metrics['vel_max_chg_dir'] = get_vel_of_max_chg_dir_btw_ts(chg_dirs, 0, len(chg_dirs), vels)

    

    vels_maj_chg_dir = get_vel_of_maj_chg_dir_btw_ts(chg_dirs, 0, len(chg_dirs), vels)

    if len(vels_maj_chg_dir) != 0:

        metrics['mean_vel_maj_chg_dir'] = np.mean(vels_maj_chg_dir)

        metrics['std_vel_maj_chg_dir'] = np.std(vels_maj_chg_dir)

        metrics['max_vel_maj_chg_dir'] = np.max(vels_maj_chg_dir)

    else:

        metrics['mean_vel_maj_chg_dir'] = 0

        metrics['std_vel_maj_chg_dir'] = 0

        metrics['max_vel_maj_chg_dir'] = 0

        

    return metrics



def construct_metrics_rec(motion_metric, play_id, inj):

    game_id = play_id[:play_id.rindex('-')]

    m_rec = [game_id, play_id, inj]

    for i in range(3, len(header_keys)):

        m_rec.append(motion_metric[header_keys[i]])

    return m_rec
import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

import matplotlib.pyplot as plt
def read_ds(f):

    ds = pd.read_csv(f)

    #shuffle

    ds = ds.sample(frac=1).reset_index(drop=True)

    #D = np.array(ds)

    #return D

    return ds



ds = read_ds('../input/all-injnoninjplaytrackmetrics/all_InjNonInjPlayTrackMetrics.csv')

col_names = list(ds.columns)
dist_col_names = ['mean_dist', 'std_dist', 

                  'mean_dist_bf_max_vel', 'std_dist_bf_max_vel', 

                  'mean_dist_af_max_vel', 'std_dist_af_max_vel']

disp_col_names = ['disp', 'disp_bf_max_vel', 'disp_af_max_vel']

vel_col_names = ['mean_vel', 'std_vel', 'max_vel', 'diff_vel', 

                 'mean_vel_bf_max_vel', 'std_vel_bf_max_vel', 'max_vel_bf_max_vel', 

                 'diff_vel_bf_max_vel', 'mean_vel_af_max_vel', 'std_vel_af_max_vel', 

                 'max_vel_af_max_vel', 'vel_max_chg_dir', 'mean_vel_maj_chg_dir', 

                 'std_vel_maj_chg_dir', 'max_vel_maj_chg_dir']

acc_col_names = ['mean_acc', 'std_acc', 

                 'mean_pos_acc', 'std_pos_acc', 'mean_neg_acc', 'std_neg_acc']

chgdir_col_names = ['mean_chg_dir', 'std_chg_dir']



def plot_mean_dat_for_injnoninj_plays(ds, game_id, col_type='dist'):

    inj_play_inf = []

    non_inj_play_inf = []

    plays_in_game = ds.loc[ds['game_id'] == game_id]

    if col_type == 'dist':

        cols = dist_col_names

    if col_type == 'disp':

        cols = disp_col_names

    if col_type == 'vel':

        cols = vel_col_names

    if col_type == 'acc':

        cols = acc_col_names

    if col_type == 'chgdir':

        cols = chgdir_col_names

    

    for i in range(len(cols)):

        grp = plays_in_game.groupby("injured")[cols[i]].mean()

        non_inj_play_inf.append(grp.iloc[0])

        inj_play_inf.append(grp.iloc[1])

    

    n_groups = len(cols)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)

    bar_width = 0.35

    opacity = 0.8



    rects1 = plt.bar(index, non_inj_play_inf, bar_width, 

                     alpha=opacity, color='g', label='non injury')



    rects2 = plt.bar(index + bar_width, inj_play_inf, bar_width,

                     alpha=opacity, color='r', label='injury')



    plt.xlabel('metrics')

    plt.ylabel('values')

    plt.title(game_id)

    plt.xticks(index + bar_width, cols, rotation=90)

    plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(0.95, 1.0))



    plt.tight_layout()

    plt.show()
#list all games from the csv file

games = ds.iloc[:, 0]
plot_mean_dat_for_injnoninj_plays(ds, games[2], 'dist')
plot_mean_dat_for_injnoninj_plays(ds, games[2], 'vel')
plot_mean_dat_for_injnoninj_plays(ds, games[2], 'acc')
plot_mean_dat_for_injnoninj_plays(ds, games[2], 'chgdir')
def compare_metric_across_plays(ds, m_idx):

    col_names = list(ds.columns)

    hist = ds.hist(column=col_names[m_idx], by='injured', sharey=True, sharex=True, 

               normed=True, range=(0,ds.iloc[:,m_idx].max()))

    plt.suptitle(col_names[m_idx])

    plt.show()
compare_metric_across_plays(ds, 21)
compare_metric_across_plays(ds, 27)
from sklearn import svm

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize, scale
def preproc_dat(X, method="mm"):

    if method == "mm":

        normed_X = normalize(X, axis=0, norm='max')

    if method == "ms":

        normed_X = scale(X, axis=0)

    return normed_X
bds = read_ds('../input/bs-injnoninjplaytrackmetrics2/bs_InjNonInjPlayTrackMetrics2.csv')

y = bds.iloc[:, 1]

X = bds.iloc[:,2:]

normed_X = preproc_dat(X, 'ms')
x_train, x_test, y_train, y_test = train_test_split(normed_X, y, test_size=0.3)
sel = SelectKBest(mutual_info_classif, 5)

sel.fit(x_train, y_train)

sel_fs = []

col_names = list(X.columns)

support = sel.get_support()

for i in range(len(support)):

    if support[i] == True:

        sel_fs.append(col_names[i])

print(sel_fs)
new_x_train = sel.transform(x_train)

new_x_test = sel.transform(x_test)



s_rsv_clf = svm.SVC(kernel='rbf', gamma=0.6)

s_rsv_clf.fit(new_x_train, y_train)

print(s_rsv_clf.score(new_x_test, y_test))
import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))