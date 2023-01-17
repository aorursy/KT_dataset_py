import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
PATH = '../input/stanford-covid-vaccine'

train = pd.read_json(f'{PATH}/train.json',lines=True).drop(columns=['index'])
test = pd.read_json(f'{PATH}/test.json', lines=True).drop(columns=['index'])
submission = pd.read_csv(f'{PATH}/sample_submission.csv')
def get_structure_mean_value(row, col):
    r_d = {'S': [], 'M': [], 'I': [], 'B': [], 'H': [], 'E': [], 'X': []}
    for p, r in zip(row['predicted_loop_type'], row[col]):
        r_d[p].append(r)

    r_m = {}
    for k in r_d.keys():
        r_m[k] = np.mean(r_d[k])
    return r_m['S'], r_m['M'], r_m['I'], r_m['B'], r_m['H'], r_m['E'], r_m['X']
r_vals = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
e_vals = ['reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C']
for col in r_vals:
    train[f'S_{col}'], train[f'M_{col}'], train[f'I_{col}'], train[f'B_{col}'], train[f'H_{col}'], train[f'E_{col}'], train[f'X_{col}'] = zip(*train.apply(lambda x: get_structure_mean_value(x, col), axis=1))
def plot_loop_type_values(df, col, xlim):
    df[[f'S_{col}', f'M_{col}', f'I_{col}', f'B_{col}', f'H_{col}', f'E_{col}', f'X_{col}']].plot.kde(title=col, xlim=xlim)
for col in r_vals:
    plot_loop_type_values(train, col, xlim=[-2, 3])
loop_type = ['S', 'M', 'I', 'B', 'H', 'E', 'X']
all_mean_loop_vals = {}
sn_train = train[train['SN_filter']==1]
for col in r_vals:
    mean_loop_vals = {}
    for loop in loop_type:
        v = sn_train[f'{loop}_{col}']
        mean_loop_vals[loop] = np.nanmean(v.values)
    all_mean_loop_vals[col] = mean_loop_vals
# Only use the ones that qualify according the Signal to Noise Filter
cv_train = train[train['SN_filter']==1]
cv_train = cv_train[['id', 'predicted_loop_type'] + r_vals]
cv_out = {}
for col in r_vals:
    cv_out[col] = np.array([np.array(x) for x in cv_train[col].values])
# Get the predicted values according to the dumb model
cv_preds = {}
for col in r_vals:
    data = []
    for i, loop in enumerate(cv_train['predicted_loop_type']):
        vals = np.zeros(len(loop))
        for j, nt in enumerate(loop):
            vals[j] = all_mean_loop_vals[col][nt]
        data.append(vals[:68])
    cv_preds[col] = np.array(data)
def mcrmse(y_grd, y_hat):
    r_vals = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    cv_score = []
    for col in r_vals:
        cv_score.append(np.sqrt(np.mean(np.square(y_grd[col] - y_hat[col]))))
    return np.mean(cv_score)
mcrmse(cv_out, cv_preds)
all_rows = []
for j,r in test[['id', 'predicted_loop_type']].iterrows():
    for i, loop in enumerate(r['predicted_loop_type']):
        #print(loop)
        row = {}
        row['id_seqpos'] = f'{r["id"]}_{i}'
        for col in r_vals:
            row[col] = all_mean_loop_vals[col][loop]
        all_rows.append(row)
sub = pd.DataFrame(all_rows)
sub.to_csv('submission.csv', index=False)
