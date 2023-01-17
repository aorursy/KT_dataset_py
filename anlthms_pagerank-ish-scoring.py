# Apply a PageRank-like algorithm to bubble up the best attacks and defenses.



import os

import numpy as np

import pandas as pd





acc = pd.read_csv('../input/accuracy_matrix.csv', index_col=0)

err = pd.read_csv('../input/error_matrix.csv', index_col=0)

hts = pd.read_csv('../input/hit_target_class_matrix.csv', index_col=0)



# Uncomment this to match the official ranking for defenses.

# acc.values[np.where((acc.values + err.values) != 1000)] = 0



acc_vals = np.float32(acc.values)

err_vals = np.float32(err.values[:, :85])

hts_vals = np.float32(hts.values[:, 85:])



K = 10



names_list = [acc.columns[:85], acc.columns[85:], acc.index]

res_list = [pd.read_csv(os.path.join('../input', filename)) for filename in [

    'non_targeted_attack_results.csv',  'targeted_attack_results.csv', 'defense_results.csv']]

types_list = ['Non-targeted attack', 'Targeted attack', 'Defense']



def_weights = np.ones(acc.shape[0]) / acc.shape[0]

att_weights = np.ones(err.shape[1]) / err.shape[1]



for iter_idx in range(8):

    print('\nIteration %d' % iter_idx)

    # Weight the attack scores according to the defenses they are attacking.

    w_err_vals = err_vals * def_weights.reshape((def_weights.shape[0], 1))

    w_hts_vals = hts_vals * def_weights.reshape((def_weights.shape[0], 1))

    # Weight the defense scores according to the attacks they are defending against.

    w_acc_vals = acc_vals * att_weights.reshape((1, att_weights.shape[0]))

    err_sum = w_err_vals.sum(axis=0)

    hts_sum = w_hts_vals.sum(axis=0)

    acc_sum = w_acc_vals.sum(axis=1)

    scores_list = [err_sum, hts_sum, acc_sum]

    for i in range(3):

        scores, names, res = scores_list[i], names_list[i], res_list[i]

        topk = np.argsort(scores)[-K:][::-1]

        name_map = {res['KaggleTeamId'][i]: res['TeamName'][i] for i in range(res.shape[0])}

        top_names = [name_map[key] for key in names[topk]]

        print('\n%s:\nTeam Name, Team ID, Score' % types_list[i])

        for ident, name, score in zip(names[topk], top_names, scores[topk]):

            print('%s,%s,%d' % (ident, name, score))



    # Update the weights according to the scores.

    def_weights = acc_sum / np.sum(acc_sum)

    att_weights = np.hstack((err_sum, hts_sum)) / (np.sum(err_sum) + np.sum(hts_sum))