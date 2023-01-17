import pandas as pd

import numpy as np

from sklearn.manifold import TSNE

import sklearn.linear_model

import matplotlib.pyplot as plt
def do_tsne(ax, perpelxity, errors, top, show, random_state=0):

    tsne = TSNE(n_components=2, perplexity=perpelxity, random_state=random_state)

    embedding = tsne.fit_transform(errors/np.mean(errors))

    ax.scatter(embedding[:,0],embedding[:,1])



    for i in range(5):

        ax.scatter(

            embedding[top[i]][0],

            embedding[top[i]][1],

            c='r')

        ax.annotate('#{}'.format(i+1),xy=(embedding[top[i]][0],embedding[top[i]][1]))

    for s in show:

        ax.scatter(embedding[s[1]][0],embedding[s[1]][1],c='g')

        ax.annotate(s[0],xy=(embedding[s[1]][0],embedding[s[1]][1]))
folders = [

    'nips17-adversarial-learning-1st-round-results',

    'nips17-adversarial-learning-2nd-round-results',

    'nips17-adversarial-learning-3rd-round-results',

    'nips17-adversarial-learning-final-results'

]



num_non_targeted = [57, 62, 82, 95]
error_matrices = [pd.read_csv('../input/{}/error_matrix.csv'.format(f)) for f in folders]

for em in error_matrices:

    em['total'] = em.sum(axis=1)

top_defenses = [np.argsort(em['total']) for em in error_matrices]

num_defenses = [len(em) for em in error_matrices]



errors = [em.as_matrix()[:,1:] for em in error_matrices]

untargeted_attacks = [e[:,:num].transpose() for e, num in zip(errors, num_non_targeted)]

untargeted_scores = [np.sum(ut, axis=1) for ut in untargeted_attacks]

top_untargeted_attacks = [np.argsort(-uts) for uts in untargeted_scores]

num_untargeted = [len(ut) for ut in untargeted_attacks]



hit_target_class_dfs = [pd.read_csv('../input/{}/hit_target_class_matrix.csv'.format(f)) for f in folders]

targeted_attacks = [df.as_matrix()[:,num:].transpose() for df, num in zip(hit_target_class_dfs, num_non_targeted)]

targeted_scores = [np.sum(ta,axis=1) for ta in targeted_attacks]

top_targeted_attacks = [np.argsort(-ts) for ts in targeted_scores]

num_targeted = [len(ta) for ta in targeted_attacks]
df_defense = [pd.read_csv('../input/{}/defense_results.csv'.format(f)) for f in folders]

df_targeted = [pd.read_csv('../input/{}/targeted_attack_results.csv'.format(f)) for f in folders]

df_nontarg = [pd.read_csv('../input/{}/non_targeted_attack_results.csv'.format(f)) for f in folders]



for dfs in [df_defense, df_targeted, df_nontarg]:

    for df in dfs:

        df.set_index('KaggleTeamId', inplace=True)
plt.figure(figsize=(8,6))

plt.plot(num_defenses, label='Defenses')

plt.plot(num_untargeted, label='Non-targeted')

plt.plot(num_targeted, label='Targeted')

plt.xticks(range(4))

plt.legend()

plt.title('Number of Teams per Round')

plt.xlabel('Round')



print('Defenses: {}'.format(num_defenses))

print('Non-targeted: {}'.format(num_untargeted))

print('Targeted: {}'.format(num_targeted))
figs = [plt.subplots(figsize=[6,4]) for _ in range(3)]



for type_idx, comp_df in enumerate([df_defense, df_targeted, df_nontarg]):

    for r in range(4):

        figs[type_idx][1].plot(range(len(comp_df[r])), comp_df[r]['NormalizedScore'], label='Round {}'.format(r))



for i, ylim in enumerate([[0,1],[0,0.45],[0,0.8]]):

    figs[i][1].set_ylim(ylim)

        

figs[0][1].set_title('Defense: Normalized Scores by Rank by Round')

figs[1][1].set_title('Targeted Attack: Normalized Scores by Rank by Round')

figs[2][1].set_title('Non-Targeted Attack: Normalized Scores by Rank by Round')

        

for f in figs:

    handles, labels = f[1].get_legend_handles_labels()

    f[1].legend(handles, labels)

    f[0].tight_layout()
df_defense_all = df_defense[0]

for i, df in enumerate(df_defense[1:]):

    df_defense_all = df_defense_all.join(df, rsuffix='_{}'.format(i+1), how='outer')
df_plot = df_defense_all.sort_values(by='NormalizedScore_3', ascending=False)

mat_plot = df_plot[['NormalizedScore', 'NormalizedScore_1', 'NormalizedScore_2', 'NormalizedScore_3']].as_matrix()[:10,:]



plt.figure(figsize=(10,8))

plt.plot(mat_plot.transpose())

plt.ylim([0.75,1])

plt.title('Defense Scores for Top 10 Finishers by Round')

f, axes = plt.subplots(4, 3, figsize=(12,16))



defense_perplexity = 20

non_targ_perplexity = 20

targ_perpelxity = 20



do_tsne(axes[0,0], defense_perplexity, errors[0], top_defenses[0],

    [('adv_inc_v3',-2),('ens_adv_inc_rn_v2',-1),('inc_v3',-3)],

   random_state=2)

do_tsne(axes[0,1], non_targ_perplexity, untargeted_attacks[0], top_untargeted_attacks[0],

    [('baseline_fgsm',-3),('baseline_noop',-2),('baseline_randnoise',-1)],

    random_state=3)

do_tsne(axes[0,2], targ_perpelxity, targeted_attacks[0], top_targeted_attacks[0],

    [('baseline_itertc_10',-3),('baseline_itertc_20',-2),('baseline_steptc',-1)],

    random_state=4)



do_tsne(axes[1,0], defense_perplexity, errors[1], top_defenses[1],

    [('adv_inc_v3',-3),('ens_adv_inv_rn_v2',-2),('inc_v3',-1)],

   random_state=2)

do_tsne(axes[1,1], non_targ_perplexity, untargeted_attacks[1], top_untargeted_attacks[1],

    [('baseline_fgsm',-3),('baseline_noop',-2),('baseline_randnoise',-1)],

    random_state=3)

do_tsne(axes[1,2], targ_perpelxity, targeted_attacks[1], top_targeted_attacks[1],

    [('baseline_itertc_10',-3),('baseline_itertc_20',-2),('baseline_steptc',-1)],

    random_state=3)



do_tsne(axes[2,0], defense_perplexity, errors[2], top_defenses[2],

    [('adv_inc_v3',-3),('ens_adv_inv_rn_v2',-2),('inc_v3',-1)],

   random_state=1)

do_tsne(axes[2,1], non_targ_perplexity, untargeted_attacks[2], top_untargeted_attacks[2],

    [('baseline_fgsm',-3),('baseline_noop',-2),('baseline_randnoise',-1)],

    random_state=3)

do_tsne(axes[2,2], targ_perpelxity, targeted_attacks[2], top_targeted_attacks[2],

    [('baseline_itertc_10',-3),('baseline_itertc_20',-2),('baseline_steptc',-1)],

    random_state=3)



do_tsne(axes[3,0], defense_perplexity, errors[3], top_defenses[3],

    [('adv_inc_v3',-3),('ens_adv_inv_rn_v2',-2),('inc_v3',-1)],

   random_state=1)

do_tsne(axes[3,1], non_targ_perplexity, untargeted_attacks[3], top_untargeted_attacks[3],

    [('baseline_fgsm',-3),('baseline_noop',-2),('baseline_randnoise',-1)],

    random_state=3)

do_tsne(axes[3,2], targ_perpelxity, targeted_attacks[3], top_targeted_attacks[3],

    [('baseline_itertc_10',-3),('baseline_itertc_20',-2),('baseline_steptc',-1)],

    random_state=3)



axes[0,0].set_title('Defenses')

axes[0,1].set_title('Non_Targeted')

axes[0,2].set_title('Targeted')



axes[0,0].set_ylabel('Round 1', rotation=0, size='large')

axes[1,0].set_ylabel('Round 2', rotation=0, size='large')

axes[2,0].set_ylabel('Round 3', rotation=0, size='large')

axes[3,0].set_ylabel('Final', rotation=0, size='large')



plt.suptitle('TSNE Clustering Based on Pairwise Results', fontsize=16)

plt.tight_layout()

f.subplots_adjust(top=0.88)
for r in range(4):

    print('Round {}'.format(r+1))

    print('--------')

    

    X = untargeted_attacks[r][:,-3:]

    y = untargeted_scores[r] / num_defenses[r]

    baselines = [str(x) for x in error_matrices[r].iloc[-3:,0]]

    model = sklearn.linear_model.LinearRegression(fit_intercept=True)

    model.fit(X, y)

    print('Coefficients:')

    for b, c in zip(baselines, model.coef_):

        print('{}: {}'.format(b, c))

    print('Intercept: {}'.format(model.intercept_))

    print('Fit Score: {}'.format(model.score(X,y)))

    print()
for r in range(4):

    print('Round {}'.format(r+1))

    print('--------')

    

    X = targeted_attacks[r][:,-3:]

    y = targeted_scores[r] / num_defenses[r]

    baselines = [str(x) for x in error_matrices[r].iloc[-3:,0]]

    model = sklearn.linear_model.LinearRegression(fit_intercept=True)

    model.fit(X, y)

    print('Coefficients:')

    for b, c in zip(baselines, model.coef_):

        print('{}: {}'.format(b, c))

    print('Intercept: {}'.format(model.intercept_))

    print('Fit Score: {}'.format(model.score(X,y)))

    print()