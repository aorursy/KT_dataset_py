import pandas as pd
non_targeted_attack_results = pd.read_csv('../input/non_targeted_attack_results.csv', index_col=0)

disqualified = non_targeted_attack_results.index[non_targeted_attack_results['Score'] < 0]

targeted_attack_results = pd.read_csv('../input/targeted_attack_results.csv', index_col=0)

disqualified_t = targeted_attack_results.index[targeted_attack_results['Score'] < 0]

hit_target_class_matrix = pd.read_csv('../input/hit_target_class_matrix.csv', index_col=0)

defense_results = pd.read_csv('../input/defense_results.csv', index_col=0)
accuracy_matrix = pd.read_csv('../input/accuracy_matrix.csv', index_col=0)

accuracy_matrix.drop(disqualified, axis=1, inplace=True)

accuracy_matrix.drop(disqualified_t, axis=1, inplace=True)

defense_score = accuracy_matrix.sum(axis=1)
defense_results = defense_results.merge(defense_score.to_frame(name='Computed Score'), left_index=True, right_index=True).sort_values(by='Computed Score',ascending=False)
defense_results[['Score','Computed Score']].sort_values(by='Computed Score',ascending=False)
targeted_attack_score = hit_target_class_matrix.transpose().sum(axis=1)

targeted_attack_results = targeted_attack_results.merge(targeted_attack_score.to_frame(name='Computed Score'), left_index=True, right_index=True)
targeted_attack_results[['Score','Computed Score']].sort_values(by='Computed Score',ascending=False)