import pandas as pd
df_1 = pd.read_csv('/kaggle/input/model-melanoma-2020/submission.csv')
df_2a = pd.read_csv('/kaggle/input/model-metadata-melanoma-2020/submission_best_mean.csv')
df_2b = pd.read_csv('/kaggle/input/model-metadata-melanoma-2020/submission_ensemble_mean.csv')
df_2c = pd.read_csv('/kaggle/input/model-metadata-melanoma-2020/submission_weighted_ensemble_mean.csv')

df_3a = pd.read_csv('/kaggle/input/melanoma-effnet-metdata/blended_effnets.csv')
df_3b = pd.read_csv('/kaggle/input/melanoma-effnet-metdata/ensembled.csv')
df_4 = pd.read_csv('/kaggle/input/minmax-melanoma-9619/submission.csv')
df_5 = pd.read_csv('/kaggle/input/melanoma-2020-9648/submission_9648.csv')
df_submission1 = df_1.copy()
df_submission1['target'] = 0.41 * df_1['target'] + 0.04 * df_2c['target'] + 0.55 * df_3b['target']
df_submission1.to_csv('submission1.csv', index=False)
df_submission1
df_submission2 = df_1.copy()
df_submission2['target'] = 0.075 * df_1['target'] + 0.1 * df_3b['target'] + 0.375 * df_4['target'] + 0.45 * df_5['target']
df_submission2.to_csv('submission2.csv', index=False)
df_submission2
df_submission3 = df_1.copy()
df_submission3['target'] = 0.9 * df_1['target'] + 0.1 * df_2b['target']
df_submission3.to_csv('submission3.csv', index=False)
df_submission3
df_submission4 = df_1.copy()
df_submission4['target'] = 0.3 * df_1['target'] + 0.1 * df_2b['target'] + 0.3 * df_3b['target'] + 0.3 * df_4['target']
df_submission4.to_csv('submission4.csv', index=False)
df_submission4