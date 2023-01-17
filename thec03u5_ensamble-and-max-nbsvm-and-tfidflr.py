import numpy as np, pandas as pd

nbsvm = '../input/nbsvm/submissionNBSVM.csv'
tfidflr = '../input/tfidf-and-lr/word_submission.csv'
p_nbsvm = pd.read_csv(nbsvm)
p_tfidflr = pd.read_csv(tfidflr)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_nbsvm.copy()
p_res[label_cols] = (p_tfidflr[label_cols] + p_nbsvm[label_cols]) / 2
p_res2 = p_nbsvm.copy()
for i, j in enumerate(p_res2):
    if j == 'id':
        continue
    temp_df = pd.concat([p_tfidflr[j], p_nbsvm[j]], axis=1)
    print (temp_df.head())
    p_res2[j] = temp_df.apply( max, axis=1 )
p_res3 = p_nbsvm.copy()
for i, j in enumerate(p_res2):
    if j == 'id':
        continue
    temp_df = pd.concat([p_tfidflr[j], p_nbsvm[j]], axis=1)
    print (temp_df.head())
    p_res3[j] = temp_df.apply(lambda r: np.max(r) if np.mean(r) > 0.5 else np.mean(r), axis=1)
p_res3.head()
p_res.to_csv('submissionAvg.csv', index=False)
p_res2.to_csv('submissionMax.csv', index=False)
p_res3.to_csv('submissionCondMax.csv', index=False)
p_res2.head()
