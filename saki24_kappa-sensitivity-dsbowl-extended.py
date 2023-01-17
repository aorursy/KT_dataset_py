import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import cohen_kappa_score

import random

random.seed(42)

np.random.seed(42)

gt = np.random.choice([0,1,2,3], 1000, p=(0.24, 0.14, 0.12, 0.50))

pred = gt.copy()

noise = np.random.choice([- 3,-2, -1, 0, 1, 2, 3], 1000, p=(0.05,0.1, 0.15, 0.4, 0.15, 0.1, 0.05))

pred_noisy = pred + noise

pred_noisy[pred_noisy > 3]  -=3

pred_noisy[pred_noisy < 0] += 3
df = pd.DataFrame({'reality':pred, 'pred_noisy': pred_noisy})

df[['reality', 'pred_noisy']].groupby(['reality', 'pred_noisy'])['reality'].count().to_frame('count').reset_index()
cohen_kappa_score(gt, pred_noisy, weights='quadratic')
for i in range(10):

    errs = np.where(pred_noisy != gt)[0]

    pred_noisy[errs[:10]] = gt[errs[:10]]

    print(cohen_kappa_score(gt, pred_noisy, weights='quadratic'))

    
df = pd.DataFrame({'reality':pred, 'pred_noisy': pred_noisy})

df[['reality', 'pred_noisy']].groupby(['reality', 'pred_noisy'])['reality'].count().to_frame('count').reset_index()
print('The original kappa was: ' + str(cohen_kappa_score(gt, df['pred_noisy'], weights='quadratic')))

to_fix = [0,1,2,3]

for h in to_fix:

    for i in range(0,4):

        if i != h:

            errs = np.where((df['pred_noisy'] != df['reality']) & (df['reality'] == h) & (df['pred_noisy'] == i))[0]

            df['new_preds'] = df['pred_noisy']

            df['new_preds'][errs[:10]] = df['reality'][errs[:10]]

            print('Kappa after fixing another ten ' + str(h) + ' missclassified as ' + str(i) + ': ' + str(cohen_kappa_score(gt, df['new_preds'], weights='quadratic')) + '. This is '+str(cohen_kappa_score(gt, df['new_preds'], weights='quadratic') - cohen_kappa_score(gt, df['pred_noisy'], weights='quadratic')) + ' improvement from the original')
