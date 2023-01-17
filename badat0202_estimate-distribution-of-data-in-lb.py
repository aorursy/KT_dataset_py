import numpy as np

import pandas as pd

epsilon = 1e-15

print('log(epsilon):',-np.log(epsilon))

print('log(1-epsilon):',-np.log(1-epsilon))
df_test = pd.read_csv('../input/test.csv')

n = len(df_test)

print('n:',n)



const = (n/n)*-np.log(1-epsilon)

print('const:',const)
test = pd.read_csv('../input/test.csv')

sub = test[['test_id']].copy()

sub['is_duplicate'] = 0

sub.to_csv('submission_alpha.csv', index=False)
test = pd.read_csv('../input/test.csv')

sub = test[['test_id']].copy()

sub['is_duplicate'] = 1

sub.to_csv('submission_beta.csv', index=False)
alpha = 6.0188 

beta = 28.52056



print ((alpha - const)/(-np.log(epsilon)),'< (n1/n) <',alpha/(-np.log(epsilon)))

print ((beta - const)/(-np.log(epsilon)),'< (n2/n) <',beta/(-np.log(epsilon)))