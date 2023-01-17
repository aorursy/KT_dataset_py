import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import train_test_split



from sklearn.metrics import mean_squared_error



from sklearn.metrics.pairwise import polynomial_kernel

from sklearn.metrics.pairwise import rbf_kernel

from sklearn.metrics.pairwise import laplacian_kernel



x_columns = [i for i in train.columns if i not in list(['id','formation_energy_ev_natom','bandgap_energy_ev'])]



label1 = 'formation_energy_ev_natom'

label2 = 'bandgap_energy_ev'



X = train[x_columns]

y = train[[label1,label2]]



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=2017)



X_train = X_train.as_matrix()

X_valid = X_valid.as_matrix()



y_train_values1 = np.log1p(y_train['formation_energy_ev_natom'].values)

y_train_values2 = np.log1p(y_train['bandgap_energy_ev'].values)

y_valid_values1 = np.log1p(y_valid['formation_energy_ev_natom'].values)

y_valid_values2 = np.log1p(y_valid['bandgap_energy_ev'].values)
clf1 = KernelRidge(kernel ='linear', alpha=1.0)

clf2 = KernelRidge(kernel ='linear', alpha=1.0)



clf1.fit(X_train,y_train_values1)

clf2.fit(X_train,y_train_values2)



preds1 = clf1.predict(X_valid)

preds2 = clf2.predict(X_valid)



y_pred1 = np.exp(preds1)-1

y_pred2 = np.exp(preds2)-1



rsme_valid1 = np.sqrt(mean_squared_error(y_valid_values1,preds1))

rsme_valid2 = np.sqrt(mean_squared_error(y_valid_values2,preds2))



rsme_total = np.sqrt(rsme_valid1*rsme_valid1+rsme_valid2*rsme_valid2)

print('RSME for formation energy:')

print(rsme_valid1)

print('RSME for band gap:')

print(rsme_valid2)

print('RSME for total:')

print(rsme_total)
clf3 = KernelRidge(kernel ='polynomial', alpha=1.0)

clf4 = KernelRidge(kernel ='polynomial', alpha=1.0)



clf3.fit(X_train,y_train_values1)

clf4.fit(X_train,y_train_values2)



preds1 = clf3.predict(X_valid)

preds2 = clf4.predict(X_valid)



y_pred1 = np.exp(preds1)-1

y_pred2 = np.exp(preds2)-1



rsme_valid1 = np.sqrt(mean_squared_error(y_valid_values1,preds1))

rsme_valid2 = np.sqrt(mean_squared_error(y_valid_values2,preds2))



rsme_total = np.sqrt(rsme_valid1*rsme_valid1+rsme_valid2*rsme_valid2)

print('RSME for formation energy:')

print(rsme_valid1)

print('RSME for band gap:')

print(rsme_valid2)

print('RSME for total:')

print(rsme_total)
clf5 = KernelRidge(kernel ='rbf', alpha=1.0)

clf6 = KernelRidge(kernel ='rbf', alpha=1.0)



clf5.fit(X_train,y_train_values1)

clf6.fit(X_train,y_train_values2)



preds1 = clf5.predict(X_valid)

preds2 = clf6.predict(X_valid)



y_pred1 = np.exp(preds1)-1

y_pred2 = np.exp(preds2)-1



rsme_valid1 = np.sqrt(mean_squared_error(y_valid_values1,preds1))

rsme_valid2 = np.sqrt(mean_squared_error(y_valid_values2,preds2))



rsme_total = np.sqrt(rsme_valid1*rsme_valid1+rsme_valid2*rsme_valid2)

print('RSME for formation energy:')

print(rsme_valid1)

print('RSME for band gap:')

print(rsme_valid2)

print('RSME for total:')

print(rsme_total)
clf7 = KernelRidge(kernel ='laplacian', alpha=1.0)

clf8 = KernelRidge(kernel ='laplacian', alpha=1.0)



clf7.fit(X_train,y_train_values1)

clf8.fit(X_train,y_train_values2)



preds1 = clf7.predict(X_valid)

preds2 = clf8.predict(X_valid)



y_pred1 = np.exp(preds1)-1

y_pred2 = np.exp(preds2)-1



rsme_valid1 = np.sqrt(mean_squared_error(y_valid_values1,preds1))

rsme_valid2 = np.sqrt(mean_squared_error(y_valid_values2,preds2))



rsme_total = np.sqrt(rsme_valid1*rsme_valid1+rsme_valid2*rsme_valid2)

print('RSME for formation energy:')

print(rsme_valid1)

print('RSME for band gap:')

print(rsme_valid2)

print('RSME for total:')

print(rsme_total)
X_test = test[x_columns]

X_test = X_test.as_matrix()



preds1 = clf3.predict(X_test)

preds2 = clf4.predict(X_test)

y_pred1 = np.exp(preds1)-1

y_pred2 = np.exp(preds2)-1



krr = pd.DataFrame()

krr['id'] = test['id']

krr['formation_energy_ev_natom'] = y_pred1

krr['bandgap_energy_ev'] = y_pred2

krr.to_csv("krr_sub.csv", index=False)