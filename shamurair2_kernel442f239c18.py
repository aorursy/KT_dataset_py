# --- library
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

from ase import Atoms
from ase.io import write
from pymatgen.io.vasp import Poscar
from pymatgen import Composition
from xenonpy.descriptor import Compositions, Structures

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

%precision 3
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train = pd.concat([train.iloc[:,0], train.iloc[:,-2:]], axis=1)
test = pd.DataFrame(test.iloc[:,0])
images = []

#for folder in range(1, 600+1):
for folder in range(1, 2400+1):
    positions = []
    cell = []
    symbols = []
    
    #with open("../input/test/" + str(folder) + "/geometry.xyz") as f:
    with open("../input/train/" + str(folder) + "/geometry.xyz") as f:
        for row, line in enumerate(f):
            fields = line.split(' ')
            # 3 line headers
            if row < 3:
                continue
            # unit cell
            elif row < 6:
                cell.append(fields[1:4])
            # atomic positions
            else:
                positions.append(fields[1:4])
                symbols.append(fields[4].replace('\n', ''))
    # atoms object
    atoms = Atoms(positions=np.array(positions, dtype=float),
                  symbols=symbols,
                  cell=np.array(cell, dtype=float))
    index = folder-1
    images.append(atoms)
for _ in range(len(images)):
    write('../poscar/train_{}_poscar'.format(_+1), images=images[_], \
          format='vasp', vasp5=True, sort=True)
    
    """
    write('../poscar/test_{}_poscar'.format(_+1), images=images[_], \
          format='vasp', vasp5=True, sort=True)
    """
ls_c, ls_s = [], []
for _ in range(len(images)):
    poscar = Poscar.from_file('../poscar/train_{}_poscar'.format(_+1))
    #poscar = Poscar.from_file('../poscar/test_{}_poscar'.format(_+1))
    s = poscar.structure
    
    c = Composition(s.composition)
    c = dict(c.get_el_amt_dict())
    
    ls_c.append(c)
    ls_s.append(s)
df_ = pd.DataFrame()
df_['composition'] = ls_c
df_['structure'] = ls_s
cal_comp = Compositions()
df_c = cal_comp.transform(df_['composition'])
cal_struc = Structures()
df_s = cal_struc.transform(df_['structure'])
df_all = pd.concat([train, df_c, df_s], axis=1)
#df_all.to_csv('../input/train_ver2.csv', index=False)
# --- data
train = pd.read_csv('../input/train_ver2.csv')
test = pd.read_csv('../input/test_ver2.csv')
train.head(3)
test.head(3)
df = train.copy()

X = df.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)
y1 = df.formation_energy_ev_natom
y2 = df.bandgap_energy_ev
std = StandardScaler()
std.fit(X)
X = std.transform(X)  
X_train, X_test, y1_train, y1_test\
= train_test_split(X, y1, train_size=0.8, random_state=2525)

X_train, X_test, y2_train, y2_test\
= train_test_split(X, y2, train_size=0.8, random_state=2525)
# formation_energy_ev_natom
model_1 = RidgeCV(cv=3, alphas=10**np.arange(-1, 2, 0.01))
model_1.fit(X_train, y1_train)
# bandgap_energy_ev
model_2 = RidgeCV(cv=3, alphas=10**np.arange(-1, 2, 0.01))
model_2.fit(X_train, y2_train)
print(model_1.alpha_)
print(model_2.alpha_)
y1_hat_train = model_1.predict(X_train)
y1_hat_test = model_1.predict(X_test)
fig = plt.figure(figsize = (12, 6), dpi=200)

ax = fig.add_subplot(121)
x_ = np.arange(-0.05, 0.45, 0.01)
y_ = x_
ax.scatter(y1_hat_train, y1_train,  color='lightgreen', edgecolor='black', lw=0.5, s=50)
ax.plot(x_, y_, color='blue', linewidth=2)
ax.set_xlabel('Predicted', fontsize='18')
ax.set_ylabel('Real', fontsize='18')
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.title("training_data", fontsize='20')

ax = fig.add_subplot(122)
ax.scatter(y1_hat_test, y1_test,  color='lightgreen', edgecolor='black', lw=0.5, s=50)
ax.plot(x_, y_, color='blue', linewidth=2)
ax.set_xlabel('Predicted', fontsize='18')
ax.set_ylabel('Real', fontsize='18')
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.title("test_data", fontsize='20')
y1_hat_train = np.where(y1_hat_train<0, 0, y1_hat_train)
y1_hat_test = np.where(y1_hat_test<0, 0, y1_hat_test)

rmsle_train = np.sqrt(msle(y1_train, y1_hat_train))
rmsle_test = np.sqrt(msle(y1_test, y1_hat_test))

print('RMSLE of training data: {:.3f}'.format(rmsle_train))
print('RMSLE of test data: {:.3f}'.format(rmsle_test))
r2_train = r2_score(y1_train, y1_hat_train)
r2_test = r2_score(y1_test, y1_hat_test)

print('R2 of training data: {:.3f}'.format(r2_train))
print('R2 of test data: {:.3f}'.format(r2_test))
y2_hat_train = model_2.predict(X_train)
y2_hat_test = model_2.predict(X_test)
fig = plt.figure(figsize = (12, 6), dpi=200)

ax = fig.add_subplot(121)
x_ = np.arange(0.0, 5.5, 0.1)
y_ = x_
ax.scatter(y2_hat_train, y2_train,  color='lightgreen', edgecolor='black', lw=0.5, s=50)
ax.plot(x_, y_, color='blue', linewidth=2)
ax.set_xlabel('Predicted', fontsize='18')
ax.set_ylabel('Real', fontsize='18')
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.title("training_data", fontsize='20')

ax = fig.add_subplot(122)
ax.scatter(y2_hat_test, y2_test,  color='lightgreen', edgecolor='black', lw=0.5, s=50)
ax.plot(x_, y_, color='blue', linewidth=2)
ax.set_xlabel('Predicted', fontsize='18')
ax.set_ylabel('Real', fontsize='18')
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
plt.title("test_data", fontsize='20')
y2_hat_train = np.where(y2_hat_train<0, 0, y2_hat_train)
y2_hat_test = np.where(y2_hat_test<0, 0, y2_hat_test)

rmsle_train = np.sqrt(msle(y2_train, y2_hat_train))
rmsle_test = np.sqrt(msle(y2_test, y2_hat_test))

print('RMSLE of training data: {:.3f}'.format(rmsle_train))
print('RMSLE of test data: {:.3f}'.format(rmsle_test))
r2_train = r2_score(y2_train, y2_hat_train)
r2_test = r2_score(y2_test, y2_hat_test)

print('R2 of training data: {:.3f}'.format(r2_train))
print('R2 of test data: {:.3f}'.format(r2_test))
df = test.copy()

X_ = df.drop(['id'], axis=1)
X_ = std.transform(X_)
y1_pred = model_1.predict(X_)
y1_pred = np.where(y1_pred<0, 0, y1_pred)
y2_pred = model_2.predict(X_)
y2_pred = np.where(y2_pred<0, 0, y2_pred)
submit = pd.DataFrame()
submit['id'] = df.id
submit['formation_energy_ev_natom'] = y1_pred
submit['bandgap_energy_ev'] = y2_pred
submit
#submit.to_csv('../submit/ridge_model.csv', index=False)