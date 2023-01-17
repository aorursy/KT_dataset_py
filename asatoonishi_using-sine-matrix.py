# calculation and plot
import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# data processing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# predictor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
# train
df_train = pd.read_csv('../input/train.csv')
df_train['dataset'] = 'train'
# test
df_test = pd.read_csv('../input/test.csv')
df_test['dataset'] = 'test'
test_len = len(df_test)
# merge train and test
df = pd.concat([df_train, df_test], axis=0, ignore_index=True, sort=False)
df_len = len(df)
df.head()
df.tail()
def get_xyz(filename):    
    row = []
    xyz = []
    lattice = []
    
    with open(filename) as f:
        for line in f.readlines():
            row = line.split()
            if row[0] == 'atom':
                xyz.append((np.array(row[1:4], dtype=np.float), row[4]))
            elif row[0] == 'lattice_vector':
                lattice.append(np.array(row[1:4], dtype=np.float))
    
    return xyz, lattice
def get_sine_matrix(xyz, lattice):
    
    n_atom = len(xyz) # number of atoms in the cell
    
    distance_matrix = np.ones((n_atom, n_atom))
    A = np.transpose(lattice) # A = (a_1, a_2, a_3), defined as above
    B = LA.inv(A) # inverse matrix of A
    
    # matrix of distance
    for i in range(n_atom):
        for j in range(i):
            r_ij = np.dot(B, xyz[i][0] - xyz[j][0])
            sin_sq_r = (np.sin(np.pi * r_ij))**2
            distance = LA.norm(np.dot(A, sin_sq_r))
            distance_matrix[i, j], distance_matrix[j, i] = distance, distance
    # Note that diagonal components remain 1
    
    # matrix of charge by charge
    labels = np.transpose(xyz)[1] # element symbol labels
    labels = labels.reshape(-1,1)
    for at, charge in zip(['O', 'Al', 'Ga', 'In'], [8, 13, 31, 49]): # convert symbols into electric charges
        labels = np.where(labels==at, charge, labels)
    charge_matrix = np.dot(labels, np.transpose(labels)).astype(np.float)
    charge_matrix -= np.diag(np.diag(charge_matrix)) # let diagonal components zero
    charge_matrix += np.diag(0.5 * labels**2.4).astype(float) # from the definition
    
    # sine matrix
    sine_matrix = charge_matrix / distance_matrix
    
    return sine_matrix
def get_eigenspectrum(matrix):
    spectrum = LA.eigvalsh(matrix)
    spectrum = np.sort(spectrum)[::-1]
    
    return spectrum
spectrum_list = []

for index in range(df_len):
    
    dataset_label = df.dataset.values[index]
    row_id = df.id.values[index]
    filename = "../input/{}/{}/geometry.xyz".format(dataset_label, row_id)
    
    # file processing
    xyz, lattice = get_xyz(filename)
    # sine matrix
    sine_matrix = get_sine_matrix(xyz, lattice)
    # eigen spectrum
    spectrum = get_eigenspectrum(sine_matrix)
    
    spectrum_list.append(spectrum)
fig, axs = plt.subplots(1,5, figsize=(15, 6))
for i in range(5):
    ax = axs[i]
    plot_data = spectrum_list[i]
    ax.plot(range(len(plot_data)), plot_data)
    ax.hlines(0, 0, 80, colors='r')
plt.show()
spectrum_df = pd.DataFrame(spectrum_list).astype(np.float)
spectrum_df = spectrum_df.fillna(0)
# standard scaling
ss = StandardScaler()
spectrum_std_df = pd.DataFrame(ss.fit_transform(spectrum_df.values))
# PCA
pca = PCA(n_components=80)
spectrum_pca_df = pd.DataFrame(pca.fit_transform(spectrum_std_df.values))

spectrum_pca_df.head()
plt.scatter(x=spectrum_pca_df.loc[:100,0], y=spectrum_pca_df.loc[:100,1])
plt.show()
df.number_of_total_atoms = df.number_of_total_atoms.astype('int')
df['group_natoms'] = df.spacegroup.astype('str') + '_' + df.number_of_total_atoms.astype('str')
sns.lmplot(x='lattice_vector_1_ang', y='bandgap_energy_ev', hue='group_natoms', data=df, fit_reg=False)
plt.show()
df = df.join(pd.get_dummies(df.group_natoms))
df.drop(['group_natoms'], axis=1, inplace=True)
df.columns
df.drop(['dataset'], axis=1, inplace=True)
df.drop(['id', 'spacegroup', 'number_of_total_atoms'], axis=1, inplace=True)
df.columns
df_new = pd.concat([spectrum_pca_df, df], axis=1)
df_new.head()
df_new.drop([395, 126, 1215, 1886, 2075, 353, 308, 2154, 531, 1379, 2319, 2337, 2370, 2333], axis=0, inplace=True)
df_new.shape
df_len = len(df_new)
train_len = df_len - test_len
# X
X_train = df_new.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)[:train_len].values
X_test = df_new.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1)[train_len:].values
# y
y_formation = df_new['formation_energy_ev_natom'][:train_len].values
y_bandgap = df_new['bandgap_energy_ev'][:train_len].values
xgb_formation = xgb.XGBRegressor()
parameters = {
    'max_depth': [2, 3, 4],
    'n_estimators' : [100, 200, 300],
             }

cv_formation = GridSearchCV(xgb_formation, param_grid=parameters, cv=4, verbose=1)
cv_formation.fit(X_train, y_formation)
cv_formation.best_params_
xgb_bandgap = xgb.XGBRegressor()
parameters = {
    'max_depth': [2, 3, 4],
    'n_estimators' : [100, 200, 300],
             }

cv_bandgap = GridSearchCV(xgb_bandgap, param_grid=parameters, cv=4, verbose=1)
cv_bandgap.fit(X_train, y_bandgap)
cv_bandgap.best_params_
def plot_features(estimator, features):
    importances = estimator.feature_importances_
    plt.figure(figsize=(10, 15))
    plt.barh(range(len(importances)), importances , align='center')
    plt.yticks(np.arange(len(features)), features)
    plt.show()
features = df_new.drop(['formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1).columns

plot_features(cv_formation.best_estimator_, features)
plot_features(cv_bandgap.best_estimator_, features)
formation_pred = cv_formation.predict(X_test)
bandgap_pred = cv_bandgap.predict(X_test)
submission = pd.DataFrame(np.arange(1, test_len + 1), columns=['id'])
submission['formation_energy_ev_natom'] = formation_pred
submission['bandgap_energy_ev'] = bandgap_pred
submission.shape
submission.head()
submission.to_csv('submission.csv', index=False)