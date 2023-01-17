import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

print(os.listdir("../input"))
train=pd.read_csv("../input/training.csv")

test=pd.read_csv("../input/test.csv")



print("train.shape:{} test.shape:{}".format(train.shape, test.shape))
train.head()
print("Missing values in train: ", train.isnull().sum().sum())
print("Missing values in test: ", train.isnull().sum().sum())
uncommon_features = []

for i in train.columns:

    if i not in test.columns:

        uncommon_features.append(i)

        

print("Extra features in train: ", uncommon_features)
def add_features(data):

    df = data.copy()

    df['NEW_FD_SUMP'] = df['FlightDistance'] / (df['p0_p'] + df['p1_p'] + df['p2_p'])

    df['NEW5_lt'] = df['LifeTime'] * (df['p0_IP'] + df['p1_IP'] + df['p2_IP']) / 3

    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)

    df['flight_dist_sig2'] = (df['FlightDistance'] / df['FlightDistanceError']) ** 2

    df['flight_dist_sig'] = df['FlightDistance'] / df['FlightDistanceError']

    df['NEW_IP_dira'] = df['IP'] * df['dira']

    df['p0p2_ip_ratio'] = df['IP'] / df['IP_p0p2']

    df['p1p2_ip_ratio'] = df['IP'] / df['IP_p1p2']

    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)

    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)

    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)

    df['NEW_iso_abc'] = df['isolationa'] * df['isolationb'] * df['isolationc']

    df['NEW_iso_def'] = df['isolationd'] * df['isolatione'] * df['isolationf']

    df['NEW_pN_IP'] = df['p0_IP'] + df['p1_IP'] + df['p2_IP']

    df['NEW_pN_p']  = df['p0_p'] + df['p1_p'] + df['p2_p']

    df['NEW_IP_pNpN'] = df['IP_p0p2'] * df['IP_p1p2']

    df['NEW_pN_IPSig'] = df['p0_IPSig'] + df['p1_IPSig'] + df['p2_IPSig']

    df['NEW_FD_LT'] = df['FlightDistance'] / df['LifeTime']

    return df
train_added = add_features(train)

test_added = add_features(test)

print("Total Number of Features: ", train_added.shape[1])
print("Eliminate features")

filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',

              'SPDhits','CDF1', 'CDF2', 'CDF3',

              'isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt',

              'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',

              'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf',

              'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT',

              'p0_IP', 'p1_IP', 'p2_IP',

              'IP_p0p2', 'IP_p1p2',

              'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof',

              'p0_IPSig', 'p1_IPSig', 'p2_IPSig',

              'DOCAone', 'DOCAtwo', 'DOCAthree']
features = list(f for f in train_added.columns if f not in filter_out)
scaler = StandardScaler()

X_train = scaler.fit_transform(train_added[features])

X_test = scaler.fit_transform(test_added[features])
y_train = train['signal']
print("Shape of Training data: ", X_train.shape, "\nShape of Testing data: ", X_test.shape, 

      "\nShape of Training Labels: ", y_train.shape)
pca = PCA().fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
def pca_summary(pca, standardized_data, out=True):

    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]

    a = list(np.std(pca.transform(standardized_data), axis=0))

    b = list(pca.explained_variance_ratio_)

    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]

    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), 

                                         ("varprop", "Proportion of Variance"), 

                                         ("cumprop", "Cumulative Proportion")])

    summary = pd.DataFrame(list(zip(a, b, c)), index=names, columns=columns)

    

    if out:

        print("Importance of components:")

        display(summary)

    return summary
summary = pca_summary(pca, X_train_pca)
def screeplot(pca, standardized_values):

    y = np.std(pca.transform(standardized_values), axis=0)**2

    x = np.arange(len(y)) + 1

    plt.plot(x, y, "o-")

    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)

    plt.ylabel("Variance")

    plt.show()



screeplot(pca, X_train)
X_train_pca_df = pd.DataFrame(X_train_pca[:,0:15])

X_train_pca_df.head()
X_test_pca_df = pd.DataFrame(X_test_pca[:,0:15])

X_test_pca_df.head()
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import to_categorical

from keras.datasets import mnist

from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG

from keras.utils import np_utils
X_train_pca_df.shape
y_train_nn = y_train.values.reshape(1, -1)
model = Sequential()

model.add(Dense(128, input_dim=15, kernel_initializer='uniform', activation='relu'))

model.add(Dense(64, kernel_initializer='uniform', activation='relu'))

model.add(Dense(32, kernel_initializer='uniform', activation='elu'))

model.add(Dense(16, kernel_initializer='uniform', activation='relu'))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Fit model

model.fit(X_train_pca_df, y_train_nn.T, epochs=10, batch_size=32)
preds_nn = model.predict(X_test_pca_df)
sub_nn = pd.DataFrame({"id": test['id'].values,"prediction": preds_nn.reshape(-1)})

sub_nn.to_csv("submit_nn.csv", index=False)