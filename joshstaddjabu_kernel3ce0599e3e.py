# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.manifold import TSNE
import scipy
import time
import tensorflow as tf
from tensorflow import keras
import datetime, os, io
from google.colab import files
%matplotlib inline
%load_ext tensorboard
# Uploading data
uploaded = files.upload()
df_og = pd.read_csv(io.BytesIO(uploaded['high_diamond_ranked_10min.csv']))
df_og.columns = map(lambda x:x.lower(), df_og.columns)
df_og['gameid'] = df_og['gameid'].astype(str)
numeric_columns = df_og.select_dtypes(['int64', 'float64']).columns
FILL_LIST = []
for cols in df_og[:]:
    if cols in numeric_columns:
        FILL_LIST.append(cols)
plt.figure(figsize=(35, 95))
plt.subplots_adjust(hspace=1, wspace=1)
for i, col in enumerate(FILL_LIST):
    try:
        plt.subplot(len(FILL_LIST), 7, i+1)
        sns.distplot(df_og[col], kde=False)
        plt.title(col)
    except TypeError:
        pass
plt.tight_layout()
# Just incase
df1 = df_og.copy()

# Feature Engineering comparative team performance variables
df1['bcspermin_diff'] = df1['bluecspermin'] - df1['redcspermin']
df1['btotexp_diff'] = df1['bluetotalexperience'] - df1['redtotalexperience']
df1['bavglvl_diff'] = df1['blueavglevel'] - df1['redavglevel']
df1['bwardsplaced_diff'] = df1['bluewardsplaced'] - df1['redwardsplaced']
df1['bwardsdestroyed_diff'] = df1['bluewardsdestroyed'] - df1['redwardsdestroyed']
df1['btowerdeaths_diff'] = df1['bluetowersdestroyed'] - df1['redtowersdestroyed']
df1['bkills_diff'] = df1['bluekills'] - df1['redkills']
df1['bdeaths_diff'] = df1['bluedeaths'] - df1['reddeaths']
df1['bgold_per_min_diff'] = df1['bluegoldpermin'] - df1['redgoldpermin']
df1['belite_diff'] = df1['blueelitemonsters'] - df1['redelitemonsters']
df1['bdrag_diff'] = df1['bluedragons'] - df1['reddragons']
df1['bheralds_diff'] = df1['blueheralds'] - df1['redheralds']
df1['blaneminions_diff'] = df1['bluetotalminionskilled'] - df1['redtotalminionskilled']
df1['bjgmionions_diff'] = df1['bluetotaljungleminionskilled'] - df1['redtotaljungleminionskilled']
df1['bteamtotminionsdiff'] = (df1['blueelitemonsters'] + df1['bluedragons'] + df1['blueheralds'] + df1['bluetotalminionskilled'] + df1['bluetotaljungleminionskilled']) - (df1['redelitemonsters'] + df1['reddragons'] + df1['redheralds'] + df1['redtotalminionskilled'] + df1['redtotaljungleminionskilled'])

df_ana = df1.loc[:, ['gameid', 'bluewins', 'bluegolddiff', 'blueexperiencediff', 'bkills_diff',
                     'bavglvl_diff', 'bluegoldpermin', 'bluetotalexperience',
                     'blueavglevel', 'bteamtotminionsdiff', 'bluekills',
                     'bcspermin_diff', 'blaneminions_diff', 'bluetotalgold']]
# Uploading data
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['data_chi3_anova9.csv']))
df = df.drop(['Unnamed: 0'], axis = 1)
df.columns = map(lambda x:x.lower(), df.columns)
df['gameid'] = df['gameid'].astype(str)
numeric_columns = df.select_dtypes(['int64', 'float64']).columns
FILL_LIST = []
for cols in df[:]:
    if cols in numeric_columns:
        FILL_LIST.append(cols)
plt.figure(figsize=(35, 95))
plt.subplots_adjust(hspace=1, wspace=1)
for i, col in enumerate(FILL_LIST):
    try:
        plt.subplot(len(FILL_LIST), 7, i+1)
        sns.distplot(df[col], kde=False)
        plt.title(col)
    except TypeError:
        pass
plt.tight_layout()
mask = np.zeros_like(df_ana.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(dpi=155)
sns.heatmap(abs(df_ana.corr()), mask=mask, vmin=0.4, cmap='gist_heat_r')
df_ana.corr()['bluewins'].sort_values(ascending=False)
X = df.iloc[:, 2:]
y = df['bluewins'].values.ravel()
scaler = preprocessing.StandardScaler()
X_stand = scaler.fit_transform(X)
X_stand_df = pd.DataFrame(X_stand, columns=X.columns)
feat_cols = [ X_stand_df.columns[i] for i in range(X_stand_df.shape[1]) ]
data2 = pd.DataFrame(X_stand_df,columns=feat_cols)
data2['y'] = y
data2['label'] = data2['y'].apply(lambda i: str(i))

# For reproducability of the results
np.random.seed(57)

# random observation selection
rndperm = np.random.permutation(data2.shape[0])

# Number of observations to use during cluster selection
N = 9000

# dataframe obj holding randomly selected data
data_subset = data2.loc[rndperm[:N],:].copy()

# data values obj from dataframe
df_subset = data_subset[feat_cols].values

time_start = time.time()

tsne = TSNE(n_components=2,
            verbose=1,
            n_iter=1000,
            perplexity=30,
            learning_rate=300,
            early_exaggeration=12)

tsne_results = tsne.fit_transform(data_subset)

print('t-SNE done! Time elasped: {} seconds'.format(time.time() - time_start))

data_subset['tsne-2d-one'] = tsne_results[:,0]
data_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 2),
    data=data_subset,
    legend="full",
    alpha=0.3)
# Binning features in new Dataframes for Visualizing
df_bins = df.copy()
df_bins['bluegolddiff_bins'] = pd.qcut(df_bins['bluegolddiff'], q=10)
pw_gd = df_bins.groupby('bluegolddiff_bins')['bluewins'].mean()
pw_gd = pw_gd.reset_index()
pw_gd.columns = ['bluegolddiff_bins', 'wp_bluegolddiff']
pw_gd['Delta'] = pw_gd['wp_bluegolddiff'].shift(-1) - pw_gd['wp_bluegolddiff']

df_bins['blueexperiencediff_bins'] = pd.qcut(df_bins['blueexperiencediff'], q=10)
pw_be = df_bins.groupby('blueexperiencediff_bins')['bluewins'].mean()
pw_be = pw_be.reset_index()
pw_be.columns = ['blueexperiencediff_bins', 'wp_blueexperiencediff']
pw_be['Delta'] = pw_be['wp_blueexperiencediff'].shift(-1) - pw_be['wp_blueexperiencediff']

df_bins['bluegoldpermin_bins'] = pd.qcut(df_bins['bluegoldpermin'], q=10)
pw_gm = df_bins.groupby('bluegoldpermin_bins')['bluewins'].mean()
pw_gm = pw_gm.reset_index()
pw_gm.columns = ['bluegoldpermin_bins', 'wp_bluegoldpermin']
pw_gm['Delta'] = pw_gm['wp_bluegoldpermin'].shift(-1) - pw_gm['wp_bluegoldpermin']

df_bins['bluetotalexperience_bins'] = pd.qcut(df_bins['bluetotalexperience'], q=10)
pw_te = df_bins.groupby('bluetotalexperience_bins')['bluewins'].mean()
pw_te = pw_te.reset_index()
pw_te.columns = ['bluetotalexperience_bins', 'wp_bluetotalexperience']
pw_te['Delta'] = pw_te['wp_bluetotalexperience'].shift(-1) - pw_te['wp_bluetotalexperience']

df_bins['bteamtotminionsdiff_bins'] = pd.qcut(df_bins['bteamtotminionsdiff'], q=10)
pw_tm = df_bins.groupby('bteamtotminionsdiff_bins')['bluewins'].mean()
pw_tm = pw_tm.reset_index()
pw_tm.columns = ['bteamtotminionsdiff_bins', 'wp_bteamtotminionsdiff']
pw_tm['Delta'] = pw_tm['wp_bteamtotminionsdiff'].shift(-1) - pw_tm['wp_bteamtotminionsdiff']

df_bins['blaneminions_diff_bins'] = pd.qcut(df_bins['blaneminions_diff'], q=10)
pw_lm = df_bins.groupby('blaneminions_diff_bins')['bluewins'].mean()
pw_lm = pw_lm.reset_index()
pw_lm.columns = ['blaneminions_diff_bins', 'wp_blaneminions_diff']
pw_lm['Delta'] = pw_lm['wp_blaneminions_diff'].shift(-1) - pw_lm['wp_blaneminions_diff']

df_bins['bluetotalgold_bins'] = pd.qcut(df_bins['bluetotalgold'], q=10)
pw_tg = df_bins.groupby('bluetotalgold_bins')['bluewins'].mean()
pw_tg = pw_tg.reset_index()
pw_tg.columns = ['bluetotalgold_bins', 'wp_bluetotalgold']
pw_tg['Delta'] = pw_tg['wp_bluetotalgold'].shift(-1) - pw_tg['wp_bluetotalgold']
plt.figure(figsize=(20,10))
sns.barplot(x='bluegolddiff_bins', y='wp_bluegolddiff', data=pw_gd)
pw_gd
plt.figure(figsize=(20,10))
sns.barplot(x='blueexperiencediff_bins', y='wp_blueexperiencediff', data=pw_be)
pw_be
plt.figure(figsize=(20,10))
sns.barplot(x='bluegoldpermin_bins', y='wp_bluegoldpermin', data=pw_gm)
pw_gm
plt.figure(figsize=(20,10))
sns.barplot(x='bluetotalexperience_bins', y='wp_bluetotalexperience', data=pw_te)
pw_te
plt.figure(figsize=(20,10))
sns.barplot(x='bteamtotminionsdiff_bins', y='wp_bteamtotminionsdiff', data=pw_tm)
pw_tm
plt.figure(figsize=(20,10))
sns.barplot(x='blaneminions_diff_bins', y='wp_blaneminions_diff', data=pw_lm)
pw_lm
plt.figure(figsize=(20,10))
sns.barplot(x='bluetotalgold_bins', y='wp_bluetotalgold', data=pw_tg)
pw_tg
X = df.iloc[:, 2:]
scaler = StandardScaler()
X_stand = scaler.fit_transform(X)
X_df = pd.DataFrame(X_stand, columns=X.columns)
y = df.iloc[:, 1:2].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# show model hyperameters
gbc_tuned = GradientBoostingClassifier(learning_rate=0.01,
                                       n_estimators=300,
                                       subsample=0.1,
                                       min_samples_leaf=2,
                                       min_samples_split=2, 
                                       max_depth=4,
                                       max_features=6,
                                       min_impurity_decrease=0.1)
gbc_tuned.fit(X_train, y_train)
# we are making predictions here
y_preds_train = gbc_tuned.predict(X_train)
y_preds_test = gbc_tuned.predict(X_test)
feature_importance = gbc_tuned.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
%%time
def build_mlp():
  model = keras.Sequential([keras.layers.Dense(70, input_shape=(None, 7), kernel_initializer='glorot_normal',
                                               activation='sigmoid',),
                            keras.layers.Dense(70, activation='selu', kernel_initializer='normal'),
                            keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')])
  optomizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.1)
  model.compile(optimizer=optomizer, loss='binary_crossentropy', metrics=['accuracy'])
  return model

logdir9 = os.path.join("logs23", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir9, histogram_freq=1)
  
mlp_model = build_mlp()
mlp_model.fit(X_train, y_train, batch_size=15, epochs=100, verbose=0, callbacks=[tensorboard_callback])

y_mlp = mlp_model.predict(X_test).ravel()
%tensorboard --logdir logs23
## Grid to determine each layer's activation function
def create_model(activation='softmax'):
  model = keras.Sequential([keras.layers.Dense(70, input_shape=(None, 7), activation='sigmoid'),
                            keras.layers.Dense(70, activation='selu'),
                            keras.layers.Dense(1, activation=activation)])
  opt = tf.keras.optimizers.Adagrad(learning_rate=0.005, initial_accumulator_value=0.1, epsilon=1e-07)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

# fix random seed for reproductibility
seed = 7
np.random.seed(seed)
# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)

# defining grid search parameters
activation = ["tanh", "sigmoid", 'relu', 'selu', 'softsign', 'softmax']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print("%f (%f) with: %r" % (mean, stdev, param))
# STAR GRID CELL
def create_model(learning_rate=0.001, momentum=0.0, optimizer='SGD',
                 loss='binary_crossentropy', init_mode='uniform', neurons=1,
                 activation='sigmoid', dropout_rate=0.0, weight_constraint=0):
  model = keras.Sequential([keras.layers.Dense(neurons, input_shape=(None, 7), kernel_initializer='glorot_normal', activation=activation, kernel_constraint=maxnorm(weight_constraint),),
                            keras.layers.Dropout(dropout_rate),
                            keras.layers.Dense(neurons, activation=activation, kernel_initializer='normal'),
                            keras.layers.Dropout(dropout_rate),
                            keras.layers.Dense(1, activation='sigmoid', kernel_initializer='he_normal')])
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
  return model

model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
weight_constraint = [0, 2, 4, 6]
dropout_rate = [0.0, 0.2, 0.4, 0.6, 0.8]
activation = ["tanh", "sigmoid", 'relu', 'softmax'] # could add 'selu'
init_mode = ['uniform', 'normal', 'glorot_normal', 'he_normal', 'he_uniform'] # ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
optimizer = ['RMSprop']
loss = ['binary_crossentropy']
neurons = [70, 85, 100]
learning_rate = [0.001, 0.01, 0.05, 0.1]
momentum = [.0, .2, .4]
batch_size = [14, 34, 72]
epochs = [200]

# Create Parameter Grid Object & Train Model
param_grid = dict(learning_rate=learning_rate, momentum=momentum,
                  batch_size=batch_size, epochs=epochs,
                  loss=loss, optimizer=optimizer,
                  activation=activation,
                  weight_constraint=weight_constraint,
                  dropout_rate=dropout_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
df['bluewins'].value_counts()
fpr_gbc, tpr_gbc, thresholds_gbc = roc_curve(y_test, y_preds_test)
auc_gbc = auc(fpr_gbc, tpr_gbc)

fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, y_mlp)
auc_mlp = auc(fpr_mlp, tpr_mlp)

plt.plot(fpr_gbc, tpr_gbc, label='GBC (area = {:.3f})'.format(auc_gbc))
plt.plot(fpr_mlp, tpr_mlp, label='MLP (area = {:.3f})'.format(auc_mlp))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
scores = cross_val_score(gbc_tuned, X_train, y_train, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print('GBC AUC : ', auc_gbc)
print()
print('Confusion Matrix GBC')
print(confusion_matrix(y_test, y_preds_test))
print()
print('Classification Matrix GBC')
print(classification_report(y_test, y_preds_test))
# Add cross_validation_score for mlp
print('MLP AUC : ', auc_mlp)
print()
print('Confusion Matrix MLP')
print(confusion_matrix(y_test, y_mlp.round(decimals=0, out=None)))
print()
print('Classification Matrix MLP')
print(classification_report(y_test, y_mlp.round(decimals=0, out=None)))
plt.title('Blue Wins Distribution')
sns.distplot(y)
plt.title("MLP Prediction Distribution")
sns.distplot(y_mlp)
