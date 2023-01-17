# Imports

from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier

import random
df_train = pd.read_csv('/kaggle/input/equipfails/equip_failures_training_set.csv')

df_test = pd.read_csv('/kaggle/input/equipfails/equip_failures_test_set.csv')
df_train.target.value_counts()
# analyzing one of the sensors:

hist_cols = [i for i in df_train.columns if 'sensor7_' in i]

hist_cols
df_sen = df_train[hist_cols]
# # agger 0s

# for i in range(0, int(len(df_train[df_train.target==0])/10), 10):

#     print(df_train.loc[:10])
# dropping all hists for now

non_hist_cols_train = [i for i in df_train.columns if 'histogram' not in i]

non_hist_cols_test = [i for i in df_test.columns if 'histogram' not in i]



df_train = df_train[non_hist_cols_train]

df_test = df_test[non_hist_cols_test]
# dealing with the datatypes

df_train = df_train.applymap(lambda x: np.nan if x == 'na' else float(x))

df_test = df_test.applymap(lambda x: np.nan if x == 'na' else float(x))
# the number of rows and columns before dropping the nans:

print("TRAIN: ", "rows: ", len(df_train), "cols: ", len(df_train.columns))

print("TEST: ", "rows: ", len(df_test), "cols: ", len(df_test.columns))
nan_dist_train = {}

nan_dist_test = {}



for i in df_train.columns:    

    nan_dist_train[i] = df_train[i].isna().sum()

        

for i in df_test.columns:    

    nan_dist_test[i] = df_test[i].isna().sum()
# plotting nan dist.

plt.scatter(range(len(nan_dist_train)), list(nan_dist_train.values()))

plt.xticks(range(len(nan_dist_train)), list(nan_dist_train.keys()))

plt.title('NaN vs Column')

plt.show()
plt.scatter(range(len(nan_dist_test)), list(nan_dist_test.values()))

plt.xticks(range(len(nan_dist_test)), list(nan_dist_test.keys()))

plt.title('NaN vs Column')

plt.show()
# drop the column if a percentage of the column (its rows) is na

# 6 percent

for i in df_train.columns:

    if df_train[i].isna().sum() > 0.06 * len(df_train):

        df_train.drop(i, axis=1, inplace=True)
# drop the column if a percentage of the column (its rows) is na

# 6 percent

for i in df_test.columns:

    if df_test[i].isna().sum() > 0.06 * len(df_test):

        df_test.drop(i, axis=1, inplace=True)
df_train.fillna(value=0, axis=1, inplace=True)

df_test.fillna(value=0, axis=1, inplace=True)
# the number of rows and columns after dropping the nans:

print("rows: ", len(df_train), "cols: ", len(df_train.columns))

print("rows: ", len(df_test), "cols: ", len(df_test.columns))
# See if there are any nans left:

print(df_train.isna().sum().sum())

print(df_test.isna().sum().sum())
zero_dist_train = {}

zero_dist_test = {}



for i in df_train.columns:

    if 0 in df_train[i].value_counts():    

        zero_dist_train[i] = df_train[i].value_counts()[0]

        

for i in df_test.columns:

    if 0 in df_train[i].value_counts():    

        zero_dist_test[i] = df_train[i].value_counts()[0]
plt.scatter(range(len(zero_dist_train)), list(zero_dist_train.values()))

plt.xticks(range(len(zero_dist_train)), list(zero_dist_train.keys()))

plt.title('Number of Zeroes vs Columns')

plt.show()
plt.scatter(range(len(zero_dist_test)), list(zero_dist_test.values()))

plt.xticks(range(len(zero_dist_test)), list(zero_dist_test.keys()))

plt.title('Number of Zeroes vs Columns')

plt.show()
# the number of rows and columns before dropping the nans:

print("TRAIN: ", "rows: ", len(df_train), "cols: ", len(df_train.columns))

print("TEST: ", "rows: ", len(df_test), "cols: ", len(df_test.columns))
# the number of rows and columns before dropping the nans:

print("TRAIN: ", "rows: ", len(df_train), "cols: ", len(df_train.columns))

print("TEST: ", "rows: ", len(df_test), "cols: ", len(df_test.columns))
# 50

zero_index = np.array([i for i in df_train[df_train.target == 0].index])

zero_index = np.random.choice(zero_index, int(0.50 * len(zero_index)), replace = False)

df_train = pd.concat([df_train[df_train.target == 1], df_train.loc[zero_index]])
df_train.target.value_counts()
d = {}



df_train_1 = df_train[df_train.target == 1]



for i in df_train_1.columns:

    if (i != 'target') or (i != 'id'):

        mu = df_train_1[i].mean()

        std = df_train_1[i].std()

        

        signal = np.random.normal(mu, std, 4000)

        

        d[i] = signal



new_df = pd.concat([df_train, pd.DataFrame(d)])    
new_df.target.value_counts()
import seaborn as sns

#Using Pearson Correlation

plt.figure(figsize=(12,10))

cor = df_train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
df_train_0 = new_df[new_df.target == 0]

new_df = new_df[new_df.target == 1]

new_df = pd.concat([new_df, df_train_0.groupby(df_train_0.index // 10).mean()])
# # # sensor61_measure, sensor17_measure, sensor35_measure

# df_train['sensor35_measure_2'] = df_train.sensor35_measure * df_train.sensor17_measure

# df_train['sensor35_measure_3'] = df_train.sensor35_measure * df_train.sensor35_measure

# df_train['sensor35_measure_4'] = df_train.sensor17_measure * df_train.sensor61_measure

# df_train['sensor35_measure_5'] = df_train.sensor1_measure * df_train.sensor1_measure

# df_train['sensor35_measure_6'] = df_train.sensor17_measure * df_train.sensor17_measure



# df_train['sensor35_measure_5'] = df_train.sensor1_measure * df_train.sensor1_measure

# df_train['sensor35_measure_6'] = df_train.sensor1_measure * df_train.sensor35_measure

# df_train['sensor35_measure_7'] = df_train.sensor16_measure * df_train.sensor35_measure



# df_train['sensor17_measure_2'] = df_train.sensor17_measure * df_train.sensor61_measure

# df_train['sensor35_measure_2'] = df_train.sensor35_measure * df_train.sensor17_measure
# # Create correlation matrix

# corr_matrix = df_train.corr().abs()



# # Select upper triangle of correlation matrix

# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# # Find index of feature columns with correlation greater than 0.95

# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# df_train.drop(df_train[to_drop], axis=1, inplace=True)
# the number of rows and columns before dropping the nans:

print("TRAIN: ", "rows: ", len(df_train), "cols: ", len(df_train.columns))

print("TEST: ", "rows: ", len(df_test), "cols: ", len(df_test.columns))
# train set

y_train = df_train['target']

ids_train = df_train['id']



df_train.drop('id', axis=1, inplace=True)

df_train.drop('target', axis=1, inplace=True)



# print(df_train.head())



X_train = df_train.values



# test set

ids_test = df_test['id']

# print(df_test.head())

df_test.drop('id', axis=1, inplace=True)



X_test = df_test.values
# Scaler models:

# scaler_model = StandardScaler()

scaler_model = MinMaxScaler()
inter = [i for i in df_train.columns if i in df_test.columns]

df_train = df_train[inter]

df_test = df_test[inter]

len(df_test.columns)
X_train_internal, X_test_internal, y_train_interal, y_test_interal = train_test_split(X_train, y_train, test_size=0.10, random_state=0)
scaler_model.fit(X_train_internal)



X_train_internal = scaler_model.transform(X_train_internal)

X_test_internal = scaler_model.transform(X_test_internal)

X_test = scaler_model.transform(X_test)
# ADHOC RF

from sklearn.ensemble import AdaBoostClassifier

ad_hocModel_2 = AdaBoostClassifier()

ad_hocModel_2.fit(X_train_internal, y_train_interal)
y_adhoc_pred = ad_hocModel_2.predict(X_test_internal)



ADA_F1 = f1_score(y_pred=y_adhoc_pred, y_true=y_test_interal, average='binary')



def plot_confusion_matrix(y_true, y_pred,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

#     classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

#            xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

plot_confusion_matrix(y_test_interal, y_adhoc_pred, normalize=True,

                      title='Normalized confusion matrix')





y_pred_test = ad_hocModel_2.predict(X_test)

df_out = pd.DataFrame({'id': ids_test, 'target': y_pred_test}, dtype=int)

df_out.to_csv('out.csv', index=None)
# ADHOC RF

ad_hocModel = RandomForestClassifier(max_depth=10, random_state=0)

ad_hocModel.fit(X_train_internal, y_train_interal)


y_adhoc_pred = ad_hocModel.predict(X_test_internal)



def plot_confusion_matrix(y_true, y_pred,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

#     classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

#            xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

plot_confusion_matrix(y_test_interal, y_adhoc_pred, normalize=True,

                      title='Normalized confusion matrix')





y_pred_test = ad_hocModel.predict(X_test)

df_out = pd.DataFrame({'id': ids_test, 'target': y_pred_test}, dtype=int)

df_out.to_csv('out.csv', index=None)
import pandas as pd

feature_importances = pd.DataFrame(ad_hocModel.feature_importances_,

                                   index = df_train.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)



feature_importances[:10]
# plt.scatter(df_train.sensor61_measure, df_train.sensor1_measure, c=y_train_interal)
pca_train_internal = PCA(n_components=2).fit(X_train_internal)

pca_test_internal = PCA(n_components=2).fit(X_test_internal)

pca_test = PCA(n_components=2).fit(X_test)



X_train_internal_pca = pca_train_internal.transform(X_train_internal)

X_test_internal_pca = pca_test_internal.transform(X_test_internal)

X_test_pca = pca_test.transform(X_test)
pca_test.explained_variance_ratio_
plt.scatter(X_train_internal_pca[:, 0], X_train_internal_pca[:, 1])
# model = RandomForestClassifier(max_depth=4)

model = RandomForestClassifier(max_depth=8)

model.fit(X_train_internal_pca, y_train_interal)
y_pred_train_interal = model.predict(X_train_internal_pca)

y_pred_test_interal = model.predict(X_test_internal_pca)
f1_score(y_pred=y_pred_train_interal, y_true=y_train_interal, average='binary') 
F1_RF = f1_score(y_pred=y_pred_test_interal, y_true=y_test_interal, average='binary') 

F1_RF
accuracy_score(y_pred=y_pred_train_interal, y_true=y_train_interal)
accuracy_score(y_pred=y_pred_test_interal, y_true=y_test_interal)
plt.scatter(X_train_internal_pca[:, 0], X_train_internal_pca[:, 1], c=y_train_interal)
def plot_confusion_matrix(y_true, y_pred,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

#     classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

#            xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

plot_confusion_matrix(y_test_interal, y_pred_test_interal, normalize=True,

                      title='Normalized confusion matrix')

y_pred_test = model.predict(X_test_pca)
df_out = pd.DataFrame({'id': ids_test, 'target': y_pred_test}, dtype=int)

df_out.to_csv('out.csv', index=None)
(y_pred_test == 1).sum()
y_pred_test_prob = model.predict_proba(X_test_pca)
y_pred_test_prob
import pickle

filename = "RF_model.pkl"

pickle.dump(model, open(filename, "wb"))
np.savetxt("test.csv", X_test_pca, delimiter=",")
model = MLPClassifier(alpha=0.5)

model.fit(X_train_internal_pca, y_train_interal)
y_pred_train_interal = model.predict(X_train_internal_pca)

y_pred_test_interal = model.predict(X_test_internal_pca)
f1_score(y_pred=y_pred_train_interal, y_true=y_train_interal, average='binary') 
F1_NN = f1_score(y_pred=y_pred_test_interal, y_true=y_test_interal, average='binary') 

F1_NN
accuracy_score(y_pred=y_pred_train_interal, y_true=y_train_interal)
accuracy_score(y_pred=y_pred_test_interal, y_true=y_test_interal)
y_pred_test = model.predict(X_test_pca)
df_out = pd.DataFrame({'id': ids_test, 'target': y_pred_test}, dtype=int)

df_out.to_csv('out.csv', index=None)
(y_pred_test == 1).sum()
def plot_confusion_matrix(y_true, y_pred,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

#     classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

#            xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

plot_confusion_matrix(y_test_interal, y_pred_test_interal, normalize=True,

                      title='Normalized confusion matrix')

from sklearn.naive_bayes import GaussianNB



# model = RandomForestClassifier(max_depth=4)

model = GaussianNB()

model.fit(X_train_internal_pca, y_train_interal)



y_pred_train_interal = model.predict(X_train_internal_pca)

y_pred_test_interal = model.predict(X_test_internal_pca)



F1_GNB = f1_score(y_pred=y_pred_test_interal, y_true=y_test_interal, average='binary') 

F1_GNB



y_pred_test = model.predict(X_test_pca)



df_out = pd.DataFrame({'id': ids_test, 'target': y_pred_test}, dtype=int)

df_out.to_csv('out.csv', index=None)



def plot_confusion_matrix(y_true, y_pred,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

#     classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

#            xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

plot_confusion_matrix(y_test_interal, y_pred_test_interal, normalize=True,

                      title='Normalized confusion matrix')

plot_help = list(zip(range(4), [F1_RF, F1_NN, F1_GNB, ADA_F1]))

plt.bar(['RF', 'NN', 'GNB', 'ADA_BOOST'], [F1_RF, F1_NN, F1_GNB, ADA_F1])

plt.title("F1 Score Comparison")