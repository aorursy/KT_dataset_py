# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd



test_df = pd.read_csv("/kaggle/input/mf-accelerator/contest_test.csv", index_col="ID")

subm_df = pd.read_csv("/kaggle/input/mf-accelerator/sample_subm.csv", index_col="ID")

df = pd.read_csv("/kaggle/input/mf-accelerator/contest_train.csv", index_col="ID")



df.head()
%matplotlib inline

_ = df["TARGET"].plot.hist()
feature_n = [col for col in df if col.startswith('FEATURE')]



len_df = len(df)

len_test_df = len(test_df)



df = df.fillna(df.median())

test_df = test_df.fillna(test_df.median())





merged_df = pd.concat([df, test_df], axis=0)

merged_df = merged_df.fillna(merged_df.median())

assert len_df + len_test_df == len(merged_df)



dumm_columns = []

drop_columns = []

norm_columns = []



one_hot = pd.get_dummies(merged_df["TARGET"].astype(int), prefix="TARGET")

merged_df = one_hot.join(merged_df)



for feature in feature_n:

        if len(merged_df[feature].unique()) == 1:

            drop_columns.append(feature)

            merged_df = merged_df.drop(feature, axis=1)

        elif 2 < len(df[feature].unique()) <= 10:

            dumm_columns.append(feature)

            one_hot = pd.get_dummies(merged_df[feature], prefix=feature)

            merged_df = merged_df.join(one_hot)

            merged_df = merged_df.drop(feature, axis=1)

        else:

            norm_columns.append(feature)

            merged_df[feature] = (merged_df[feature] - merged_df[feature].mean()) / merged_df[feature].std()



df = merged_df.iloc[:len_df]

test_df = merged_df.iloc[len_df:]



df.head()
#OVERSAMPLING

count_class_0, count_class_1, count_class_2 = df.TARGET.value_counts()



df_class_0 = df[df["TARGET"] == 0]

df_class_1 = df[df["TARGET"] == 1]

df_class_2 = df[df["TARGET"] == 2]



df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_class_2_over = df_class_2.sample(count_class_0, replace=True)



df = pd.concat([df_class_0, df_class_1_over, df_class_2_over], axis=0)



_ = df["TARGET"].plot.hist()
import seaborn as sns



feature_n = [col for col in df if col.startswith('FEATURE')]

df = df.fillna(df.median())



for feature in feature_n:

    if len(df[feature].unique()) == 1:

        df = df.drop(feature, axis=1)



corr = df.corr()



#fig, ax = plt.subplots(figsize=(20, 10))



numbers = np.array(list(range(len(df))))



# plot the heatmap

_ = sns.heatmap(corr)
feature_n = [col for col in df if col.startswith('FEATURE')]



for i, feature in enumerate(feature_n):

    print (i, len(df[feature].unique()))



#noncategorial features [0:235], categorial [235:] (359)
from sklearn.model_selection import train_test_split



X = df.drop([col for col in df if col.startswith("TARGET")], axis=1)



Y = df["TARGET"]



#print (X.head())



X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=0.2, random_state=42)



print (X_train.shape, Y_train.shape)

print (X_test.shape, Y_test.shape)
from sklearn.decomposition import PCA



pca = PCA(n_components=20, random_state=42)



data_for_pca = X_train.to_numpy()[:,0:235]



pca.fit(data_for_pca)



components = pca.components_



# noncategorical to pca

X_nc = X_train.to_numpy()[:, 0:235] @ components.T #noncatecorial_features

X_ca = X_train.to_numpy()[:, 235:] #categorial features

X_train = np.concatenate([X_nc, X_ca], axis=1)



X_nc = X_test.to_numpy()[:, 0:235] @ components.T #noncatecorial_features

X_ca = X_test.to_numpy()[:, 235:] #categorial features

X_test = np.concatenate([X_nc, X_ca], axis=1)
from catboost import CatBoostClassifier



cat = CatBoostClassifier()



cat.fit(X_train, Y_train)
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



predict = cat.predict(X_test)



conf_mat = confusion_matrix(y_true=Y_test, y_pred=predict)



print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1', 'Class 2']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
from sklearn.metrics import f1_score



predict = cat.predict(X_test)



f1_score(y_true=Y_test, y_pred=predict, average="macro")
test_df.head()
X_test = test_df.to_numpy()



X_nc = X_test[:, 0:235] @ components.T #noncatecorial_features

X_ca = X_test[:, 235:] #categorial features

X_test = np.concatenate([X_nc, X_ca], axis=1)



predict = cat.predict(X_test)