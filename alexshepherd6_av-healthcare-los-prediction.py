# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras as ks
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
%matplotlib inline

plt.rcParams['figure.figsize'] = (7, 6)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.chdir("/kaggle/input/av-healthcare-analytics-ii/healthcare/")
# training dataset
train = pd.read_csv("train_data.csv")

print("Available Features : {}".format(train.columns))

train.head()
train["Stay"].value_counts()
# Converting certain features to categorical variables for ease of analysis

cat_cols = list(set(list(train.columns)) - set(['Available Extra Rooms in Hospital', 'Visitors with Patient', 'Admission_Deposit']))
ordered_cols = ['Bed Grade', 'Age', 'Stay', "Severity of Illness"]
stay_order = ['0-10', '11-20', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90','91-100', 'More than 100 Days']
y_feature = "Stay"

for c in cat_cols:
    if c in ordered_cols:
        if c in ordered_cols[-2:]:
            if c == "Age":
                # Age 
                train[c] = pd.Categorical(train[c], ordered=True, categories=stay_order[:-1])
            elif c == "Stay":
                # Stay
                train[c] = pd.Categorical(train[c], ordered=True, categories=stay_order)
            elif c == "Severity of Illness":
                train[c] = pd.Categorical(train[c], ordered=True, categories=["Minor", "Moderate", "Extreme"])
        else:
            # Bed Grade
            train[c] = pd.Categorical(train[c], ordered=True)
    else:
        train[c] = pd.Categorical(train[c])
# Class Distribution

sns.countplot(train["Stay"])

plt.title("Class Distribution of Training Dataset (n = {})".format(len(train)))
plt.ylabel("Frequency")
plt.xticks(rotation = 45)
plt.show()
train.describe()
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))

# ax1 -------
sns.countplot(train["Severity of Illness"], ax = ax1)

ax1.set_title("""Distribution of Severity of Illness feature in training dataset
(n = {})""".format(len(train)))
ax1.set_ylabel("Frequency")

# ax2 -------
c = train.groupby(["Severity of Illness", "Stay"]).size()
c = (c/c.groupby(level=1).sum()).reset_index()

sns.lineplot(x = "Stay", y = 0, hue = "Severity of Illness", data = c, ax = ax2)

ax2.set_title("""Proportion of Length of Stay Category by Severity of Illness
(n = {})""".format(len(train)))
ax2.set_ylabel("Proportion of Length of Stay Category")
ax2.set_xticklabels(stay_order, rotation = 45)

plt.tight_layout()
plt.show()
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))

# ax1 -------
sns.countplot(train["Type of Admission"], ax = ax1)

ax1.set_title("""Distribution of Type of Admission feature in training dataset
(n = {})""".format(len(train)))
ax1.set_ylabel("Frequency")

# ax2 -------
c = train.groupby(["Type of Admission", "Stay"]).size()
c = (c/c.groupby(level=1).sum()).reset_index()

sns.lineplot(x = "Stay", y = 0, hue = "Type of Admission", data = c, ax = ax2)

ax2.set_title("""Proportion of Length of Stay Category by Type of Admission Category
(n = {})""".format(len(train)))
ax2.set_ylabel("Proportion of Type of Admission Category")
ax2.set_xticklabels(stay_order, rotation = 45)

plt.tight_layout()
plt.show()
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))

# ax1 -------
sns.countplot(train["Age"], ax = ax1)

ax1.set_title("""Distribution of Age feature in training dataset
(n = {})""".format(len(train)))
ax1.set_ylabel("Frequency")

# ax2 -------
c = train.groupby(["Age", "Stay"]).size()
c = (c/c.groupby(level=0).sum()).reset_index()
c = c.values[:,-1].reshape(c.Age.unique().size, c["Stay"].unique().size).astype(float)

sns.heatmap(c, ax = ax2, cmap = "Blues")

ax2.set_title("""Proportion of Age Category by Length of Stay Category
(n = {})""".format(len(train)))
ax2.set_ylabel("Age Category")
ax2.set_yticklabels(stay_order[:-1], rotation = 0)
ax2.set_xticklabels(stay_order, rotation = 45)

plt.tight_layout()
plt.show()
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))

# ax1 -------
sns.countplot(train["Department"], ax = ax1)

ax1.set_title("""Distribution of Department feature in training dataset
(n = {})""".format(len(train)))
ax1.set_ylabel("Frequency")

# ax2 -------
c = train.groupby(["Department", "Stay"]).size()
c = (c/c.groupby(level=0).sum()).reset_index()
yticks = c.Department.unique()
c = c.values[:,-1].reshape(c.Department.unique().size, c["Stay"].unique().size).astype(float)

sns.heatmap(c, ax = ax2, cmap = "Blues")

ax2.set_title("""Proportion of Department Category by Length of Stay Category 
(n = {})""".format(len(train)))
ax2.set_ylabel("Department")
ax2.set_yticklabels(yticks, rotation = 0)
ax2.set_xticklabels(stay_order, rotation = 45)

plt.tight_layout()
plt.show()
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))

# ax1 -------
sns.distplot(train["Visitors with Patient"], kde = False, ax = ax1)

ax1.set_title("""Distribution of Visitors with Patient feature in training dataset
(n = {})""".format(len(train)))
ax1.set_ylabel("Frequency")

# ax2 -------
sns.boxplot(x = "Stay", y = "Visitors with Patient", data = train, ax = ax2)

ax2.set_title("""Distribution of Visitors with Patient by Length of Stay Category
(n = {})""".format(len(train)))
ax2.set_ylabel("Visitors with Patient")
ax2.set_xticklabels(stay_order, rotation = 45)

plt.tight_layout()
plt.show()
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))

# ax1 -------
sns.countplot(train["Hospital_code"], ax = ax1)

ax1.set_title("""Distribution of Hospital code feature in training dataset
(n = {})""".format(len(train)))
ax1.set_ylabel("Frequency")

# ax2 -------
c = train.groupby(["Hospital_code", "Stay"]).size()
c = (c/c.groupby(level=0).sum()).reset_index()
yticks = c.Hospital_code.unique()
c = c.values[:,-1].reshape(c.Hospital_code.unique().size, c.Stay.unique().size).astype(float)

sns.heatmap(c, ax = ax2, cmap = "Blues")

ax2.set_title("""Proportion of Hospital code Category by Length of Stay Category 
(n = {})""".format(len(train)))
ax2.set_ylabel("Hospital code")
ax2.set_yticklabels((np.arange(yticks.size) * 2) + 1, rotation = 0)
ax2.set_xticklabels(stay_order, rotation = 45)

plt.tight_layout()
plt.show()
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))

# ax1 -------
sns.countplot(train["Ward_Type"], ax = ax1)

ax1.set_title("""Distribution of Ward Type feature in training dataset
(n = {})""".format(len(train)))
ax1.set_ylabel("Frequency")

# ax2 -------
c = train.groupby(["Ward_Type", "Stay"]).size()
c = (c/c.groupby(level=0).sum()).reset_index()
yticks = c.Ward_Type.unique()
c = c.values[:,-1].reshape(c.Ward_Type.unique().size, c.Stay.unique().size).astype(float)

sns.heatmap(c, ax = ax2, cmap = "Blues")

ax2.set_title("""Proportion of Ward Type by Length of Stay Category 
(n = {})""".format(len(train)))
ax2.set_ylabel("Ward_Type")
ax2.set_yticklabels(yticks, rotation = 0)
ax2.set_xticklabels(stay_order, rotation = 45)

plt.tight_layout()
plt.show()
input_features = ["Severity of Illness", "Age", "Type of Admission", "New Ward_Type",
                  "Hospital_code", "Department", "Visitors with Patient"]

ordered_cats = input_features[0:2]
cats = input_features[2:-1]
num = [input_features[-1]]
# Concatenating Ward types as explained in EDA

train.loc[:, "New Ward_Type"] = train.loc[:, "Ward_Type"].astype(str)
combine_cats = {"PQRU": ["P", "Q", "R", "U"], "ST": ["S", "T"]}

for k in combine_cats:
    _idxs = train[train["Ward_Type"].isin(combine_cats[k])].index.values
    train.at[_idxs, "New Ward_Type"] = k
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14,6))

# ax1 -------
sns.countplot(train["New Ward_Type"], ax = ax1)

ax1.set_title("""Distribution of New Ward Type feature in training dataset
(n = {})""".format(len(train)))
ax1.set_ylabel("Frequency")

# ax2 -------
c = train.groupby(["New Ward_Type", "Stay"]).size()
c = (c/c.groupby(level=0).sum()).reset_index()
yticks = c["New Ward_Type"].unique()
c = c.values[:,-1].reshape(c["New Ward_Type"].unique().size, c.Stay.unique().size).astype(float)

sns.heatmap(c, ax = ax2, cmap = "Blues")

ax2.set_title("""Proportion of New Ward Type by Length of Stay Category 
(n = {})""".format(len(train)))
ax2.set_ylabel("New Ward_Type")
ax2.set_yticklabels(yticks, rotation = 0)
ax2.set_xticklabels(stay_order, rotation = 45)

plt.tight_layout()
plt.show()
X_train = train.loc[:, input_features + [y_feature]]
X_train.head()
def balance_classes(data, y):
    """
    Balances class in dataset
    
    :param data: (Pandas DataFrame) dataset to balance
    :param y: (String) class column name
    """
    d = pd.DataFrame(data[y].value_counts())
    max_class = d.idxmax(axis = 0).values[0]
    max_class_count = d.loc[max_class][0]

    new_data = data[data[y] == max_class]

    for c in list(set(data[y].unique()) - set([max_class])):
        
        try:
            c_idxs = data[data[y] == c].index.values
            c_idxs = np.random.choice(c_idxs, max_class_count)
            new_data = pd.concat([new_data, data.loc[c_idxs,:]], ignore_index = True)
        except Exception as e:
            print(e)
            pass
        
    return new_data
X_train = balance_classes(X_train, "Stay")
X_train.Stay.value_counts()
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical 
from sklearn.model_selection import train_test_split


def prepare_dataset():

    # Convert categorical variables to indices
    X_trn = None
    m = MinMaxScaler()

    for c in X_train.columns:
        print(c)

        if c in cats or c in ordered_cats or c == y_feature:
            _ = pd.factorize(X_train[c], sort=True)[0]
            if c in cats:
                _ = to_categorical(_, num_classes=X_train[c].unique().size)
            elif c in ordered_cats:
                _ = _/np.max(_)
        else:
            #print(c)
            _ = m.fit_transform(X_train[c].values.reshape(-1,1))[:, 0]

        try:
            print("Xtrn: ", X_trn.shape) 
        except:
            pass
        print("_: ",_.shape)

        try:
            if len(_.shape) == 1:
                    print("1D")
                    X_trn = np.hstack((X_trn, _.reshape(-1,1)))
            else:
                X_trn = np.hstack((X_trn, _))  
        except Exception as e:
            print("Error")
            if len(_.shape) == 1:
                    print("1D")
                    X_trn = _.reshape(-1,1)
            else:
                X_trn = _

    X_trn = X_trn


    y_trn = X_trn[:, -1]
    X_trn = X_trn[:, :-1]

    # adding Visitors Number ^ 2 to add another feature
    #X_trn = np.vstack((X_trn.T, m.fit_transform(((X_train["Visitors with Patient"]**2).values).reshape(-1,1))[:, 0])).T

    print(X_trn.shape)
    
    y_trn = to_categorical(y_trn, num_classes=train.Stay.unique().size)
    
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=0.3)
    
    return X_trn, X_val, y_trn, y_val
X_trn, X_val, y_trn, y_val = prepare_dataset()
METRICS = [ 
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Accuracy(name='acc')
]
baseline = tf.keras.Sequential([
    layers.Dense(input_dim = X_trn.shape[1], units = 128, activation = "relu"),
    layers.Dense(128, activation = "relu"),
    layers.Dense(train.Stay.unique().size, activation = "softmax")
], name = "baseline")

baseline.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=METRICS)

baseline.summary()
bs_history = baseline.fit(X_trn, y_trn,
                          epochs=20,
                          batch_size = 700,
                          validation_data=(X_val, y_val),
                          validation_steps=5)
def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
plot_loss(bs_history, "Baseline", 0)
plot_loss(inc_layers_feat_history, "Inc layers feature", 1)
baseline.evaluate(X_val, y_val)
def cm(model, classes = None, datasets = None):
    
    if datasets is None:
        datasets = [X_val, y_val]
        
    prediction = np.argmax(model.predict(datasets[0]), axis = 1)
    tst = np.vstack((np.argmax(datasets[1], axis = 1), prediction)).T

    cm = np.zeros((len(classes),len(classes)))

    for i in tst:
        cm[i[0], i[1]] += 1

    f = plt.figure(figsize = (7,6))
    
    f = sns.heatmap(cm)#, annot = True)
    
    
    plt.title("Confusion Matrix over validation set")
    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation = 0)
    plt.ylabel("Actual")
    plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation = 45)
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()
f, (ax1, ax2) = plt.subplots(1,2, figsize = (14, 6))

# ax1 ---

c = X_train.groupby(["Stay", "Severity of Illness"]).size()
c = (c/c.groupby(level = 1).sum()).reset_index()

sns.barplot(x = "Stay", y = 0, hue = "Severity of Illness", data = c, ax = ax1)
ax1.set_title("""Proportion of Severity of Illnesses by LOS in model training data
(n = {})""".format(X_train.shape[0]))
#ax1.set_xticks(rotation = 45)
ax1.set_ylabel("Proportion of Severity of Illnesses")

# ax2 ---

# Aggregating extreme and moderate categories into above moderate category
X_train.loc[:, "Severity of Illness_v2"] = X_train["Severity of Illness"].astype(str)
non_minor_idxs = X_train[X_train["Severity of Illness_v2"] != "Minor"].index.values
X_train.at[non_minor_idxs, "Severity of Illness_v2"] = "Above Moderate"

c = X_train.groupby(["Stay", "Severity of Illness_v2"]).size()
c = (c/c.groupby(level = 1).sum()).reset_index()

sns.barplot(x = "Stay", y = 0, hue = "Severity of Illness_v2", data = c, ax = ax2)
ax2.set_title("""Proportion of Severity of Illnesses v2 by LOS in model training data
(n = {})""".format(X_train.shape[0]))
#ax2.set_xticks(rotation = 45)
ax2.set_ylabel("Proportion of Severity of Illnesses")

plt.show()
input_features = ["Severity of Illness_v2", "Age", "Type of Admission", "New Ward_Type",
                  "Hospital_code", "Department", "Visitors with Patient"]

ordered_cats = input_features[0:2]
cats = input_features[2:-1]
num = [input_features[-1]]

X_train[input_features[0]] = pd.Categorical(X_train[input_features[0]], ordered=True, categories=["Minor", "Above Moderate"])

X_train = X_train.loc[:, input_features + [y_feature]]

X_trn, X_val, y_trn, y_val = prepare_dataset()
"""
earlystop_callback = tf.keras.callbacks.EarlyStopping(
  monitor='val_acc', min_delta=0.0001, mode = "min",
  patience=10, verbose=1)
"""

inc_layers_feat = tf.keras.Sequential([
    layers.Dense(input_dim = X_trn.shape[1], units = 128, activation = "relu"),
    layers.Dense(128, activation = "relu"),
    layers.Dense(128, activation = "relu"),
    layers.Dense(128, activation = "relu"),
    layers.Dense(train.Stay.unique().size, activation = "softmax")
], name = "inc_layers_feat")

inc_layers_feat.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=METRICS)



inc_layers_feat.summary()
inc_layers_feat_history = inc_layers_feat.fit(X_trn, y_trn,
                          epochs=20,
                          batch_size = 700,
                          validation_data=(X_val, y_val),
                          validation_steps=5)
baseline_feat.evaluate(X_val, y_val)
cm(baseline, classes = stay_order)
cm(baseline_feat, classes = stay_order)
