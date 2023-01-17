import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/heart.csv")

df.info()
df.head()
corr = df.corr()['target'].abs().sort_values()

corr
# Helper function for plotting side by side

def sideplot(df, col, kind="bar", title=None):

    assert kind in ["bar", "hist"]

    fig = plt.figure(figsize=(10, 6))

    if kind == "bar":

        ax1 = plt.subplot(2, 2, 1)

        df[df.target == 1][['target', col]].groupby(col).count().plot(kind='bar', rot=0, legend=False, ax=ax1, color="#268bd2")

        ax2 = plt.subplot(2, 2, 2)

        df[df.target == 0][['target', col]].groupby(col).count().plot(kind='bar', rot=0, legend=False, ax=ax2, color="#268bd2")

    else:

        ax1 = plt.subplot(2, 2, 1)

        plt.hist(df[df.target == 1][col], color="#268bd2")

        plt.xlabel(col)

        ax2 = plt.subplot(2, 2, 2)

        plt.hist(df[df.target == 0][col], color="#268bd2")

        plt.xlabel(col)

    # Re-adjusting

    ylim = (0, max(ax1.get_ylim()[1], ax2.get_ylim()[1]))

    ax1.set_ylim(ylim)

    ax2.set_ylim(ylim)

    xlim = (min(ax1.get_xlim()[0], ax2.get_xlim()[0]), max(ax1.get_xlim()[1], ax2.get_xlim()[1]))

    ax1.set_xlim(xlim)

    ax2.set_xlim(xlim)

    if title is not None:

        fig.suptitle(title)

    #plt.subplots_adjust(top=0.99)
sideplot(df, "fbs", kind="bar", title="Comparison of fasting blood sugar")
sideplot(df, "chol", kind="hist", title="Comparison of serum cholestoral")
sideplot(df, "restecg", kind="bar", title="Comparison of resting ECG results")
sideplot(df, "trestbps", kind="hist", title="Comparison of resting blood pressure")
sideplot(df, "age", kind="hist", title="Comparison of age")
sideplot(df, "sex", kind="bar", title="Comparison of sex")
sideplot(df, "thal", kind="bar", title="Comparison of (thal)")
sideplot(df, "slope", kind="bar", title="Comparison of the slope of the peak exercise ST segment")
sideplot(df, "ca", kind="bar", title="Comparison of the number of major visible vessels under fluorosopy")
sideplot(df, "thalach", kind="hist", title="Comparison of maximum heart rate achieved")
sideplot(df, "oldpeak", kind="hist", title="Comparison of ST depression induced by exercise relative to rest")
sideplot(df, "cp", kind="bar", title="Comparison of chest pain type")
sideplot(df, "exang", kind="bar", title="Comparison of exercise induced angina")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

clf = LDA(n_components=1)



y = df["target"].values

X = clf.fit(df[df.columns[:-1]].values, y).transform(df[df.columns[:-1]].values)

X = X[:, 0]



sns.swarmplot(X[y == 0], color="b", label="without HD")

sns.swarmplot(X[y == 1], color="r", label="with HD")

plt.title("LDA analysis of heart disease classification")

plt.legend()
def onehot(ser, num_classes=None):

    """

    One-hot encode the series.

    Example: 

    >>> onehot([1, 0, 2], 3)

    array([[0., 1., 0.],

       [1., 0., 0.],

       [0., 0., 1.]])

    """

    if num_classes == None:

        num_classes = len(np.unique(ser))

    return np.identity(num_classes)[ser]



new_col_names = []

need_encode_col = ["restecg", "thal", "slope", "cp"]

no_encode_col = [col for col in df.columns if col not in need_encode_col]

new_df = df[no_encode_col]

for col in need_encode_col:

    num_classes = len(df[col].unique())

    new_col_names = [f"{col}_{i}" for i in range(num_classes)]

    encoded = pd.DataFrame(onehot(df[col], num_classes), columns=new_col_names, dtype=int)

    new_df = pd.concat([new_df, encoded], axis=1)

new_df.head()
from sklearn.decomposition import PCA

clf = PCA(n_components=2)

data_cols = [col for col in new_df.columns if col != "target"]

X = new_df[data_cols]

y = new_df["target"]

X_trans = clf.fit(X, y).transform(X)

sns.scatterplot(X_trans[y == 0][:, 0], X_trans[y == 0][:, 1], color="b", label="without HD")

sns.scatterplot(X_trans[y == 1][:, 0], X_trans[y == 1][:, 1], color="r", label="with HD")

plt.title("PCA analysis of heart disease classification")

plt.legend()
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



new_df_shfl = shuffle(new_df, random_state=443)

X = new_df_shfl[data_cols].values

y = new_df_shfl["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=80)
num_epochs = 5000

log_inteval = 250

total_losses = []

total_val_losses = []

lr = 1e-4

lr_decay_inteval = 2500

lr_decay_rate = 0.3
import torch

from torch import nn, optim

model = nn.Sequential(

    nn.Linear(len(data_cols), 80),

    nn.ReLU(),

    nn.Dropout(0.3),

    nn.Linear(80, 256),

    nn.ReLU(),

    nn.Dropout(0.6),

    nn.Linear(256, 1),

)

loss_fn = torch.nn.BCELoss()

opt = optim.Adam(model.parameters(), lr=lr)



def init_normal(m):

    if type(m) == nn.Linear:

        nn.init.xavier_normal_(m.weight, 0.06)



model.apply(init_normal)
for epoch in range(1, num_epochs+1):

    y_pred = model(torch.tensor(X_train, dtype=torch.float))

    y_pred = torch.sigmoid(y_pred)

    opt.zero_grad()

    loss = loss_fn(y_pred[:, 0], torch.tensor(y_train, dtype=torch.float))

    loss.backward()

    opt.step()

    total_losses.append(loss.item())

    if epoch % log_inteval == 0: # Logging

        epochs_ran = epoch

        model.eval()

        with torch.no_grad():

            y_pred = model(torch.tensor(X_test, dtype=torch.float))

            y_pred = torch.sigmoid(y_pred)

            val_loss = loss_fn(y_pred[:, 0], torch.tensor(y_test, dtype=torch.float))

            total_val_losses.append(val_loss.item())

        model.train()

        print(f"total loss in epoch {epoch} = {'%.4f'%loss}, validation loss = {'%.4f'%val_loss}, lr = {'%.2e'%lr}")

        if len(total_val_losses) > 3 and val_loss.item() > total_val_losses[-2] and val_loss.item() > total_val_losses[-3]:

            print(f"Validation loss not improving for {log_inteval * 2} epochs, stopping...")

            break

    if epoch % lr_decay_inteval == 0: # Learning rate decay

        lr *= lr_decay_rate

        for param_group in opt.param_groups:

            param_group['lr'] = lr
plt.plot(total_losses, 'b', label="train")

plt.plot(np.array(range(epochs_ran // log_inteval)) * log_inteval + log_inteval, total_val_losses, 'r', label="valid")

plt.ylim([0, 1])

plt.title("Learning curve")

plt.legend()
from sklearn.metrics import confusion_matrix

with torch.no_grad():

    model.eval()

    y_pred = model(torch.tensor(X_test, dtype=torch.float))

    y_pred_lbl = np.where(y_pred.numpy() > 0, 1, 0)

cm = pd.DataFrame(confusion_matrix(y_test, y_pred_lbl), columns=["T", "F"], index=["P", "N"])

print("Accuracy = %.2f%%" % ((cm.iloc[1, 1] + cm.iloc[0, 0]) / cm.values.sum() * 100))

cm
import lightgbm as lgb



lgb_train = lgb.Dataset(X_train, y_train)

lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
params = {

    "num_iterations": 1000,

    "num_leaves": 63,

    "max_depth": 7,

    "max_bin": 500,

    "learning_rate": 0.001,

    "min_data_in_leaf": 1,

    "objective": "binary",

    "metric": ["binary"],

}



bst = lgb.train(params, lgb_train, valid_sets=[lgb_valid], early_stopping_rounds=50, verbose_eval=100)
y_pred = bst.predict(X_test)

y_pred_lbl = np.where(y_pred > 0.5, 1, 0)

cm = pd.DataFrame(confusion_matrix(y_test, y_pred_lbl), columns=["T", "F"], index=["P", "N"])

print("Accuracy = %.2f%%" % ((cm.iloc[1, 1] + cm.iloc[0, 0]) / cm.values.sum() * 100))

cm