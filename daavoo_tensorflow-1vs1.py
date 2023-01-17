import numpy as np

import pandas as pd 

import seaborn as sns

import tensorflow as tf



import itertools

import os

import tflearn





from matplotlib import pyplot as plt

from sklearn.decomposition.kernel_pca import KernelPCA

from sklearn.metrics import classification_report

from sklearn.preprocessing import PolynomialFeatures
%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



submission = pd.read_csv("../input/sample_submission.csv")

submission["type"] = "Unknown"
sns.pairplot(train.drop("id",axis=1), hue="type", diag_kind="kde")
# interaction_only to avoid x**2, etc

poly_features = PolynomialFeatures(interaction_only=True)
# [:,5:] to discard columns of original features

try_comb = pd.DataFrame(

    poly_features.fit_transform(train.drop(["id", "color", "type"], axis=1))[:,5:],

    columns=["boneXrotting", "boneXhair", "boneXsoul",

             "rottingXhair", "rottingXsoul", 

             "hairXsoul"]

)

try_comb["type"] = train.type
sns.pairplot(try_comb, hue="type", diag_kind="kde")
for i in ["boneXhair", "boneXsoul", "hairXsoul"]:

    train[i] = try_comb[i].copy()

try_comb = None
try_comb = pd.DataFrame(

    poly_features.fit_transform(test.drop(["id", "color"], axis=1))[:,5:],

    columns=["boneXrotting", "boneXhair", "boneXsoul",

             "rottingXhair", "rottingXsoul", 

             "hairXsoul"]

)



for i in ["boneXhair", "boneXsoul", "hairXsoul"]:

    test[i] = try_comb[i].copy()

    

try_comb = None
train.loc[train.type == "Ghost", "type"] = 0

train.loc[train.type == "Ghoul", "type"] = 1

train.loc[train.type == "Goblin", "type"] = 2

train["type"] = train["type"].astype("int")
is_ghost = (train.type == 0).values

is_ghoul = (train.type == 1).values

is_goblin = (train.type == 2).values
ghost_ghoul = train.loc[is_ghost | is_ghoul].copy()

ghost_goblin = train.loc[is_ghost | is_goblin].copy()

ghoul_goblin = train.loc[is_ghoul| is_goblin].copy()
def plot_KPCA(df, transf, labels={"A":0, "B":1}):

    plt.figure(figsize=(10,8))



    for label,marker,color in zip(list(labels.keys()),('x', 'o'),('blue', 'red')):



        plt.scatter(x=transf[:,0][(df.type == labels[label]).values],

                    y=transf[:,1][(df.type == labels[label]).values],

                    marker=marker,

                    color=color,

                    alpha=0.7,

                    label='class {}'.format(label)

                    )



    plt.legend()

    plt.title('KernelPCA projection')



    plt.show()
X = ghost_ghoul.drop(["id", "color", "type"], axis=1).values

y = ghost_ghoul.type.values



KPCA = KernelPCA(n_components=2, kernel="rbf", gamma=1)

ghost_ghoul_KPCA= KPCA.fit(X,y)



ghost_ghoul_transf = ghost_ghoul_KPCA.transform(X)



plot_KPCA(ghost_ghoul, ghost_ghoul_transf, labels={"Ghost":0, "Ghoul":1})
X = ghost_goblin.drop(["id", "color", "type"], axis=1).values

y = ghost_goblin.type.values



KPCA = KernelPCA(n_components=2, kernel="rbf")

ghost_goblin_KPCA= KPCA.fit(X,y)



ghost_goblin_transf = ghost_goblin_KPCA.transform(X)



plot_KPCA(ghost_goblin, ghost_goblin_transf, labels={"Ghost":0, "Ghoblin":2})
X = ghoul_goblin.drop(["id", "color", "type"], axis=1).values

y = ghoul_goblin.type.values



KPCA = KernelPCA(n_components=2, kernel="rbf", gamma=3)

ghoul_goblin_KPCA= KPCA.fit(X,y)



ghoul_goblin_transf = ghoul_goblin_KPCA.transform(X)



plot_KPCA(ghoul_goblin, ghoul_goblin_transf, labels={"Ghoul":1, "Ghoblin":2})
ghost_ghoul["KPCA_0"] = ghost_ghoul_transf[:,0]

ghost_ghoul["KPCA_1"] = ghost_ghoul_transf[:,1]



ghost_goblin["KPCA_0"] = ghost_goblin_transf[:,0]

ghost_goblin["KPCA_1"] = ghost_goblin_transf[:,1]



ghoul_goblin["KPCA_0"] = ghoul_goblin_transf[:,0]

ghoul_goblin["KPCA_1"] = ghoul_goblin_transf[:,1]
def neural_net(X_train, y_train, X_test, layers=[1024], dropout=0.8, n_epoch=30):

    

    if isinstance(X_train, pd.DataFrame):

        X_train = X_train.values

        

    if isinstance(y_train, pd.DataFrame):

        y_train = y_train.values

        

    if isinstance(X_test, pd.DataFrame):

        X_test = X_test.values

        

    with tf.Graph().as_default():



        net = tflearn.input_data(shape=[None, X_train.shape[1]])

        for layer_size in layers:            

            net = tflearn.fully_connected(net, layer_size,

                                          activation='relu',

                                          weights_init='xavier',

                                          regularizer='L2')

            net = tflearn.dropout(net, dropout)

        net = tflearn.fully_connected(net, y_train.shape[1], activation='softmax')

        net = tflearn.regression(net)



        model = tflearn.DNN(net, tensorboard_verbose=0)

        model.fit(X_train, y_train, validation_set=0.2, n_epoch=n_epoch)



        probs = np.array(model.predict(X_test))    

        

    return probs
X_train = ghost_ghoul.drop(["id", "color", "type"], axis=1)

y_train = pd.get_dummies(ghost_ghoul["type"])

X_test = test.drop(["id", "color"], axis=1)



# Apply the KPCA transformer to test

transf = ghost_ghoul_KPCA.transform(X_test)



X_test["KPCA_0"] = transf[:,0]

X_test["KPCA_1"] = transf[:,1]
ghost_ghoul_probs = neural_net(X_train, y_train, X_test, dropout=0.8, n_epoch=40)
X_train = ghost_goblin.drop(["id", "color", "type"], axis=1)

y_train = pd.get_dummies(ghost_goblin["type"])

X_test = test.drop(["id", "color"], axis=1)



# Apply the KPCA transformer to test

transf = ghost_goblin_KPCA.transform(X_test)



X_test["KPCA_0"] = transf[:,0]

X_test["KPCA_1"] = transf[:,1]
ghost_goblin_probs = neural_net(X_train, y_train, X_test, layers=[512], dropout=0.8, n_epoch=60)
X_train = ghoul_goblin.drop(["id", "color", "type"], axis=1)

y_train = pd.get_dummies(ghoul_goblin["type"])

X_test = test.drop(["id", "color"], axis=1)



# Apply the KPCA transformer to test

transf = ghoul_goblin_KPCA.transform(X_test)



X_test["KPCA_0"] = transf[:,0]

X_test["KPCA_1"] = transf[:,1]
ghoul_goblin_probs = neural_net(X_train, y_train, X_test, dropout=0.5, n_epoch=40)
global_predictions =  np.zeros((X_test.values.shape[0], 3))



global_predictions[:,[0,1]] += ghost_ghoul_probs

global_predictions[:,[0,2]] += ghost_goblin_probs

global_predictions[:,[1,2]] += ghoul_goblin_probs
test["global_pred"] = np.argmax(global_predictions, axis=1).astype("str")



test.loc[test.global_pred == "0", "global_pred"] = "Ghost"

test.loc[test.global_pred == "1", "global_pred"] = "Ghoul"

test.loc[test.global_pred == "2", "global_pred"] = "Ghoblin"
sns.pairplot(test.drop("id",axis=1), hue="global_pred", diag_kind="kde")
submission["type"] = test["global_pred"] 

submission.to_csv("sub.csv", index=False)