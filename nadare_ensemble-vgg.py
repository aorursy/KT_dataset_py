import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from tqdm import tqdm

%matplotlib inline
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
y_train, X_train = train.iloc[:, 0].values, train.iloc[:, 1:].values
X_test = test.values
X_train = X_train/255
X_test = X_test/255

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
def generate_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu', input_shape = (28,28,1)))


    for filters, convs in [[32, 2], [64, 3], [128, 3], [256, 3]]:
        for _ in range(convs):
            model.add(Conv2D(filters = filters, kernel_size = (3, 3), padding = 'Same'))
            model.add(BatchNormalization(axis=-1))
            model.add(Activation("relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = "softmax"))

    optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    return model

def make_grouped_samples(y, groups):
    y = y
    labels = []
    vec = [len(groups)]*10
    remain = set(list(range(10)))

    for i, g in enumerate(groups):
        labels.append("{}".format("&".join(map(str, g))))
        for j in g:
            vec[j] = i
        remain = remain - set(g)
    labels.append("&".join(map(str, remain)))
    
    new_y = np.vectorize(lambda x: vec[x])(y)
    num_classes = np.unique(new_y).shape[0]
    new_y = to_categorical(new_y, num_classes=num_classes)
    return  new_y, labels[:num_classes]


def fit_evaluate_save(X_train, y_train, X_test, model, labels, model_name, balanced):
    if balanced:
        class_weight = {k:v for k, v in enumerate(y_train.shape[0]/y_train.sum(axis=0))}
        model_name += "_balanced"
    else:
        class_weight = {k:1. for k in range(len(labels))}
        
    X, X_val, y, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=2018)
    labels = [model_name + "_" + label for label in labels]

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=4, 
                                                verbose=0, 
                                                factor=0.5, 
                                                min_lr=0.0001)
    early_stopping = EarlyStopping(monitor="val_acc", patience=15)
    
    hist = model.fit(X, y, batch_size=2500, epochs=100, verbose=0,
                     validation_data=(X_val, y_val), callbacks=[learning_rate_reduction, early_stopping])

    model.save("models/" + model_name + ".h5")

    y_pred = model.predict(X_test)
    pd.DataFrame(y_pred, columns=labels).to_csv("ensemble/csv/" + model_name + "_test.csv")

    y_val_pred = model.predict(X_val)
    pd.DataFrame(y_val_pred, columns=labels).to_csv("ensemble/csv/" + model_name + "_val.csv")
    
    y_val_pred_class = np.argmax(y_val_pred, axis=1)
    y_true = np.argmax(y_val, axis=1)
    cfmtx = confusion_matrix(y_true, y_val_pred_class)
    sns.heatmap(cfmtx, cmap="Blues", annot=True)
    plt.savefig("./ensemble/pictures/" + model_name + "_heatmap.png")
    
    plt.figure()
    plt.plot(range(len(hist.history["acc"])), hist.history["acc"])
    plt.plot(range(len(hist.history["val_acc"])), hist.history["val_acc"])
    plt.savefig("./ensemble/pictures/" + model_name + "_accplot.png")

    return hist
Q = [[[(i,)], str(i) + "vsALL"] for i in range(10)]
Q.append([[(i,) for i in range(10)], "all_in_one"])
Q.append([[(5, 7, 9), (0, 2, 3, 4, 6), (1,)] ,"type"])
Q.append([[(5,), (7,), (9,)], "shoose_type"])
Q.append([[(0,), (3,), (2,), (4,), (6,)], "shirt_type"])
Q.append([[(0,), (6,)], "t_shirt_OR_shirt"])
Q.append([[(2,), (6,)], "pullover_OR_shirt"])
Q.append([[(3,), (6,)], "dress_OR_shirt"])
Q.append([[(4,), (6,)], "Coat_OR_shirt"])
Q.append([[(2,), (4,)], "pullover_OR_coat"])
Q.append([[(3,), (4,)], "dress_OR_coat"])
Q.append([[(0,), (3,)], "t_shirt_OR_dress"])

hists = []
for groups, model_name in tqdm(Q):
    y_train_grouped, labels = make_grouped_samples(y_train, groups)
    model = generate_model(len(labels))
    hist = fit_evaluate_save(X_train, y_train_grouped, X_test, model, labels, model_name, False)
    hists.append(hist)
    model = generate_model(len(labels))
    hist = fit_evaluate_save(X_train, y_train_grouped, X_test, model, labels, model_name, True)
    hists.append(hist)
    
ens_val_df = pd.DataFrame()
ens_test_df = pd.DataFrame()
for d in os.listdir("ensemble/csv/"):
    if d.split("_")[-1] == "test.csv":
        ens_test_df = pd.concat([ens_test_df, pd.read_csv("ensemble/csv/" + d, index_col=0)], axis=1)
    else:
        ens_val_df = pd.concat([ens_val_df, pd.read_csv("ensemble/csv/" + d, index_col=0)], axis=1)
        
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
parameters = {'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],
              'max_features'      : ["auto", "log2", 0.5],
              'random_state'      : [2018],
              'n_jobs'            : [1],
              'max_depth'         : list(range(2, 15))}

kf = KFold(n_splits=10, shuffle=True, random_state=2018)

clf = RandomForestClassifier()
gs = GridSearchCV(clf, param_grid=parameters, n_jobs=-1, cv=kf, verbose=10, scoring="accuracy")
gs.fit(ens_val_df, y_val)
_, _, _, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=2018)
gs.best_score_
gs.best_params_
rfc = RandomForestClassifier(max_depth=9, max_features="auto", n_estimators=500, n_jobs=-1, random_state=2018)
rfc.fit(ens_val_df, y_val)
y_ans = rfc.predict(ens_test_df)
pd.DataFrame({"id": range(1, len(y_ans)+1),
              "label": y_ans}).to_csv("output/5th_1_sub.csv", index=False)
