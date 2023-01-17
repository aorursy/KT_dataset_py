# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn import datasets
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from tqdm import tqdm

import os
import gc
from itertools import combinations, chain
from datetime import datetime

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
X=np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
y=np.array([1,1,1,2,2,2])
skf=StratifiedKFold(n_splits=3)

for train_index,test_index in skf.split(X,y):
    print("Train Index:",train_index,",Test Index:",test_index)
    X_train,X_test=X[train_index],X[test_index]
    y_train,y_test=y[train_index],y[test_index]
import tensorflow as tf

filepath = "../input/"
INPUT_NODE = 47
OUTPUT_NODE = 7

REGULARIZATION_RATE = 0.0001
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

TRAINING_STEP = 1500
train_df = pd.read_csv(filepath + "train.csv")
test_df = pd.read_csv(filepath + "test.csv")
smpsb = pd.read_csv(filepath + "sample_submission.csv")
# print(train_df)
def main(train_df, test_df):
    # this is public leaderboard ratio
    start = datetime.now()
    type_ratio = np.array([0.37053, 0.49681, 0.05936, 0.00103, 0.01295, 0.02687, 0.03242])
    
    total_df = pd.concat([train_df.iloc[:, :-1], test_df])
    
    # Aspect
    total_df["Aspect_Sin"] = np.sin(np.pi*total_df["Aspect"]/180)
    total_df["Aspect_Cos"] = np.cos(np.pi*total_df["Aspect"]/180)
    print("Aspect", (datetime.now() - start).seconds)
    
    # Hillshade
    hillshade_col = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    for col1, col2 in combinations(hillshade_col, 2):
        total_df[col1 + "_add_" + col2] = total_df[col2] + total_df[col1]
        total_df[col1 + "_dif_" + col2] = total_df[col2] - total_df[col1]
        total_df[col1 + "_div_" + col2] = (total_df[col2]+0.01) / (total_df[col1]+0.01)
        total_df[col1 + "_abs_" + col2] = np.abs(total_df[col2] - total_df[col1])
    
    total_df["Hillshade_mean"] = total_df[hillshade_col].mean(axis=1)
    total_df["Hillshade_std"] = total_df[hillshade_col].std(axis=1)
    total_df["Hillshade_max"] = total_df[hillshade_col].max(axis=1)
    total_df["Hillshade_min"] = total_df[hillshade_col].min(axis=1)
    print("Hillshade", (datetime.now() - start).seconds)
    
    # Hydrology ** I forgot to add arctan
    total_df["Degree_to_Hydrology"] = ((total_df["Vertical_Distance_To_Hydrology"] + 0.001) /
                                       (total_df["Horizontal_Distance_To_Hydrology"] + 0.01))
    
    # Holizontal
    horizontal_col = ["Horizontal_Distance_To_Hydrology",
                      "Horizontal_Distance_To_Roadways",
                      "Horizontal_Distance_To_Fire_Points"]
    
    
    for col1, col2 in combinations(hillshade_col, 2):
        total_df[col1 + "_add_" + col2] = total_df[col2] + total_df[col1]
        total_df[col1 + "_dif_" + col2] = total_df[col2] - total_df[col1]
        total_df[col1 + "_div_" + col2] = (total_df[col2]+0.01) / (total_df[col1]+0.01)
        total_df[col1 + "_abs_" + col2] = np.abs(total_df[col2] - total_df[col1])
    print("Holizontal", (datetime.now() - start).seconds)
    
    
    def categorical_post_mean(x):
        p = (x.values)*type_ratio
        p = p/p.sum()*x.sum() + 10*type_ratio
        return p/p.sum()
    
    # Wilder
    wilder = pd.DataFrame([(train_df.iloc[:, 11:15] * np.arange(1, 5)).sum(axis=1),
                          train_df.Cover_Type]).T
    wilder.columns = ["Wilder_Type", "Cover_Type"]
    wilder["one"] = 1
    piv = wilder.pivot_table(values="one",
                             index="Wilder_Type",
                             columns="Cover_Type",
                             aggfunc="sum").fillna(0)
    
    tmp = pd.DataFrame(piv.apply(categorical_post_mean, axis=1).tolist()).reset_index()
    tmp["index"] = piv.sum(axis=1).index
    tmp.columns = ["Wilder_Type"] + ["Wilder_prob_ctype_{}".format(i) for i in range(1, 8)]
    tmp["Wilder_Type_count"] = piv.sum(axis=1).values
    
    total_df["Wilder_Type"] = (total_df.filter(regex="Wilder") * np.arange(1, 5)).sum(axis=1)
    total_df = total_df.merge(tmp, on="Wilder_Type", how="left")
    
    for i in range(7):
        total_df.loc[:, "Wilder_prob_ctype_{}".format(i+1)] = total_df.loc[:, "Wilder_prob_ctype_{}".format(i+1)].fillna(type_ratio[i])
    total_df.loc[:, "Wilder_Type_count"] = total_df.loc[:, "Wilder_Type_count"].fillna(0)
    print("Wilder_type", (datetime.now() - start).seconds)
    
    
    # Soil type
    soil = pd.DataFrame([(train_df.iloc[:, -41:-1] * np.arange(1, 41)).sum(axis=1),
                          train_df.Cover_Type]).T
    soil.columns = ["Soil_Type", "Cover_Type"]
    soil["one"] = 1
    piv = soil.pivot_table(values="one",
                           index="Soil_Type",
                           columns="Cover_Type",
                           aggfunc="sum").fillna(0)
    
    tmp = pd.DataFrame(piv.apply(categorical_post_mean, axis=1).tolist()).reset_index()
    tmp["index"] = piv.sum(axis=1).index
    tmp.columns = ["Soil_Type"] + ["Soil_prob_ctype_{}".format(i) for i in range(1, 8)]
    tmp["Soil_Type_count"] = piv.sum(axis=1).values
    
    total_df["Soil_Type"] = (total_df.filter(regex="Soil") * np.arange(1, 41)).sum(axis=1)
    total_df = total_df.merge(tmp, on="Soil_Type", how="left")
    
    for i in range(7):
        total_df.loc[:, "Soil_prob_ctype_{}".format(i+1)] = total_df.loc[:, "Soil_prob_ctype_{}".format(i+1)].fillna(type_ratio[i])
    total_df.loc[:, "Soil_Type_count"] = total_df.loc[:, "Soil_Type_count"].fillna(0)
    print("Soil_type", (datetime.now() - start).seconds)
    
    icol = total_df.select_dtypes(np.int64).columns
    fcol = total_df.select_dtypes(np.float64).columns
    total_df.loc[:, icol] = total_df.loc[:, icol].astype(np.int32)
    total_df.loc[:, fcol] = total_df.loc[:, fcol].astype(np.float32)
    return total_df

total_df = main(train_df, test_df)
one_col = total_df.filter(regex="(Type\d+)|(Area\d+)").columns
total_df = total_df.drop(one_col, axis=1)
y = train_df["Cover_Type"].values
X = total_df[total_df["Id"] <= 15120].drop("Id", axis=1)
X_test = total_df[total_df["Id"] > 15120].drop("Id", axis=1)

# 训练集的特征列，验证集的特征列，训练集的label列，
# 验证集的label列 = train_test_split(数据集的特征列，数据集的label列，期望划分的验证集的大小，划分的随机种子值，是否打乱，分层抽样)
# x_train,x_vali,y_train,y_vali = train_test_split(X,y,test_size=0.1,random_state=0,stratify=y)

gc.collect() #内存回收
print(X.shape)
# skf=StratifiedKFold(n_splits=3)
# features = X.values
# labels = processLabel(y)
# testFeature = X_test.values
# # print(type(testFeature))

# for train_index,vali_index in skf.split(features,y):
#     X_train, X_vali = features[train_index], features[test_index]
#     y_train, y_vali = labels[train_index], labels[test_index]
#     print(y_train.shape)
#     train_feed = {x:X_train, y_:y_train}
#     validate_feed = {x:X_vali, y_:y_vali}
def processLabel(labels):
    t = np.zeros(shape=[labels.shape[0],7])
    for i in range(labels.shape[0]):
        t[i, int(labels[i])-1] = 1.0
    return t

def forward(input_tensor, weight, bias):
    return tf.matmul(input_tensor, weight) + bias

def train(features, labels, testFeature, originLabel):
    x = tf.placeholder(tf.float32, shape=[None, features.shape[1]], name="input")
    y_ = tf.placeholder(tf.float32, shape=[None, labels.shape[1]], name="y-input")

    Id = np.linspace(features.shape[0]+1, features.shape[0] + testFeature.shape[0], testFeature.shape[0])

    weight = tf.Variable(tf.truncated_normal([INPUT_NODE, OUTPUT_NODE], stddev=0.1))
    bias = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))  
    y = forward(x, weight, bias)

    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weight)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, features.shape[0] / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    correct_predection = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predection, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
#         validate_feed = {x:features[14000:,:], y_:labels[14000:]}
#         train_feature = features[0:14000,:]
#         train_lable = labels[0:14000]    
        test_feed = {x:testFeature}
    
        for i in range(TRAINING_STEP):
            skf=StratifiedKFold(n_splits=10)
            for train_index,vali_index in skf.split(features,originLabel):
                X_train, X_vali = features[train_index], features[vali_index]
                y_train, y_vali = labels[train_index], labels[vali_index]
                train_feed = {x:X_train, y_:y_train}
                validate_feed = {x:X_vali, y_:y_vali}
                sess.run(train_step, feed_dict=train_feed)
            if i%30==0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("%d now the vali_acc is %g" %(i, validate_acc))
                
#             xs = train_feature[i%140*100:i%140*100+100, :]
#             ys = train_lable[i%140*100:i%140*100+100]
#             sess.run(train_step, feed_dict={x:xs, y_:ys})
            
        validate_acc = sess.run(accuracy, feed_dict=validate_feed)
        print("After training the vali_acc is %g" %validate_acc)
        
        outputTenor = sess.run(y, feed_dict=test_feed)
        result = sess.run(tf.arg_max(outputTenor, 1)+1)
        Id = sess.run(tf.to_int32(Id, name="ToInt32"))
        dataframe = pd.DataFrame({'Id': Id, 'Cover_Type': result}, columns=['Id', 'Cover_Type'])
        dataframe.to_csv("sample_submission.csv", index=False)
        
trainFeature = X.values
trainLabel = processLabel(y)
testFeature = X_test.values
train(trainFeature, trainLabel, testFeature, y)

all_set =  [['Elevation', 500],
            ['Horizontal_Distance_To_Roadways', 500],
            ['Horizontal_Distance_To_Fire_Points', 500],
            ['Horizontal_Distance_To_Hydrology', 500],
            ['Hillshade_9am', 500],
            ['Aspect', 500],
            ['Hillshade_3pm', 500],
            ['Slope', 500],
            ['Hillshade_Noon', 500],
            ['Vertical_Distance_To_Hydrology', 500],
            ['Elevation_PLUS_Vertical_Distance_To_Hydrology', 200],
            ['Elevation_PLUS_Hillshade_9am_add_Hillshade_Noon', 200],
            ['Elevation_PLUS_Aspect', 200],
            ['Elevation_PLUS_Hillshade_Noon_dif_Hillshade_3pm', 200],
            ['Elevation_PLUS_Hillshade_Noon_abs_Hillshade_3pm', 200],
            ['Elevation_PLUS_Hillshade_9am', 200],
            ['Elevation_PLUS_Horizontal_Distance_To_Hydrology', 200],
            ['Elevation_PLUS_Horizontal_Distance_To_Roadways', 100],
            ['Elevation_PLUS_Vertical_Distance_To_Hydrology', 200],
            ['Wilder_Type_PLUS_Elevation', 500],
            ['Wilder_Type_PLUS_Hillshade_Noon_div_Hillshade_3pm', 500],
            ['Wilder_Type_PLUS_Degree_to_Hydrology', 200],
            ['Wilder_Type_PLUS_Hillshade_9am_div_Hillshade_3pm', 500],
            ['Wilder_Type_PLUS_Aspect_Cos', 500],
            ['Hillshade_9am_dif_Hillshade_Noon_PLUS_Hillshade_Noon_dif_Hillshade_3pm', 200],
            ['Hillshade_Noon_PLUS_Hillshade_3pm', 200],
            ['Hillshade_Noon_add_Hillshade_3pm_PLUS_Hillshade_Noon_dif_Hillshade_3pm', 200]]


def simple_feature_scores2(clf, cols, test=False, **params):
    scores = []
    bscores = []
    lscores = []
    
    X_preds = np.zeros((len(y), 7))
    scl = StandardScaler().fit(X.loc[:, cols])
    
    for train, val in StratifiedKFold(n_splits=10, shuffle=True, random_state=2018).split(X, y):
        X_train = scl.transform(X.loc[train, cols])
        X_val = scl.transform(X.loc[val, cols])
        y_train = y[train]
        y_val = y[val]
        C = clf(**params) 

        C.fit(X_train, y_train)
        X_preds[val] = C.predict_proba(X_val)
        #scores.append(accuracy_score(y_val, C.predict(X_val)))
        #bscores.append(balanced_accuracy_score(y_val, C.predict(X_val)))
        #lscores.append(log_loss(y_val, C.predict_proba(X_val), labels=list(range(1, 8))))
    
    if test:
        X_test_select = scl.transform(X_test.loc[:, cols])
        C = clf(**params)
        C.fit(scl.transform(X.loc[:, cols]), y)
        X_test_preds = C.predict_proba(X_test_select)
    else:
        X_test_preds = None
    return scores, bscores, lscores, X_preds, X_test_preds
import warnings
import gc
from multiprocessing import Pool

warnings.filterwarnings("ignore")

preds = []
test_preds = []
for colname, neighbor in tqdm(all_set):
    gc.collect()
    #print(colname, depth)
    ts, tbs, ls, pred, test_pred = simple_feature_scores2(KNeighborsClassifier,
                                                          colname.split("_PLUS_"),
                                                          test=True,
                                                          n_neighbors=neighbor)
    preds.append(pred)
    test_preds.append(test_pred)
cols = list(chain.from_iterable([[col[0] + "_KNN_{}".format(i) for i in range(1, 8)] for col in all_set]))
knn_train_df = pd.DataFrame(np.hstack(preds)).astype(np.float32)
knn_train_df.columns = cols
knn_test_df = pd.DataFrame(np.hstack(test_preds)).astype(np.float32)
knn_test_df.columns = cols
all_set = [['Elevation', 4],
           ['Horizontal_Distance_To_Roadways', 4],
           ['Horizontal_Distance_To_Fire_Points', 3],
           ['Horizontal_Distance_To_Hydrology', 4],
           ['Hillshade_9am', 3],
           ['Vertical_Distance_To_Hydrology', 3],
           ['Slope', 4],
           ['Aspect', 4],
           ['Hillshade_3pm', 3],
           ['Hillshade_Noon', 3],
           ['Degree_to_Hydrology', 3],
           ['Hillshade_Noon_dif_Hillshade_3pm', 3],
           ['Hillshade_Noon_abs_Hillshade_3pm', 3],
           ['Elevation_PLUS_Hillshade_9am_add_Hillshade_Noon', 5],
           ['Elevation_PLUS_Hillshade_max', 5],
           ['Elevation_PLUS_Horizontal_Distance_To_Hydrology', 5],
           ['Aspect_Sin_PLUS_Aspect_Cos_PLUS_Elevation', 5],
           ['Elevation_PLUS_Horizontal_Distance_To_Fire_Points', 5],
           ['Wilder_Type_PLUS_Elevation', 5],
           ['Elevation_PLUS_Hillshade_9am', 5],
           ['Elevation_PLUS_Degree_to_Hydrology', 5],
           ['Wilder_Type_PLUS_Horizontal_Distance_To_Roadways', 5],
           ['Wilder_Type_PLUS_Hillshade_9am_add_Hillshade_Noon', 4],
           ['Wilder_Type_PLUS_Horizontal_Distance_To_Hydrology', 5],
           ['Wilder_Type_PLUS_Hillshade_Noon_abs_Hillshade_3pm', 4],
           ['Hillshade_9am_add_Hillshade_Noon_PLUS_Hillshade_std', 4],
           ['Hillshade_9am_PLUS_Hillshade_9am_add_Hillshade_Noon', 4],
           ['Hillshade_9am_add_Hillshade_Noon_PLUS_Hillshade_Noon_add_Hillshade_3pm', 5]]

def simple_feature_scores(clf, cols, test=False, **params):
    scores = []
    bscores = []
    lscores = []
    
    X_preds = np.zeros((len(y), 7))
    
    
    for train, val in StratifiedKFold(n_splits=10, shuffle=True, random_state=2018).split(X, y):
        X_train = X.loc[train, cols]
        X_val = X.loc[val, cols]
        y_train = y[train]
        y_val = y[val]
        C = clf(**params) 

        C.fit(X_train, y_train)
        X_preds[val] = C.predict_proba(X_val)
        #scores.append(accuracy_score(y_val, C.predict(X_val)))
        #bscores.append(balanced_accuracy_score(y_val, C.predict(X_val)))
        #lscores.append(log_loss(y_val, C.predict_proba(X_val), labels=list(range(1, 8))))
    
    if test:
        X_test_select = X_test.loc[:, cols]
        C = clf(**params)
        C.fit(X.loc[:, cols], y)
        X_test_preds = C.predict_proba(X_test_select)
    else:
        X_test_preds = None
    return scores, bscores, lscores, X_preds, X_test_preds
preds = []
test_preds = []
for colname, depth in tqdm(all_set):
    #print(colname, depth)
    ts, tbs, ls, pred, test_pred = simple_feature_scores(DecisionTreeClassifier,
                                                         colname.split("_PLUS_"),
                                                         test=True,
                                                         max_depth=depth)
    preds.append(pred)
    test_preds.append(test_pred)

cols = list(chain.from_iterable([[col[0] + "_DT_{}".format(i) for i in range(1, 8)] for col in all_set]))
dt_train_df = pd.DataFrame(np.hstack(preds)).astype(np.float32)
dt_train_df.columns = cols

dt_test_df = pd.DataFrame(np.hstack(test_preds)).astype(np.float32)
dt_test_df.columns = cols
# target encoding features(1.2.3)
te_train_df = total_df.filter(regex="ctype").iloc[:len(train_df)]
te_test_df = total_df.filter(regex="ctype").iloc[len(train_df):]
train_level2 = train_df[["Id"]]
test_level2 = test_df[["Id"]]
y = train_df["Cover_Type"].values
X = total_df[total_df["Id"] <= 15120].drop("Id", axis=1)
X_test = total_df[total_df["Id"] > 15120].drop("Id", axis=1)
type_ratio = np.array([0.37053, 0.49681, 0.05936, 0.00103, 0.01295, 0.02687, 0.03242])
class_weight = {k: v for k, v in enumerate(type_ratio, start=1)}
RFC1_col = ["RFC1_{}_proba".format(i) for i in range(1, 8)]
for col in RFC1_col:
    train_level2.loc[:, col] = 0
    test_level2.loc[:, col] = 0
print(RFC1_col)
