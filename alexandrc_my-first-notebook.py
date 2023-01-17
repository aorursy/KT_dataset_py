import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output 

# Any results you write to the current directory are saved as output.
# Load data

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")



y_train = data_train['Survived']

X_train = data_train.drop(labels=["Survived"], axis=1)

X_test = data_test

n_train = X_train.shape[0]

n_test = X_test.shape[0]

print ("train size", n_train)

print ("test_size", n_test)



X_train.head()
X_all = X_train.append(X_test, ignore_index=True)

print (X_all.info())
# Name, Ticket and Cabin are decided to be useless. Drop them

X_train = X_train.drop(labels=["Name", "Ticket", "Cabin"], axis=1)

X_test = X_test.drop(labels=["Name", "Ticket", "Cabin"], axis=1)

X_all = X_all.drop(labels=["Name", "Ticket", "Cabin"], axis=1)
# ------------------------------------------------------------------------

# Methods of filling NAN values

# ------------------------------------------------------------------------

def my_fillna(X, method):

    if method=="bfill":

        res = X.fillna(method="bfill")

        res = res.fillna(method="ffill")  # if last one is nan

    elif method=="ffill":

        res = X.fillna(method="ffill")

        res = res.fillna(method="bfill")  # if first one is nan

    elif method=="zero":

        res = X.fillna({"Age":0.0, "Fare":0.0, "Embarked":"<NAN>"})

    elif method=="avg":

        res = X.fillna({"Embarked":"<NAN>"})

        age_avg = res['Age'].mean()

        fare_avg = res['Fare'].mean()

        res = res.fillna({"Age":age_avg, "Fare":fare_avg})

    elif method=="drop":

        res = X.dropna()

    return res   
# ------------------------------------------------------------------------

# Convert all categorical features to numeric/real

# ------------------------------------------------------------------------

from sklearn.feature_extraction import DictVectorizer as DV



def encode_cat(X):

    encoder = DV(sparse = False)

    X_cat = encoder.fit_transform(X.T.to_dict().values())

    return X_cat

    

def encode_cat_test(X):

    print('\nSource data:\n')

    print(X.shape)

    print(X[:10])

    encoder = DV(sparse = False)

    X_cat = encoder.fit_transform(X.T.to_dict().values())

    print('\nEncoded data:\n')

    print(X_cat.shape)

    print(X_cat[:10])

    print('\nVocabulary:\n')

    print(encoder.vocabulary_)

    print(encoder.feature_names_)

    return X_cat
zzz = encode_cat_test(my_fillna(X_train,"bfill"))



#make list for non-cat column of features (manually, for a while)

real_ind_list_A = [0,5,6,7,8,11]  # if fillna="avg" or "zero"

real_ind_list_B = [0,4,5,6,7,10]  # if fillna="bfill" or "ffill"
# ------------------------------------------------------------------------

# Feature scaling

# ------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler



#create standard scaler

def scale_features(X_tr, X_tt, real_ind_list, y):

    X_tr_scaled = np.array(X_tr)

    X_tt_scaled = np.array(X_tt)

    

    X_tr_real = X_tr[:,real_ind_list]

    X_tt_real = X_tt[:,real_ind_list]

    

    scaler = StandardScaler()    

    scaler.fit(X_tr_real, y)  # set scaled parameters relatively to train data

    

    X_tr_scaled[:,real_ind_list] = scaler.transform(X_tr_real)

    X_tt_scaled[:,real_ind_list] = scaler.transform(X_tt_real)  

    

    return X_tr_scaled, X_tt_scaled
# ------------------------------------------------------------------------

# Implementation of "greedy" algorithm for feature selection 

# ------------------------------------------------------------------------

from sklearn.cross_validation import cross_val_score



def reduce_features(X, y, cls):    

    n_features = X.shape[1]

    list_features_ind = [] #initial

    max_prev_score = 0.0

    list_iter_score = []    

    flag_stop = False

    flag_del = False

    n_iter = 0

    

    while flag_stop == False:

        

        n_iter = n_iter + 1

        

        # try to add one feature

        res_score = np.zeros(n_features)        

        for i in range(n_features):

            if i in list_features_ind:

                pass

            else:

                list_tmp_ind = list(list_features_ind)

                list_tmp_ind.append(i)

                X_tmp = X[:,list_tmp_ind]

                res_cv = cross_val_score(cls, X_tmp, y)    # cross validation score

                res_score[i] = res_cv.mean()

                

        #print n_iter, "add", res_score

        max_ind = np.argmax(res_score)

        max_val = res_score[max_ind]

        #print max_ind, max_val

        if max_val > max_prev_score:

            list_features_ind.append(max_ind)

            max_prev_score = max_val

            list_iter_score.append(max_val)

            flag_del = False

        else:            

            flag_del = True        

            

        # if adding one feature wasn't effective, try to delete one chosen feature

        if flag_del == True:

            if len(list_features_ind) <= 1:

                flag_stop = True

                break

                

            res_score = np.zeros(n_features)        

            for i in range(n_features):

                if i in list_features_ind:

                    list_tmp_ind = list(list_features_ind)

                    list_tmp_ind.remove(i)

                    X_tmp = X[:,list_tmp_ind]

                    res_cv = cross_val_score(cls, X_tmp, y)   # cross validation score

                    res_score[i] = res_cv.mean()                        



            #print n_iter, "del", res_score

            max_ind = np.argmax(res_score)

            max_val = res_score[max_ind]

            #print max_ind, max_val

            if max_val > max_prev_score:

                list_features_ind.remove(max_ind)

                max_prev_score = max_val

                list_iter_score.append(max_val)

            else:              

                flag_stop = True            

                break       



    return list_features_ind, list_iter_score                 
# ------------------------------------------------------------------------

# Reduce number of features using PCA

# ------------------------------------------------------------------------

from sklearn.decomposition import PCA

def reduce_features_PCA(X, y):

    mdl = PCA()

    mdl.fit(X)

    print (mdl.explained_variance_ratio_)

    #TODO:

    # define the optimal number of principal components

    # do some tests for three suitable numbers via 'cross_val_score'    

    # return reduced_dataset, list of remained features, and list of dropped features
# ------------------------------------------------------------------------

# Classifier initialization

# ------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression as LR

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.ensemble import RandomForestClassifier as RandomForest

from sklearn.ensemble import BaggingClassifier as Bagging

from sklearn.tree import DecisionTreeClassifier as DecisionTree

from sklearn.linear_model import Perceptron

from sklearn.neural_network import MLPClassifier as MLP



def get_classifier(cls, param):

    if cls=="LR":

        return LR(C=param, random_state=123)

    elif cls=="KNN":

        return KNN(n_neighbors=param)

    elif cls=="RForest":

        return RandomForest(n_estimators=75, max_depth=param, random_state=123)

    elif cls=="BagTree":

        return Bagging(base_estimator=DecisionTree(max_depth=param, random_state=123), random_state=123)

    elif cls=="Perceptron":    

        return Perceptron(eta0=param, random_state=123)

    elif cls=="MLP":

        return MLP(hidden_layer_sizes=(20,), alpha=param, max_iter=40, solver='lbfgs') #too slow

    else:

        pass
#----------------------------------------------------------------------

# Main cell: Research !

#----------------------------------------------------------------------

import warnings

warnings.filterwarnings("ignore")



research_list = [     

{    

    'classifier' : "LR",    

    'param_values' : [0.07,0.1,0.13,0.17,0.2],

    'param_name' : "C" 

},

{    

    'classifier' : "KNN",    

    'param_values' : [1,2,3,4,5],

    'param_name' : "K" 

},    

{    

    'classifier' : "RForest",    

    'param_values' : [3,4,5,7,10],

    'param_name' : "max_depth" 

},    

{    

    'classifier' : "BagTree",    

    'param_values' : [3,4,5,7,10],

    'param_name' : "max_depth" 

},

{    

    'classifier' : "Perceptron",    

    'param_values' : [0.001,0.01,0.1,1],

    'param_name' : "eta0" 

}]



fillna_meth_list = ["bfill", "ffill", "zero", "avg"]



view_log = False

scaling = True



for item in research_list:

    

    cls_max_score = 0.0

    cls_max_ind = []

    cls_max_param = 0.0



    for meth in fillna_meth_list:    

        X_train_na = my_fillna(X_train, meth)

        X_test_na = my_fillna(X_test, meth)

        X_all_na = X_train_na.append(X_test_na, ignore_index=True)

        X_all_cat = encode_cat(X_all_na)         

        X_train_cat = X_all_cat[:n_train]

        X_test_cat = X_all_cat[n_train:] 



        if scaling == True:

            if meth=="zero" or meth=="avg":

                X_train_cat, X_test_cat = scale_features(X_train_cat, X_test_cat, y_train, real_ind_list_A)

            else:

                X_train_cat, X_test_cat = scale_features(X_train_cat, X_test_cat, y_train, real_ind_list_B)        



        for pval in item['param_values']:

            cls = get_classifier(item['classifier'], pval)

            l_ind, l_scores = reduce_features(X_train_cat, y_train, cls)

            if view_log == True:

                print ("---", meth, ",", item['param_name'], "=", pval, "---")

                print (l_ind)

                print (l_scores[-1:][0])

            if cls_max_score <= max(l_scores):

                cls_max_meth = meth

                cls_max_score = l_scores[-1:][0]  # last one must be the best

                cls_max_ind = list(l_ind)

                cls_max_param = pval



    print ("---", item['classifier'], ":", cls_max_meth, ",", item['param_name'], "=", cls_max_param, "---")

    print ("features: ", cls_max_ind)

    print ("score: ", cls_max_score)  # <- cross_val_score().mean
#----------------------------------------------------------------------

# BONUS: Visualization of dataset

#----------------------------------------------------------------------

from sklearn.manifold import TSNE, MDS

from sklearn.metrics.pairwise import pairwise_distances

from sklearn.preprocessing import scale

%matplotlib inline



# prepare data

X_train_na = my_fillna(X_train, 'avg')

X_test_na = my_fillna(X_test, 'avg')

X_all_na = X_train_na.append(X_test_na, ignore_index=True)

X_all_cat = encode_cat(X_all_na)         

X_train_cat = X_all_cat[:n_train]

X_test_cat = X_all_cat[n_train:]



X_train_cat, X_test_cat = scale_features(X_train_cat, X_test_cat, y_train, real_ind_list_A) #avg



# calculate projection on 2 dimensions using three different algorithms

X_tsne_view = TSNE().fit_transform(X_train_cat)

X_mds_view = MDS().fit_transform(X_train_cat)

X_pca_view = PCA(n_components=2).fit_transform(X_train_cat)



# draw plots

fig = plt.figure(figsize=(12, 5))

plt.subplot(131)

for response, color in zip([0,1],['red', 'blue']):

    plt.scatter(X_tsne_view[y_train.values==response, 0], 

                X_tsne_view[y_train.values==response, 1], c=color, alpha=1)

plt.legend(["died","surv."])

plt.xlabel("t-NSE algoritm")

    

plt.plot()

plt.subplot(132)

for response, color in zip([0,1],['red', 'blue']):

    plt.scatter(X_mds_view[y_train.values==response, 0], 

                X_mds_view[y_train.values==response, 1], c=color, alpha=1)    

plt.legend(["died","surv."])    

plt.xlabel("MDS algorithm (metric=cos)")



plt.plot()

plt.subplot(133)

for response, color in zip([0,1],['red', 'blue']):

    plt.scatter(X_pca_view[y_train.values==response, 0], 

                X_pca_view[y_train.values==response, 1], c=color, alpha=1)    

plt.legend(["died","surv."])    

plt.xlabel("PCA method (n=2)")
# use the best scored model and calculate prediction for y_test (submission)

meth = "avg"                          # method of filling nan values

feature_list = [9, 11, 0, 8, 5, 4]    # used encoded features

cls = get_classifier("RForest", 7)    # used classifier 



# prepare data with chosen params

X_train_na = my_fillna(X_train, meth)

X_test_na = my_fillna(X_test, meth)

X_all_na = X_train_na.append(X_test_na, ignore_index=True)

X_all_cat = encode_cat(X_all_na)         

X_train_cat = X_all_cat[:n_train]

X_test_cat = X_all_cat[n_train:]



X_train_cat, X_test_cat = scale_features(X_train_cat, X_test_cat, y_train, real_ind_list_A) #avg



# use chosen classifier

cls.fit(X_train_cat[:, feature_list], y_train)

y_test = cls.predict(X_test_cat[:, feature_list])

print("cross_val_score", cross_val_score(cls, X_train_cat[:, feature_list], y_train).mean())

print("train score", cls.score(X_train_cat[:, feature_list], y_train))
# write submission to file

fout = open("answer_1.csv", "w")

i = 891

fout.write("PassengerId,Survived\n")

for y in y_test:

    i = i + 1

    fout.write(str(i)+","+str(y)+"\n")

fout.close()