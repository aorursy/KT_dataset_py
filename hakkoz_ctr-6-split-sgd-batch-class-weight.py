import pandas as pd

import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier

from tqdm import tqdm

import pickle

from sklearn.utils import class_weight

from sklearn.metrics import roc_auc_score



#import utility functions

from utils3 import display_test_scores_v2



#hide warnings

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
dtype_mapping = pd.read_csv("../input/ctr-train-test-split-0/dtype_mapping.csv", index_col=0)

dtype_mapping = dtype_mapping.iloc[:,0].to_dict()
def read_jackie_chunk(model, csv_path_list, chunksize, dtype_mapping):

    for csv_path in csv_path_list:

        for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=50):

            if len(chunk) == 1:

                continue

            chunk = chunk.drop(columns=[chunk.columns[i] for i in range(1,38)])

            chunk = chunk.dropna()

            y = chunk["label"]

            # drop uid_tenc also

            X = chunk.iloc[:, 1:]

            # drop pt_d_tenc also

            X = X.drop(columns=["pt_d_tenc"])

            

            # class weight technique 1: compute_class_weight 

            weights = class_weight.compute_class_weight("balanced", np.array([0,1]), y)

            weights_dict = {key: value for key,value in zip(list(range(len(weights))),list(weights))}

           

            # class weight technique 2: counts to length ratio

            #weights = len(y) / y.value_counts()

            #weights_dict = weights.to_dict()

            

            # class weight technique 3: smoothen weights (log)

            #mu = 0.15

            #labels_dict = y.value_counts().to_dict()

            #weights_dict = dict()

            #for key in labels_dict.keys():

            #    score = np.log(mu*len(y)/float(labels_dict[key]))

            #    weights_dict[key] = score if score > 1 else 1

            

            model.set_params(class_weight=weights_dict)

            model = model.partial_fit(X,y, classes = np.array([0,1]))

    return model
csv_path_list = ['../input/ctr-train-test-split-1/train_df1.csv',

                '../input/ctr-train-test-split-1-5/train_df1_5.csv',

                 '../input/ctr-train-test-split-2/train_df3.csv',

                '../input/ctr-train-test-split-2-5/train_df2_5.csv',

                 '../input/ctr-train-test-split-3/train_df3.csv',

                 '../input/ctr-train-test-split-3-5/train_df3_5.csv',

                 '../input/ctr-train-test-split-4v2/train_df4',

                 '../input/ctr-train-test-split-4-5/train_df4_5'

                ]



chunksize = 10 ** 5



model = SGDClassifier(loss = "log", n_jobs = -1, random_state = 0, warm_start = True)



model_final = read_jackie_chunk(model, csv_path_list, chunksize, dtype_mapping)
pred_list=[]

y_test_list=[]

for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-0/test_df.csv', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=20):

    chunk = chunk.drop(columns=[chunk.columns[i] for i in range(1,38)])

    chunk = chunk.dropna()

    y = chunk["label"]

    # drop uid_tenc also

    X = chunk.iloc[:, 1:]

    # drop pt_d_tenc also

    X = X.drop(columns=["pt_d_tenc"])

    y_pred = model_final.predict_proba(X)

    pred_list.append(y_pred)

    y_test_list.append(y)
def display_test_scores(test, pred):

    str_out = ""

    str_out += ("TEST SCORES\n")

    str_out += ("\n")



    #print AUC score

    auc = roc_auc_score(test, pred)

    str_out += ("AUC: {:.4f}\n".format(auc))

    str_out += ("\n")

    

    false_indexes = np.where(test != pred)

    return str_out, false_indexes
pred_array = np.concatenate(pred_list)

y_test = pd.concat(y_test_list)

results, false = display_test_scores(y_test, pred_array[:,1])

print(results)
# train whole dataset (including test_df.csv)

for chunk in tqdm(pd.read_csv('../input/ctr-train-test-split-0/test_df.csv', chunksize=chunksize,index_col=0, dtype=dtype_mapping), total=20):

    chunk = chunk.drop(columns=[chunk.columns[i] for i in range(1,38)])

    chunk = chunk.dropna()

    y = chunk["label"]

    # drop uid_tenc also

    X = chunk.iloc[:, 1:]

    # drop pt_d_tenc also

    X = X.drop(columns=["pt_d_tenc"])

    # class weight technique 1: compute_class_weight 

    weights = class_weight.compute_class_weight("balanced", np.array([0,1]), y)

    weights_dict = {key: value for key,value in zip(list(range(len(weights))),list(weights))}

    model.set_params(class_weight=weights_dict)

    model_final_final = model_final.partial_fit(X,y, classes = np.array([0,1]))
# save model to file

pickle.dump(model_final_final, open("sgd_model.pkl", "wb"))