# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
averages = train_labels.groupby('title')['accuracy_group'].agg(['median','count'])
ans = {'Bird Measurer (Assessment)':1,

       'Cart Balancer (Assessment)': 3,

       'Cauldron Filler (Assessment)':3,

       'Chest Sorter (Assessment)': 0,

       'Mushroom Sorter (Assessment)':3

      }
sample_submission1 = sample_submission
sample_submission1['accuracy_group'] = test.groupby('installation_id').last().title.map(ans).reset_index(drop=True)

sample_submission1.to_csv('submission_0.csv', index=None)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
def test_in_sub(test):

    tgms = test.groupby('installation_id').last().game_session

    tgms1 = tgms.reset_index()

    test_ass = test[test.type == "Assessment"]

    tgms1["title"] = str(test_ass[test_ass.game_session == tgms1["game_session"][0]].title.reset_index(drop=True)[0])

    

    for i in range(0,len(tgms1)):

        tgms1["title"][i] = str(test_ass[test_ass.game_session==tgms1["game_session"][i]].title.reset_index(drop=True)[0])

    return tgms1
def c_accuracy_group(df):

    df["accuracy_group"]=0

    for i in range(0,len(df)):

        acc = float(df["accuracy"][i])

        if (acc == float(0)):

            df["accuracy_group"][i]=0

        elif (acc < float(0.5)):

            df["accuracy_group"][i]=1

        elif (acc < float(1)):

            df["accuracy_group"][i]=2

        elif (acc == float(1)):

            df["accuracy_group"][i]=3

        else:

            df["accuracy_group"][i] = None

    return df

            
def test_to_label(test):

    print("Converting to label format, as of submissions done in assessment")

    test_ass = test[test.type == "Assessment"]

    test_ass_sub = test_ass[(((test.event_code == 4100) & (test.title != 'Bird Measurer (Assessment)'))) | (((test.event_code == 4110) & (test.title == 'Bird Measurer (Assessment)')))]

    test_ass_sub_inf = test_ass_sub[["installation_id","game_session","timestamp","title","event_data"]]

    test_ass_sub_inf0 = test_ass_sub_inf

    test_ass_sub_inf0["correct"] = 0

    test_ass_sub_inf0["incorrect"] = 0

    

    for i in range(0,len(test_ass_sub_inf0)):

        if "\"correct\":true" in test_ass_sub_inf0["event_data"][test_ass_sub_inf0.index[i]]:

            test_ass_sub_inf0["correct"][test_ass_sub_inf0.index[i]] = 1

        else:

            test_ass_sub_inf0["incorrect"][test_ass_sub_inf0.index[i]] = 1

    test_ass_sub_inf1 = test_ass_sub_inf0.groupby(by=["installation_id","game_session","title"],sort=False).sum()

    test_ass_sub_inf2 = test_ass_sub_inf1

    test_ass_sub_inf2 = test_ass_sub_inf2.reset_index()

    test_ass_sub_inf2["accuracy"] =float(0)

    

    for i in range(0,len(test_ass_sub_inf2)):

        corr = test_ass_sub_inf2["correct"][i]

        incor = test_ass_sub_inf2["incorrect"][i]

        test_ass_sub_inf2["accuracy"][i] = float(corr)/(incor+corr)

    

    test_ass_sub_inf3 = test_ass_sub_inf2

    test_ass_sub_inf3 = c_accuracy_group(test_ass_sub_inf3)

    return test_ass_sub_inf3

    
def get_time_gm(train, train_labels):

    print("Adding cumulative game played time for each session")

    train_data_1 = train[["installation_id", "game_session", "title","game_time"]]

    train_time_god_2 = train[["installation_id", "game_session", "title","game_time"]]

    ttg_time = train_time_god_2.groupby(by=['game_session'], sort=False).last().game_time.reset_index()

    ttg_time0 = train_time_god_2[["installation_id", "game_session", "title"]].merge(ttg_time, on = 'game_session', how = 'left')

    ttg_time00 = ttg_time0.groupby(by=['installation_id','game_session'], sort=False).sum().groupby(level=[0]).cumsum()

    ttg_time1 = ttg_time00.reset_index()

    ttg_time2 = ttg_time1[["game_session","game_time"]]

    train_labels1 = train_labels

    # join train with train labels

    train_labels_t = train_labels1.merge(ttg_time2, on = 'game_session', how = 'left')



    return train_labels_t

def get_final_feat(train, train_labels_derive_time):   

    # train, train_labels_derive_time

    print("Adding more time sensitive features such as: all correct, all incorrect, score, average score, corr to incorr ratio and vice versa until any game session")

    

    train_edit_c = train[["installation_id", "game_session", "title", "event_data"]]

    trc_ic = train_edit_c[(train_edit_c.event_data.str.contains("\"correct\":true") | train_edit_c.event_data.str.contains("\"correct\":false")) ]

    trc_ic1 = trc_ic.groupby(["installation_id", "game_session", "title"]).size().reset_index().drop(columns = [0])

    

    trc = train_edit_c[train_edit_c.event_data.str.contains("\"correct\":true")]

    trc_edit = trc[["installation_id","game_session"]]

    trc_edit["correct_all"] = 1

    trc_edit_all1 = trc_edit.groupby(by=['installation_id','game_session'], sort = False).sum().groupby(level=[0]).cumsum()

    trc_edit_all1 = trc_edit_all1.reset_index()

    

    tric = train_edit_c[train_edit_c.event_data.str.contains("\"correct\":false")]

    tric_edit = tric[["installation_id","game_session"]]

    tric_edit["incorrect_all"] = 1

    tric_edit_all1 = tric_edit.groupby(by=['installation_id','game_session'], sort=False).sum().groupby(level=[0]).cumsum()

    tric_edit_all1 = tric_edit_all1.reset_index()

    

    print("Adding correct all and incorrect all feature, later we might wanna add specific accuracy groups of titles/assesssments, to record history of gameplay of user")

    # join train with train labels

    train_c_1 = trc_ic1.merge(trc_edit_all1[["game_session","correct_all"]], on = 'game_session', how = 'left')

    train_c_2 = train_c_1.merge(tric_edit_all1[["game_session","incorrect_all"]], on = 'game_session', how = 'left')

    

    # join train with train labels

    train_c_2["correct_all"].fillna(0, inplace=True)

    train_c_2["incorrect_all"].fillna(0, inplace=True)

    to_get_acc = train_c_2 # contains all the gms with either true or false

    

    print("Adding score and score count")

    to_get_acc1 = to_get_acc

    

    to_get_acc1["score"] = 0.000001

    to_get_acc1["score_c"] = 0

   # to_get_acc1["acc_r"] = 0.000001

    #to_get_acc1["inacc_r"] = 0.000001

    

   

    for i in range(0,len(to_get_acc1)):

        acc = to_get_acc1["correct_all"][i]

        ina = to_get_acc1["incorrect_all"][i]

        if((acc == 0) and (ina) == 0):

            to_get_acc1["score_c"][i] = 0

            to_get_acc1["score"][i] = 0

         #   to_get_acc1["acc_r"][i] = 0

         #   to_get_acc1["inacc_r"][i] = 0

            continue

        elif(acc == 0):

            to_get_acc1["score"][i] = round(float(ina),3)*(-5)

            to_get_acc1["score_c"][i] = 1

            #to_get_acc1["acc_r"][i] = 0

          #  to_get_acc1["inacc_r"][i] = ina

        elif(ina == 0):

            to_get_acc1["score"][i] = round(float(acc),3)*(5)

            to_get_acc1["score_c"][i] = 1

          #  to_get_acc1["acc_r"][i] = acc

          #  to_get_acc1["inacc_r"][i] = 0

        elif((ina != 0) and (acc != 0)):

            to_get_acc1["score"][i] = round((float(acc)),3)*3-round((float(ina)),3)*1

            to_get_acc1["score_c"][i] = 1

           # to_get_acc1["acc_r"][i] = round(float(acc)/ina,3)

           # to_get_acc1["inacc_r"][i] = round(float(ina)/acc,3)

            

    #to_get_acc1

    

    train_copy = train[["installation_id", "game_session", "title"]].groupby(by = ["installation_id","game_session","title"], sort=False).size().reset_index()

    # join train with train labels

    train_t_1 = train_copy.drop(columns=[0]).merge(to_get_acc1[["game_session","correct_all", "incorrect_all", "score", "score_c"]], on = 'game_session', how = 'left')

    train_t_2 = train_t_1

    train_t_2["correct_all"].fillna(0, inplace=True)

    train_t_2["incorrect_all"].fillna(0, inplace=True)

    train_t_2["score"].fillna(0, inplace=True)

    train_t_2["score_c"].fillna(0, inplace=True)

    

    train_t_3 = train_t_2

    train_t_3 = train_t_3.groupby(by=['installation_id','game_session','title'],sort=False).sum().groupby(level=[0]).cumsum()

    train_t_3 = train_t_3.reset_index()

    

    print("Adding average score")

    train_t_4 = train_t_3

    # now count average score till that point, acc_r, inacc_r

    train_t_4["average_score"] = float(0)

    train_t_4["acc_r"] = float(0)

    train_t_4["inacc_r"] = float(0)

    for i in range(0,len(train_t_4)):

        acc = train_t_4["correct_all"][i]

        inacc = train_t_4["incorrect_all"][i]

        score = round(float(train_t_4["score"][i]))

        count = train_t_4["score_c"][i]

        if (count!=0):

            train_t_4["average_score"][i] = round(float(score)/count,3)

        else:

            train_t_4["average_score"][i] = 0

        if((inacc == 0)&(acc == 0)):

            train_t_4["acc_r"][i] = 0

            train_t_4["inacc_r"][i] = 0

        elif(inacc == 0):

            train_t_4["acc_r"][i] = acc

            train_t_4["inacc_r"][i] = 0

        elif(acc == 0):

            train_t_4["acc_r"][i] = 0

            train_t_4["inacc_r"][i] = inacc

        elif((inacc != 0) & (acc != 0)):

            train_t_4["acc_r"][i] = round(float(acc)/inacc,3)

            train_t_4["inacc_r"][i] = round(float(inacc)/acc,3)

                

    train_t_5 = train_t_4        

    

    print("Almost done, combining all into label format")

    # join train with train labels

    train_labels_derive_time_corr = train_labels_derive_time.merge(train_t_5[["game_session","correct_all","incorrect_all","score","score_c","average_score","acc_r","inacc_r"]], on = 'game_session', how = 'left')

    return train_labels_derive_time_corr
def get_all(train):

    train_labels_derive = test_to_label(train)

    train_labels_derive_time = get_time_gm(train,train_labels_derive)

    get = get_final_feat(train, train_labels_derive_time)

    return get
def get_sub(test):

    test_labels_derive = test_in_sub(test)

    test_labels_derive_time = get_time_gm(test,test_labels_derive)

    get = get_final_feat(test, test_labels_derive_time)

    return get
    %%time

    get_train = get_all(train)
    %%time

    get_test = get_all(test)
%%time

test_sub_a = get_sub(test)
test_sub_a
get_sub = test_sub_a.drop(columns = ["game_session", "installation_id"])
get_train.to_csv("train_converted.csv", index = None)

get_test.to_csv("test_converted_train.csv", index = None)

test_sub_a.to_csv("test_sub_converted.csv", index = None)
get_train
get_trainA = get_train.drop(columns = ["correct", "incorrect", "accuracy"])
get_testA = get_test.drop(columns = ["correct", "incorrect", "accuracy"])
labels_map = {"Mushroom Sorter (Assessment)":1,"Bird Measurer (Assessment)":2,"Cauldron Filler (Assessment)":3,"Chest Sorter (Assessment)":4,"Cart Balancer (Assessment)":5}
get_trainB = get_trainA.drop(columns =["installation_id","game_session"])

get_trainB['title'] = get_trainB['title'].map(labels_map)
get_trainC = get_trainB
# Shuffling randomly for better variability and better chances of prediction, NOTE We havent normalized the distribution yet...!!!!

get_trainD = get_trainC.sample(frac=1).reset_index(drop=True)
get_testB = get_testA.drop(columns =["installation_id","game_session"])

get_testB['title'] = get_testB['title'].map(labels_map)
get_testC = get_testB
# Shuffling randomly for better variability and better chances of prediction, NOTE We havent normalized the distribution yet...!!!!

get_testD = get_testC.sample(frac=1).reset_index(drop=True)
get_sub
get_subA = get_sub

get_subA['title'] = get_subA['title'].map(labels_map) 
get_subA
X1 = get_trainD.drop(columns = ["accuracy_group"])

Y1 = get_trainD["accuracy_group"]
from pandas import read_csv

from numpy import set_printoptions

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

X1r = scaler.fit_transform(X1)
X_test1 = pd.DataFrame(get_testD.drop(columns = ["accuracy_group"]))

X_test1
#Y_test1 = pd.DataFrame(["accuracy_group"])

Y_test1 = get_testD["accuracy_group"]

Y_test1
from sklearn.metrics import mean_squared_error

from math import sqrt



#rms = sqrt(mean_squared_error(y_actual, y_predicted))
import numpy as np

import pandas as pd

import datetime

from catboost import CatBoostClassifier

from time import time

from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix

# this function is the quadratic weighted kappa (the metric used for the competition submission)

def qwk(act,pred,n=4,hist_range=(0,3)):

    

    # Calculate the percent each class was tagged each label

    O = confusion_matrix(act,pred)

    # normalize to sum 1

    O = np.divide(O,np.sum(O))

    

    # create a new matrix of zeroes that match the size of the confusion matrix

    # this matriz looks as a weight matrix that give more weight to the corrects

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            # makes a weird matrix that is bigger in the corners top-right and botton-left (= 1)

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    # make two histograms of the categories real X prediction

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    # multiply the two histograms using outer product

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E)) # normalize to sum 1

    

    # apply the weights to the confusion matrix

    num = np.sum(np.multiply(W,O))

    # apply the weights to the histograms

    den = np.sum(np.multiply(W,E))

    

    return 1-np.divide(num,den)

    
# Cross Validation Classification Report

#from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

#filename = 'pima-indians-diabetes.data.csv'

#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#dataframe = read_csv(filename, names=names)

#array = dataframe.values

X_train = get_trainD.drop(columns = ["accuracy_group"])

Y_train = get_trainD["accuracy_group"]

model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

predicted1 = model.predict(X_test1)

report1 = classification_report(Y_test1, predicted1)

print(report1)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms1 = sqrt(mean_squared_error(Y_test1, predicted1))

print(rms1)

print(str(rms1**2))
# oof is an zeroed array of the same size of the input dataset

oof2 = np.zeros(len(X_test1))

oof2 = model.predict(X_test1)

print('OOF QWK:', qwk(Y_test1, oof2))
# Cross Validation Classification Report

#from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

#filename = 'pima-indians-diabetes.data.csv'

#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#dataframe = read_csv(filename, names=names)

#array = dataframe.values

#X = get_trainD.drop(columns = ["accuracy_group"])

#Y = get_trainD["accuracy_group"]

test_size = 0.33

seed = 7



X_1 = get_trainD.drop(columns = ["accuracy_group"])

Y_1 = get_trainD["accuracy_group"]

X_train, X_test, Y_train, Y_test = train_test_split(X_1, Y_1, test_size=test_size,

random_state=seed)

model2 = DecisionTreeClassifier()

model2.fit(X_train, Y_train)

predicted2 = model2.predict(X_test1)

#from sklearn.metrics import classification_report

report2 = classification_report(Y_test1, predicted2)

print(report2)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms2 = sqrt(mean_squared_error(Y_test1, predicted2))

print(rms2)

print(str(rms2**2))
# oof is an zeroed array of the same size of the input dataset

oof1 = np.zeros(len(X_test1))

oof1 = modellgr.predict(X_test1)

print('OOF QWK:', qwk(Y_test1, oof1))
# Cross Validation Classification Report

#from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

#filename = 'pima-indians-diabetes.data.csv'

#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#dataframe = read_csv(filename, names=names)

#array = dataframe.values

#X = get_trainD.drop(columns = ["accuracy_group"])

#Y = get_trainD["accuracy_group"]

#test_size = 0.33

#seed = 7

modellgrr = LogisticRegression()

modellgrr.fit(X1r, Y1)

predicted2r = modellgr.predict(X_test1)

#from sklearn.metrics import classification_report

reportlgrr = classification_report(Y_test1, predicted2r)

print(reportlgrr)
import numpy as np

import pandas as pd

import datetime

from catboost import CatBoostClassifier

from time import time

from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix

# this function is the quadratic weighted kappa (the metric used for the competition submission)

def qwk(act,pred,n=4,hist_range=(0,3)):

    

    # Calculate the percent each class was tagged each label

    O = confusion_matrix(act,pred)

    # normalize to sum 1

    O = np.divide(O,np.sum(O))

    

    # create a new matrix of zeroes that match the size of the confusion matrix

    # this matriz looks as a weight matrix that give more weight to the corrects

    W = np.zeros((n,n))

    for i in range(n):

        for j in range(n):

            # makes a weird matrix that is bigger in the corners top-right and botton-left (= 1)

            W[i][j] = ((i-j)**2)/((n-1)**2)

            

    # make two histograms of the categories real X prediction

    act_hist = np.histogram(act,bins=n,range=hist_range)[0]

    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]

    

    # multiply the two histograms using outer product

    E = np.outer(act_hist,prd_hist)

    E = np.divide(E,np.sum(E)) # normalize to sum 1

    

    # apply the weights to the confusion matrix

    num = np.sum(np.multiply(W,O))

    # apply the weights to the histograms

    den = np.sum(np.multiply(W,E))

    

    return 1-np.divide(num,den)

    
all_features = [x for x in get_trainD.columns if x not in ['accuracy_group']]

# this list comprehension create the list of features that will be used on the input dataset X

# all but accuracy_group, that is the label y

# this cat_feature must be declared to pass later as parameter to fit the model

cat_features = ['title']

# here the dataset select the features and split the input ant the labels

Xc, yc = get_trainD[all_features], get_trainD['accuracy_group']
# this function makes the model and sets the parameters

# for configure others parameter consult the documentation below:

# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html

def make_classifier():

    clf = CatBoostClassifier(

                               loss_function='MultiClass',

                                # eval_metric="AUC",

                               task_type="CPU",

                               learning_rate=0.01,

                               iterations=2000,

                               od_type="Iter",

                                # depth=8,

                               early_stopping_rounds=500,

                                #l2_leaf_reg=1,

                                #border_count=96,

                               random_seed=42

                              )

        

    return clf
# oof is an zeroed array of the same size of the input dataset

#oof = np.zeros(len(Xc))

#oof[test_idx] = clf.predict(Xc.loc[test_idx, all_features]).reshape(len(test_idx))

#print('OOF QWK:', qwk(yc, oof))
%%time

# CV

from sklearn.model_selection import KFold

# oof is an zeroed array of the same size of the input dataset

oof = np.zeros(len(Xc))

NFOLDS = 5

# here the KFold class is used to split the dataset in 5 diferents training and validation sets

# this technique is used to assure that the model isn't overfitting and can performs aswell in 

# unseen data. More the number of splits/folds, less the test will be impacted by randomness

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

training_start_time = time()



for fold, (trn_idx, test_idx) in enumerate(folds.split(Xc, yc)):

    # each iteration of folds.split returns an array of indexes of the new training data and validation data

    start_time = time()

    print(f'Training on fold {fold+1}')

    # creates the model

    clf = make_classifier()

    # fits the model using .loc at the full dataset to select the splits indexes and features used

    clf.fit(Xc.loc[trn_idx, all_features],yc.loc[trn_idx], eval_set=(Xc.loc[test_idx, all_features], yc.loc[test_idx]),

                          use_best_model=True, verbose=500, cat_features=cat_features)

    

    # then, the predictions of each split is inserted into the oof array

    oof[test_idx] = clf.predict(Xc.loc[test_idx, all_features]).reshape(len(test_idx))

    

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

    

print('-' * 30)

# and here, the complete oof is tested against the real data using que metric (quadratic weighted kappa)

print('OOF QWK:', qwk(yc, oof))

print('-' * 30)
%%time

# train model on all data once

clf1 = make_classifier()

clf1.fit(Xc, yc, verbose=500, cat_features=cat_features)

predicted3 = clf1.predict(X_test1)

from sklearn.metrics import classification_report

reportcat = classification_report(Y_test1, predicted3)

print(reportcat)
from sklearn.metrics import mean_squared_error

from math import sqrt



rms3 = sqrt(mean_squared_error(Y_test1, predicted3))

print(rms3)

print(str(rms3**2))
Y_test1.index
# oof is an zeroed array of the same size of the input dataset

oof3 = np.zeros(len(X_test1))

oof3 = clf.predict(X_test1)

print('OOF QWK:', qwk(Y_test1, oof3))
# oof is an zeroed array of the same size of the input dataset

oof3 = np.zeros(len(X_test1))

oof3 = clf1.predict(X_test1)

print('OOF QWK:', qwk(Y_test1, oof3))
rms2 = round(rms2,4)

ms2 = rms2**2

print("Logistic regression : "+str(rms2)+" , "+str(ms2))

rms1 = round(rms1,4)

ms1 = rms1**1

print("CART : "+str(rms1)+" , "+str(ms1))

rms3 = round(rms3,4)

ms3 = rms3**2

print("Cat booster : "+str(rms3)+" , "+str(ms3)) 
get_trainD
get_trainD0 = get_trainD[get_trainD.accuracy_group == 0]

get_trainD0
get_trainD1 = get_trainD[get_trainD.accuracy_group == 1]

get_trainD1
get_trainD2 = get_trainD[get_trainD.accuracy_group == 2]

get_trainD2
get_trainD3 = get_trainD[get_trainD.accuracy_group == 3]

get_trainD3
get_trainD0_s = get_trainD0.sample(n=2205,random_state = 1).reset_index(drop = True)

get_trainD1_s = get_trainD1.sample(n=2205,random_state = 1).reset_index(drop = True)

get_trainD2_s = get_trainD2.sample(frac = 1, random_state = 1).reset_index()

get_trainD3_s = get_trainD3.sample(n=2205,random_state = 1).reset_index(drop = True)

train_equal_data = pd.concat([get_trainD0_s,get_trainD1_s,get_trainD2_s,get_trainD3_s], ignore_index=True, sort =False).reset_index(drop=True)
train_equal_data1 = train_equal_data.drop(columns = ["index"]).sample(frac = 1, random_state = 1)
train_equal_data1
train_equal_data1.groupby(["accuracy_group"]).size()
Xa = train_equal_data1.drop(columns = ["accuracy_group"])

Xa
Xa1 = Xa[["title","average_score","acc_r","inacc_r"]]

Xa1
ya = train_equal_data1["accuracy_group"]

ya
ya1 = ya
# this function makes the model and sets the parameters

# for configure others parameter consult the documentation below:

# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html

def make_classifier2():

    clf1 = CatBoostClassifier(

                               loss_function='MultiClass',

                                # eval_metric="AUC",

                               task_type="CPU",

                               learning_rate=0.01,

                               iterations=500,

                               od_type="Iter",

                                # depth=8,

                               early_stopping_rounds=100,

                                #l2_leaf_reg=1,

                                #border_count=96,

                               random_seed=42

                              )

        

    return clf1
# this function makes the model and sets the parameters

# for configure others parameter consult the documentation below:

# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html

def make_classifier2aa():

    clf1 = CatBoostClassifier(

                               loss_function='MultiClass',

                                # eval_metric="AUC",

                               task_type="CPU",

                               learning_rate=0.01,

                               iterations=2000,

                               od_type="Iter",

                                # depth=8,

                               early_stopping_rounds=100,

                                #l2_leaf_reg=1,

                                #border_count=96,

                               random_seed=42

                              )

        

    return clf1
%%time

# CV

from sklearn.model_selection import KFold

# oof is an zeroed array of the same size of the input dataset

oof = np.zeros(len(Xa))

NFOLDS = 5

# here the KFold class is used to split the dataset in 5 diferents training and validation sets

# this technique is used to assure that the model isn't overfitting and can performs aswell in 

# unseen data. More the number of splits/folds, less the test will be impacted by randomness

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

training_start_time = time()



for fold, (trn_idx, test_idx) in enumerate(folds.split(Xa, ya)):

    # each iteration of folds.split returns an array of indexes of the new training data and validation data

    start_time = time()

    print(f'Training on fold {fold+1}')

    # creates the model

    clf2 = make_classifier2()

    # fits the model using .loc at the full dataset to select the splits indexes and features used

    clf2.fit(Xa.loc[trn_idx, all_features],ya.loc[trn_idx], eval_set=(Xa.loc[test_idx, all_features], ya.loc[test_idx]),

                          use_best_model=True, verbose=500, cat_features=cat_features)

    

    # then, the predictions of each split is inserted into the oof array

    oof[test_idx] = clf2.predict(Xa.loc[test_idx, all_features]).reshape(len(test_idx))

    print(clf2.get_feature_importance())

    print('OOF QWK:', qwk(ya.loc[test_idx], oof[test_idx]))

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

    

print('-' * 30)

# and here, the complete oof is tested against the real data using que metric (quadratic weighted kappa)

print('OOF QWK:', qwk(ya, oof))

print('-' * 30)
X_test1 = get_testD[["title","average_score","acc_r","inacc_r"]]

X_test1
Y_test1 = get_testD["accuracy_group"]

Y_test1
%%time

# CV

from sklearn.model_selection import KFold

# oof is an zeroed array of the same size of the input dataset

oof = np.zeros(len(Xa))

NFOLDS = 5

# here the KFold class is used to split the dataset in 5 diferents training and validation sets

# this technique is used to assure that the model isn't overfitting and can performs aswell in 

# unseen data. More the number of splits/folds, less the test will be impacted by randomness

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

training_start_time = time()



for fold, (trn_idx, test_idx) in enumerate(folds.split(Xa1, ya1)):

    # each iteration of folds.split returns an array of indexes of the new training data and validation data

    start_time = time()

    print(f'Training on fold {fold+1}')

    # creates the model

    clf21 = make_classifier2()

    # fits the model using .loc at the full dataset to select the splits indexes and features used

    clf21.fit(Xa1.loc[trn_idx, all_features],ya1.loc[trn_idx], eval_set=(Xa1.loc[test_idx, all_features], ya1.loc[test_idx]),

                          use_best_model=True, verbose=500, cat_features=cat_features)

    

    # then, the predictions of each split is inserted into the oof array

    oof[test_idx] = clf21.predict(Xa1.loc[test_idx, all_features]).reshape(len(test_idx))

    print(clf21.get_feature_importance())

    print('OOF QWK:', qwk(ya1.loc[test_idx], oof[test_idx]))

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

    

print('-' * 30)

# and here, the complete oof is tested against the real data using que metric (quadratic weighted kappa)

print('OOF QWK:', qwk(ya1, oof))

print('-' * 30)
%%time

# train model on all data once

clf21 = make_classifier2()

clf21.fit(Xa1, ya1, verbose=300, cat_features=cat_features)

predicteda = clf21.predict(X_test1)

from sklearn.metrics import classification_report

reportcata = classification_report(Y_test1, predicteda)

print(reportcat)
%%time

# train model on all data once

clf2 = make_classifier2()

clf2.fit(Xa, ya, verbose=500, cat_features=cat_features)

%%time

# train model on all data once

clf2 = make_classifier()

clf2.fit(Xa, ya, verbose=500, cat_features=cat_features)

%%time

# train model on all data once

clf2a = make_classifier2()

clf2a.fit(Xa, ya, verbose=100, cat_features=cat_features)

%%time

# train model on all data once

clf2aa = make_classifier2aa()

clf2aa.fit(Xa, ya, verbose=100, cat_features=cat_features)

clf2.get_feature_importance()
clf2a.get_feature_importance()
clf2aa.get_feature_importance()
Xa.columns
predicteda1 = clf21.predict(X_test1[["title","average_score","acc_r","inacc_r"]])

from sklearn.metrics import classification_report

reportcata1 = classification_report(Y_test1, predicteda1)

print(reportcata1)
# oof is an zeroed array of the same size of the input dataset

oofa21 = np.zeros(len(X_test1[["title","average_score","acc_r","inacc_r"]]))

oofa21 = clf21.predict(X_test1[["title","average_score","acc_r","inacc_r"]])

print('OOF QWK:', qwk(Y_test1, oofa21))
get_testD55 = get_testD

get_testD55["wrongly_labeled"] = 0

for i in range(0,len(get_testD)):

    if (get_testD["accuracy_group"][i] != int(oofa21[i])):

        get_testD55["wrongly_labeled"] = 1

incorr55 = get_testD55[get_testD55.wrongly_labeled == 1]
len(incorr55)
from pandas import set_option

pd.set_option('display.max_rows', 500)

incorr55
predicteda2 = clf2a.predict(X_test1)

from sklearn.metrics import classification_report

reportcata2 = classification_report(Y_test1, predicteda2)

print(reportcata2)
predicteda2a = clf2aa.predict(X_test1)

from sklearn.metrics import classification_report

reportcata2a = classification_report(Y_test1, predicteda2a)

print(reportcata2a)
from sklearn.metrics import mean_squared_error

from math import sqrt



rmsa = sqrt(mean_squared_error(Y_test1, predicteda))

print(rmsa)

print(str(rmsa**2))
from sklearn.metrics import mean_squared_error

from math import sqrt



rmsa2 = sqrt(mean_squared_error(Y_test1, predicteda2))

print(rmsa2)

print(str(rmsa2**2))
from sklearn.metrics import mean_squared_error

from math import sqrt



rmsa2a = sqrt(mean_squared_error(Y_test1, predicteda2a))

print(rmsa2a)

print(str(rmsa2a**2))
# oof is an zeroed array of the same size of the input dataset

oofa = np.zeros(len(X_test1))

oofa = clf2.predict(X_test1)

print('OOF QWK:', qwk(Y_test1, oofa))
# oof is an zeroed array of the same size of the input dataset

oofa = np.zeros(len(X_test1))

oofa = clf2a.predict(X_test1)

print('OOF QWK:', qwk(Y_test1, oofa))
# oof is an zeroed array of the same size of the input dataset

oofa = np.zeros(len(X_test1))

oofa = clf2aa.predict(X_test1)

print('OOF QWK:', qwk(Y_test1, oofa))
# this function makes the model and sets the parameters

# for configure others parameter consult the documentation below:

# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html

def make_classifier3():

    clf1 = CatBoostClassifier(

                               loss_function='MultiClass',

                                # eval_metric="AUC",

                               task_type="CPU",

                               learning_rate=0.01,

                               iterations=2000,

                               od_type="Iter",

                                # depth=8,

                               early_stopping_rounds=500,

                                #l2_leaf_reg=1,

                                #border_count=96,

                               random_seed=42

                              )

        

    return clf1
%%time

# CV

from sklearn.model_selection import KFold

# oof is an zeroed array of the same size of the input dataset

oof = np.zeros(len(Xa))

NFOLDS = 5

# here the KFold class is used to split the dataset in 5 diferents training and validation sets

# this technique is used to assure that the model isn't overfitting and can performs aswell in 

# unseen data. More the number of splits/folds, less the test will be impacted by randomness

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

training_start_time = time()



for fold, (trn_idx, test_idx) in enumerate(folds.split(Xa, ya)):

    # each iteration of folds.split returns an array of indexes of the new training data and validation data

    start_time = time()

    print(f'Training on fold {fold+1}')

    # creates the model

    clf3 = make_classifier3()

    # fits the model using .loc at the full dataset to select the splits indexes and features used

    clf3.fit(Xa.loc[trn_idx, all_features],ya.loc[trn_idx], eval_set=(Xa.loc[test_idx, all_features], ya.loc[test_idx]),

                          use_best_model=True, verbose=500, cat_features=cat_features)

    

    # then, the predictions of each split is inserted into the oof array

    oof[test_idx] = clf3.predict(Xa.loc[test_idx, all_features]).reshape(len(test_idx))

    print(clf3.get_feature_importance())

    print('OOF QWK:', qwk(ya.loc[test_idx], oof[test_idx]))

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

    

print('-' * 30)

# and here, the complete oof is tested against the real data using que metric (quadratic weighted kappa)

print('OOF QWK:', qwk(ya, oof))

print('-' * 30)
%%time

# train model on all data once

clf3 = make_classifier3()

clf3.fit(Xa, ya, verbose=500, cat_features=cat_features)

train_equal_data1
clf2.get_feature_importance()
Xa.columns
%%time

import math

train_equal_data2 = train_equal_data1

train_equal_data2["acc_r_log"] = 0.001

train_equal_data2["inacc_r_log"] = 0.001

for i in range(0,len(train_equal_data2)):

    acc = train_equal_data2["acc_r"][i]

    inacc = train_equal_data2["inacc_r"][i]

    if (acc!= 0):

        train_equal_data2["acc_r_log"][i] = round(math.log(acc),3)

    if (inacc!=0):

        train_equal_data2["inacc_r_log"][i] = round(math.log(inacc),3)

train_equal_data2
get_testD
import math

test_l2 = get_testD

test_l2["acc_r_log"] = 0.001

test_l2["inacc_r_log"] = 0.001

for i in range(0,len(test_l2)):

    acc = test_l2["acc_r"][i]

    inacc = test_l2["inacc_r"][i]

    if (acc!= 0):

        test_l2["acc_r_log"][i] = round(math.log(acc),3)

    if (inacc!=0):

        test_l2["inacc_r_log"][i] = round(math.log(inacc),3)

test_l2
Xl1 = train_equal_data2.drop(columns = ["accuracy_group"])

Yl1 = train_equal_data2["accuracy_group"]

X_testl1 = test_l2.drop(columns = ["accuracy_group"])

Y_testl1 = test_l2["accuracy_group"]
# this function makes the model and sets the parameters

# for configure others parameter consult the documentation below:

# https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html

def make_classifier4():

    clf1 = CatBoostClassifier(

                               loss_function='MultiClass',

                                # eval_metric="AUC",

                               task_type="CPU",

                               learning_rate=0.01,

                               iterations=2000,

                               od_type="Iter",

                                # depth=8,

                               early_stopping_rounds=100,

                                #l2_leaf_reg=1,

                                #border_count=96,

                               random_seed=42

                              )

        

    return clf1
%%time

# CV

from sklearn.model_selection import KFold

# oof is an zeroed array of the same size of the input dataset

oof = np.zeros(len(Xl1))

NFOLDS = 5

# here the KFold class is used to split the dataset in 5 diferents training and validation sets

# this technique is used to assure that the model isn't overfitting and can performs aswell in 

# unseen data. More the number of splits/folds, less the test will be impacted by randomness

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=2019)

training_start_time = time()



for fold, (trn_idx, test_idx) in enumerate(folds.split(Xl1, Yl1)):

    # each iteration of folds.split returns an array of indexes of the new training data and validation data

    start_time = time()

    print(f'Training on fold {fold+1}')

    # creates the model

    clf4 = make_classifier4()

    # fits the model using .loc at the full dataset to select the splits indexes and features used

    clf4.fit(Xl1.loc[trn_idx, all_features],Yl1.loc[trn_idx], eval_set=(Xl1.loc[test_idx, all_features], Yl1.loc[test_idx]),

                          use_best_model=True, verbose=500, cat_features=cat_features)

    

    # then, the predictions of each split is inserted into the oof array

    oof[test_idx] = clf4.predict(Xl1.loc[test_idx, all_features]).reshape(len(test_idx))

    print(clf4.get_feature_importance())

    print('OOF QWK:', qwk(ya.loc[test_idx], oof[test_idx]))

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))

    

print('-' * 30)

# and here, the complete oof is tested against the real data using que metric (quadratic weighted kappa)

print('OOF QWK:', qwk(Yl1, oof))

print('-' * 30)
%%time

# train model on all data once

clf4 = make_classifier3()

clf4.fit(Xl1, Yl1, verbose=300, cat_features=cat_features)

clf4.get_feature_importance()
predicteda4 = clf4.predict(X_testl1)

from sklearn.metrics import classification_report

reportcata4 = classification_report(Y_testl1, predicteda4)

print(reportcata4)
from sklearn.metrics import mean_squared_error

from math import sqrt



rmsa2a4 = sqrt(mean_squared_error(Y_testl1, predicteda4))

print(rmsa2a4)

print(str(rmsa2a4**2))
# oof is an zeroed array of the same size of the input dataset

oofa4 = np.zeros(len(X_testl1))

oofa4 = clf4.predict(X_testl1)

print('OOF QWK:', qwk(Y_testl1, oofa4))
oofa4
# oof is an zeroed array of the same size of the input dataset

oofa4l = np.zeros(len(Xl1))

oofa4l = clf4.predict(Xl1)

print('OOF QWK:', qwk(Yl1, oofa4l))
get_subA