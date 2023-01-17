!wget -O /usr/share/fonts/truetype/liberation/simhei.ttf https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf
import matplotlib

import matplotlib.pyplot as plt



zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/liberation/simhei.ttf')

#plt.rcParams['axes.unicode_minus'] = False 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import xgboost as xgb

from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.model_selection import KFold, cross_val_score , train_test_split

from sklearn.metrics import confusion_matrix , recall_score , roc_auc_score 

from sklearn.metrics import mean_squared_error as MSE



from time import time

import datetime

train_feature_csv = '/kaggle/input/quoraprogram2/content/drive/Shared drives/zuyukun576/GraduationProject/program2/data/train_feature2.csv'

test_feature_csv = '/kaggle/input/quoraprogram2/content/drive/Shared drives/zuyukun576/GraduationProject/program2/data/test_feature2.csv'

train_label_csv = '/kaggle/input/quoraprogram2/content/drive/Shared drives/zuyukun576/GraduationProject/program2/data/trainlabel.csv'



train = pd.read_csv(train_feature_csv)

test = pd.read_csv(test_feature_csv)

trainlabel = pd.read_csv(train_label_csv)
display(train.head())

display(test.head())

display(trainlabel.head())
dtrain = xgb.DMatrix(train, label = trainlabel)
p = 0.369197853026293

pos_public = (0.55410 + np.log(1 - p)) / np.log((1 - p) / p)

pos_private = (0.55525 + np.log(1 - p)) / np.log((1 - p) / p)

average = (pos_public + pos_private) / 2

print (pos_public, pos_private, average)
w0 = average * (1 - p) / ((1 - average) * p)

print(w0)
w1 = average / p

w2 = (1 - average) / (1 - p)

print(w1, w2)
def weighted_log_loss(preds, dtrain):

    label = dtrain.get_label()

    return "weighted_logloss", -np.mean(w1 * label * np.log(preds) + w2 * (1 - label) * np.log(1 - preds))
params = {}

params["silent"] = True

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.1

params["max_depth"] = 6

params["min_child_weight"] = 1

params["gamma"] = 0

params["subsample"] = 0.8

params["colsample_bytree"] = 0.9

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"  # 使用GPU加速的直方图算法

params['max_bin'] = 256





time0 = time()

model1 = xgb.cv(params, dtrain, 

                num_boost_round = 5000, 

                nfold = 10, 

                feval = weighted_log_loss, 

                early_stopping_rounds = 200, 

                verbose_eval = 50)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
print('Best number of trees = {}'.format(model1.shape[0]))
print("min train-weighted_logloss-mean", min(model1.iloc[:,2]), (model1.iloc[:,2].tolist().index(min(model1.iloc[:,2]))+1))

print("min test-weighted_logloss-mean", min(model1.iloc[:,6]), (model1.iloc[:,6].tolist().index(min(model1.iloc[:,6]))+1))
fig,ax = plt.subplots(1,figsize=(12,9))

#ax.set_ylim(top=5)

ax.grid()

ax.plot(range(1,model1.shape[0]+1),model1.iloc[:,2],c="red",label="train,original")

ax.plot(range(1,model1.shape[0]+1),model1.iloc[:,6],c="orange",label="test,original")

ax.legend(fontsize="xx-large")



#标注train 最低点

min_train, min_train_indx = round(min(model1.iloc[:,2]), 6), (model1.iloc[:,2].tolist().index(min(model1.iloc[:,2]))+1)

min_train_show_x = min_train_indx

show_min_train='min:['+str(min_train_indx)+' '+str(min_train)+']'

ax.annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

ax.plot(min_train_indx, min_train, 'gs')



#标注test 最低点

min_test, min_test_indx = round(min(model1.iloc[:,6]), 6), (model1.iloc[:,6].tolist().index(min(model1.iloc[:,6]))+1)

min_test_show_x = min_train_indx

show_min_test='min:['+str(min_test_indx)+' '+str(min_test)+']'

ax.annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

ax.plot(min_test_indx, min_test, 'gs')



ax.set_title("交叉验证曲线-num_boost_round", fontproperties=zhfont, fontsize=20)

ax.set_xlabel('num_boost_round' , fontproperties=zhfont, fontsize=20)

ax.set_ylabel('weighted_logloss' , fontproperties=zhfont, fontsize=20)



plt.show()
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.1

fix_params["gamma"] = 0

fix_params["subsample"] = 0.8

fix_params["colsample_bytree"] = 0.9

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_bin"] = 256



time0 = time()

evaluation_list = []

for depth in [5, 6]:

    for child_weight in [1, 2.5, 4]:

        params = {**fix_params, **{"max_depth": depth, "min_child_weight": child_weight}}

        evaluation = xgb.cv(params, dtrain, num_boost_round = 847, nfold = 10, 

                            feval = weighted_log_loss, early_stopping_rounds = 100)

        # evaluation记录了每一轮迭代的交叉验证结果

        evaluation_list.append(evaluation)

        

for depth in [7, 8]:

    for child_weight in [4, 5, 6]:

        params = {**fix_params, **{"max_depth": depth, "min_child_weight": child_weight}}

        evaluation = xgb.cv(params, dtrain, num_boost_round = 847, nfold = 10, 

                            feval = weighted_log_loss, early_stopping_rounds = 100)

        # evaluation记录了每一轮迭代的交叉验证结果

        evaluation_list.append(evaluation)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

        

evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    # evaluation的最后一行即相应参数组合的结果

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
title_depth = [5,5,5,6,6,6,7,7,7,8,8,8]

title_child_weight = [1, 2.5, 4, 1, 2.5, 4, 4, 5, 6, 4, 5, 6]

def show_cv(title_depth, title_child_weight, evaluation_list):

    fig,ax = plt.subplots(4,3,figsize=(30,20))

    for i in range(len(evaluation_list)):

        

        r = i//3

        c = i%3

        

        ax[r][c].set_ylim(0.15,0.3)

        ax[r][c].grid()

        

        #未调整前

        ax[r][c].plot(range(1,model1.shape[0]+1),model1.iloc[:,2],c="red",label="train,original")

        ax[r][c].plot(range(1,model1.shape[0]+1),model1.iloc[:,6],c="orange",label="test,original")

        #调整后

        ax[r][c].plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,2],c="green",label="train,cv{}")

        ax[r][c].plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,6],c="blue",label="test,cv{}")

        

        ax[r][c].legend(fontsize="10")

        

        #未调整前-标注train 最低点

        min_train, min_train_indx = round(min(model1.iloc[:,2]), 6), (model1.iloc[:,2].tolist().index(min(model1.iloc[:,2]))+1)

        min_train_show_x = min_train_indx

        show_min_train_original ='min:['+str(min_train_indx)+' '+str(min_train)+']'

        #ax[r][c].annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

        ax[r][c].plot(min_train_indx, min_train, 'gs')



        #未调整前-标注test 最低点

        min_test, min_test_indx = round(min(model1.iloc[:,6]), 6), (model1.iloc[:,6].tolist().index(min(model1.iloc[:,6]))+1)

        min_test_show_x = min_train_indx

        show_min_test_original ='min:['+str(min_test_indx)+' '+str(min_test)+']'

        #ax[r][c].annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

        ax[r][c].plot(min_test_indx, min_test, 'gs')

        

        

        #调整后-标注train 最低点

        min_train, min_train_indx = round(min(evaluation_list[i].iloc[:,2]), 6), (evaluation_list[i].iloc[:,2].tolist().index(min(evaluation_list[i].iloc[:,2]))+1)

        min_train_show_x = min_train_indx

        show_min_train_cv='min:['+str(min_train_indx)+' '+str(min_train)+']'

        #ax[r][c].annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

        ax[r][c].plot(min_train_indx, min_train, 'gs')



        #调整后-标注test 最低点

        min_test, min_test_indx = round(min(evaluation_list[i].iloc[:,6]), 6), (evaluation_list[i].iloc[:,6].tolist().index(min(evaluation_list[i].iloc[:,6]))+1)

        min_test_show_x = min_train_indx

        show_min_test_cv='min:['+str(min_test_indx)+' '+str(min_test)+']'

        #ax[r][c].annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

        ax[r][c].plot(min_test_indx, min_test, 'gs')

        

        ax[r][c].set_title("交叉验证曲线{} max_depth:{} min_child_weight:{}".format(i,title_depth[i],title_child_weight[i]), fontproperties=zhfont, fontsize=20)



    plt.show()



show_cv(title_depth, title_child_weight, evaluation_list)
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.1

fix_params["gamma"] = 0

fix_params["subsample"] = 0.8

fix_params["colsample_bytree"] = 0.9

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_bin"] = 256



time0 = time()

evaluation_list = []

for depth in [5, 6]:

    for child_weight in [4, 4.5, 5, 5.5]:

        params = {**fix_params, **{"max_depth": depth, "min_child_weight": child_weight}}

        evaluation = xgb.cv(params, dtrain, num_boost_round = 5000, nfold = 10, 

                            feval = weighted_log_loss, early_stopping_rounds = 100)

        evaluation_list.append(evaluation)

print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))



evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel





title_depth = []

title_child_weight = []



title_list = []

for depth in [5, 6]:

    for child_weight in [4, 4.5, 5, 5.5]:

        title_depth.append(depth)

        title_child_weight.append(child_weight)



def show_cv(title_depth, title_child_weight, evaluation_list):

    fig,ax = plt.subplots(3,3,figsize=(30,15))

    prop_cycle = plt.rcParams['axes.prop_cycle']

    colors = prop_cycle.by_key()['color']

    for i in range(len(evaluation_list)):

        

        r = i//3

        c = i%3

        

        ax[r][c].set_ylim(0.15,0.3)

        ax[r][c].grid()

        #未调整前

        ax[r][c].plot(range(1,model1.shape[0]+1),model1.iloc[:,2],c="red",label="train,original")

        ax[r][c].plot(range(1,model1.shape[0]+1),model1.iloc[:,6],c="orange",label="test,original")

        #调整后

        ax[r][c].plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,2],c="green",label="train,cv{}".format(i))

        ax[r][c].plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,6],c="blue",label="test,cv{}".format(i))

        

        ax[r][c].legend(fontsize="10")

        

        #未调整前-标注train 最低点

        min_train, min_train_indx = round(min(model1.iloc[:,2]), 6), (model1.iloc[:,2].tolist().index(min(model1.iloc[:,2]))+1)

        min_train_show_x = min_train_indx

        show_min_train_original ='min:['+str(min_train_indx)+' '+str(min_train)+']'

        #ax[r][c].annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

        ax[r][c].plot(min_train_indx, min_train, 'gs')



        #未调整前-标注test 最低点

        min_test, min_test_indx = round(min(model1.iloc[:,6]), 6), (model1.iloc[:,6].tolist().index(min(model1.iloc[:,6]))+1)

        min_test_show_x = min_train_indx

        show_min_test_original ='min:['+str(min_test_indx)+' '+str(min_test)+']'

        #ax[r][c].annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

        ax[r][c].plot(min_test_indx, min_test, 'gs')

        

        

        #调整后-标注train 最低点

        min_train, min_train_indx = round(min(evaluation_list[i].iloc[:,2]), 6), (evaluation_list[i].iloc[:,2].tolist().index(min(evaluation_list[i].iloc[:,2]))+1)

        min_train_show_x = min_train_indx

        show_min_train_cv='min:['+str(min_train_indx)+' '+str(min_train)+']'

        #ax[r][c].annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

        ax[r][c].plot(min_train_indx, min_train, 'gs')



        #调整后-标注test 最低点

        min_test, min_test_indx = round(min(evaluation_list[i].iloc[:,6]), 6), (evaluation_list[i].iloc[:,6].tolist().index(min(evaluation_list[i].iloc[:,6]))+1)

        min_test_show_x = min_train_indx

        show_min_test_cv='min:['+str(min_test_indx)+' '+str(min_test)+']'

        #ax[r][c].annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

        ax[r][c].plot(min_test_indx, min_test, 'gs')



        ax[r][c].set_title("交叉验证曲线{} max_depth:{} min_child_weight:{}".format(i,title_depth[i],title_child_weight[i]), fontproperties=zhfont, fontsize=15)



    #fig.suptitle('depth in [5, 6, 7], child_weight in [3, 3.5, 4, 4.5]')

    #plt.subplots_adjust(hspace = 0.25)

    plt.show()



show_cv(title_depth, title_child_weight, evaluation_list)
model2 = evaluation_list[1]



print('Best number of trees = {}'.format(model2.shape[0]))

print("min train-weighted_logloss-mean", min(model2.iloc[:,2]), (model2.iloc[:,2].tolist().index(min(model2.iloc[:,2]))+1))

print("min test-weighted_logloss-mean", min(model2.iloc[:,6]), (model2.iloc[:,6].tolist().index(min(model2.iloc[:,6]))+1))





fig,ax = plt.subplots(1,figsize=(12,9))

#ax.set_ylim(top=5)

ax.grid()

ax.plot(range(1,model2.shape[0]+1),model2.iloc[:,2],c="red",label="train,original")

ax.plot(range(1,model2.shape[0]+1),model2.iloc[:,6],c="orange",label="test,original")

ax.legend(fontsize="xx-large")



#标注train 最低点

min_train, min_train_indx = round(min(model2.iloc[:,2]), 6), (model2.iloc[:,2].tolist().index(min(model2.iloc[:,2]))+1)

min_train_show_x = min_train_indx

show_min_train='min:['+str(min_train_indx)+' '+str(min_train)+']'

ax.annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

ax.plot(min_train_indx, min_train, 'gs')



#标注test 最低点

min_test, min_test_indx = round(min(model2.iloc[:,6]), 6), (model2.iloc[:,6].tolist().index(min(model2.iloc[:,6]))+1)

min_test_show_x = min_train_indx

show_min_test='min:['+str(min_test_indx)+' '+str(min_test)+']'

ax.annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

ax.plot(min_test_indx, min_test, 'gs')



ax.set_title("交叉验证曲线-num_boost_round:1280,max_depth:5,min_child_weight:4.5", fontproperties=zhfont, fontsize=20)

ax.set_xlabel('num_boost_round' , fontproperties=zhfont, fontsize=20)

ax.set_ylabel('weighted_logloss' , fontproperties=zhfont, fontsize=20)



plt.show()
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.1

fix_params["gamma"] = 0

fix_params["subsample"] = 0.8

fix_params["colsample_bytree"] = 0.9

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_depth"] = 5

fix_params["min_child_weight"] = 4.5





time0 = time()



evaluation_list = []

#for bin in [216, 226, 236, 246, 256, 266, 276, 286]:

for max_bin in [150, 180, 200, 256, 286, 300, 350, 400]:

    params = {**fix_params, **{"max_bin": max_bin}}

    evaluation = xgb.cv(params, dtrain, num_boost_round = 1280, nfold = 10, 

                        feval = weighted_log_loss, early_stopping_rounds = 100)

    evaluation_list.append(evaluation)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

    

evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel

title_max_bin = [150, 180, 200, 256, 286, 300, 350, 400]



def show_cv(title_bin, evaluation_list):

    fig,ax = plt.subplots(3,3,figsize=(30,15))

    prop_cycle = plt.rcParams['axes.prop_cycle']

    colors = prop_cycle.by_key()['color']

    for i in range(len(evaluation_list)):

        

        r = i//3

        c = i%3

        

        ax[r][c].set_ylim(0.15,0.3)

        

        ax[r][c].grid()

        #未调整前

        ax[r][c].plot(range(1,model2.shape[0]+1),model2.iloc[:,2],c="red",label="train,original")

        ax[r][c].plot(range(1,model2.shape[0]+1),model2.iloc[:,6],c="orange",label="test,original")

        #调整后

        ax[r][c].plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,2],c="green",label="train,cv{}".format(i))

        ax[r][c].plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,6],c="blue",label="test,cv{}".format(i))

        ax[r][c].legend(fontsize="10")

        

        #未调整前-标注train 最低点

        min_train, min_train_indx = round(min(model2.iloc[:,2]), 6), (model2.iloc[:,2].tolist().index(min(model2.iloc[:,2]))+1)

        min_train_show_x = min_train_indx

        show_min_train_original ='min:['+str(min_train_indx)+' '+str(min_train)+']'

        #ax[r][c].annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

        ax[r][c].plot(min_train_indx, min_train, 'gs')



        #未调整前-标注test 最低点

        min_test, min_test_indx = round(min(model2.iloc[:,6]), 6), (model2.iloc[:,6].tolist().index(min(model2.iloc[:,6]))+1)

        min_test_show_x = min_train_indx

        show_min_test_original ='min:['+str(min_test_indx)+' '+str(min_test)+']'

        #ax[r][c].annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

        ax[r][c].plot(min_test_indx, min_test, 'gs')

        

        

        #调整后-标注train 最低点

        min_train, min_train_indx = round(min(evaluation_list[i].iloc[:,2]), 6), (evaluation_list[i].iloc[:,2].tolist().index(min(evaluation_list[i].iloc[:,2]))+1)

        min_train_show_x = min_train_indx

        show_min_train_cv='min:['+str(min_train_indx)+' '+str(min_train)+']'

        #ax[r][c].annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

        ax[r][c].plot(min_train_indx, min_train, 'gs')



        #调整后-标注test 最低点

        min_test, min_test_indx = round(min(evaluation_list[i].iloc[:,6]), 6), (evaluation_list[i].iloc[:,6].tolist().index(min(evaluation_list[i].iloc[:,6]))+1)

        min_test_show_x = min_train_indx

        show_min_test_cv='min:['+str(min_test_indx)+' '+str(min_test)+']'

        #ax[r][c].annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

        ax[r][c].plot(min_test_indx, min_test, 'gs')



        ax[r][c].set_title("交叉验证曲线{} max_bin:{}".format(i,title_max_bin[i]), fontproperties=zhfont, fontsize=15)



    plt.subplots_adjust(hspace = 0.25)

    plt.show()



show_cv(title_max_bin, evaluation_list)
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.1

fix_params["gamma"] = 0

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_depth"] = 5

fix_params["min_child_weight"] = 4.5

fix_params["max_bin"] = 200





time0 = time()

evaluation_list = []

for row in [0.7, 0.8, 0.9]:

    for col in [0.7, 0.8, 0.9]:

        params = {**fix_params, **{"subsample": row, "colsample_bytree": col}}

        evaluation = xgb.cv(params, dtrain, num_boost_round = 1280, nfold = 10, 

                            feval = weighted_log_loss, early_stopping_rounds = 100)

        evaluation_list.append(evaluation)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

        

evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
title_sub = [0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9]

title_col = [0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9]



def show_cv(title_sub, title_col, evaluation_list):

    fig,ax = plt.subplots(3,3,figsize=(30,15))

    prop_cycle = plt.rcParams['axes.prop_cycle']

    colors = prop_cycle.by_key()['color']

    for i in range(len(evaluation_list)):

        

        r = i//3

        c = i%3

        

        ax[r][c].set_ylim(0.15,0.3)

        

        ax[r][c].grid()

        #未调整前

        ax[r][c].plot(range(1,model2.shape[0]+1),model2.iloc[:,2],c="red",label="train,original")

        ax[r][c].plot(range(1,model2.shape[0]+1),model2.iloc[:,6],c="orange",label="test,original")

        #调整后

        ax[r][c].plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,2],c="green",label="train,cv{}".format(i))

        ax[r][c].plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,6],c="blue",label="test,cv{}".format(i))

        ax[r][c].legend(fontsize="10")

        

        #未调整前-标注train 最低点

        min_train, min_train_indx = round(min(model2.iloc[:,2]), 6), (model2.iloc[:,2].tolist().index(min(model2.iloc[:,2]))+1)

        min_train_show_x = min_train_indx

        show_min_train_original ='min:['+str(min_train_indx)+' '+str(min_train)+']'

        #ax[r][c].annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

        ax[r][c].plot(min_train_indx, min_train, 'gs')



        #未调整前-标注test 最低点

        min_test, min_test_indx = round(min(model2.iloc[:,6]), 6), (model2.iloc[:,6].tolist().index(min(model2.iloc[:,6]))+1)

        min_test_show_x = min_train_indx

        show_min_test_original ='min:['+str(min_test_indx)+' '+str(min_test)+']'

        #ax[r][c].annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

        ax[r][c].plot(min_test_indx, min_test, 'gs')

        

        

        #调整后-标注train 最低点

        min_train, min_train_indx = round(min(evaluation_list[i].iloc[:,2]), 6), (evaluation_list[i].iloc[:,2].tolist().index(min(evaluation_list[i].iloc[:,2]))+1)

        min_train_show_x = min_train_indx

        show_min_train_cv='min:['+str(min_train_indx)+' '+str(min_train)+']'

        #ax[r][c].annotate(show_min_train,xytext=(min_train_indx-100,min_train-0.015),xy=(min_train_show_x,min_train))

        ax[r][c].plot(min_train_indx, min_train, 'gs')



        #调整后-标注test 最低点

        min_test, min_test_indx = round(min(evaluation_list[i].iloc[:,6]), 6), (evaluation_list[i].iloc[:,6].tolist().index(min(evaluation_list[i].iloc[:,6]))+1)

        min_test_show_x = min_train_indx

        show_min_test_cv='min:['+str(min_test_indx)+' '+str(min_test)+']'

        #ax[r][c].annotate(show_min_test,xytext=(min_test_indx-100,min_test+0.015),xy=(min_test_show_x,min_test))

        ax[r][c].plot(min_test_indx, min_test, 'gs')



        ax[r][c].set_title("交叉验证曲线{} subsample:{} colsample_bytree:{}".format(i, title_sub[i], title_col[i]), fontproperties=zhfont, fontsize=15)



    plt.subplots_adjust(hspace = 0.25)

    plt.show()



show_cv(title_sub, title_col, evaluation_list)
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["gamma"] = 0

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_depth"] = 5

fix_params["min_child_weight"] = 4.5

fix_params["max_bin"] = 200

fix_params["subsample"] = 0.7

fix_params["colsample_bytree"] = 0.9

#fix_params["eta"] = 0.02



time0 = time()



evaluation_list = []

for eta in [0.06, 0.04, 0.02]:

    params = {**fix_params, **{"eta": eta}}

    evaluation = xgb.cv(params, dtrain, num_boost_round = 10000, nfold = 10, 

                        feval = weighted_log_loss, early_stopping_rounds = 150)

    evaluation_list.append(evaluation)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

    

evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
title_eta = [0.06, 0.04, 0.02]



def show_cv(title_eta, evaluation_list):

    fig,ax = plt.subplots(1,figsize=(25,9))

    prop_cycle = plt.rcParams['axes.prop_cycle']

    colors = prop_cycle.by_key()['color']

    ax.set_ylim(0.15,0.3)

    ax.grid()

    color_index = [0, 1, 2, 3, 4, 5]

    for i in range(len(title_eta)):

        ax.plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,2],c="green",label="train,eta{}".format(title_eta[i]), color=colors[color_index[i]])

        ax.plot(range(1,evaluation_list[i].shape[0]+1),evaluation_list[i].iloc[:,6],c="blue",label="test,eta{}".format(title_eta[i]), color=colors[color_index[i+3]])



    ax.legend(fontsize="10")

    ax.set_title("交叉验证曲线 eta", fontproperties=zhfont, fontsize=15)

    plt.show()



show_cv(title_eta, evaluation_list)
evaluation_list_eta1 = evaluation_list
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["gamma"] = 0

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_depth"] = 5

fix_params["min_child_weight"] = 4.5

fix_params["max_bin"] = 200

fix_params["subsample"] = 0.7

fix_params["colsample_bytree"] = 0.9

#fix_params["eta"] = 0.02



time0 = time()



evaluation_list = []

for eta in [0.03, 0.025]:

    params = {**fix_params, **{"eta": eta}}

    evaluation = xgb.cv(params, dtrain, num_boost_round = 10000, nfold = 10, 

                        feval = weighted_log_loss, early_stopping_rounds = 150)

    evaluation_list.append(evaluation)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

    

evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
title_eta = [0.06, 0.04, 0.03, 0.025, 0.02]



evaluation_eta = []

evaluation_eta.append(evaluation_list_eta1[0])

evaluation_eta.append(evaluation_list_eta1[1])

evaluation_eta.append(evaluation_list[0])

evaluation_eta.append(evaluation_list[1])

evaluation_eta.append(evaluation_list_eta1[2])



def show_cv(title_eta, evaluation_list):

    fig,ax = plt.subplots(1,figsize=(25,9))

    prop_cycle = plt.rcParams['axes.prop_cycle']

    colors = prop_cycle.by_key()['color']

    ax.grid()

    ax.set_ylim(0.15,0.3)

    color_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for i in range(len(title_eta)):

        ax.plot(range(1,evaluation_eta[i].shape[0]+1),evaluation_eta[i].iloc[:,2],c="green",label="train,eta{}".format(title_eta[i]), color=colors[color_index[i]])

        ax.plot(range(1,evaluation_eta[i].shape[0]+1),evaluation_eta[i].iloc[:,6],c="blue",label="test,eta{}".format(title_eta[i]), color=colors[color_index[i+5]])



    ax.legend(fontsize="10")

    

    ax.set_title("交叉验证曲线 eta", fontproperties=zhfont, fontsize=15)

    plt.show()



show_cv(title_eta, evaluation_list)
params = {}

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.03

params["gamma"] = 0

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"

params["max_depth"] = 5

params["min_child_weight"] = 4.5

params["max_bin"] = 200

params["subsample"] = 0.7

params["colsample_bytree"] = 0.9



params['n_estimators'] = 4500



time0 = time()

X_train, X_test, y_train, y_test = train_test_split(train, trainlabel, test_size=0.3, random_state=420)



clf = XGBClassifier(**params).fit(X_train, y_train.values.ravel())



y_pred = clf.predict(X_test)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))



print("\tAccuracy:{}".format(clf.score(X_test, y_test)))

print("\tRecall:{}".format(recall_score(y_test, y_pred)))

print("\tAUC:{}".format(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])))



display(confusion_matrix(y_test, y_pred, labels=[1,0]))
params = {}

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.025

params["gamma"] = 0

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"

params["max_depth"] = 5

params["min_child_weight"] = 4.5

params["max_bin"] = 200

params["subsample"] = 0.7

params["colsample_bytree"] = 0.9



params['n_estimators'] = 5600



time0 = time()

X_train, X_test, y_train, y_test = train_test_split(train, trainlabel, test_size=0.3, random_state=420)



clf = XGBClassifier(**params).fit(X_train, y_train.values.ravel())



y_pred = clf.predict(X_test)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))



print("\tAccuracy:{}".format(clf.score(X_test, y_test)))

print("\tRecall:{}".format(recall_score(y_test, y_pred)))

print("\tAUC:{}".format(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])))



display(confusion_matrix(y_test, y_pred, labels=[1,0]))
params = {}

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.02

params["gamma"] = 0

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"

params["max_depth"] = 5

params["min_child_weight"] = 4.5

params["max_bin"] = 200

params["subsample"] = 0.7

params["colsample_bytree"] = 0.9



params['n_estimators'] = 6900



time0 = time()

X_train, X_test, y_train, y_test = train_test_split(train, trainlabel, test_size=0.3, random_state=420)



clf = XGBClassifier(**params).fit(X_train, y_train.values.ravel())



y_pred = clf.predict(X_test)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))



print("\tAccuracy:{}".format(clf.score(X_test, y_test)))

print("\tRecall:{}".format(recall_score(y_test, y_pred)))

print("\tAUC:{}".format(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])))



display(confusion_matrix(y_test, y_pred, labels=[1,0]))
params = {}

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.02

params["gamma"] = 0

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"

params["max_depth"] = 5

params["min_child_weight"] = 4.5

params["max_bin"] = 200

params["subsample"] = 0.7

params["colsample_bytree"] = 0.9



params['n_estimators'] = 6900



time0 = time()

clf = XGBClassifier(**params).fit(train, trainlabel.values.ravel())

y_pred = clf.predict(test)



print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))



#t = pd.read_csv('/kaggle/input/quora-question-pairs/test.csv')



sub = pd.DataFrame()

#sub['test_id'] = test["test_id"]

sub['is_duplicate'] = y_pred

#sub.to_csv('submission6900.csv', index=False)
sub.shape
sub.hist()
fig,ax = plt.subplots(figsize=(15,15))

plot_importance(clf, height=0.5, ax=ax, max_num_features=64)

plt.show()