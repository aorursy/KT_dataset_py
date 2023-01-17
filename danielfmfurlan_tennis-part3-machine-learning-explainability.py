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
def get_data_splits(dataframe, valid_fraction=0.1):

    valid_fraction = 0.1

    valid_size = int(len(dataframe) * valid_fraction)



    train = dataframe[:-valid_size * 2]

    valid = dataframe[-valid_size * 2:-valid_size]

    test = dataframe[-valid_size:]

    #print(f"Train size : {len(train)}\nValidation size : {len(valid)}\nTest size : {len(test)}")

    return train, valid, test
import lightgbm as lgb

from sklearn import metrics

from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

import seaborn as sns

import matplotlib.pyplot as plt 

import joblib

dire1 = "../input/model1/Model 1.txt"

bst1 = joblib.load(dire1)



dire2 = "../input/model2/Model 2.txt"

bst2 = joblib.load(dire2)



dire3 = "../input/model3/Model 3.txt"

bst3 = joblib.load(dire3)



dire4 = "../input/model4/Model 4.txt"

bst4 = joblib.load(dire4)
# df_atp = pd.read_csv("../input/df-atp/df_atp.csv",index_col=0)

# df = pd.read_csv("../input/data-4qcsv/data_4q.csv",index_col=0)

data_model1 = pd.read_csv("../input/data-model1/data_model1.csv",index_col=0)

data_model2 = pd.read_csv("../input/data-model2/data_model2.csv",index_col=0)

data_model3 = pd.read_csv("../input/data-model3/data_model3.csv",index_col=0)

data_model4 = pd.read_csv("../input/data-model4/data_model4.csv",index_col=0)
import numpy as np

import shap

valid_size = int(len(data_model1) * 0.1)

        

train1, valid1, test1 = get_data_splits(data_model1)



train2, valid2, test2 = get_data_splits(data_model2)

train3, valid3, test3 = get_data_splits(data_model3)

train4, valid4, test4 = get_data_splits(data_model4)



bag = {1:{"train":train1,"valid":valid1,"test":test1,"bst":bst1},\

       2:{"train":train2,"valid":valid2,"test":test2,"bst":bst2},\

       3:{"train":train3,"valid":valid3,"test":test3,"bst":bst3},\

       4:{"train":train4,"valid":valid4,"test":test4,"bst":bst4}}


    

shap.initjs()

for i in range(1,5):

    #### SINGLE PREDICTION

    feature_cols = bag[i]["valid"].columns.drop('Labels')

    

    row_to_show = 0      ############# I'm looking here at the very first sample of our validation set! Change it to play with other examples! 

    

    data_for_prediction = bag[i]["valid"].drop("Labels",axis=1).iloc[row_to_show]

    data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

    idx = bag[i]["valid"].index[0]

    print(f"Model number : {i}\nprediction of model : ",bag[i]["bst"].predict(data_for_prediction_array), " ground truth : ", bag[i]["valid"]["Labels"][idx + row_to_show])



    data_ser = data_for_prediction.values.reshape(1, -1)

    

    ###### We call the explainer!!

    explainer = shap.TreeExplainer(bag[i]["bst"])



    shap_values = explainer.shap_values(data_ser)



    #display(explainer.expected_value,shap_values)    #### Uncomment to see what the shap_values are about!! It's an array with all the variables' shap values!



    display(shap.force_plot(explainer.expected_value[1], shap_values[1], data_ser, feature_names = feature_cols))





def most_import(shap_values,feature_cols,data):

    feature_cols = feature_cols

    display(shap.summary_plot(shap_values, data,feature_names = feature_cols))

def get_shap(m,part):

    

    feature_cols = bag[m]["valid"].columns.drop('Labels') 

    explainer = shap.TreeExplainer(bag[m]["bst"])

    data = bag[m][part][feature_cols]

    shap_values = explainer.shap_values(data)

    

    return shap_values, feature_cols, data



def get_idx(n,m):

    return [idx2 for idx2,x in enumerate(bag[m]["valid"].columns.drop("Labels")) if x == feat_3[n]]
### Lets take the Model 2 to analyse how the features play their importance in the prediction.

model = 3



shap_values, feat_cols, data = get_shap(model,'valid')



most_import(shap_values,feat_cols,data)
#### OVERALL OUTPUT:

def overall(m1,m2):

    """

    m1 : number of model 1

    m2 : number of model 2

    """

    for i in range(m1,m2+1):

        feature_cols = bag[i]["valid"].columns.drop('Labels')

        explainer = shap.TreeExplainer(bag[i]["bst"])

        data = bag[i]['valid'][feature_cols]

        shap_values = explainer.shap_values(data)

    

        display(shap.force_plot(explainer.expected_value[1], shap_values[1][:1000,:], feature_cols)) #### we're taking 1000 samples from our valid set!

        



        ##  for name in valid.columns.drop("Labels"): ################# scatter plots between the model features!

        #     shap.dependence_plot(name, shap_values[1], valid[feature_cols])
overall(1,3) ### We're just visualizing the first 3 models!
def roundIt(dic,n,r):

    for i in range(1,n):

        dic[i]['tp'] = round(float(dic[i]['tp']),r)

        dic[i]['tn'] = round(float(dic[i]['tn']),r)

        dic[i]['fp'] = round(float(dic[i]['fp']),r)

        dic[i]['fn'] = round(float(dic[i]['fn']),r)

    return dic



def get_conf_dict(n):

    y = [round(bag_ft[n]['tn'],3),bag_ft[n]['fn'],bag_ft[n]['tp'],bag_ft[n]['fp']]

    return y
i = 0

th = 0.5

feat_3 = ["Oponent_cat","Player1_cat", "WRank"] ### Most impactful features!

n_feat = len(feat_3)





m = 2 ### model number



shap_values, feature_cols, _ = get_shap(m,'valid')



bag_ft = {1:{"tp":0,"tn":0,"fp":0,"fn":0,"idx":get_idx(0,m),"feature":feat_3[0]},\

          2:{"tp":0,"tn":0,"fp":0,"fn":0,"idx":get_idx(1,m),"feature":feat_3[1]},\

          3:{"tp":0,"tn":0,"fp":0,"fn":0,"idx":get_idx(2,m),"feature":feat_3[2]}}



#print(bag_ft[1]["idx"])



y_true = bag[m]["valid"]["Labels"][:1000]

y_pred = bag[m]['bst'].predict(bag[m]["valid"][feature_cols])



for each in shap_values[1]:

    #print(each)

    

    while i < 1000:

        val1 = round(float(each[bag_ft[1]["idx"]]),3)

        val2 = round(float(each[bag_ft[2]["idx"]]),3)

        val3 = round(float(each[bag_ft[3]["idx"]]),3)

        idx = y_true.index[i]

        #print("ytrue, ypred, i : ",y_true[idx],y_pred3[i],i)

        if y_true[idx] == 1 and y_pred[i] >= th: #### TP

            #tpO = tpO + abs(val)

            bag_ft[1]["tp"] = bag_ft[1]["tp"] + abs(val1)

            bag_ft[2]["tp"] = bag_ft[2]["tp"] + abs(val2)

            bag_ft[3]["tp"] = bag_ft[3]["tp"] + abs(val3)

            i = i+ 1

            continue

        if y_true[idx] == 1 and y_pred[i] < th: #### FN

            #fnO = fnO + abs(val)

            bag_ft[1]["fn"] = bag_ft[1]["fn"] + abs(val1)

            bag_ft[2]["fn"] = bag_ft[2]["fn"] + abs(val2)

            bag_ft[3]["fn"] = bag_ft[3]["fn"] + abs(val3)

            i = i+ 1

            continue

        if y_true[idx] == 0 and y_pred[i] >= th: #### FP

            #fpO = fpO + abs(val)

            bag_ft[1]["fp"] = bag_ft[1]["fp"] + abs(val1)

            bag_ft[2]["fp"] = bag_ft[2]["fp"] + abs(val2)

            bag_ft[3]["fp"] = bag_ft[3]["fp"] + abs(val3)

            i = i+ 1

            continue

        if y_true[idx] == 0 and y_pred[i] < th: #### TN

            #tnO = tnO + abs(val)

            bag_ft[1]["tn"] = bag_ft[1]["tn"] + abs(val1)

            bag_ft[2]["tn"] = bag_ft[2]["tn"] + abs(val2)

            bag_ft[3]["tn"] = bag_ft[3]["tn"] + abs(val3)

            i = i+ 1



bag_ft = roundIt(bag_ft,4,None)  

print(bag_ft)
# data = [get_conf_dict(1), get_conf_dict(2), get_conf_dict(3)] 

  

# # Create the pandas DataFrame 

# df = pd.DataFrame(data, columns = ['TN', 'FN','TP','FP']) 

  

# # print dataframe. 

# df
# def get_bar_posneg():

#     fig = plt.figure(figsize=(20,10))

#     ind = np.arange(1,5)

    
#-5.420885622420833 -5.184509795861786 -2.8049931418340392 -2.347999877153214

fig = plt.figure(figsize=(20,10))

ind = np.arange(1,5)

cat = 3

y = get_conf_dict(cat)

print(y,ind)



plt.bar(ind, y)

plt.ylabel('Scores SHAP values',fontsize=20)

plt.title(f'Impact of {bag_ft[cat]["feature"]} on samples',fontsize=20)

plt.xticks(ind, ('TN', 'FN', 'TP', 'FP'),fontsize=20)

plt.yticks(np.arange(0.0, 22.0, 1.0),fontsize=20)

#plt.legend((p1[0], p2[0]), (f'Model {m1}', f'Model {m2}'),fontsize=20)

for i, v in enumerate(y):

    print(i,v)

    v = round(v,3)

    plt.text(i+0.95, v/2, str(v), color='black', fontweight='bold',fontsize=20)

plt.show()
##### Get the stacked bar plot of most important features impact on pos/neg cases

#################################################################################



fig = plt.figure(figsize=(20,10))

ind = np.arange(4)



y1 = get_conf_dict(2)

y2 = get_conf_dict(1)

y3 = get_conf_dict(3)

width = 0.45



p1 = plt.bar(ind, y1,width)           ### Player1

p2 = plt.bar(ind, y2,width,bottom=y1) ### Oponent_cat

p3 = plt.bar(ind, y3,width,bottom=y2) ### WRank





plt.xticks(ind, ('TN', 'FN', 'TP', 'FP'),fontsize=20)

plt.yticks(np.arange(0, 150, 10),fontsize=20)

plt.legend((p1[0], p2[0],p3[0]), (f'{bag_ft[2]["feature"]}',\

                                  f'{bag_ft[1]["feature"]}',\

                                  f'{bag_ft[3]["feature"]}'),fontsize=20)



plt.ylabel('Scores SHAP values',fontsize=20)

plt.title(f'SHAP Impact of most important features on pos/neg cases on model {m}',fontsize=20)





for i, v in enumerate(y1):

    #print(i,v)

    plt.text(i - 0.02, v/2, str(v), color='black', fontweight='bold',fontsize=20)

    

for i, v in enumerate(y2):

    #print(i,v)

    plt.text(i - 0.04, v/2 + y1[i], str(v), color='pink', fontweight='bold',fontsize=20)

for i, v in enumerate(y3):

    #print(i,v)

    plt.text(i - 0.04, v/2 + y2[i], str(v), color='white', fontweight='bold',fontsize=20)





plt.show()
def conf_mat(n,a,b):

    

    feature_cols = bag[n]["valid"].columns.drop('Labels')

    y_pred = bag[n]['bst'].predict(bag[n]["valid"][feature_cols][a:b])

    

    th = 0.5

    y_pred_class = y_pred > th



    cm = confusion_matrix(bag[n]["valid"]["Labels"][a:b], y_pred_class)

    tn, fp, fn, tp = cm.ravel()

    #display(tn,fn,tp, fp)

    

    return [tn, fp, fn, fp], y_pred
##### Get the differences in pos/neg between 2 MODELS and in a particular period analysed on the "overall" function

N = 4



##### Model 1 & 2

##### Period: 190:250

a = 160

b = 190

m1= 1

m2= 2

fig = plt.figure(figsize=(35,10))

conf1, y_pred1 = conf_mat(m1,a,b)

conf2, y_pred2 = conf_mat(m2,a,b)



# menMeans = (20, 35, 30, 35, 27)

# womenMeans = (25, 32, 34, 20, 25)

model1 = conf1

model2 = conf2



menStd = (2, 3, 4, 1)

womenStd = (3, 5, 2, 3 )

ind = np.arange(N)    # the x locations for the groups

width = 0.35       # the width of the bars: can also be len(x) sequence



p1 = plt.bar(ind, model1, width,)

p2 = plt.bar(ind, model2, width,

             bottom=model1)

for i, v in enumerate(model1):

    #print(i,v)

    plt.text(i - 0.025, v/2, str(v), color='black', fontweight='bold',fontsize=20)

    

for i, v in enumerate(model2):

    #print(i,v)

    plt.text(i - 0.025, v/2 + model1[i], str(v), color='gray', fontweight='bold',fontsize=20)

    

plt.ylabel('Scores',fontsize=20)

plt.title(f'Scores models {m1} & {m2} for samples in {a}:{b}',fontsize=20)

plt.xticks(ind, ('TN', 'FN', 'TP', 'FP'),fontsize=20)

plt.yticks(np.arange(0, 51, 5),fontsize=20)

plt.legend((p1[0], p2[0]), (f'Model {m1}', f'Model {m2}'),fontsize=20)





plt.show()
# #We can notice that in that region we have exactly the same amount of

# #positive and negative cases (30 per each). That is to say: 30 times the "1" and 30 times the "0"

# y_true = bag[2]["valid"]["Labels"][:1000]

# y_true[190:250].sum()