# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy.stats import pearsonr
from sklearn.utils import shuffle
data = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv", index_col = "Serial No." )

data.head()
data.describe()
# fig, ax = plt.subplots(1,7,figsize=(16,6))
# for i in range(4,10):
#     val = data.loc[(data["Chance of Admit "] > 0.1*i) & (data["Chance of Admit "] < 0.1*(1+i))]
# #print(val)
#     sns.scatterplot(y= val["CGPA"], x=val["Chance of Admit "],ax=ax[i-3])
# #sns.scatterplot(x= data["CGPA"], y=data.loc["Chance of Admit "],ax=ax[1])
# fig, ax = plt.subplots(1,7,figsize=(16,8))
# for i in range(4,10):
#     val = data.loc[(data["Chance of Admit "] > 0.1*i) & (data["Chance of Admit "] < 0.1*(1+i))]
# #print(val)
#     sns.scatterplot(y= val["TOEFL Score"], x=val["Chance of Admit "],ax=ax[i-3])
### ONLY FOR TOEFL = 92
toefMax = data.loc[(data["TOEFL Score"]==92)]["Chance of Admit "].max()
toefMin = data.loc[(data["TOEFL Score"]==92)]["Chance of Admit "].min()
print("max and min for 92: {}, {}".format(toefMax,toefMin))
t_u = sorted(data["TOEFL Score"].unique())

t_val = pd.Series(index=t_u)
for un in data["TOEFL Score"].unique():
    t_max = data.loc[(data["TOEFL Score"]==un)]["Chance of Admit "].max()
    t_min = data.loc[(data["TOEFL Score"]==un)]["Chance of Admit "].min()
    t_val[un] = t_max - t_min
t_val
cor = []
for one,two in itertools.combinations(data.columns,2):
    cor.append(f"Correlation btw {one} and {two} is: {pearsonr(data[one],data[two])}")

from heapq import nlargest
from operator import itemgetter
all_cor = []
all_cord = {}
for one,two in itertools.combinations(data.columns,2):

    v,_ = pearsonr(data[one],data[two])
    all_cor.append(f"Correlation btw {one} and {two} is: {v:.4f}")
    all_cord[one+"-"+two] = round(v,4)

all_cor

m = dict(sorted(all_cord.items(), key = itemgetter(1), reverse = True)[:4])
print("The 4 strongest correlations are : ",m)

data.columns
adm_cor = []
for col in data.columns.drop(["Chance of Admit "]):
    #plt.figure(figsize=(15,15))
    v,_ = pearsonr(data[col],data["Chance of Admit "])
    adm_cor.append(f"Correlation btw {col} and Chance of Admit is: {v:.4f}")
    #sns.regplot(x=one,y=two,data=data)
adm_cor
results = []
results_x = [] ## for "xentropy" metric
def split_data(data):
    # shuffle data:
    #data = shuffle(data)
    fr = 0.1
    vsize = int(len(data)*fr)
    print("vsize = ", vsize)
    train = data[:-2*vsize]
    valid = data[-2*vsize:-vsize]
    test = data[-vsize:]
    
    return train,valid,test
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lbg

m = False
i = 0
plots = []
val_pred = []
ground = []
def training(train, valid, obj, met, boost, test=None):
    global m 
    global Ax
    global i
    global val_pred
    global ground
    if m == False:
        print("training...")
        fig = plt.figure(figsize=(15,15))
        Ax = fig.add_subplot(1,1,1)  
        m = True
        
    feat_cols = train.columns.drop("Chance of Admit ")
    dtrain = lbg.Dataset(train[feat_cols],label=train["Chance of Admit "])
    dvalid = lbg.Dataset(valid[feat_cols],label=valid["Chance of Admit "])
    
    objective = obj 
    Metric = met
    boosting = boost
    param = {"num_leaves" : 64, "objective":objective, "boosting-type":boosting, "metric":Metric,"seed":7, "verbose":-1}
    num_rounds = 300       
     
    evals_result = {} 
    bst = lbg.train(param, dtrain, num_rounds, valid_sets=[dvalid,dtrain],evals_result=evals_result, early_stopping_rounds=10, verbose_eval=0)

    
    if param["metric"] == "rmse" and i < 1:        
        print('Plot metrics during training... Our metric : ', param["metric"])
        
        lbg.plot_metric(evals_result, metric=Metric, ax = Ax, figsize=(15,15))
        plt.xlabel('Iterations',fontsize=20)
        plt.ylabel('RMSE',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title("RMSE during training",fontsize=20)
        
        plt.text(20, 0.1, 'Objective: {}. Boost: {}'.format(objective,boosting), style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        
        plt.legend(fontsize=20)
        i += 1

        
    valid_pred = bst.predict(valid[feat_cols])
    val_pred = valid_pred
    ground = valid["Chance of Admit "]
    
    valid_err = mean_squared_error(valid["Chance of Admit "], valid_pred)

    if Metric == "xentropy":
        results_x.append(f"Model with objective as {objective}, metric as {Metric}, and boosting as {boosting}. Score/error : {valid_err:.8f}")
    else:
        results.append(f"Model with objective as {objective}, metric as {Metric}, and boosting as {boosting}. Score/error : {valid_err:.8f}")
    
    if test is not None: 
        test_pred = bst.predict(test[feat_cols])
        print("our test_pred : ", test_pred)
        print("our ground truth : ", test['Chance of Admit '])
        test_score = mean_squared_error(test['Chance of Admit '], test_pred)
        return bst, valid_err, test_score
    else:
        return bst, valid_err, evals_result
    
    
obj = ["xentropy" ,"regression"]
boost = ["random_forest","gbdt",None]
met = ["xentropy","rmse"]
train,valid,test = split_data(data)
#all = obj + boost
for me in met:
    for ob in obj:
        for bo in boost:
            #eval_result = {}
            #global m 
            training(train,valid,ob,me,bo,test)
            
plt.show()
#https://matplotlib.org/3.1.1/tutorials/text/text_intro.html
#https://python4astronomers.github.io/plotting/advanced.html
#training(data)
results
#type(results[0])

def find_min(data):
    baseline_err = float(1)
    mod = ""
    if type(data) is list:
        for each in data:
        #print("each ", each, " type : ", type(each))
            v = re.findall("\d+\.\d+",each)
        #print("our v : ", v)
            if float(v[0]) < baseline_err:
                baseline_err = float(v[0])
                mod = each
    else:
        for each in data:
            v = re.findall("\d+\.\d+",str(data[each]))
        #print("our v : ", v)
            if float(v[0]) < baseline_err:
                baseline_err = float(v[0])
                mod = each + " : " + str(data[each])
            
    return mod
best_mod = find_min(results)
print("our best model : ", best_mod)
baseline_err = re.findall("\d+\.\d+",best_mod)
print("base ", baseline_err[0])
### SAVE THE PARAMETERS INTO A DICTIONARY
par = {"objective":"","metric":"","boosting":""}
para = []
def get_par():
    idx = 0
    vir = 0
    while idx < len(best_mod) and idx != -1:
        idx = best_mod.find("as", idx+1)
        vir = best_mod.find(",", vir+1)
        if vir == -1:
            vir = best_mod.find(".")
        if idx == -1:
            break
        obj = best_mod[(idx+3):vir]
        para.append(obj)
        #print("our obj :", obj)
    i = 0
    for key in par:
        par[key] = para[i]
        i+=1
    print(par)
get_par()
import category_encoders as ce
all_res = {} # dictionary with anders models and ihr results
all_res["baseline"] = baseline_err[0]
#train,valid,test = split_data(data)    
from sklearn.preprocessing import LabelEncoder
inter = pd.DataFrame(index=data.index) #### DATA
var = pd.DataFrame(index=data.index)#### DATA
#train,valid,_ = split_data(data)

lb_enc = LabelEncoder()
inter_ft = ['GRE Score', 'TOEFL Score']
for each in inter_ft:
    name = "CGPA_"+each
    nameVar = each+"_admitVar"
    inter[name] = lb_enc.fit_transform(data["CGPA"].apply(str)+"_"+data[each].apply(str)) ### DATA
    data[name] = inter[name] #### DATA
    #valid[name] = 
#     var[nameVar]= train[each].apply(fun)
#     train[nameVar] = var[nameVar]
    
train.head()
data.head()
train,valid,test = split_data(data)
res = training(train,valid,par["objective"],par["metric"],par["boosting"],test)
all_res["interaction"] = res[1].round(8)
#print(f"Our model performance error: {res[1]:.4f}")
print(f"Models erros : {all_res}")
print("Performance quality (>1 worse): ", float(res[1])/float(baseline_err[0]))
print("parameters : ", par["metric"])

print('Plot feature importances...')
ax = lbg.plot_importance(res[0], max_num_features=10,figsize=(10,10))
plt.show()
print("Plot tree: ")
ax = lbg.plot_tree(res[0],figsize=(15,15))
plt.show()
#res[0]
feat = ['GRE Score', 'TOEFL Score','CGPA']
enc = pd.DataFrame(index=train.index)


c_enc = ce.CountEncoder(cols=feat)
c_enc.fit(data[feat]) #### TRAIN
try:
    #data = data.join(c_enc.transform(data[feat]).add_suffix("_count"))
    train = train.join(c_enc.transform(train[feat]).add_suffix("_count"))
    valid = valid.join(c_enc.transform(valid[feat]).add_suffix("_count"))
    train.head()
except:
    print("we have already create the columns!")
    print(train.head())

train.tail()
train.isnull().sum()
valid.isnull().sum()
#train,valid,test = split_data(data)
res = training(train,valid,par["objective"],par["metric"],par["boosting"])
print(f"Our model performance error: {res[1]:.6f}")

all_res["count"] = res[1].round(8)
#print(f"Our model performance error: {res[1]:.4f}")
print(f"Models erros : {all_res}")
print("Performance quality (>1 worse): ", float(res[1])/float(baseline_err[0]))
def fun(dat,name):
    
    def repl(ser):
        return t_val[ser]
    
    #print("our dat : ", dat)
    t_u = sorted(dat[name].unique())

    t_val = pd.Series(index=t_u)
    for un in dat[name].unique():
        t_max = dat.loc[(dat[name]==un)]["Chance of Admit "].max()
        t_min = dat.loc[(dat[name]==un)]["Chance of Admit "].min()
        t_val[un] = t_max - t_min

    datafr = pd.DataFrame(index=dat.index)
    
    datafr = dat[name].apply(repl)    
    
    return datafr
from sklearn.preprocessing import LabelEncoder

#train,valid,_ = split_data(data)

inter = pd.DataFrame(index=train.index)
var = pd.DataFrame(index=train.index)
# inter = pd.DataFrame(index=data.index)
# var = pd.DataFrame(index=data.index)

inter_ft = ['GRE Score', 'TOEFL Score', "CGPA"]
for each in inter_ft:
    #print("our each is : ", each)
    nameVar = each+"_admitVar"

    var[nameVar]= fun(train,each)# train[each].apply(fun)
    train[nameVar] = var[nameVar]
    valid[nameVar] = var[nameVar]
    #data[nameVar] = var[nameVar]
    #train = train.join(var[nameVar].add_suffix(nameVar))
    
train.head()
#train,valid,test = split_data(data)
res = training(train,valid,par["objective"],par["metric"],par["boosting"])
print(f"Our model performance error: {res[1]:.8f}")

all_res["admi_var"] = res[1].round(8)
#print(f"Our model performance error: {res[1]:.4f}")
print(f"Models erros : {all_res}")
print("Performance quality (>1 worse): ", float(res[1])/float(baseline_err[0]))
col = train.columns[-3:]
print("col : ", col)
feat = col
enc2 = pd.DataFrame(index=train.index)


c_enc = ce.CountEncoder(cols=feat)
c_enc.fit(train[feat]) ##################################" !!!!!"
try:
    #data = data.join(c_enc.transform(data[feat]).add_suffix("_countVar"))
    train = train.join(c_enc.transform(train[feat]).add_suffix("_countVar"))
    valid = valid.join(c_enc.transform(valid[feat]).add_suffix("_countVar"))
    train.head()
except:
    print("we have already create the columns!")
    print(train.head())

data.head()
train.head()

#train,valid,test = split_data(data)
res = training(train,valid,par["objective"],par["metric"],par["boosting"])
print(f"Our model performance error: {res[1]:.8f}")

all_res["CountAdmi_var"] = res[1].round(8)
#print(f"Our model performance error: {res[1]:.4f}")
print(f"Models erros : {all_res}")
print("Performance quality (>1 worse): ", float(res[1])/float(baseline_err[0]))

plt.figure(figsize=(15,10))
x = np.linspace(1,50,50)

val_pred = [float(b) for b in val_pred]
yerr = (((val_pred) - (ground))).to_numpy()
yerr = [float(a) for a in yerr]

plt.errorbar(x,val_pred,yerr=yerr,marker = 'o',fmt='go',markersize=5,uplims=True,lolims = False, capsize=0,capthick=0,label="prediction")
plt.title("Difference between ground_truth and predictions for 50 values")
plt.ylabel("Chance of Admit")
plt.xlabel("Example number")
plt.legend()
plt.show()

diferr = pd.DataFrame(columns=["Prediction", "Ground_Truth", "Error"])
diferr["Ground_Truth"] = ground
diferr["Prediction"] = val_pred
diferr["Error"] = yerr

plt.figure(figsize=(15,10))
plt.hist(yerr)
plt.title("Error density")
diferr.head()
train = train.drop(col, axis=1)
valid = valid.drop(col, axis=1)
#data.head()

#train,valid,test = split_data(data)
res = training(train,valid,par["objective"],par["metric"],par["boosting"])
print(f"Our model performance error: {res[1]:.8f}")

all_res["Count_none_Admi_var"] = res[1].round(8)
#print(f"Our model performance error: {res[1]:.4f}")
print(f"Models erros : {all_res}")
print("Performance quality (>1 worse): ", float(res[1])/float(baseline_err[0]))
best_mod = find_min(all_res)
print("best_mod : ", best_mod)
from sklearn.feature_selection import f_regression, mutual_info_regression

f_test, _ = f_regression(train, train["Chance of Admit "])
f_test /= np.max(f_test)

mi = mutual_info_regression(train, train["Chance of Admit "])
mi /= np.max(mi)

plt.figure(figsize=(35, 5))
for i in range(int(len(train.columns)/4)):
    if i < 8:
        #plt.figure(figsize=(25, 15))
        plt.subplot(1, len(train.columns)/2, i + 1)
        plt.scatter(train.iloc[:, i], train["Chance of Admit "], edgecolor='black', s=20)
        plt.xlabel("${}$".format(train.columns[i]), fontsize=14)
        if i == 0:
            plt.ylabel("Chance of Admit", fontsize=14)
        plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
              fontsize=16)
plt.show()
plt.figure(figsize=(35, 5))
for i in range(4):
    #print("value of i :", i)
    plt.subplot(1, len(train.columns)/2, i + 1)
    plt.scatter(train.iloc[:, i+4], train["Chance of Admit "], edgecolor='black', s=20)
    plt.xlabel("${}$".format(train.columns[i+4]), fontsize=14)
    if i == 0:
        plt.ylabel("Chance of Admit", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i+4], mi[i+4]),fontsize=16)
plt.show()
plt.figure(figsize=(35, 5))
for i in range(4):
    #print("value of i :", i)
    plt.subplot(1, len(train.columns)/2, i + 1)
    plt.scatter(train.iloc[:, i+8], train["Chance of Admit "], edgecolor='black', s=20)
    plt.xlabel("${}$".format(train.columns[i+8]), fontsize=14)
    if i == 0:
        plt.ylabel("Chance of Admit", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i+8], mi[i+8]),fontsize=16)
plt.show()
plt.figure(figsize=(35, 5))
for i in range(4):
    #print("value of i :", i)
    plt.subplot(1, len(train.columns)/2, i + 1)
    plt.scatter(train.iloc[:, i+12], train["Chance of Admit "], edgecolor='black', s=20)
    plt.xlabel("${}$".format(train.columns[i+12]), fontsize=14)
    if i == 0:
        plt.ylabel("Chance of Admit", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i+12], mi[i+12]),fontsize=16)
plt.show()
print("ftest : ", f_test)
print("mitest : ", mi)

sel = {train.columns[key]:value for key,value in enumerate(f_test) if value > 0.50}

sel2 = {train.columns[key]:value for key,value in enumerate(mi) if value > 0.23}

print("sel1", sel)
print("sel2", sel2)
selected_feat = sel
train.head()
train = train.drop([train.columns[i] for i in range(len(train.columns)) if train.columns[i] not in selected_feat.keys() and train.columns[i] != "Chance of Admit "], axis=1)
valid = valid.drop([valid.columns[i] for i in range(len(valid.columns)) if valid.columns[i] not in selected_feat.keys() and valid.columns[i] != "Chance of Admit "], axis=1)

#train,valid,test = split_data(data)
res = training(train,valid,par["objective"],par["metric"],par["boosting"])
print(f"Our model performance error: {res[1]:.8f}")

all_res["FEAT_SEL"] = res[1].round(8)
#print(f"Our model performance error: {res[1]:.4f}")
print(f"Models erros : {all_res}")
print("Performance quality (>1 worse): ", float(res[1])/float(baseline_err[0]))
all_res
plt.figure()
X = []
error = []

for t, er in all_res.items():
    X.append(t)
    error.append(float(er))
print(X)
print(error)
plt.figure(figsize=(15, 10))

#plt.subplot(131)
plt.ylabel("RMSE")
e = plt.bar(X, error)#color=['black', 'red', 'green', 'blue', 'cyan','orange','gray'])
e[0].set_color('r')
e[4].set_color("g")
plt.xticks()
plt.ylim(0.0,0.003)
plt.tick_params(axis='x', colors='red')

plt.suptitle('RMSE performance for each model')
plt.show()
data = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv", index_col = "Serial No." )

data.head()

from sklearn.preprocessing import LabelEncoder

#train,valid,_ = split_data(data)

inter = pd.DataFrame(index=train.index)
var = pd.DataFrame(index=train.index)
# inter = pd.DataFrame(index=data.index)
# var = pd.DataFrame(index=data.index)

inter_ft = ['GRE Score', 'TOEFL Score', "CGPA"]
for each in inter_ft:

    nameVar = each+"_admitVar"
    var[nameVar]= fun(train,each)# train[each].apply(fun)
    train[nameVar] = var[nameVar]
    valid[nameVar] = var[nameVar]
    #data[nameVar] = var[nameVar]
    #train = train.join(var[nameVar].add_suffix(nameVar))
    
train.head()
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

xs = {"lab":"TOEFL Score_admitVar", 1:train["TOEFL Score_admitVar"]}
zs = {"lab":"TOEFL Score",1:train["TOEFL Score"]}
ys = {"lab":"Chance of Admit",1:train["Chance of Admit "]}

fig = px.scatter_3d(train, x=xs[1], y=ys[1], z=zs[1],
              color=zs[1])
fig.show()

#https://plot.ly/python/3d-scatter-plots/
fig,ax = plt.subplots(1,3,figsize=(10,10))
var = ["TOEFL Score_admitVar", "GRE Score_admitVar", "CGPA_admitVar"]
var2 = ["TOEFL Score", "GRE Score", "CGPA"]
for i in range(3):

    xs = {"lab":var[i],1:train[var[i]]}
    zs = {"lab":var2[i],1:train[var2[i]]}
    ys = {"lab":"Chance of Admit",1:train["Chance of Admit "]}
    fig = px.scatter_3d(train, x=xs[1], y=ys[1], z=zs[1],
              color=zs[1])
    
    fig.show()

var = ["TOEFL Score_admitVar", "GRE Score_admitVar", "CGPA_admitVar"]
for i in range(3):
    sns.jointplot(train[var[i]],train["Chance of Admit "],data=train, space=0, kind="kde")
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

fig = plt.figure(figsize=(8, 6))
#ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
xs = {"lab":"TOEFL Score_admitVar", 1:train["TOEFL Score_admitVar"]}
ys = {"lab":"TOEFL Score",1:train["TOEFL Score"]}
zs = {"lab":"Chance of Admit",1:train["Chance of Admit "]}

ax.scatter(xs[1], ys[1], zs[1], s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel(xs["lab"])
ax.set_ylabel(ys["lab"])
ax.set_zlabel(zs["lab"])

plt.show()

# https://plot.ly/python/3d-surface-plots/
# https://jovianlin.io/data-visualization-seaborn-part-3/
plt.scatter(x = train["TOEFL Score_admitVar"], 
            y = train['Chance of Admit '], 
            s = train['TOEFL Score']*0.525, # <== 😀 Look here!
            alpha=0.4, 
            edgecolors='w')

plt.xlabel('TOEFL Score_admitVar')
plt.ylabel('Chance of Admit ')
plt.title('TOEF Score_admVar, Chance of Admit, TOEFL Sore', y=1.05)

var = ["TOEFL Score_admitVar", "GRE Score_admitVar", "CGPA_admitVar"]
var2 = ["TOEFL Score", "GRE Score", "CGPA"]
i=1
xs = {"lab":var2[0],1:train[var2[0]]}
zs = {"lab":var2[1],1:train[var2[1]]}
ys = {"lab":"Chance of Admit",1:train[var2[2]]}

fig = px.scatter_3d(train, x=xs[1], y=ys[1], z=zs[1],
              color=train["Chance of Admit "])
    
fig.show()
plt.figure(figsize=(16,8))
val = data.loc[(data["CGPA"] > 8) & (data["CGPA"] < 10)]

sns.scatterplot(y= val["CGPA"], x=val["TOEFL Score"])
