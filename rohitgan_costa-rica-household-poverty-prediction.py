import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics
from collections import Counter
from plotly.offline import  iplot,init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)
"""
focus more on the EDA and feature engineering.
I used classic grid-search for hyper-parameter tuning 
"""
data=pd.read_csv("../input/costa-rican-household-poverty-prediction/train.csv")
test=pd.read_csv("../input/costa-rican-household-poverty-prediction/test.csv")

"""
mapping target variable to its meaning
"""
label_dict={1:"extreme poverty",2:"moderate poverty",3:"vulnerable households",4:"non-vulnerable",}


"""
dictionary for description of all the variables
"""
features=data.columns
info={}
x=open("../input/description/description.txt")

for x,line in enumerate(x):
    info[line.split(",",1)[0]]= line.split(",",1)[1].replace("\n","")
info    


len(test)
# area2(rural) is redundant because area1(urban) is already there
data=data.drop(columns=["area2","female"])
#info that might be important is if the person is a widow or not
data=data.drop(columns=["estadocivil1","estadocivil2","estadocivil3","estadocivil4","estadocivil5"])

# area2(rural) is redundant because area1(urban) is already there
test=test.drop(columns=["area2","female"])
#info that might be important is if the person is a widow or not
test=test.drop(columns=["estadocivil1","estadocivil2","estadocivil3","estadocivil4","estadocivil5"])
data["depend"] = (data["hogar_nin"]+data["hogar_mayor"]+1)/(1+data["hogar_adul"])
test["depend"] = (test["hogar_nin"]+test["hogar_mayor"]+1)/(1+test["hogar_adul"])
set(data["depend"])
data[["depend","Target"]].corr(method="spearman")
info["depend"]=info["dependency"]
data=data.drop(columns="dependency")
test=test.drop(columns="dependency")
id_ = ['Id', 'idhogar', 'Target']

ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 
            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 
            'instlevel9', 'mobilephone', 'rez_esc']

ind_ordered = ['rez_esc', 'escolari', 'age']


hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 
           'paredpreb','pisocemento', 'pareddes', 'paredmad',
           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 
           'pisonatur', 'pisonotiene', 'pisomadera',
           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 
           'abastaguadentro', 'abastaguafuera', 'abastaguano',
            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 
           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',
           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 
           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 
           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',
           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 
           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 
           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',
           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1']


hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 
              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',
              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']

hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
# Groupby the household and figure out the number of unique values
all_equal = data.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
households_leader = data.groupby('idhogar')['parentesco1'].sum()

# Find households without a head
households_no_head = households_leader[households_leader==0]

print('There are {} households without a head.'.format(len(households_leader[households_leader==0])))
# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(data[(data['idhogar'] == household) & (data['parentesco1'] == 1.0)]['Target'])
    
    # Set the correct label for all members in the household
    data.loc[data['idhogar'] == household, 'Target'] = true_target
    
    
# Groupby the household and figure out the number of unique values
all_equal = data.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))
data=data.drop(columns=[i for i in data.columns if i.startswith("SQ")==True]) #squared variables add no extra info
data=data.drop(columns=["agesq"]) #squared variables add no extra info
test=test.drop(columns=[i for i in test.columns if i.startswith("SQ")==True]) #squared variables add no extra info
test=test.drop(columns=["agesq"]) #squared variables add no extra info
null=data.isnull().any()
null[null==True]
data["v2a1"]=data["v2a1"].fillna(0)# For NaN, assuming guy owns house
data["v18q1"]=data["v18q1"].fillna(0)# For NaN, assuming 0 tablets

test["v2a1"]=test["v2a1"].fillna(0)# For NaN, assuming guy owns house
test["v18q1"]=test["v18q1"].fillna(0)# For NaN, assuming 0 tablets
data["Target"][data["meaneduc"].isnull()==True] 
#so we need to impute meaneducation with mode of meaneducation for people with poverty level=4
mode_edu=statistics.mode(data["meaneduc"][data["Target"]==4][data["meaneduc"].isnull()==False])
data["meaneduc"]=data["meaneduc"].fillna(mode_edu)#fill Nan with mode because 0 already exists
test["meaneduc"]=test["meaneduc"].fillna(mode_edu)#fill Nan with mode because 0 already exists

set(data["rez_esc"].fillna(0)) # dropping the variable. There are more informative education related variables
data=data.drop(columns=["rez_esc"])
test=test.drop(columns=["rez_esc"])
len(test)
features=["escolari","meaneduc","instlevel1","instlevel2","instlevel3","instlevel4","instlevel5","instlevel6",
          "instlevel7","instlevel8","instlevel9"]
# poverty level vs mean education

traces =[go.Box(
    y = data["meaneduc"][data["Target"]==i],
    name = label_dict[i],
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = False,
    marker = dict(
        color = 'rgb(7,40,89)'),
    line = dict(
        color = 'rgb(7,40,89)')
) for i in set(data["Target"])]

layout = go.Layout(
    title = "poverty level Vs mean education"
)

fig = go.Figure(data=traces,layout=layout)
iplot(fig)
encoding={i:int(i[-1]) for i in [j for j in features if "inst" in j]} # convert instlevel to an ordinal variable
insti=["instlevel1","instlevel2","instlevel3","instlevel4","instlevel5","instlevel6",
          "instlevel7","instlevel8","instlevel9"]

for i in range(len(data)):
    for j in insti:
        if data.loc[i,j]==1:
            data.loc[i,j]=encoding[j]
            
for i in range(len(test)):
    for j in insti:
        if test.loc[i,j]==1:
            test.loc[i,j]=encoding[j]
            
data["qualified"]=data[insti].sum(axis=1) 
test["qualified"]=test[insti].sum(axis=1) 
info["qualified"]="sum of all instlevels"
data=data.drop(columns=insti)
test=test.drop(columns=insti)
traces =[go.Box(
    y = data["qualified"][data["Target"]==i],
    name = label_dict[i],
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = False,
    marker = dict(
        color = 'rgb(7,40,89)'),
    line = dict(
        color = 'rgb(7,40,89)')
) for i in set(data["Target"])]

layout = go.Layout(
    title = "poverty level Vs highest qualification"
)

fig = go.Figure(data=traces,layout=layout)
iplot(fig)
avg_qual=data.groupby("idhogar")["qualified"].apply(lambda x: x.mean())
avg_qual_test=test.groupby("idhogar")["qualified"].apply(lambda x: x.mean())

info["avg_qual"]="average academic qualification level of a household"
data["avg_qual"]=""

for i in range(len(data)):
   data.loc[i,"avg_qual"]=avg_qual[data.loc[i,"idhogar"]]# create a new household level feature for average insti level
data["avg_qual"]=data["avg_qual"].astype(float)

for i in range(len(test)):
   test.loc[i,"avg_qual"]=avg_qual_test[test.loc[i,"idhogar"]]# create a new household level feature for average insti level
test["avg_qual"]=test["avg_qual"].astype(float)
data[["escolari","meaneduc","avg_qual","Target"]].corr(method="pearson")
data=data.drop(columns=["escolari","meaneduc"])
test=test.drop(columns=["escolari","meaneduc"])
len(test)
walls={"epared1":1,"epared2":2,"epared3":3}
roof={"etecho1":1,"etecho2":2,"etecho3":3}
floor={"eviv1":1,"eviv2":2,"eviv3":3}

data["walls"]=""
data["roof"]=""
data["floor"]=""

test["walls"]=""
test["roof"]=""
test["floor"]=""

for i in range(len(data)):
    for j in walls.keys():
        if data.loc[i,j]==1:
            data.loc[i,"walls"]=walls[j]
            
for i in range(len(data)):
    for j in roof.keys():
        if data.loc[i,j]==1:
            data.loc[i,"roof"]=roof[j]

for i in range(len(data)):
    for j in floor.keys():
        if data.loc[i,j]==1:
            data.loc[i,"floor"]=floor[j]

for i in range(len(test)):
    for j in walls.keys():
        if test.loc[i,j]==1:
            test.loc[i,"walls"]=walls[j]
            
for i in range(len(test)):
    for j in roof.keys():
        if test.loc[i,j]==1:
            test.loc[i,"roof"]=roof[j]

for i in range(len(test)):
    for j in floor.keys():
        if test.loc[i,j]==1:
            test.loc[i,"floor"]=floor[j]

data["housing"]=data["walls"]+data["floor"]+data["roof"]
test["housing"]=test["walls"]+test["floor"]+test["roof"]
data[["walls","floor","roof","housing","Target"]].corr(method="spearman")
data=data.drop(columns=["walls","floor","roof"])
test=test.drop(columns=["walls","floor","roof"])
features1=["mobilephone","computer","television","refrig","v18q"]
data["equipped"] = data[features1].sum(axis=1) 
test["equipped"] = test[features1].sum(axis=1) 

info["equipped"]="how tech savvy a family is"
# created a new feature called equipped that indicates how tech savvy a household is
# poverty level vs how equipped a household is

traces =[go.Box(
    y = data["Target"][data["equipped"]==i],
    name = "equipped level"+str(i),
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'suspectedoutliers',
    marker = dict(
        color = 'rgb(7,40,89)'),
    line = dict(
        color = 'rgb(7,40,89)')
) for i in set(data["Target"])]

layout = go.Layout(
    title = "poverty level Vs how equipped a household is"
)

fig = go.Figure(data=traces,layout=layout)
iplot(fig)
data[["equipped","v18q1","qmobilephone","Target"]].corr(method="spearman")
data=data.drop(columns=features1)
test=test.drop(columns=features1)
data["tab_cap"]= (data["v18q1"]/data["hogar_total"]).fillna(0)
data["phone_cap"]= (data["qmobilephone"]/data["hogar_total"]).fillna(0)
data["tech_cap"]=data["tab_cap"]+data["phone_cap"]


test["tab_cap"]= (test["v18q1"]/test["hogar_total"]).fillna(0)
test["phone_cap"]= (test["qmobilephone"]/test["hogar_total"]).fillna(0)
test["tech_cap"]=test["tab_cap"]+test["phone_cap"]
data[["tab_cap","phone_cap","tech_cap","qmobilephone","v18q1","Target"]].corr()
data=data.drop(columns=["tab_cap","phone_cap","qmobilephone","v18q1"])
test=test.drop(columns=["tab_cap","phone_cap","qmobilephone","v18q1"])
data[["tamhog","tamviv","hhsize","hogar_total","Target"]].corr()
data=data.drop(columns=["tamhog","tamviv","hhsize"])
test=test.drop(columns=["tamhog","tamviv","hhsize"])
features=["bedrooms","overcrowding","hacdor","rooms","hacapo"]
data[features+["Target"]].corr()
data=data.drop(columns=["hacapo","hacdor","bedrooms"])
test=test.drop(columns=["hacapo","hacdor","bedrooms"])
features=['tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4',"tipovivi5","v2a1"]
#plot status of non-rent payers 

x=[info[i].split(" ",2)[2] for i in features[:-1]]
y=[Counter(data[data["v2a1"]==0][i])[1] for i in features[:-1]]

dat=[go.Bar( x=x,y=y,name="housing")]
layout = go.Layout(title='distribution of zero-rent payers ',xaxis=dict(title='status'),yaxis=dict(title='number'),
                  titlefont=dict(family='Courier New, monospace',size=25,color='#7f7f7f'))
fig = go.Figure(data=dat, layout=layout)
iplot(fig)
# poverty condition of people who fully own their homes

y=Counter(data["Target"][data["tipovivi1"]==1])
x= [label_dict[i] for i in y.keys()]

dat=[go.Bar( x=x,y=list(y.values()),name="fully paid house")]
layout = go.Layout(title='poverty distribution of people who fully own their house',xaxis=dict(title='poverty level'),yaxis=dict(title='number'),
                  titlefont=dict(family='Courier New, monospace',size=25,color='#7f7f7f'))
fig = go.Figure(data=dat, layout=layout)
iplot(fig)
# poverty distribution of rent-payers

y=Counter(data["Target"][data["tipovivi3"]==1])
x= [label_dict[i] for i in y.keys()]

dat=[go.Bar( x=x,y=list(y.values()),name="rent payers")]
layout = go.Layout(title='poverty distribution of rent payers',xaxis=dict(title='poverty level'),yaxis=dict(title='number'),
                  titlefont=dict(family='Courier New, monospace',size=25,color='#7f7f7f'))
fig = go.Figure(data=dat, layout=layout)
iplot(fig)
data=data.drop(columns="v2a1")
test=test.drop(columns="v2a1")
data[["r4h1","r4h2","r4m1","r4m2","r4m3","r4t1","r4t2","r4t3","Target"]].corr(method="spearman")
info["r4t1"] #more babies more poverty
data=data.drop(columns=["r4h1","r4h2","r4m1","r4m2","r4m3","r4t2","r4t3"])
test=test.drop(columns=["r4h1","r4h2","r4m1","r4m2","r4m3","r4t2","r4t3"])
# poverty level vs locality

traces =[go.Box(
    y = data["Target"][data[i]==1],
    name = info[i],
    jitter = 0.3,
    pointpos = -1.8,
    boxpoints = 'suspectedoutliers',
    marker = dict(
        color = 'rgb(7,40,89)'),
    line = dict(
        color = 'rgb(7,40,89)')
) for i in ["lugar1","lugar2","lugar3","lugar4","lugar5","lugar6"] ]

layout = go.Layout(
    title = "poverty level Vs location"
)

fig = go.Figure(data=traces,layout=layout)
iplot(fig)
data=data.drop(columns=["lugar1","lugar2","lugar3","lugar4","lugar5","lugar6"])
test=test.drop(columns=["lugar1","lugar2","lugar3","lugar4","lugar5","lugar6"])
x=Counter(data["Target"])

trace1=[go.Bar( x=["level "+str(i) for i in list(x.keys())],y=list(x.values()) ,name="class imbalancee")]



layout = go.Layout(title='class imbalance',xaxis=dict(title='poverty level'),yaxis=dict(title='number'),
                  titlefont=dict(family='Courier New, monospace',size=25,color='#7f7f7f'))
fig = go.Figure(data=trace1, layout=layout)
iplot(fig)
    

data=data[data["parentesco1"]==1]
len(test)
from imblearn.over_sampling import RandomOverSampler
y=data["Target"]
x=data.drop(columns=["Target","Id"])
test_labels=test["Id"]
test=test.drop(columns=["Id"])

x=x.drop(columns=["edjefe","edjefa"])
test=test.drop(columns=["edjefe","edjefa"])

x=x.drop(columns=["hogar_nin","hogar_adul","hogar_mayor"])
test=test.drop(columns=["hogar_nin","hogar_adul","hogar_mayor"])

x=x.drop(columns=["idhogar"])
test=test.drop(columns=["idhogar"])

ros = RandomOverSampler(random_state=0)
x_res, y_res = ros.fit_resample(x, y)

Counter(y_res) # imbalance has been gotten rid of!
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
scorer2 = make_scorer(accuracy_score, greater_is_better=True)
pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 
                      ('scaler', MinMaxScaler())])
len(test)
x_res=pipeline.fit_transform(x_res)
test=pipeline.transform(test)
model = RandomForestClassifier(n_estimators=500, random_state=10, n_jobs = -1)
cv_score = cross_val_score(model, x_res, y_res, cv = 10, scoring = scorer)
cv2_score = cross_val_score(model, x_res, y_res, cv = 10, scoring = scorer2)
rf_model = RandomForestClassifier(n_estimators=500, random_state=10, n_jobs = -1)
rf_trained=rf_model.fit(x_res,y_res)
predicted=rf_model.predict(test)
subm=pd.DataFrame()
subm["Id"]=test_labels
subm["Target"]=predicted
subm.to_csv("results.csv",index=False)

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV as grid_search
xg_model=XGBClassifier(gamma=0.1,max_depth=5,objective='multi:softmax',n_estimators=1200)
xg_model
param_grid={"max_depth":[3,4,5,6],"gamma":[0.001,0.01,0.1,0.2,0,3],"n_estimators":[100,150,200,250,300,350,400,450,500,550,600,650], "objective":['multi:softmax']}
#grid_xg=grid_search(XGBClassifier(), param_grid, scoring=scorer2,cv=6)
#grid_xg.fit(x_res, y_res)
#cv_score_xg = cross_val_score(xg_model, x_res, y_res, cv = 10, scoring = scorer)
#cv2_score_xg = cross_val_score(xg_model, x_res, y_res, cv = 10, scoring = scorer2)
xg_model.fit(x_res,y_res)

subm=pd.DataFrame()
subm["Id"]=test_labels
subm["Target"]=xg_model.predict(test)
subm.to_csv("results.csv",index=False)

