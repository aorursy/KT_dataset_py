# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from plotly import tools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
print(train.shape)
train.head()
test = pd.read_csv("../input/test.csv")
print(test.shape)
test.head()

missing = train.isnull().sum().to_frame().sort_values(0, ascending = False)
missing.head()
def plot_pie(col,name):
    levels = train[col].unique()
    values = []
    labels = []
    for level in levels:
        labels.append(level)
        temp = train.loc[train[col] == level]
        val = temp.Target.value_counts().sum()
        values.append(val)
        
    
    fig = {
      "data": [
        {
          "values": values,
          "labels": labels,
#           "domain": {"x": [0,0]},
          "name": name,
          "hoverinfo":"label+percent+name",
          "hole": .4,
          "type": "pie"
        }],
      "layout": {
            "title":name,
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": col,
                    "x": 0.5,
                    "y": 0.5
                }
            ]
        }
    }
    iplot(fig, filename='donut')
#     return values,labels
plot_pie("Target","Levels of Poverty")
train.head()
# train.drop(["r4m3","r4h3","tamhog","tamviv","hhsize","r4t1","r4t2","r4t3","SQBescolari","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq"],axis=1,inplace=True)
# train.shape
# test.drop(["r4m3","r4h3","tamhog","tamviv","hhsize","r4t1","r4t2","r4t3","SQBage","SQBhogar_total","SQBedjefe","SQBhogar_nin","SQBovercrowding","SQBdependency","SQBmeaned","agesq"],axis=1,inplace=True)
# test.shape
extreme = train.loc[train.Target == 1]
moderate = train.loc[train.Target == 2]
vulnerable = train.loc[train.Target == 3]
non_vulnerable = train.loc[train.Target == 4]

def get_data(col):
    extremes = extreme[col].value_counts().to_dict()
    moderates = moderate[col].value_counts().to_dict()
    vulnerables = vulnerable[col].value_counts().to_dict()
    non_vulnerables = non_vulnerable[col].value_counts().to_dict()
    y = [extremes[0],moderates[0],vulnerables[0],non_vulnerables[0]]
    y2 = [extremes[1],moderates[1],vulnerables[1],non_vulnerables[1]]
    return y,y2

def plot_bar(x,y,y2,name1,name2):
    
    trace1 = go.Bar(
        x=x,
        y=y,
        text=y,
        textposition = 'auto',
        name=name1,
        marker=dict(
            color='rgb(158,202,225)',
            line=dict(
                color='rgb(8,48,107)',
                width=1.5),
            ),
        opacity=0.6
    )

    trace2 = go.Bar(
        x=x,
        y=y2,
        text=y2,
        name = name2,
        textposition = 'auto',
        marker=dict(
            color='rgb(58,200,225)',
            line=dict(
                color='rgb(8,48,107)',
                width=1.5),
            ),
        opacity=0.8
    )

    data = [trace1,trace2]

    iplot(data, filename='grouped-bar-direct-labels')
    

x = ['extreme poverty ', 'moderate poverty ', 'vulnerable households ','non-vulnerable households']
y ,y2= get_data("male")  #
plot_bar(x,y,y2,"female","male")
y ,y2= get_data("dis")  #
plot_bar(x,y,y2,"Not Disabled","Disabled")

def single_bar(y,name):
    trace0 = go.Bar(
    x=x,
    y=y,
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
        opacity=0.6
    )

#     data = [trace0]
#     layout = go.Layout(
#         title=name,
#     )

#     fig = go.Figure(data=data)#, layout=layout)
    return trace0
    #     py.iplot(fig, filename='text-hover-bar')
    
# edu_feats = ["instlevel1","instlevel2","instlevel3","instlevel4","instlevel5","instlevel6","instlevel7","instlevel8"]#,"instlevel9"]
# figs = []
# i = 0
# for feat in edu_feats:
#         print(i)
#         y,y1 = get_data(feat)
#         fig = single_bar(y,feat)
#         figs.append(fig)
#         i = i +1
def pair_plots(col):
    y = extreme[col].value_counts().to_dict()
    y1 = moderate[col].value_counts().to_dict()
    y2 = vulnerable[col].value_counts().to_dict()
    y3 = non_vulnerable[col].value_counts().to_dict()
    
    trace1 = go.Bar(y=[y[0], y1[0], y2[0], y3[0]],
                    name="Does Not Own", 
                    x=x,
                    marker=dict(
                        color="rgb(158,202,225)",
                        opacity=0.6))
    trace2 = go.Bar(y=[y[1], y1[1], y2[1], y3[1]],
                    name="Owns",
                    x=x, 
                    marker=dict(
                        color="rgb(58,200,225)",
                        opacity=0.6))
    
    return trace1, trace2 
    
trace1, trace2 = pair_plots("v18q")
trace3, trace4 = pair_plots("refrig")
trace5, trace6 = pair_plots("computer")
trace7, trace8 = pair_plots("television")
trace9, trace10 = pair_plots("mobilephone")
titles = ["Tablet", "Refrigirator", "Computer", "Television", "MobilePhone"]

fig = tools.make_subplots(rows=3, cols=2, print_grid=False, subplot_titles=titles)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)
fig.append_trace(trace4, 1, 2)
fig.append_trace(trace5, 2, 1)
fig.append_trace(trace6, 2, 1)
fig.append_trace(trace7, 2, 2)
fig.append_trace(trace8, 2, 2)
fig.append_trace(trace9, 3, 1)
fig.append_trace(trace10, 3, 1)

fig['layout'].update(height=1000, title="Ammenities for each division", showlegend=False)
iplot(fig)
import matplotlib_venn as venn
# plt.subplot(121)

def return_sets(x,col):
    temp = x.loc[x[col]==1]
    idss = []
    for ids in temp.Id:
        idss.append(ids)
    return idss

def make_sets(computer,tv,tablet):
    computer=list(set([a for a in computer]))
    tv=list(set([a for a in tv]))
    tablet= list(set([a for a in tablet]))

    comp_tv=list(set(computer).intersection(tv))
    comp_tablet=list(set(computer).intersection(tablet))
    tv_tablet=list(set(tv).intersection(tablet))

    all_intersect = list(set(computer).intersection(tv).intersection(tablet))
    return computer,tv,tablet,comp_tv,comp_tablet,tv_tablet,all_intersect

def make_venn3(set1,set2,set3,l1,l2,l3):
    
    plt.figure(figsize=(22,12))
    plt.subplot(141)

    computer = return_sets(extreme,set1)
    tv = return_sets(extreme,set2)
    tablet = return_sets(extreme,set3)
    computer,tv,tablet,comp_tv,comp_tablet,tv_tablet,all_intersect = make_sets(computer,tv,tablet)
    v = venn.venn3(subsets=(len(computer),len(tv),len(comp_tv),len(tablet),len(comp_tablet),len(tv_tablet),len(all_intersect)),set_labels=(l1,l2,l3))
    plt.title("Ammenities of extreme")
    plt.subplot(142)


    computer = return_sets(moderate,set1)
    tv = return_sets(moderate,set2)
    tablet = return_sets(moderate,set3)
    computer,tv,tablet,comp_tv,comp_tablet,tv_tablet,all_intersect = make_sets(computer,tv,tablet)
    v = venn.venn3(subsets=(len(computer),len(tv),len(comp_tv),len(tablet),len(comp_tablet),len(tv_tablet),len(all_intersect)),set_labels=(l1,l2,l3))
    plt.title("Ammenities of Moderate")
    plt.subplot(143)

    computer = return_sets(vulnerable,set1)
    tv = return_sets(vulnerable,set2)
    tablet = return_sets(vulnerable,set3)
    computer,tv,tablet,comp_tv,comp_tablet,tv_tablet,all_intersect = make_sets(computer,tv,tablet)
    v = venn.venn3(subsets=(len(computer),len(tv),len(comp_tv),len(tablet),len(comp_tablet),len(tv_tablet),len(all_intersect)),set_labels=(l1,l2,l3))
    plt.title("Ammenities of Vulnerable")
    plt.subplot(144)

    computer = return_sets(non_vulnerable,set1)
    tv = return_sets(non_vulnerable,set2)
    tablet = return_sets(non_vulnerable,set3)
    computer,tv,tablet,comp_tv,comp_tablet,tv_tablet,all_intersect = make_sets(computer,tv,tablet)
    v = venn.venn3(subsets=(len(computer),len(tv),len(comp_tv),len(tablet),len(comp_tablet),len(tv_tablet),len(all_intersect)),set_labels=(l1,l2,l3))
    plt.title("Ammenities of non_vulnerable")

    plt.show()
    print(extreme.shape) 
    print(moderate.shape)
    print(vulnerable.shape) 
    print(non_vulnerable.shape)

make_venn3("refrig","television","mobilephone","Fridge","TV","Mobile")
make_venn3("refrig","computer","mobilephone","Fridge","Computer","Mobile")

make_venn3("computer","television","v18q","Computer","TV","Tablet")
def target_name(x):
    if x == 1:
        return "Extreme"
    if x == 2:
        return "Moderate"
    if x == 3:
        return "Vulnerable"
    if x == 4:
        return "Non Vulnerable"
train["Target_name"]= train.Target.apply(lambda x: target_name(x))
train["tablets_total"] = train.apply(lambda x:x.v18q1/x.hogar_total,axis=1)
train["phones_total"] = train.apply(lambda x:x.qmobilephone/x.hogar_total,axis=1)
train.drop(["v18q1","qmobilephone"],axis=1,inplace=True)

test["tablets_total"] = test.apply(lambda x:x.v18q1/x.hogar_total,axis=1)
test["phones_total"] = test.apply(lambda x:x.qmobilephone/x.hogar_total,axis=1)
test.drop(["v18q1","qmobilephone"],axis=1,inplace=True)

train.head()
extreme = train.loc[train.Target == 1]
moderate = train.loc[train.Target == 2]
vulnerable = train.loc[train.Target == 3]
non_vulnerable = train.loc[train.Target == 4]

def plot_violin(col,name):
    data = []
    for i in range(0,len(pd.unique(train['Target']))):
        trace = {
                "type": 'violin',
                "x": train['Target'][train['Target'] == pd.unique(train['Target'])[i]],
                "y": train[col][train['Target'] == pd.unique(train['Target'])[i]],
                "name": pd.unique(train['Target_name'])[i],
                "box": {
                    "visible": True
                },
                "meanline": {
                    "visible": True
                }
            }
        data.append(trace)


    fig = {
        "data": data,
        "layout" : {
            "title": name,
            "yaxis": {
                "zeroline": False,
            }
        }
    }


    iplot(fig, filename='violin/multiple', validate = False)
plot_violin("tablets_total","tablets vs houshold_members")
plot_violin("phones_total","Phones vs Household members")
def get_data1(col):
    extremes = extreme.loc[extreme[col] ==1]
    moderates = moderate.loc[moderate[col] ==1]
    vulnerables = vulnerable.loc[vulnerable[col] ==1]
    non_vulnerables = non_vulnerable.loc[non_vulnerable[col] ==1]
    extremes = extremes[col].value_counts().to_dict()
    moderates = moderates[col].value_counts().to_dict()
    vulnerables = vulnerables[col].value_counts().to_dict()
    non_vulnerables = non_vulnerables[col].value_counts().to_dict()
    y = [extremes[0],moderates[0],vulnerables[0],non_vulnerables[0]]
    y2 = [extremes[1],moderates[1],vulnerables[1],non_vulnerables[1]]
    return y,y2

edu_feats = ["abastaguano","sanitario1","energcocinar1","epared1","etecho1","eviv1"]#,"instlevel9"]
figs = []
i = 0
for feat in edu_feats:
        print(i)
        y,y2 = get_data(feat)
        fig = single_bar(y2,feat)
        figs.append(fig)
        i = i +1

titles = ["Water Supply", "Sanitation", "Fuel Energy", "Walls","Roof","Floor"]

fig = tools.make_subplots(rows=3, cols=2, print_grid=False, subplot_titles=titles)
fig.append_trace(figs[0], 1, 1)
fig.append_trace(figs[1], 1, 2)
fig.append_trace(figs[2], 2, 1)
fig.append_trace(figs[3], 2, 2)
fig.append_trace(figs[4], 3, 1)
fig.append_trace(figs[5], 3, 2)
# fig.append_trace(tr9, 3, 1)
# fig.append_trace(tr10, 3, 1)

fig['layout'].update(height=1000, title="Available Facilities", showlegend=False)
iplot(fig)
x = ['extreme poverty ', 'moderate poverty ', 'vulnerable households ','non-vulnerable households']
y ,y2= get_data("area1")  #
plot_bar(x,y,y2,"Urban","Rural")
plot_violin("v2a1","Monthly Rent")
plot_violin("dependency","Dependancy")
plot_violin("meaneduc","Mean Years of Education of all Adults in a Household")
plot_violin("edjefe","Years of Education of Male Head of Household")
plot_violin("edjefa","Years of Education of Female Head of Household")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train.Target = le.fit_transform(train.Target_name)
train.drop(["Target_name"],axis=1,inplace=True)
# train.to_csv("train.csv",index=False)
# test.to_csv("test.csv",index=False)
depend = []
for dependency, children, olds, total in zip(train['dependency'], train['hogar_nin'], train['hogar_mayor'], train['hogar_total']):
    calc_depend = False
    if depend != depend:
        calc_depend = True
    elif (dependency == "yes" or dependency == "no"):
        calc_depend = True

    #Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
    if calc_depend:
        i = (children + olds) / (total - children - olds)
    else:
        i = float(dependency)

    depend += [i]

train['dependency'] = depend

chw = []
for nin, adul in zip(train['hogar_nin'], train['hogar_adul']):
    if adul == 0:
        chw += [nin * 2]
    else:
        chw += [nin / adul]

train['child_weight'] = (train['hogar_nin'] + train['hogar_mayor']) / train['hogar_total']
train['child_weight2'] = chw
train['child_weight3'] = train['r4t1'] / train['r4t3']
train['work_power'] = train['dependency'] * train['hogar_adul']
train['SQBworker'] = train['hogar_adul'] ** 2
train['rooms_per_person'] = train['rooms'] / (train['tamviv'])
train['bedrooms_per_room'] = train['bedrooms'] / train['rooms']
train['female_weight'] = train['r4m3'] / train['r4t3']


depend = []
for dependency, children, olds, total in zip(test['dependency'], test['hogar_nin'], test['hogar_mayor'], test['hogar_total']):
    calc_depend = False
    if depend != depend:
        calc_depend = True
    elif (dependency == "yes" or dependency == "no"):
        calc_depend = True

    #Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
    if calc_depend:
        i = (children + olds) / (total - children - olds)
    else:
        i = float(dependency)

    depend += [i]

test['dependency'] = depend

chw = []
for nin, adul in zip(test['hogar_nin'], test['hogar_adul']):
    if adul == 0:
        chw += [nin * 2]
    else:
        chw += [nin / adul]




test['child_weight'] = (test['hogar_nin'] + test['hogar_mayor']) / test['hogar_total']
test['child_weight2'] = chw
test['child_weight3'] = test['r4t1'] / test['r4t3']
test['work_power'] = test['dependency'] * test['hogar_adul']
test['SQBworker'] = test['hogar_adul'] ** 2
test['rooms_per_person'] = test['rooms'] / (test['tamviv'])
test['bedrooms_per_room'] = test['bedrooms'] / test['rooms']
test['female_weight'] = test['r4m3'] / test['r4t3']
categorical_feats = [
    f for f in train.columns if len(train[f].value_counts()) == 2
]


train_cat = train[categorical_feats]
train_cat = train_cat.fillna('XNA')
print(train_cat.shape)
train_cat.head()

from contextlib import contextmanager
import time

timer_depth = -1
@contextmanager
def timer(name):
    t0 = time.time()
    global timer_depth
    timer_depth += 1
    yield
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print('----'*timer_depth + f'>>[{name}] done in {time.time() - t0:.0f} s ---> memory used: {memoryUse:.4f} GB', '')
    if(timer_depth == 0):
        print('\n')
    timer_depth -= 1


import psutil

def cal_woe(app_train, app_train_target):
    num_events = app_train_target.sum()
    num_non_events = app_train_target.shape[0] - app_train_target.sum()

    feature_list = []
    feature_iv_list = []
    for col in app_train.columns:
        if app_train[col].unique().shape[0] == 1:
            del app_train[col]
            print('remove constant col', col)

        with timer('cope with %s' % col):
            feature_list.append(col)

            woe_df = pd.DataFrame()
            woe_df[col] = app_train[col]
            woe_df['target'] = app_train_target
            events_df = woe_df.groupby(col)['target'].sum().reset_index().rename(columns={'target' : 'events'})
            events_df['non_events'] = woe_df.groupby(col).count().reset_index()['target'] - events_df['events']
            def cal_woe(x):
                return np.log( ((x['non_events']+0.5)/num_non_events) / ((x['events']+0.5)/num_events)  )
            events_df['WOE_'+col] = events_df.apply(cal_woe, axis=1)

            def cal_iv(x):
                return x['WOE_'+col]*(x['non_events'] / num_non_events - x['events'] / num_events)
            events_df['IV_'+col] = events_df.apply(cal_iv, axis=1)

            feature_iv = events_df['IV_'+col].sum()
            feature_iv_list.append(feature_iv)

            events_df = events_df.drop(['events', 'non_events', 'IV_'+col], axis=1)
            app_train = app_train.merge(events_df, how='left', on=col)
    iv_df = pd.DataFrame()
    iv_df['feature'] = feature_list
    iv_df['IV'] = feature_iv_list
    iv_df = iv_df.sort_values(by='IV', ascending=False)
    return app_train, iv_df
train_cat_target = train.Target
with timer('calculate WOE and IV'):
    train_cat, iv_df = cal_woe(train_cat, train_cat_target)
selected_cats =iv_df.loc[iv_df['IV']>0.001]
selected_cats = selected_cats.feature.tolist()
iv_df = iv_df.feature.tolist()
excluded_feats = ["Target"]
Id = test.Id
y = train.Target
features = [f_ for f_ in train.columns if f_ not in excluded_feats]
features = [f_ for f_ in features if f_ not in iv_df ]
for f in selected_cats:
    features.append(f)
len(features)
train = train[features]

train.head()
test = test[features]
test.head()

merged = pd.concat([train,test])
merged.shape
merged.idhogar = le.fit_transform(merged.idhogar)
merged.Id = le.fit_transform(merged.Id)
merged.dependency = le.fit_transform(merged.dependency)
merged.edjefe = le.fit_transform(merged.edjefe)
merged.edjefa = le.fit_transform(merged.edjefa)
train = merged.iloc[:train.shape[0],:]
test = merged.iloc[train.shape[0]:,:]
print(test.shape)
train.shape

train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)
# news = pd.read_csv("newFinal.csv")
# news