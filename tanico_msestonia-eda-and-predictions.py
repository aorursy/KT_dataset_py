import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import copy
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.Survived = df.Survived.apply(lambda x: 'Survived' if x==1 else "NotSurvived")
df_survived =(df.groupby('Survived')[['PassengerId']].count()*100/df.shape[0]).reset_index()
df_survived.columns=['Survived','Survived_Ratio']
fig = px.bar(df_survived,x='Survived', y='Survived_Ratio',color ='Survived',title='Survived Ratio(ALL)')
fig.show()

ls_survived = ['Survived','NotSurvived']
fig = make_subplots(rows=1, cols=2, subplot_titles=('#Passenger','Survived Rate[%]'))
fig.add_trace(
    go.Histogram(
        x=df['Age'], 
        xbins=dict(start=df.Age.min(),end=df.Age.max(),size=10),
        hovertemplate="Age:%{x}, Count:%{y}", name="ALL"),
    row=1, col=1)

for is_survive in ls_survived :
    fig.add_trace(
        go.Histogram(
            x=df.loc[df.Survived==is_survive,'Age'], 
            xbins=dict(start=df.Age.min(),end=df.Age.max(),size=10),
            hovertemplate="Age:%{x}, Count:%{y}",
            name=is_survive),
        row=1, col=2)
fig.update_xaxes(title_text='Age')
fig.update_layout(barmode='stack',bargap=0.2, title_text="Age")
fig.show()
ls_age_bins = [-0.1, 20, 30, 40, 50, 60, 70, 100 ]
df['AgeBin'] = pd.cut(df.Age,bins =ls_age_bins, labels=["<{0}".format(ls_age_bins[i+1]) for i,_ in enumerate(ls_age_bins[:-1])] )
df_ageBin = df.groupby(['AgeBin',"Survived"])[["PassengerId"]].count().fillna(0)
df_ageBin_ratio = (100*df_ageBin/df_ageBin.sum(level="AgeBin"))#.reset_index()#

fig = make_subplots(rows=1, cols=1)
df_ageBin_positive = df_ageBin_ratio.xs('Survived', level='Survived')
fig.add_trace(
    go.Bar(
        x=df_ageBin_positive.index, y=df_ageBin_positive.loc[:,'PassengerId'],
        hovertemplate="AgeBin:%{x}, Count:%{y}", name="ALL"),
    row=1, col=1)
fig.update_xaxes(title_text='AgeBin')
fig.update_yaxes(title_text='Survived Rate[%]')
fig.update_layout(title_text="Age : Survived Rate")
ls_sex = ['F','M']
df_sex = df.groupby(['Survived','Sex'])[['PassengerId']].count()
df_sex_ratio = 100*df_sex/df_sex.sum(level='Sex')
fig = make_subplots(rows=1, cols=2, subplot_titles=('Sex by Survived','Sex by Survived(Ratio)'))
for is_survive in ls_survived:
    fig.add_trace(
        go.Bar(x=ls_sex,y=df_sex.loc[is_survive,'PassengerId'],name=is_survive ),
        row=1, col=1)
for is_survive in ls_survived:
    fig.add_trace(
        go.Bar(x=ls_sex,y=df_sex_ratio.loc[is_survive,'PassengerId'],name=is_survive+'(Ratio)' ),
        row=1, col=2)
fig.update_layout(barmode='stack')   
fig.show()
fig = px.box(df, x="Sex", y="Age", color="Survived")
fig.update_traces() # or "inclusive", or "linear" by default
fig.show()
fig = make_subplots(rows=1, cols=2, subplot_titles=('Female',"Male"))
for i, _sex in enumerate(ls_sex):
    for is_survive in ls_survived:
        fig.add_trace(
            go.Histogram(
                x=df.loc[(df.Survived==is_survive)&(df.Sex==_sex),'Age'], 
                xbins=dict(start=df.Age.min(),end=df.Age.max(),size=5),
                hovertemplate="Age:%{x}, Count:%{y}",
                name=is_survive),
            row=1, col=i+1)

fig.update_xaxes(title_text='Age')
fig.update_yaxes(title_text='Count')
fig.show()
df['AgeBin'] = pd.cut(df.Age,bins =ls_age_bins, labels=["<{0}".format(ls_age_bins[i+1]) for i,_ in enumerate(ls_age_bins[:-1])] )
df_ageBin = df.groupby(['Sex',"AgeBin","Survived"])[["PassengerId"]].count().fillna(0)
df_ageBin_ratio = (100*df_ageBin/df_ageBin.sum(level=['Sex',"AgeBin",]))#.reset_index()#

df_ageBin_positive = df_ageBin_ratio.xs('Survived', level='Survived')

fig = make_subplots(subplot_titles=['Survived Rate'])
for i, _sex in enumerate(ls_sex):
    df_tmp = df_ageBin_positive.loc[_sex]
    fig.add_trace(
        go.Bar(
            x=df_tmp.index, y=df_tmp['PassengerId'],
            hovertemplate="Age:%{x}, Ratio:%{y}",
            name=_sex),
        row=1, col=1)

fig.update_xaxes(title_text='Age')
fig.update_yaxes(title_text='Survived Rate[%]')
fig.show()
df_cn = df.pivot_table(index='Country', columns='Survived',values='PassengerId', aggfunc=len, margins=True).fillna(0)
df_cn = df_cn.sort_values(by='All')
ls_cn = list(df_cn.index)
ls_cn.remove('All')

fig = make_subplots(rows=1, cols=2, subplot_titles=('#Passenger(Passenger <100)','#Passenger(Passenger >=100 (incl. ALL))'))
for is_survived in ls_survived:
    df_tmp = df_cn.loc[df_cn['All']<100,is_survived]
    fig.add_trace(
            go.Bar(x=df_tmp.index, y=df_tmp.values, name=is_survived),
        row=1, col=1)
for is_survived in ls_survived:
    df_tmp = df_cn.loc[df_cn['All']>=100,is_survived]
    fig.add_trace(
            go.Bar(x=df_tmp.index, y=df_tmp.values, name=is_survived),
        row=1, col=2)

fig.update_layout(barmode='stack')
fig.update_yaxes(title_text='#Passenger')
fig.show()
fig = make_subplots(rows=1, cols=2, subplot_titles=['Survived Rate( order by #Passenger)'])
df_cn_ratio = copy.copy(df_cn)
for is_survived in ls_survived:
    df_cn_ratio[is_survived] = 100*df_cn_ratio[is_survived]/df_cn_ratio['All']
del df_cn_ratio['All']  

df_tmp = df_cn_ratio.loc[df_cn['All']<100,"Survived"]
fig.add_trace(
        go.Bar(x=df_tmp.index, y=df_tmp.values, name="Survived"),
    row=1, col=1)

fig.update_layout(barmode='stack')
fig.update_yaxes(title_text='Survived Rate[%]')
ls_country = list(set(df['Country']))
ls_country_agg = ["Estonia","Sweden","Others"]
df['CountryOpt'] =   "Others"
df.loc[df["Country"].isin(ls_country_agg), "CountryOpt"] = df.loc[df["Country"].isin(ls_country_agg), "Country"] 
df_cn = df.groupby(['Survived','CountryOpt'])[['PassengerId']].count().fillna(0)
df_cn_ratio = 100*df_cn/df_cn.sum(level='CountryOpt')

fig = make_subplots(rows=1, cols=1, subplot_titles=['Optimized Nationality'])
for is_survived in ls_survived:
    df_tmp = df_cn.loc[is_survived,'PassengerId']
    fig.add_trace(
            go.Bar(x=df_tmp.index, y=df_tmp.values, name=is_survived),
        row=1, col=1)
fig.update_layout(barmode='stack')   
fig.update_yaxes(title_text='#Passenger')
fig.show()
fig = make_subplots(rows=1, cols=1, subplot_titles=['Optimized Nationality'])
df_tmp = df_cn_ratio.loc["Survived",'PassengerId']
fig.add_trace(
        go.Bar(x=df_tmp.index, y=df_tmp.values, name="Survived Rate"),
    row=1, col=1)
fig.update_yaxes(title_text='Survived Rate[%]')
fig.show()

df_cn = df.groupby(['Survived','Category'])[['PassengerId']].count().fillna(0)
df_cn_ratio = 100*df_cn/df_cn.sum(level='Category')

fig = make_subplots(rows=1, cols=1, subplot_titles=['Crew vs Passenger'])
for is_survived in ls_survived:
    df_tmp = df_cn.loc[is_survived,'PassengerId']
    fig.add_trace(
            go.Bar(x=df_tmp.index, y=df_tmp.values, name=is_survived),
        row=1, col=1)
fig.update_layout(barmode='stack')   
fig.update_yaxes(title_text='#Passenger')
fig.show()
fig = make_subplots(rows=1, cols=1, subplot_titles=['Crew vs Passenger'])
df_tmp = df_cn_ratio.loc["Survived",'PassengerId']
fig.add_trace(
        go.Bar(x=df_tmp.index, y=df_tmp.values, name="Survived Rate"),
    row=1, col=1)
fig.update_yaxes(title_text='Survived Rate[%]')
fig.show()

for target_col in ["CountryOpt", "Sex"]:
    df_tmp = df.pivot_table(index=['Category',target_col], columns='Survived',values='PassengerId', aggfunc=len, margins=True).fillna(0)
    for is_survived in ls_survived:
        df_tmp[is_survived]=  100*df_tmp[is_survived]/df_tmp['All']

    ls_cat =  ['C','P']
    ls_target_cat = df_tmp.xs('C', level=0).index
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Crew vs Passenger : Survival Rate in  '+target_col ])
    for cat in ls_cat:
        fig.add_trace(
                go.Bar(x=ls_target_cat, y=df_tmp.loc[cat,"Survived"], name=cat),
            row=1, col=1)
    fig.show()
df["FirstnameCleaned"] = df.Firstname.str.split(" ", expand=True).loc[:,0].str.upper()
df["LastnameCleaned"] =   df.Lastname.str.split(" ", expand=True).loc[:,0].str.upper()

# plt.plot( pd.DataFrame(df.FirstnameCleaned.value_counts()).iloc[1:,:])

df_name = pd.DataFrame(df.LastnameCleaned.value_counts())
ls_name = df_name [df_name.LastnameCleaned == 1].index.to_list()

df.loc[df.FirstnameCleaned.isin(ls_name), 'FirstnameCleaned']='OTHERS'

df_nameCnt = df_name.reset_index()
df_nameCnt.columns=["FirstnameCleaned","NameCount"]
# df = pd.concat([df, df_nameCnt],axis=1)
df_family = df.groupby(["LastnameCleaned",'Country']).count()[["PassengerId"]]
df_family = df_family.loc[df_family.PassengerId>1].reset_index()
df[df.Age<20].groupby(["LastnameCleaned",'Country']).count()[["PassengerId"]].sort_values(by='PassengerId')
df_family[(df_family.Country=="Sweden")&(df_family.LastnameCleaned=="PERSSON")]
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import lightgbm as lgb

from imblearn.over_sampling  import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
### setting ###
df_Xy = copy.copy(df)
opt = {}
opt['sample'] = None
opt['model'] = 'lgbm'
opt['scaling'] = True
# ls_catcol = ["Sex","Category","Country",'CountryOpt']
# ls_catcol = ["Sex","Category",'CountryOpt']
ls_catcol = ["Sex","Category"]

# ls_numcol = ['Age','NameCount']
ls_numcol = ['Age']

ls_Xcol = ls_numcol + ls_catcol

for _catcol in ls_catcol:
    df_dummies = pd.get_dummies(df[_catcol],prefix=_catcol)
    df_Xy = pd.concat([df_Xy, df_dummies],axis=1)
    del df_Xy[_catcol]
    ls_Xcol.remove(_catcol)
    ls_Xcol+=list(df_dummies.columns)
    

X = df_Xy[ls_Xcol]
y = df_Xy["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 31)

## Scaling 
if opt['scaling']:
    scaler = StandardScaler()
    for _numcol in ls_numcol:
        X_train.loc[:,_numcol] = scaler.fit_transform(X_train[_numcol].values.reshape(-1,1))
        X_test.loc[:,_numcol]  = scaler.transform(X_test[_numcol].values.reshape(-1,1))
# Sampling 
# # coz  unbalnces data
if opt['sample'] in ['under', 'over']:
    if opt['sample'] in ['over']:
        ros = RandomOverSampler( 
                random_state=0, sampling_strategy={0:sum(y_train==0), 1:sum(y_train==0)//4})
    elif opt['sample'] in ['under']:
        ros = RandomUnderSampler( 
                random_state=0, replacement=True,
                sampling_strategy={0:sum(y_train==1)*5, 1:sum(y_train==1)})
    else:
        raise Exception("pls set under/over")
        
    X_train_resampled, y_train_resampled= ros.fit_sample(X_train, y_train)
    print('y_train:\n{}'.format(pd.Series(y_train).value_counts()))
    print('y_train_resample:\n{}'.format(pd.Series(y_train_resampled).value_counts()))

    X_train, y_train = X_train_resampled, y_train_resampled
else:
    print("NO under/over sampling ")
params = {}
params['svm'] = {
    'C': [2 ** -5, 2 ** 15] ,
    "gamma":[2 ** -15, 2 ** 3]}

params['lgbm'] = {
    "max_depth": [2, 4, 6, 8, 10],
    "learning_rate" : [0.001,0.01, 0.05, 0.1],
    "num_leaves": [3, 15, 63, 255, 1023],
    "n_estimators": [100, 200, 500, 1000]}

model = {}
model['lgbm'] = lgb.LGBMClassifier(silent=False)
model['svm'] = svm.SVC()

skf = StratifiedKFold(n_splits=3,
                      shuffle=True,
                      random_state=0)

gscv = GridSearchCV(estimator = model[opt['model']],
                           param_grid = params[opt['model']],
                           scoring = 'balanced_accuracy',# coz  unbalnces data
                           cv = skf,
                           verbose=1,
                           return_train_score = True,
                           n_jobs = -1)

gscv.fit(X_train, y_train)
clf = gscv.best_estimator_
y_pred = clf.predict(X_test)
accuracy_score(y_pred, y_test)
print(classification_report(y_test, y_pred))
importance = pd.DataFrame(clf.feature_importances_,index=X_test.columns, columns=['importance'])
display(importance)
