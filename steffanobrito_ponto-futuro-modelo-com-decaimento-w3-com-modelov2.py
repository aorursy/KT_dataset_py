



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

import lightgbm as lgb

from lightgbm import LGBMClassifier,LGBMRegressor



from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.metrics import mean_squared_log_error,make_scorer,r2_score  

from sklearn import preprocessing

pd.set_option('display.max_rows',500)

pd.set_option('display.max_columns',900)

# from pandas_profiling import ProfileReport

import plotly

import warnings

warnings.filterwarnings('ignore')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory




train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
df= pd.concat([train, test])



df['Date'] = pd.to_datetime(df['Date'])



# Create date columns

le = preprocessing.LabelEncoder()

df['Day_num'] = le.fit_transform(df.Date)

df['Day'] = df['Date'].dt.day

df['Month'] = df['Date'].dt.month

df['Year'] = df['Date'].dt.year



df.head(5)
from plotly.offline import iplot

from plotly import tools

import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import plotly.io as pio

pio.templates.default = "plotly_dark"

py.init_notebook_mode(connected=True)
temp = train.groupby(['Date', 'Country_Region'])['ConfirmedCases'].sum().reset_index()

temp['Date'] = pd.to_datetime(temp['Date']).dt.strftime('%m/%d/%Y')

temp['size'] = temp['ConfirmedCases'].pow(0.3) * 3.5



fig = px.scatter_geo(temp, locations="Country_Region", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country_Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Cases Over Time', color_continuous_scale="greens")

fig.show()



grouped = train.groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()



fig2 = px.line(grouped, x="Date", y="ConfirmedCases", 

              title="Worldwide Confirmed Cases Over Time")

fig2.show()

groupedbr = train[train['Country_Region']=='Brazil'].groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()



fig2 = px.line(groupedbr, x="Date", y="ConfirmedCases", 

              title="Brazil Confirmed Cases Over Time")

fig2.show()

df['Province_State'].fillna('Vazio',inplace=True)

df['Local']=np.where(df['Province_State']== 'Vazio',df['Country_Region'],df['Country_Region']+'/'+df['Province_State'])

df.head()
df_test=df[df['ForecastId']>0]

df['Date']=df['Date'].astype('str')

df=df[df['Id']>0]

df['ConfirmedCases'].fillna(0,inplace=True)


print(df.dtypes)

df.sample()
df_f=df[df['Month']>2]



df0=df[(df['Day_num'].between(0,14))]

df1=df[(df['Day_num'].between(15,29))]

df2=df[(df['Day_num'].between(30,44))]

df3=df[(df['Day_num'].between(45,59))]

df4=df[(df['Day_num'].between(60,74))]



df5=df[(df['Day_num'].between(50,64))]

df6=df[(df['Day_num'].between(35,49))]

df7=df[(df['Day_num'].between(52,66))]

df8=df[(df['Day_num'].between(58,72))]

df9=df[(df['Day_num'].between(49,63))]

df10=df[(df['Day_num'].between(39,53))]

df11=df[(df['Day_num'].between(48,62))]

df12=df[(df['Day_num'].between(59,73))]

df13=df[(df['Day_num'].between(61,75))]

df14=df[(df['Day_num'].between(26,40))]

df15=df[(df['Day_num'].between(46,60))]

df16=df[(df['Day_num'].between(56,70))]

df17=df[(df['Day_num'].between(57,71))]

df18=df[(df['Day_num'].between(36,50))]

df19=df[(df['Day_num'].between(53,67))]



dfr=df[(df['Day_num'].between(62,76))]





#df6=df[((df['Month']==3)&(df['Day'].between(23,31)))|((df['Month']==4)&(df['Day'].between(1,6)))]

def make_decay(df):

    



    dft=df.pivot_table(index='Local',columns='Date',values='ConfirmedCases').reset_index()

    Lista_colunas=['Local','dia_01','dia_02','dia_03','dia_04','dia_05','dia_06','dia_07',

               'dia_08','dia_09','dia_10','dia_11','dia_12','dia_13','dia_14','dia_15']

    dft_copy=dft.copy()

    dft.columns=Lista_colunas

    C1=np.where(

        (dft.iloc[: , -15].values)==0,

    (np.power(dft.iloc[: , -8].values/((dft.iloc[: , -15].values)+1),1/7)) -(1)

    ,(np.power(dft.iloc[: , -8].values/((dft.iloc[: , -15].values)),1/7)) -(1)

    )



    C1=np.where(C1<0,0,C1)



    C2=np.where(

        (dft.iloc[: , -8].values)==0,

    (np.power(dft.iloc[: , -1].values/((dft.iloc[: , -8].values)+1),1/7)) -(1)

    ,(np.power(dft.iloc[: , -1].values/((dft.iloc[: , -8].values)),1/7)) -(1)

    )



    C2=np.where(C2<0,0,C2)



    dft['Crescimento_1']=C1

    dft['Crescimento_2']=C2

    

    #dataset adicionais

    #gdp2020 = pd.read_csv('/kaggle/input/covidinformacoes/gdp.csv')

#population2020 = pd.read_csv('/kaggle/input/population2020/population2020.csv')

    



    emprego_vul= pd.read_csv('/kaggle/input/covidinformacoes/Vulnerable employment ( of total employment).csv',skiprows=1)

    diox_carb=pd.read_csv('/kaggle/input/covidinformacoes/Carbon dioxide emissions per capita (tonnes).csv',skiprows=1)

    expec_vida=pd.read_csv('/kaggle/input/covidinformacoes/Life expectancy at birth.csv',skiprows=1)

    gastos_saude=pd.read_csv('/kaggle/input/covidinformacoes/Current health expenditure ( of GDP).csv',skiprows=1)

    idh=pd.read_csv('/kaggle/input/covidinformacoes/Human Development Index (HDI).csv',skiprows=1)

    idade_mediana=pd.read_csv('/kaggle/input/covidinformacoes/Median age (years).csv',skiprows=1)

    tuberculose=pd.read_csv('/kaggle/input/covidinformacoes/Tuberculosis incidence (per 100000 people).csv',skiprows=1)

    desigualdade_exp_vida=pd.read_csv('/kaggle/input/covidinformacoes/Inequality in life expectancy ().csv',skiprows=1)

    desigualdade_idh_ajustado=pd.read_csv('/kaggle/input/covidinformacoes/Inequality-adjusted HDI (IHDI).csv',skiprows=1)

    desigualdade_ganhos=pd.read_csv('/kaggle/input/covidinformacoes/Inequality in income ().csv',skiprows=1)

    desemprego=pd.read_csv('/kaggle/input/covidinformacoes/Unemployment total ( of labour force).csv',skiprows=1)

    #idade_mediana=pd.read_csv('/kaggle/input/covidinformacoes/Median age (years).csv',skiprows=1)

    #idade_mediana=pd.read_csv('/kaggle/input/covidinformacoes/Median age (years).csv',skiprows=1)

    

    #df_life=emprego_vul.copy()

    

    def func(x):

        x_new = 0

        try:

            x_new = float(x.replace(",", ""))

        except:

    #         print(x)

            x_new = np.nan

        return x_new

    

    tmp = emprego_vul.iloc[:,1].values.tolist()

    emprego_vul = emprego_vul[['Country', '2018']]

    emprego_vul['2018'] = emprego_vul['2018'].apply(lambda x: func(x))

    emprego_vul.columns = ['Country', 'Emprego_vulneravel']



    tmp = diox_carb.iloc[:,1].values.tolist()

    diox_carb = diox_carb[['Country', '2016']]

    diox_carb['2016'] = diox_carb['2016'].apply(lambda x: func(x))

    diox_carb.columns = ['Country', 'Dioxido_carbono']



    tmp = expec_vida.iloc[:,1].values.tolist()

    expec_vida = expec_vida[['Country', '2018']]

    expec_vida['2018'] = expec_vida['2018'].apply(lambda x: func(x))

    expec_vida.columns = ['Country', 'Expec_vida']

    

    tmp = gastos_saude.iloc[:,1].values.tolist()

    gastos_saude = gastos_saude[['Country', '2016']]

    gastos_saude['2016'] = gastos_saude['2016'].apply(lambda x: func(x))

    gastos_saude.columns = ['Country', 'Gastos_saude']

    

    tmp = idh.iloc[:,1].values.tolist()

    idh = idh[['Country', '2018']]

    idh['2018'] = idh['2018'].apply(lambda x: func(x))

    idh.columns = ['Country', 'IDH'] 

    

    tmp= idade_mediana.iloc[:,1].values.tolist()

    idade_mediana = idade_mediana[['Country', '2020']]

    idade_mediana['2020'] = idade_mediana['2020'].apply(lambda x: func(x))

    idade_mediana.columns = ['Country', 'Idade']

    

    tmp= tuberculose.iloc[:,1].values.tolist()

    tuberculose = tuberculose[['Country', '2017']]

    tuberculose['2017'] = tuberculose['2017'].apply(lambda x: func(x))

    tuberculose.columns = ['Country', 'Tuberculose']

    

    tmp= desigualdade_exp_vida.iloc[:,1].values.tolist()

    desigualdade_exp_vida = desigualdade_exp_vida[['Country', '2018']]

    desigualdade_exp_vida['2018'] = desigualdade_exp_vida['2018'].apply(lambda x: func(x))

    desigualdade_exp_vida.columns = ['Country', 'desigualdade_exp_vida']

    

    tmp= desigualdade_ganhos.iloc[:,1].values.tolist()

    desigualdade_ganhos = desigualdade_ganhos[['Country', '2018']]

    desigualdade_ganhos['2018'] = desigualdade_ganhos['2018'].apply(lambda x: func(x))

    desigualdade_ganhos.columns = ['Country', 'desigualdade_ganhos']

    

    tmp= desigualdade_idh_ajustado.iloc[:,1].values.tolist()

    desigualdade_idh_ajustado = desigualdade_idh_ajustado[['Country', '2018']]

    desigualdade_idh_ajustado['2018'] = desigualdade_idh_ajustado['2018'].apply(lambda x: func(x))

    desigualdade_idh_ajustado.columns = ['Country', 'desigualdade_idh_ajustado']

    

    tmp= desemprego.iloc[:,1].values.tolist()

    desemprego = desemprego[['Country', '2018']]

    desemprego['2018'] = desemprego['2018'].apply(lambda x: func(x))

    desemprego.columns = ['Country', 'desemprego']

    

    # Merge

    

    dft['Country']=dft['Local'].str.split('/',expand=True)[0]

    

    #train = pd.merge(train, population2020, how='left', left_on = 'Country_Region', right_on = 'name')

    train=dft.copy()

    train = pd.merge(train, desemprego, how='left',on='Country')

    train = pd.merge(train, idade_mediana, how='left', on='Country')

    train = pd.merge(train, idh, how='left', on='Country')

    train = pd.merge(train, emprego_vul, how='left', on='Country')

    train = pd.merge(train, gastos_saude, how='left', on='Country')

    train = pd.merge(train, expec_vida, how='left', on='Country')

    train = pd.merge(train, diox_carb, how='left', on='Country')

    train = pd.merge(train, tuberculose, how='left',on='Country')

    train = pd.merge(train, desigualdade_exp_vida, how='left', on='Country')

    train = pd.merge(train, desigualdade_ganhos, how='left',on='Country')

    train = pd.merge(train, desigualdade_idh_ajustado, how='left', on='Country')

    

 

 

    

    dft=train.copy()

    

    dft.rename({'Taiwan*':'Taiwan'}, axis=1)

    Beta1_RM=-0.1692

    dft['Decay']=np.where(((dft['Crescimento_1']==0)|(dft['Crescimento_2']==0)),

                          Beta1_RM,np.power(dft['Crescimento_2']/(dft['Crescimento_1']),1/7) - 1)

    dft['Decay']=np.where(((dft['Crescimento_1']==0) & (dft['Crescimento_2']==0)),

                    0,dft['Decay'])

    dft['Decay']=np.where(dft['Decay'].fillna('Vazio')=='Vazio',0,dft['Decay'])

    dft['Decay']=np.where(((dft['Decay']>0.02)&(dft['Crescimento_2']>0.20)),(1.5*Beta1_RM),dft['Decay'])

    dft['Decay']=np.where(((dft_copy.iloc[: ,1].values>100) & (dft['Crescimento_2']>0.1)),(Beta1_RM),dft['Decay'])

    return (dft)









# Aplicar funções nas diferentes janelas temporais



dft0=make_decay(df0)

dft1=make_decay(df1)

dft2=make_decay(df2)

dft3=make_decay(df3)

dft4=make_decay(df4)

dft5=make_decay(df5)

dft6=make_decay(df6)

dft7=make_decay(df7)

dft8=make_decay(df8)

dft9=make_decay(df9)

dft10=make_decay(df10)

dft11=make_decay(df11)

dft12=make_decay(df12)

dft13=make_decay(df13)

dft14=make_decay(df14)

dft15=make_decay(df15)

dft16=make_decay(df16)

dft17=make_decay(df17)

dft18=make_decay(df18)

dft19=make_decay(df19)



dftr=make_decay(dfr)





dfmodel=pd.concat([dft0,dft1,dft2,dft3,dft4,dft5,dft6

                  ,dft7,dft8,dft9,dft10,dft11,dft12

                  ,dft13,dft14,dft15,dft16,dft17

                  ,dft18,dft19],ignore_index=True)

dfmodel.head()
print(dfmodel.shape)

print(dftr.shape)

dftr.head(1)
dfmodel.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dfmodel.columns]

dftr.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dftr.columns]

resposta=dftr['Decay']

dfteste= dftr.drop(columns=['Local','Crescimento_1','Crescimento_2','Country'])
y=dfmodel['Decay']

variaveis=dfmodel.drop(columns=['Local','Crescimento_1','Crescimento_2','Country'])



X_train, X_test,y_train,y_test = train_test_split(variaveis, y,test_size=0.2)





#Hyper Parameters

params= {'boosting_type' : 'dart',

         'max_depth':-1,

         'objective':'regression',

         'nthread': 5,

         'num_leaves':64,

         'learning_rate':0.01,

         'max_bin':256,

         'subsample_for_bin':200,

         'subsample':1,

         'subsample_freq':1,

         'colsample_bytree':0.8,

         'reg_alpha':1.2,

         'reg_lambda':1.2,

         'min_split_gain':0.5,

         'min_child_weight':1,

         'min_child_samples':5,

         'metric':'l2'    

}



gridParams = {'learning_rate': [0.001,0.01,0.03,0.05,0.07,0.09]

              ,'num_leaves':[10,20,30,50,70,80]

              ,'boosting_type': ['dart']

              ,'objective': ['regression']

              ,'random_state' : [13]

              ,'min_split_gain': [0.01,0.03,0.05]

              ,'drop_rate' : [0.015,0.02,0.04,0.05,0.07]

              ,'max_bin': [64,128,256]

              ,'reg_alpha':[0,1,2,3,5,7]

              ,'reg_lambda':[0,1,2,3,5,7]

              ,'colsample_bytree':[0.03,0.05,0.6,0.7,0.8,0.9]

              ,'min_child_weight':[0.1,0.25,0.5,0.8,1,1.5,2,3]

}



mdl = lgb.LGBMRegressor(boosting_type='dart',

                       objective='regression',

                       n_jobs=-1,

                       silent=True,

                       max_depth=params['max_depth'],

                       max_bin=params['max_bin'],

                       subsample_for_bin= params['subsample_for_bin'],

                       subsample=params['subsample'],

                        subsample_freq=params['subsample_freq'],

                        min_split_gain=params['min_split_gain'],

                        min_child_weight=params['min_child_weight'],

                        min_child_samples=params['min_child_samples'],

                       )



mdl.get_params().keys()



grid=RandomizedSearchCV(mdl, gridParams, scoring=make_scorer(score_func=r2_score, greater_is_better=True)

                        ,n_iter=25,n_jobs=-1)



grid.fit(X_train,y_train)



print(grid.best_params_)

print(grid.best_score_)



#Get from Grid/RandomizedSearch



params['min_split_gain']= grid.best_params_['min_split_gain']

params['learning_rate']= grid.best_params_['learning_rate']

params['max_bin']= grid.best_params_['max_bin']

params['num_leaves']= grid.best_params_['num_leaves']

params['drop_rate']= grid.best_params_['drop_rate']

params['reg_alpha']= grid.best_params_['reg_alpha']

params['reg_lambda']= grid.best_params_['reg_lambda']

params['colsample_bytree']= grid.best_params_['colsample_bytree']

params['min_child_weight']= grid.best_params_['min_child_weight']





train_data=lgb.Dataset(X_train,label=y_train)

lgbm_cases= lgb.train(params,

               train_data,

               600,

               verbose_eval=4)
resp=lgbm_cases.predict(dfteste)

print(r2_score(resp,resposta))

dftr['Previsto']=resp

dftr.head()
import shap

shap_values = shap.TreeExplainer(lgbm_cases).shap_values(X_train)

shap.summary_plot(shap_values, X_train)
dft=df_f.pivot_table(index='Local',columns='Date',values='ConfirmedCases').reset_index()



dft_copy=dft.copy()

#dft.columns=Lista_colunas

C1=np.where(

        (dft.iloc[: , -15].values)==0,

    (np.power(dft.iloc[: , -8].values/((dft.iloc[: , -15].values)+1),1/7)) -(1)

    ,(np.power(dft.iloc[: , -8].values/((dft.iloc[: , -15].values)),1/7)) -(1)

    )



C1=np.where(C1<0,0,C1)



C2=np.where(

        (dft.iloc[: , -8].values)==0,

    (np.power(dft.iloc[: , -1].values/((dft.iloc[: , -8].values)+1),1/7)) -(1)

    ,(np.power(dft.iloc[: , -1].values/((dft.iloc[: , -8].values)),1/7)) -(1)

    )



C2=np.where(C2<0,0,C2)



dft['Crescimento_1']=C1

dft['Crescimento_2']=C2



Beta1_RM=-0.1692

dft['Decay']=np.where(((dft['Crescimento_1']==0)|(dft['Crescimento_2']==0)),

                          Beta1_RM,np.power(dft['Crescimento_2']/(dft['Crescimento_1']),1/7) - 1)

dft['Decay']=np.where(((dft['Crescimento_1']==0) & (dft['Crescimento_2']==0)),

                    0,dft['Decay'])

dft['Decay']=np.where(dft['Decay'].fillna('Vazio')=='Vazio',0,dft['Decay'])

dft['Decay']=np.where(((dft['Decay']>0.02)&(dft['Crescimento_2']>0.20)),(1.5*Beta1_RM),dft['Decay'])

dft['Decay']=np.where(((dft_copy.iloc[: ,1].values>100) & (dft['Crescimento_2']>0.1)),(Beta1_RM),dft['Decay'])



dft.head()



dt=pd.merge(dft,dftr[['Local','Previsto']],on='Local',how='left')

dt.head()
dt[dt['Local'].isin(['Brazil','US/New York','US/New Jersey','US/Illinois','US/California','Italy','Spain','France','Germany'])]
Beta0=0.941

#Beta1=-0.1692

dft=dt.copy()

dft['Decay_Calculado']=dft['Decay']

dft['Decay']=dft['Previsto']

Beta1=dft['Decay']







#dft['Cres_2020-04-01']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-02']=dft['Cres_2020-04-01']*((dft['Cres_2020-04-01']*(Beta1)+Beta0))

#dft['Cres_2020-04-03']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-04']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-05']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-06']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-07']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

dft['Cres_2020-04-08']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

dft['Cres_2020-04-09']=dft['Cres_2020-04-08']*((dft['Cres_2020-04-08']*(Beta1)+Beta0))

dft['Cres_2020-04-10']=dft['Cres_2020-04-09']*((dft['Cres_2020-04-09']*(Beta1)+Beta0))

dft['Cres_2020-04-11']=dft['Cres_2020-04-10']*((dft['Cres_2020-04-10']*(Beta1)+Beta0))

dft['Cres_2020-04-12']=dft['Cres_2020-04-11']*((dft['Cres_2020-04-11']*(Beta1)+Beta0))

dft['Cres_2020-04-13']=dft['Cres_2020-04-12']*((dft['Cres_2020-04-12']*(Beta1)+Beta0))

dft['Cres_2020-04-14']=dft['Cres_2020-04-13']*((dft['Cres_2020-04-13']*(Beta1)+Beta0))

dft['Cres_2020-04-15']=dft['Cres_2020-04-14']*((dft['Cres_2020-04-14']*(Beta1)+Beta0))

dft['Cres_2020-04-16']=dft['Cres_2020-04-15']*((dft['Cres_2020-04-15']*(Beta1)+Beta0))

dft['Cres_2020-04-17']=dft['Cres_2020-04-16']*((dft['Cres_2020-04-16']*(Beta1)+Beta0))

dft['Cres_2020-04-18']=dft['Cres_2020-04-17']*((dft['Cres_2020-04-17']*(Beta1)+Beta0))

dft['Cres_2020-04-19']=dft['Cres_2020-04-18']*((dft['Cres_2020-04-18']*(Beta1)+Beta0))

dft['Cres_2020-04-20']=dft['Cres_2020-04-19']*((dft['Cres_2020-04-19']*(Beta1)+Beta0))

dft['Cres_2020-04-21']=dft['Cres_2020-04-20']*((dft['Cres_2020-04-20']*(Beta1)+Beta0))

dft['Cres_2020-04-22']=dft['Cres_2020-04-21']*((dft['Cres_2020-04-21']*(Beta1)+Beta0))

dft['Cres_2020-04-23']=dft['Cres_2020-04-22']*((dft['Cres_2020-04-22']*(Beta1)+Beta0))

dft['Cres_2020-04-24']=dft['Cres_2020-04-23']*((dft['Cres_2020-04-23']*(Beta1)+Beta0))

dft['Cres_2020-04-25']=dft['Cres_2020-04-24']*((dft['Cres_2020-04-24']*(Beta1)+Beta0))

dft['Cres_2020-04-26']=dft['Cres_2020-04-25']*((dft['Cres_2020-04-25']*(Beta1)+Beta0))

dft['Cres_2020-04-27']=dft['Cres_2020-04-26']*((dft['Cres_2020-04-26']*(Beta1)+Beta0))

dft['Cres_2020-04-28']=dft['Cres_2020-04-27']*((dft['Cres_2020-04-27']*(Beta1)+Beta0))

dft['Cres_2020-04-29']=dft['Cres_2020-04-28']*((dft['Cres_2020-04-28']*(Beta1)+Beta0))

dft['Cres_2020-04-30']=dft['Cres_2020-04-29']*((dft['Cres_2020-04-29']*(Beta1)+Beta0))



dft['Cres_2020-05-01']=dft['Cres_2020-04-30']*((dft['Cres_2020-04-30']*(Beta1)+Beta0))

dft['Cres_2020-05-02']=dft['Cres_2020-05-01']*((dft['Cres_2020-05-01']*(Beta1)+Beta0))

dft['Cres_2020-05-03']=dft['Cres_2020-05-02']*((dft['Cres_2020-05-02']*(Beta1)+Beta0))

dft['Cres_2020-05-04']=dft['Cres_2020-05-03']*((dft['Cres_2020-05-03']*(Beta1)+Beta0))

dft['Cres_2020-05-05']=dft['Cres_2020-05-04']*((dft['Cres_2020-05-04']*(Beta1)+Beta0))

dft['Cres_2020-05-06']=dft['Cres_2020-05-05']*((dft['Cres_2020-05-05']*(Beta1)+Beta0))

dft['Cres_2020-05-07']=dft['Cres_2020-05-06']*((dft['Cres_2020-05-06']*(Beta1)+Beta0))







#dft['2020-04-01']=(1+dft['Cres_2020-04-01'])*dft['2020-03-31']

#dft['2020-04-02']=(1+dft['Cres_2020-04-02'])*dft['2020-04-01']

#dft['2020-04-03']=(1+dft['Cres_2020-04-03'])*dft['2020-04-02']

#dft['2020-04-04']=(1+dft['Cres_2020-04-04'])*dft['2020-04-03']

#dft['2020-04-05']=(1+dft['Cres_2020-04-05'])*dft['2020-04-04']

#dft['2020-04-06']=(1+dft['Cres_2020-04-06'])*dft['2020-04-05']

#dft['2020-04-07']=(1+dft['Cres_2020-04-07'])*dft['2020-04-06']

dft['2020-04-08']=(1+dft['Cres_2020-04-08'])*dft['2020-04-07']

dft['2020-04-09']=(1+dft['Cres_2020-04-09'])*dft['2020-04-08']

dft['2020-04-10']=(1+dft['Cres_2020-04-10'])*dft['2020-04-09']

dft['2020-04-11']=(1+dft['Cres_2020-04-11'])*dft['2020-04-10']

dft['2020-04-12']=(1+dft['Cres_2020-04-12'])*dft['2020-04-11']

dft['2020-04-13']=(1+dft['Cres_2020-04-13'])*dft['2020-04-12']

dft['2020-04-14']=(1+dft['Cres_2020-04-14'])*dft['2020-04-13']

dft['2020-04-15']=(1+dft['Cres_2020-04-15'])*dft['2020-04-14']

dft['2020-04-16']=(1+dft['Cres_2020-04-16'])*dft['2020-04-15']

dft['2020-04-17']=(1+dft['Cres_2020-04-17'])*dft['2020-04-16']

dft['2020-04-18']=(1+dft['Cres_2020-04-18'])*dft['2020-04-17']

dft['2020-04-19']=(1+dft['Cres_2020-04-19'])*dft['2020-04-18']

dft['2020-04-20']=(1+dft['Cres_2020-04-20'])*dft['2020-04-19']

dft['2020-04-21']=(1+dft['Cres_2020-04-21'])*dft['2020-04-20']

dft['2020-04-22']=(1+dft['Cres_2020-04-22'])*dft['2020-04-21']

dft['2020-04-23']=(1+dft['Cres_2020-04-23'])*dft['2020-04-22']

dft['2020-04-24']=(1+dft['Cres_2020-04-24'])*dft['2020-04-23']

dft['2020-04-25']=(1+dft['Cres_2020-04-25'])*dft['2020-04-24']

dft['2020-04-26']=(1+dft['Cres_2020-04-26'])*dft['2020-04-25']

dft['2020-04-27']=(1+dft['Cres_2020-04-27'])*dft['2020-04-26']

dft['2020-04-28']=(1+dft['Cres_2020-04-28'])*dft['2020-04-27']

dft['2020-04-29']=(1+dft['Cres_2020-04-29'])*dft['2020-04-28']

dft['2020-04-30']=(1+dft['Cres_2020-04-30'])*dft['2020-04-29']



dft['2020-05-01']=(1+dft['Cres_2020-05-01'])*dft['2020-04-30']

dft['2020-05-02']=(1+dft['Cres_2020-05-02'])*dft['2020-05-01']

dft['2020-05-03']=(1+dft['Cres_2020-05-03'])*dft['2020-05-02']

dft['2020-05-04']=(1+dft['Cres_2020-05-04'])*dft['2020-05-03']

dft['2020-05-05']=(1+dft['Cres_2020-05-05'])*dft['2020-05-04']

dft['2020-05-06']=(1+dft['Cres_2020-05-06'])*dft['2020-05-05']

dft['2020-05-07']=(1+dft['Cres_2020-05-07'])*dft['2020-05-06']
dfm=df_f.pivot_table(index='Local',columns='Date',values='Fatalities').reset_index()

#dft.iloc[: , -8].values/((dft.iloc[: , -15].values)

mortes_adj=dfm.iloc[: , -1].values.sum() / dft_copy.iloc[: , -1].values.sum()

dft['mortes']=dfm.iloc[: , -1].values / dft_copy.iloc[: , -1].values

 

print(mortes_adj)

dft.head()
dft[dft['Local'].isin(['Brazil','US/New York','US/New Jersey','US/Illinois','Italy','Spain','France','Germany'])]
#dft.loc(dft['mortes']>(2*mortes_adj),'mortes')=(2*mortes_adj)

#dft.loc(dft['mortes']<(mortes_adj/2),'mortes')=(mortes_adj/2)

dft['mortes']=np.where(dft['mortes']>(2*mortes_adj),(2*mortes_adj),np.where(dft['mortes']<(mortes_adj/2),(mortes_adj/2),dft['mortes']))
dfi=dft[['Local', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04',

       '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09',

       '2020-03-10', '2020-03-11', '2020-03-12', '2020-03-13', '2020-03-14',

       '2020-03-15', '2020-03-16', '2020-03-17', '2020-03-18', '2020-03-19',

       '2020-03-20', '2020-03-21', '2020-03-22', '2020-03-23', '2020-03-24',

       '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29',

       '2020-03-30', '2020-03-31',  '2020-04-01', '2020-04-02', '2020-04-03',

       '2020-04-04', '2020-04-05', '2020-04-06', '2020-04-07', '2020-04-08',

       '2020-04-09', '2020-04-10', '2020-04-11', '2020-04-12', '2020-04-13',

       '2020-04-14', '2020-04-15', '2020-04-16', '2020-04-17', '2020-04-18',

       '2020-04-19', '2020-04-20', '2020-04-21', '2020-04-22', '2020-04-23',

       '2020-04-24', '2020-04-25', '2020-04-26', '2020-04-27', '2020-04-28',

       '2020-04-29', '2020-04-30','2020-05-01','2020-05-02','2020-05-03','2020-05-04','2020-05-05','2020-05-06','2020-05-07']]

dfi.head()
dfi['2020-05-07'].sum()
df = dfi.melt('Local', var_name='Date', value_name='ConfirmedCases')





df=pd.merge(df,dft[['Local','mortes']],on='Local',how='left')

df['Fatalities']=df['ConfirmedCases']*df['mortes']

df[df['Local']=='Brazil']


df_test.head()

df_test=df_test.drop(columns=['ConfirmedCases','Fatalities'])

df_test['Date']=df_test['Date'].astype('str')

df_test=pd.merge(df_test,df[['Local','Date','ConfirmedCases','Fatalities']],on=['Local','Date'],how='left')
dff=df_test.sort_values(by='ConfirmedCases',ascending=False)

dff.head(20)


dfm = dfm.melt('Local', var_name='Date', value_name='Fatalities')



dfat=pd.merge(df_test,dfm[['Local','Fatalities','Date']],on=['Local','Date'],how='left',suffixes=('_predicted','_real'))

dfat['Fatalities_real'].fillna('Vazio',inplace=True)

dfat['Fatalities']=np.where(dfat['Fatalities_real']=='Vazio',dfat['Fatalities_predicted'],dfat['Fatalities_real'])

dfat[dfat['Local']=='Brazil']
dfat.to_csv('Regioes.csv',index=None)


groupedbr = dfat[dfat['Country_Region']=='Brazil'].groupby('Date')['Date', 'ConfirmedCases', 'Fatalities'].sum().reset_index()



fig2 = px.line(groupedbr, x="Date", y="ConfirmedCases", 

              title="Brazil Confirmed Cases + Predicted Over Time")

fig2.show()

submission=dfat[['ForecastId','ConfirmedCases','Fatalities']]

submission['ForecastId']=submission['ForecastId'].astype('int32')

submission['Fatalities']=submission['Fatalities'].astype('float')

print(submission.dtypes)

submission.sample(10)





submission.describe()
df_test[df_test['ConfirmedCases'].isna()]
dftpronto=dfat.copy()
submission.to_csv('submission.csv',index=None)

submission.sample(10)