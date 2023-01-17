



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




train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
df= pd.concat([train, test])



df['Date'] = pd.to_datetime(df['Date'])



# Create date columns

le = preprocessing.LabelEncoder()

df['Day_num'] = le.fit_transform(df.Date)

df['Day'] = df['Date'].dt.day

df['Month'] = df['Date'].dt.month

df['Year'] = df['Date'].dt.year



df.head(5)
df[(df['Province_State']=='Florida') & (df['Id']>0)].tail()
########Flórida casos correção########

#df['ConfirmedCases']=np.where(((df['Province_State']=='Florida')&(df['Day_num']==82)),21019,df['ConfirmedCases'])

df['Mortalidade']=np.where(df['ConfirmedCases']==0,0,df['Fatalities']/df['ConfirmedCases'])









df[(df['Country_Region']=='Brazil') & (df['Id']>0)].tail()
df['Province_State'].fillna('Vazio',inplace=True)

df['Local']=np.where(df['Province_State']== 'Vazio',df['Country_Region'],df['Country_Region']+'/'+df['Province_State'])
df_test=df[df['ForecastId']>0]

df['Date']=df['Date'].astype('str')

df=df[df['Id']>0]

df['ConfirmedCases'].fillna(0,inplace=True)



print(df.dtypes)

df[(df['Local']=='Brazil')].tail()
df_f=df[df['Month']>2]



df0=df[(df['Day_num'].between(0,14))]

df1=df[(df['Day_num'].between(1,15))]

df2=df[(df['Day_num'].between(2,16))]

df3=df[(df['Day_num'].between(3,17))]

df4=df[(df['Day_num'].between(4,18))]



df5=df[(df['Day_num'].between(5,19))]

df6=df[(df['Day_num'].between(6,20))]

df7=df[(df['Day_num'].between(7,21))]

df8=df[(df['Day_num'].between(8,22))]

df9=df[(df['Day_num'].between(9,23))]

df10=df[(df['Day_num'].between(10,24))]

df11=df[(df['Day_num'].between(11,25))]

df12=df[(df['Day_num'].between(12,26))]

df13=df[(df['Day_num'].between(13,27))]

df14=df[(df['Day_num'].between(14,28))]

df15=df[(df['Day_num'].between(15,29))]

df16=df[(df['Day_num'].between(16,30))]

df17=df[(df['Day_num'].between(17,31))]

df18=df[(df['Day_num'].between(18,32))]

df19=df[(df['Day_num'].between(19,33))]

df20=df[(df['Day_num'].between(20,34))]

df21=df[(df['Day_num'].between(21,35))]

df22=df[(df['Day_num'].between(22,36))]

df23=df[(df['Day_num'].between(23,37))]

df24=df[(df['Day_num'].between(24,38))]

df25=df[(df['Day_num'].between(25,39))]

df26=df[(df['Day_num'].between(26,40))]

df27=df[(df['Day_num'].between(27,41))]

df28=df[(df['Day_num'].between(28,42))]

df29=df[(df['Day_num'].between(29,43))]

df30=df[(df['Day_num'].between(30,44))]

df31=df[(df['Day_num'].between(31,45))]

df32=df[(df['Day_num'].between(32,46))]

df33=df[(df['Day_num'].between(33,47))]

df34=df[(df['Day_num'].between(34,48))]

df35=df[(df['Day_num'].between(35,49))]

df36=df[(df['Day_num'].between(36,50))]

df37=df[(df['Day_num'].between(37,51))]

df38=df[(df['Day_num'].between(38,52))]

df39=df[(df['Day_num'].between(39,53))]

df40=df[(df['Day_num'].between(40,54))]

df41=df[(df['Day_num'].between(41,55))]

df42=df[(df['Day_num'].between(42,56))]

df43=df[(df['Day_num'].between(43,57))]

df44=df[(df['Day_num'].between(44,58))]

df45=df[(df['Day_num'].between(45,59))]

df46=df[(df['Day_num'].between(46,60))]

df47=df[(df['Day_num'].between(47,61))]

df48=df[(df['Day_num'].between(48,62))]

df49=df[(df['Day_num'].between(49,63))]

df50=df[(df['Day_num'].between(50,64))]

df51=df[(df['Day_num'].between(51,65))]

df52=df[(df['Day_num'].between(52,66))]

df53=df[(df['Day_num'].between(53,67))]

df54=df[(df['Day_num'].between(54,68))]

df55=df[(df['Day_num'].between(55,69))]

df56=df[(df['Day_num'].between(56,70))]

df57=df[(df['Day_num'].between(57,71))]

df58=df[(df['Day_num'].between(58,72))]

df59=df[(df['Day_num'].between(59,73))]

df60=df[(df['Day_num'].between(60,74))]

df61=df[(df['Day_num'].between(61,75))]

df62=df[(df['Day_num'].between(62,76))]

df63=df[(df['Day_num'].between(63,77))]

df64=df[(df['Day_num'].between(64,78))]

df65=df[(df['Day_num'].between(65,79))]

df66=df[(df['Day_num'].between(66,80))]

df67=df[(df['Day_num'].between(67,81))]

df68=df[(df['Day_num'].between(68,82))]

#df69=df[(df['Day_num'].between(69,83))]















dfr=df[(df['Day_num'].between(69,83))]





#df6=df[((df['Month']==3)&(df['Day'].between(23,31)))|((df['Month']==4)&(df['Day'].between(1,6)))]

dfr.tail()
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

    populacao=pd.read_csv('/kaggle/input/covidinformacoes/Population total (millions).csv',skiprows=1)

    populacao_65=pd.read_csv('/kaggle/input/covidinformacoes/Population ages 65 and older (millions).csv',skiprows=1)

    pop_urbana=pd.read_csv('/kaggle/input/covidinformacoes/Population urban ().csv',skiprows=1)

    energia_nr=pd.read_csv('/kaggle/input/covidinformacoes/Fossil fuel energy consumption ( of total energy consumption).csv',skiprows=1)

    pop_prisao=pd.read_csv('/kaggle/input/covidinformacoes/Prison population (per 100000 people).csv',skiprows=1)

    usu_internet=pd.read_csv('/kaggle/input/covidinformacoes/Internet users total ( of population).csv',skiprows=1)

    jovens_sem_oc=pd.read_csv('/kaggle/input/covidinformacoes/Youth not in school or employment ( ages 15-24).csv',skiprows=1)

    escola_anos=pd.read_csv('/kaggle/input/covidinformacoes/Mean years of schooling (years).csv',skiprows=1)

    

    #df_life=emprego_vul.copy()

    

    def func(x):

        x_new = 0

        try:

            x_new = float(x.replace(",", ""))

        except:

    #         print(x)

            x_new = np.nan

        return x_new

    

    tmp = populacao.iloc[:,1].values.tolist()

    populacao= populacao[['Country', '2018']]

    populacao['Country']=np.where(populacao['Country']=='United States','US',populacao['Country'])

    populacao['2018'] = populacao['2018'].apply(lambda x: func(x))

    populacao.columns = ['Country', 'populacao']

    

    tmp = populacao_65.iloc[:,1].values.tolist()

    populacao_65= populacao_65[['Country', '2018']]

    populacao_65['Country']=np.where(populacao_65['Country']=='United States','US',populacao_65['Country'])

    populacao_65['2018'] = populacao_65['2018'].apply(lambda x: func(x))

    populacao_65.columns = ['Country', 'populacao_65']    



    tmp = pop_urbana.iloc[:,1].values.tolist()

    pop_urbana= pop_urbana[['Country', '2018']]

    pop_urbana['Country']=np.where(pop_urbana['Country']=='United States','US',pop_urbana['Country'])

    pop_urbana['2018'] = pop_urbana['2018'].apply(lambda x: func(x))

    pop_urbana.columns = ['Country', 'pop_urbana']    



    tmp = energia_nr.iloc[:,1].values.tolist()

    energia_nr= energia_nr[['Country', '2014']]

    energia_nr['Country']=np.where(energia_nr['Country']=='United States','US',energia_nr['Country'])

    energia_nr['2014'] = energia_nr['2014'].apply(lambda x: func(x))

    energia_nr.columns = ['Country', 'energia_nr']    



    tmp = usu_internet.iloc[:,1].values.tolist()

    usu_internet= usu_internet[['Country', '2017']]

    usu_internet['Country']=np.where(usu_internet['Country']=='United States','US',usu_internet['Country'])

    usu_internet['2017'] = usu_internet['2017'].apply(lambda x: func(x))

    usu_internet.columns = ['Country', 'usu_internet']

    

    tmp = jovens_sem_oc.iloc[:,1].values.tolist()

    jovens_sem_oc= jovens_sem_oc[['Country', '2017']]

    jovens_sem_oc['Country']=np.where(jovens_sem_oc['Country']=='United States','US',jovens_sem_oc['Country'])

    jovens_sem_oc['2017'] = jovens_sem_oc['2017'].apply(lambda x: func(x))

    jovens_sem_oc.columns = ['Country', 'jovens_sem_oc']

    

    tmp = escola_anos.iloc[:,1].values.tolist()

    escola_anos= escola_anos[['Country', '2018']]

    escola_anos['Country']=np.where(escola_anos['Country']=='United States','US',escola_anos['Country'])

    escola_anos['2018'] = escola_anos['2018'].apply(lambda x: func(x))

    escola_anos.columns = ['Country', 'escola_anos']

    

    tmp = emprego_vul.iloc[:,1].values.tolist()

    emprego_vul = emprego_vul[['Country', '2018']]

    emprego_vul['Country']=np.where(emprego_vul['Country']=='United States','US',emprego_vul['Country'])

    emprego_vul['2018'] = emprego_vul['2018'].apply(lambda x: func(x))

    emprego_vul.columns = ['Country', 'Emprego_vulneravel']



    tmp = diox_carb.iloc[:,1].values.tolist()

    diox_carb = diox_carb[['Country', '2016']]

    diox_carb['Country']=np.where(diox_carb['Country']=='United States','US',diox_carb['Country'])

    diox_carb['2016'] = diox_carb['2016'].apply(lambda x: func(x))

    diox_carb.columns = ['Country', 'Dioxido_carbono']



    tmp = expec_vida.iloc[:,1].values.tolist()

    expec_vida = expec_vida[['Country', '2018']]

    expec_vida['Country']=np.where(expec_vida['Country']=='United States','US',expec_vida['Country'])

    expec_vida['2018'] = expec_vida['2018'].apply(lambda x: func(x))

    expec_vida.columns = ['Country', 'Expec_vida']

    

    tmp = gastos_saude.iloc[:,1].values.tolist()

    gastos_saude = gastos_saude[['Country', '2016']]

    gastos_saude['Country']=np.where(gastos_saude['Country']=='United States','US',gastos_saude['Country'])

    gastos_saude['2016'] = gastos_saude['2016'].apply(lambda x: func(x))

    gastos_saude.columns = ['Country', 'Gastos_saude']

    

    tmp = idh.iloc[:,1].values.tolist()

    idh = idh[['Country', '2018']]

    idh['Country']=np.where(idh['Country']=='United States','US',idh['Country'])

    idh['2018'] = idh['2018'].apply(lambda x: func(x))

    idh.columns = ['Country', 'IDH'] 

    

    tmp= idade_mediana.iloc[:,1].values.tolist()

    idade_mediana = idade_mediana[['Country', '2020']]

    idade_mediana['Country']=np.where(idade_mediana['Country']=='United States','US',idade_mediana['Country'])

    idade_mediana['2020'] = idade_mediana['2020'].apply(lambda x: func(x))

    idade_mediana.columns = ['Country', 'Idade']

    

    tmp= tuberculose.iloc[:,1].values.tolist()

    tuberculose = tuberculose[['Country', '2017']]

    tuberculose['Country']=np.where(tuberculose['Country']=='United States','US',tuberculose['Country'])

    tuberculose['2017'] = tuberculose['2017'].apply(lambda x: func(x))

    tuberculose.columns = ['Country', 'Tuberculose']

    

    tmp= desigualdade_exp_vida.iloc[:,1].values.tolist()

    desigualdade_exp_vida = desigualdade_exp_vida[['Country', '2018']]

    desigualdade_exp_vida['Country']=np.where(desigualdade_exp_vida['Country']=='United States','US',desigualdade_exp_vida['Country'])

    desigualdade_exp_vida['2018'] = desigualdade_exp_vida['2018'].apply(lambda x: func(x))

    desigualdade_exp_vida.columns = ['Country', 'desigualdade_exp_vida']

    

    tmp= desigualdade_ganhos.iloc[:,1].values.tolist()

    desigualdade_ganhos = desigualdade_ganhos[['Country', '2018']]

    desigualdade_ganhos['Country']=np.where(desigualdade_ganhos['Country']=='United States','US',desigualdade_ganhos['Country'])

    desigualdade_ganhos['2018'] = desigualdade_ganhos['2018'].apply(lambda x: func(x))

    desigualdade_ganhos.columns = ['Country', 'desigualdade_ganhos']

    

    tmp= desigualdade_idh_ajustado.iloc[:,1].values.tolist()

    desigualdade_idh_ajustado = desigualdade_idh_ajustado[['Country', '2018']]

    desigualdade_idh_ajustado['Country']=np.where(desigualdade_idh_ajustado['Country']=='United States','US',desigualdade_idh_ajustado['Country'])

    desigualdade_idh_ajustado['2018'] = desigualdade_idh_ajustado['2018'].apply(lambda x: func(x))

    desigualdade_idh_ajustado.columns = ['Country', 'desigualdade_idh_ajustado']

    

    tmp= desemprego.iloc[:,1].values.tolist()

    desemprego = desemprego[['Country', '2018']]

    desemprego['Country']=np.where(desemprego['Country']=='United States','US',desemprego['Country'])

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

    train = pd.merge(train, populacao, how='left', on='Country')

    train = pd.merge(train, populacao_65, how='left', on='Country')

    train = pd.merge(train, pop_urbana, how='left', on='Country')

    #train = pd.merge(train, pop_prisao, how='left', on='Country')

    train = pd.merge(train, energia_nr, how='left', on='Country')

    train = pd.merge(train, usu_internet, how='left', on='Country')

    train = pd.merge(train, jovens_sem_oc, how='left', on='Country')

    train = pd.merge(train, escola_anos, how='left', on='Country')

    

    dft=train.copy()

    dft['Pop_maior65']=np.where(((dft['populacao_65']==-99) | (dft['populacao']==-99)),0,dft['populacao_65'] / dft['populacao'])

    dft.rename({'Taiwan*':'Taiwan'}, axis=1)

    Beta1_RM=-0.1692

    dft['Decay']=np.where(((dft['Crescimento_1']==0)|(dft['Crescimento_2']==0)),

                          Beta1_RM,np.power(dft['Crescimento_2']/(dft['Crescimento_1']),1/7) - 1)

    dft['Decay']=np.where(((dft['Crescimento_1']==0) & (dft['Crescimento_2']==0)),

                    0,dft['Decay'])

    dft['Decay']=np.where(dft['Decay'].fillna('Vazio')=='Vazio',0,dft['Decay'])

    dft['Decay']=np.where(((dft['Decay']>0.02)&(dft['Crescimento_2']>0.20)),(1.5*Beta1_RM),dft['Decay'])

    dft['Decay']=np.where(((dft_copy.iloc[: ,1].values>100) & (dft['Crescimento_2']>0.1)),(Beta1_RM),dft['Decay'])

    

    dft.fillna(-99,inplace=True)

    

    

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

dft20=make_decay(df20)

dft21=make_decay(df21)

dft22=make_decay(df22)

dft23=make_decay(df23)

dft24=make_decay(df24)

dft25=make_decay(df25)

dft26=make_decay(df26)

dft27=make_decay(df27)

dft28=make_decay(df28)

dft29=make_decay(df29)

dft30=make_decay(df30)

dft31=make_decay(df31)

dft32=make_decay(df32)

dft33=make_decay(df33)

dft34=make_decay(df34)

dft35=make_decay(df35)

dft36=make_decay(df36)

dft37=make_decay(df37)

dft38=make_decay(df38)

dft39=make_decay(df39)

dft40=make_decay(df40)

dft41=make_decay(df41)

dft42=make_decay(df42)

dft43=make_decay(df43)

dft44=make_decay(df44)

dft45=make_decay(df45)

dft46=make_decay(df46)

dft47=make_decay(df47)

dft48=make_decay(df48)

dft49=make_decay(df49)

dft50=make_decay(df50)

dft51=make_decay(df51)

dft52=make_decay(df52)

dft53=make_decay(df53)

dft54=make_decay(df54)

dft55=make_decay(df55)

dft56=make_decay(df56)

dft57=make_decay(df57)

dft58=make_decay(df58)

dft59=make_decay(df59)

dft60=make_decay(df60)

dft61=make_decay(df61)

dft62=make_decay(df62)

dft63=make_decay(df63)

dft64=make_decay(df64)

dft65=make_decay(df65)

dft66=make_decay(df66)

dft67=make_decay(df67)

dft68=make_decay(df68)

#dft69=make_decay(df19)







dftr=make_decay(dfr)





dfmodel=pd.concat([dft0,dft1,dft2,dft3,dft4,dft5,dft6

                  ,dft7,dft8,dft9,dft10,dft11,dft12

                  ,dft13,dft14,dft15,dft16,dft17

                  ,dft18,dft19,dft20,dft21,dft22,dft23

                  ,dft24,dft25,dft26,dft26,dft27,dft28

                  ,dft29,dft30,dft31,dft32,dft33,dft34

                  ,dft35,dft36,dft37,dft38,dft39,dft40

                  ,dft41,dft42,dft43,dft44,dft45,dft46

                  ,dft47,dft48,dft49,dft50,dft51,dft52

                  ,dft53,dft54,dft55,dft56,dft57,dft58

                  ,dft59,dft60,dft61,dft62

                   ,dft63,dft64

                   ,dft65,dft66,dft67,dft68

                   

                  

                  

                  ],ignore_index=True)

dfmodel.head()
dfmodel.describe()
dfmodel.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dfmodel.columns]

dftr.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dftr.columns]





dfmodel.fillna(-99,inplace=True)

dftr.fillna(-99,inplace=True)



resposta=dftr['Decay']

dfteste_cat= dftr.drop(columns=['Crescimento_1','Crescimento_2','Decay'])

dfteste= dftr.drop(columns=['Local','Crescimento_1','Crescimento_2','Country','Decay'])



y=dfmodel['Decay']

variaveis_cat=dfmodel.drop(columns=['Crescimento_1','Crescimento_2','Decay'])

variaveis=dfmodel.drop(columns=['Local','Crescimento_1','Crescimento_2','Country','Decay'])





X_train, X_test,y_train,y_test = train_test_split(variaveis, y,test_size=0.1)

cX_train, cX_test,cy_train,cy_test = train_test_split(variaveis_cat, y,test_size=0.1)



import catboost

from catboost import CatBoostRegressor, Pool
train_pool = Pool(X_train,

                  label=y_train

                  #cat_features=['Local','Country'])

                 )

val_pool = Pool(X_test,

                  label=y_test

                  #cat_features=['Local','Country'])

               )

test_pool = Pool(dfteste,

                  label=resposta

                  #cat_features=['Local','Country'])

                )
model = CatBoostRegressor(objective='RMSE')



model.fit(train_pool, plot=True, eval_set=val_pool, verbose=500)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clfGB = GradientBoostingRegressor(**params)



clfGB.fit(X_train, y_train)

rGB=clfGB.predict(dfteste)



clfRF = RandomForestRegressor()



clfRF.fit(X_train, y_train)

rRF=clfRF.predict(dfteste)

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

              ,'num_leaves':[10,20,30,50,70]

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
resp=model.predict(test_pool)

respLGB=lgbm_cases.predict(dfteste)

print("MSE CatBoost: %.4f"  %mean_squared_error(resp,resposta))

print("MSE GradientBoosting: %.4f" %mean_squared_error(rGB,resposta))

print("MSE RandomForest: %.4f" %mean_squared_error(rRF,resposta))

print("MSE LightGBM: %.4f" %mean_squared_error(respLGB,resposta))





dftr['Previsto']=resp

dftr['Resposta']=resposta

dftr['Erro']=np.where((((dftr['Previsto'])==0)|(dftr['Resposta'])==0),0,np.abs((dftr['Previsto'] / dftr['Resposta']) -1))

dftr.describe()
dftr['Modelado']=np.where(dftr['Erro']>0.33,dftr['Decay'],dftr['Previsto'])

dftr[dftr['Local'].isin(['Brazil','US/New York','US/New Jersey','US/Florida','Italy','Spain','France','Germany'])]
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



dt=pd.merge(dft,dftr[['Local','Modelado']],on='Local',how='left')

dt.head()
dt[dt['Local'].isin(['Brazil','US/New York','US/New Jersey','US/Illinois','US/California','Italy','Spain','France','Germany'])]
Beta0=0.9414

#Beta1=-0.1692

dft=dt.copy()

dft['Decay_Calculado']=dft['Decay']

dft['Decay']=dft['Modelado']

Beta1=dft['Decay']







#dft['Cres_2020-04-01']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-02']=dft['Cres_2020-04-01']*((dft['Cres_2020-04-01']*(Beta1)+Beta0))

#dft['Cres_2020-04-03']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-04']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-05']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-06']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-07']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-08']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-09']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-10']=dft['Cres_2020-04-09']*((dft['Cres_2020-04-09']*(Beta1)+Beta0))

#dft['Cres_2020-04-11']=dft['Cres_2020-04-10']*((dft['Cres_2020-04-10']*(Beta1)+Beta0))

#dft['Cres_2020-04-12']=dft['Cres_2020-04-11']*((dft['Cres_2020-04-11']*(Beta1)+Beta0))

#dft['Cres_2020-04-13']=dft['Cres_2020-04-12']*((dft['Cres_2020-04-12']*(Beta1)+Beta0))

#dft['Cres_2020-04-14']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

dft['Cres_2020-04-15']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

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

dft['Cres_2020-05-08']=dft['Cres_2020-05-07']*((dft['Cres_2020-05-07']*(Beta1)+Beta0))

dft['Cres_2020-05-09']=dft['Cres_2020-05-08']*((dft['Cres_2020-05-08']*(Beta1)+Beta0))

dft['Cres_2020-05-10']=dft['Cres_2020-05-09']*((dft['Cres_2020-05-09']*(Beta1)+Beta0))

dft['Cres_2020-05-11']=dft['Cres_2020-05-10']*((dft['Cres_2020-05-10']*(Beta1)+Beta0))

dft['Cres_2020-05-12']=dft['Cres_2020-05-11']*((dft['Cres_2020-05-11']*(Beta1)+Beta0))

dft['Cres_2020-05-13']=dft['Cres_2020-05-12']*((dft['Cres_2020-05-12']*(Beta1)+Beta0))

dft['Cres_2020-05-14']=dft['Cres_2020-05-13']*((dft['Cres_2020-05-13']*(Beta1)+Beta0))





#dft['2020-04-01']=(1+dft['Cres_2020-04-01'])*dft['2020-03-31']

#dft['2020-04-02']=(1+dft['Cres_2020-04-02'])*dft['2020-04-01']

#dft['2020-04-03']=(1+dft['Cres_2020-04-03'])*dft['2020-04-02']

#dft['2020-04-04']=(1+dft['Cres_2020-04-04'])*dft['2020-04-03']

#dft['2020-04-05']=(1+dft['Cres_2020-04-05'])*dft['2020-04-04']

#dft['2020-04-06']=(1+dft['Cres_2020-04-06'])*dft['2020-04-05']

#dft['2020-04-07']=(1+dft['Cres_2020-04-07'])*dft['2020-04-06']

#dft['2020-04-08']=(1+dft['Cres_2020-04-08'])*dft['2020-04-07']

#dft['2020-04-09']=(1+dft['Cres_2020-04-09'])*dft['2020-04-08']

#dft['2020-04-10']=(1+dft['Cres_2020-04-10'])*dft['2020-04-09']

#dft['2020-04-11']=(1+dft['Cres_2020-04-11'])*dft['2020-04-10']

#dft['2020-04-12']=(1+dft['Cres_2020-04-12'])*dft['2020-04-11']

#dft['2020-04-13']=(1+dft['Cres_2020-04-13'])*dft['2020-04-12']

#dft['2020-04-14']=(1+dft['Cres_2020-04-14'])*dft['2020-04-13']

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

dft['2020-05-08']=(1+dft['Cres_2020-05-08'])*dft['2020-05-07']

dft['2020-05-09']=(1+dft['Cres_2020-05-09'])*dft['2020-05-08']

dft['2020-05-10']=(1+dft['Cres_2020-05-10'])*dft['2020-05-09']

dft['2020-05-11']=(1+dft['Cres_2020-05-11'])*dft['2020-05-10']

dft['2020-05-12']=(1+dft['Cres_2020-05-12'])*dft['2020-05-11']

dft['2020-05-13']=(1+dft['Cres_2020-05-13'])*dft['2020-05-12']

dft['2020-05-14']=(1+dft['Cres_2020-05-14'])*dft['2020-05-13']
def make_decay_d(df):

    



    dft=df.pivot_table(index='Local',columns='Date',values='Mortalidade').reset_index()

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

    populacao=pd.read_csv('/kaggle/input/covidinformacoes/Population total (millions).csv',skiprows=1)

    populacao_65=pd.read_csv('/kaggle/input/covidinformacoes/Population ages 65 and older (millions).csv',skiprows=1)

    pop_urbana=pd.read_csv('/kaggle/input/covidinformacoes/Population urban ().csv',skiprows=1)

    energia_nr=pd.read_csv('/kaggle/input/covidinformacoes/Fossil fuel energy consumption ( of total energy consumption).csv',skiprows=1)

    pop_prisao=pd.read_csv('/kaggle/input/covidinformacoes/Prison population (per 100000 people).csv',skiprows=1)

    usu_internet=pd.read_csv('/kaggle/input/covidinformacoes/Internet users total ( of population).csv',skiprows=1)

    jovens_sem_oc=pd.read_csv('/kaggle/input/covidinformacoes/Youth not in school or employment ( ages 15-24).csv',skiprows=1)

    escola_anos=pd.read_csv('/kaggle/input/covidinformacoes/Mean years of schooling (years).csv',skiprows=1)

    

    #df_life=emprego_vul.copy()

    

    def func(x):

        x_new = 0

        try:

            x_new = float(x.replace(",", ""))

        except:

    #         print(x)

            x_new = np.nan

        return x_new

    

    tmp = populacao.iloc[:,1].values.tolist()

    populacao= populacao[['Country', '2018']]

    populacao['Country']=np.where(populacao['Country']=='United States','US',populacao['Country'])

    populacao['2018'] = populacao['2018'].apply(lambda x: func(x))

    populacao.columns = ['Country', 'populacao']

    

    tmp = populacao_65.iloc[:,1].values.tolist()

    populacao_65= populacao_65[['Country', '2018']]

    populacao_65['Country']=np.where(populacao_65['Country']=='United States','US',populacao_65['Country'])

    populacao_65['2018'] = populacao_65['2018'].apply(lambda x: func(x))

    populacao_65.columns = ['Country', 'populacao_65']    



    tmp = pop_urbana.iloc[:,1].values.tolist()

    pop_urbana= pop_urbana[['Country', '2018']]

    pop_urbana['Country']=np.where(pop_urbana['Country']=='United States','US',pop_urbana['Country'])

    pop_urbana['2018'] = pop_urbana['2018'].apply(lambda x: func(x))

    pop_urbana.columns = ['Country', 'pop_urbana']    



    tmp = energia_nr.iloc[:,1].values.tolist()

    energia_nr= energia_nr[['Country', '2014']]

    energia_nr['Country']=np.where(energia_nr['Country']=='United States','US',energia_nr['Country'])

    energia_nr['2014'] = energia_nr['2014'].apply(lambda x: func(x))

    energia_nr.columns = ['Country', 'energia_nr']    



    tmp = usu_internet.iloc[:,1].values.tolist()

    usu_internet= usu_internet[['Country', '2017']]

    usu_internet['Country']=np.where(usu_internet['Country']=='United States','US',usu_internet['Country'])

    usu_internet['2017'] = usu_internet['2017'].apply(lambda x: func(x))

    usu_internet.columns = ['Country', 'usu_internet']

    

    tmp = jovens_sem_oc.iloc[:,1].values.tolist()

    jovens_sem_oc= jovens_sem_oc[['Country', '2017']]

    jovens_sem_oc['Country']=np.where(jovens_sem_oc['Country']=='United States','US',jovens_sem_oc['Country'])

    jovens_sem_oc['2017'] = jovens_sem_oc['2017'].apply(lambda x: func(x))

    jovens_sem_oc.columns = ['Country', 'jovens_sem_oc']

    

    tmp = escola_anos.iloc[:,1].values.tolist()

    escola_anos= escola_anos[['Country', '2018']]

    escola_anos['Country']=np.where(escola_anos['Country']=='United States','US',escola_anos['Country'])

    escola_anos['2018'] = escola_anos['2018'].apply(lambda x: func(x))

    escola_anos.columns = ['Country', 'escola_anos']

    

    tmp = emprego_vul.iloc[:,1].values.tolist()

    emprego_vul = emprego_vul[['Country', '2018']]

    emprego_vul['Country']=np.where(emprego_vul['Country']=='United States','US',emprego_vul['Country'])

    emprego_vul['2018'] = emprego_vul['2018'].apply(lambda x: func(x))

    emprego_vul.columns = ['Country', 'Emprego_vulneravel']



    tmp = diox_carb.iloc[:,1].values.tolist()

    diox_carb = diox_carb[['Country', '2016']]

    diox_carb['Country']=np.where(diox_carb['Country']=='United States','US',diox_carb['Country'])

    diox_carb['2016'] = diox_carb['2016'].apply(lambda x: func(x))

    diox_carb.columns = ['Country', 'Dioxido_carbono']



    tmp = expec_vida.iloc[:,1].values.tolist()

    expec_vida = expec_vida[['Country', '2018']]

    expec_vida['Country']=np.where(expec_vida['Country']=='United States','US',expec_vida['Country'])

    expec_vida['2018'] = expec_vida['2018'].apply(lambda x: func(x))

    expec_vida.columns = ['Country', 'Expec_vida']

    

    tmp = gastos_saude.iloc[:,1].values.tolist()

    gastos_saude = gastos_saude[['Country', '2016']]

    gastos_saude['Country']=np.where(gastos_saude['Country']=='United States','US',gastos_saude['Country'])

    gastos_saude['2016'] = gastos_saude['2016'].apply(lambda x: func(x))

    gastos_saude.columns = ['Country', 'Gastos_saude']

    

    tmp = idh.iloc[:,1].values.tolist()

    idh = idh[['Country', '2018']]

    idh['Country']=np.where(idh['Country']=='United States','US',idh['Country'])

    idh['2018'] = idh['2018'].apply(lambda x: func(x))

    idh.columns = ['Country', 'IDH'] 

    

    tmp= idade_mediana.iloc[:,1].values.tolist()

    idade_mediana = idade_mediana[['Country', '2020']]

    idade_mediana['Country']=np.where(idade_mediana['Country']=='United States','US',idade_mediana['Country'])

    idade_mediana['2020'] = idade_mediana['2020'].apply(lambda x: func(x))

    idade_mediana.columns = ['Country', 'Idade']

    

    tmp= tuberculose.iloc[:,1].values.tolist()

    tuberculose = tuberculose[['Country', '2017']]

    tuberculose['Country']=np.where(tuberculose['Country']=='United States','US',tuberculose['Country'])

    tuberculose['2017'] = tuberculose['2017'].apply(lambda x: func(x))

    tuberculose.columns = ['Country', 'Tuberculose']

    

    tmp= desigualdade_exp_vida.iloc[:,1].values.tolist()

    desigualdade_exp_vida = desigualdade_exp_vida[['Country', '2018']]

    desigualdade_exp_vida['Country']=np.where(desigualdade_exp_vida['Country']=='United States','US',desigualdade_exp_vida['Country'])

    desigualdade_exp_vida['2018'] = desigualdade_exp_vida['2018'].apply(lambda x: func(x))

    desigualdade_exp_vida.columns = ['Country', 'desigualdade_exp_vida']

    

    tmp= desigualdade_ganhos.iloc[:,1].values.tolist()

    desigualdade_ganhos = desigualdade_ganhos[['Country', '2018']]

    desigualdade_ganhos['Country']=np.where(desigualdade_ganhos['Country']=='United States','US',desigualdade_ganhos['Country'])

    desigualdade_ganhos['2018'] = desigualdade_ganhos['2018'].apply(lambda x: func(x))

    desigualdade_ganhos.columns = ['Country', 'desigualdade_ganhos']

    

    tmp= desigualdade_idh_ajustado.iloc[:,1].values.tolist()

    desigualdade_idh_ajustado = desigualdade_idh_ajustado[['Country', '2018']]

    desigualdade_idh_ajustado['Country']=np.where(desigualdade_idh_ajustado['Country']=='United States','US',desigualdade_idh_ajustado['Country'])

    desigualdade_idh_ajustado['2018'] = desigualdade_idh_ajustado['2018'].apply(lambda x: func(x))

    desigualdade_idh_ajustado.columns = ['Country', 'desigualdade_idh_ajustado']

    

    tmp= desemprego.iloc[:,1].values.tolist()

    desemprego = desemprego[['Country', '2018']]

    desemprego['Country']=np.where(desemprego['Country']=='United States','US',desemprego['Country'])

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

    train = pd.merge(train, populacao, how='left', on='Country')

    train = pd.merge(train, populacao_65, how='left', on='Country')

    train = pd.merge(train, pop_urbana, how='left', on='Country')

    #train = pd.merge(train, pop_prisao, how='left', on='Country')

    train = pd.merge(train, energia_nr, how='left', on='Country')

    train = pd.merge(train, usu_internet, how='left', on='Country')

    train = pd.merge(train, jovens_sem_oc, how='left', on='Country')

    train = pd.merge(train, escola_anos, how='left', on='Country')

    

    dft=train.copy()

    dft['Pop_maior65']=np.where(((dft['populacao_65']==-99) | (dft['populacao']==-99)),0,dft['populacao_65'] / dft['populacao'])

    dft.rename({'Taiwan*':'Taiwan'}, axis=1)



    

    dft.fillna(-99,inplace=True)

    

    

    return (dft)









# Aplicar funções nas diferentes janelas temporais



dft0=make_decay_d(df0)

dft1=make_decay_d(df1)

dft2=make_decay_d(df2)

dft3=make_decay_d(df3)

dft4=make_decay_d(df4)

dft5=make_decay_d(df5)

dft6=make_decay_d(df6)

dft7=make_decay_d(df7)

dft8=make_decay_d(df8)

dft9=make_decay_d(df9)

dft10=make_decay_d(df10)

dft11=make_decay_d(df11)

dft12=make_decay_d(df12)

dft13=make_decay_d(df13)

dft14=make_decay_d(df14)

dft15=make_decay_d(df15)

dft16=make_decay_d(df16)

dft17=make_decay_d(df17)

dft18=make_decay_d(df18)

dft19=make_decay_d(df19)

dft20=make_decay_d(df20)

dft21=make_decay_d(df21)

dft22=make_decay_d(df22)

dft23=make_decay_d(df23)

dft24=make_decay_d(df24)

dft25=make_decay_d(df25)

dft26=make_decay_d(df26)

dft27=make_decay_d(df27)

dft28=make_decay_d(df28)

dft29=make_decay_d(df29)

dft30=make_decay_d(df30)

dft31=make_decay_d(df31)

dft32=make_decay_d(df32)

dft33=make_decay_d(df33)

dft34=make_decay_d(df34)

dft35=make_decay_d(df35)

dft36=make_decay_d(df36)

dft37=make_decay_d(df37)

dft38=make_decay_d(df38)

dft39=make_decay_d(df39)

dft40=make_decay_d(df40)

dft41=make_decay_d(df41)

dft42=make_decay_d(df42)

dft43=make_decay_d(df43)

dft44=make_decay_d(df44)

dft45=make_decay_d(df45)

dft46=make_decay_d(df46)

dft47=make_decay_d(df47)

dft48=make_decay_d(df48)

dft49=make_decay_d(df49)

dft50=make_decay_d(df50)

dft51=make_decay_d(df51)

dft52=make_decay_d(df52)

dft53=make_decay_d(df53)

dft54=make_decay_d(df54)

dft55=make_decay_d(df55)

dft56=make_decay_d(df56)

dft57=make_decay_d(df57)

dft58=make_decay_d(df58)

dft59=make_decay_d(df59)

dft60=make_decay_d(df60)

dft61=make_decay_d(df61)

dft62=make_decay_d(df62)

dft63=make_decay_d(df63)

dft64=make_decay_d(df64)

dft65=make_decay_d(df65)

dft66=make_decay_d(df66)

dft67=make_decay_d(df67)

dft68=make_decay_d(df68)

#dft69=make_decay(df19)







dftr_d=make_decay_d(dfr)





dfmodel_d=pd.concat([dft0,dft1,dft2,dft3,dft4,dft5,dft6

                  ,dft7,dft8,dft9,dft10,dft11,dft12

                  ,dft13,dft14,dft15,dft16,dft17

                  ,dft18,dft19,dft20,dft21,dft22,dft23

                  ,dft24,dft25,dft26,dft26,dft27,dft28

                  ,dft29,dft30,dft31,dft32,dft33,dft34

                  ,dft35,dft36,dft37,dft38,dft39,dft40

                  ,dft41,dft42,dft43,dft44,dft45,dft46

                  ,dft47,dft48,dft49,dft50,dft51,dft52

                  ,dft53,dft54,dft55,dft56,dft57,dft58

                  ,dft59,dft60,dft61,dft62

                   ,dft63,dft64

                   ,dft65,dft66,dft67,dft68

                   

                  

                  

                  ],ignore_index=True)



dfmodel_d.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dfmodel_d.columns]

dftr_d.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in dftr_d.columns]





dfmodel_d.fillna(-99,inplace=True)

dftr_d.fillna(-99,inplace=True)



resposta_d=dftr_d['Crescimento_2']

#dfteste_cat= dftr.drop(columns=['Crescimento_1','Crescimento_2','Decay'])

dfteste_d= dftr_d.drop(columns=['Local','Crescimento_1','Crescimento_2','Country','dia_15'])



y_d=dfmodel_d['Crescimento_2']

#variaveis_cat=dfmodel.drop(columns=['Crescimento_1','Crescimento_2','Decay'])

variaveis_d=dfmodel_d.drop(columns=['Local','Crescimento_1','Crescimento_2','Country','dia_15'])





X_trainD, X_testD,y_trainD,y_testD = train_test_split(variaveis_d, y_d,test_size=0.1)

#cX_train, cX_test,cy_train,cy_test = train_test_split(variaveis_cat, y,test_size=0.1)




from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clfGB = GradientBoostingRegressor(**params)



clfGB.fit(X_trainD, y_trainD)

rGB=clfGB.predict(dfteste_d)



clfRF = RandomForestRegressor()



clfRF.fit(X_trainD, y_trainD)

rRF=clfRF.predict(dfteste_d)



####LightGBM#######

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

              ,'num_leaves':[10,20,30,50,70]

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



grid.fit(X_trainD,y_trainD)



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





train_data=lgb.Dataset(X_trainD,label=y_trainD)

lgbm_cases= lgb.train(params,

               train_data,

               600,

               verbose_eval=4)





#####CatBoost######

train_pool_d = Pool(X_trainD,

                  label=y_trainD

                  #cat_features=['Local','Country'])

                 )

val_pool_d = Pool(X_testD,

                  label=y_testD

                  #cat_features=['Local','Country'])

               )

test_pool_d = Pool(dfteste_d,

                  label=resposta_d

                  #cat_features=['Local','Country'])

                )

model = CatBoostRegressor(objective='RMSE')



model.fit(train_pool_d, plot=True, eval_set=val_pool_d, verbose=500)
resp=model.predict(test_pool_d)

respLGB=lgbm_cases.predict(dfteste_d)

print("MSE CatBoost: %.4f"  %mean_squared_error(resp,resposta))

print("MSE GradientBoosting: %.4f" %mean_squared_error(rGB,resposta))

print("MSE RandomForest: %.4f" %mean_squared_error(rRF,resposta))

print("MSE LightGBM: %.4f" %mean_squared_error(respLGB,resposta))



dftr_d['Previsto']=resp

dftr_d['Resposta']=resposta_d



dftr_d[dftr_d['Local'].isin(['Brazil','US/New York','US/New Jersey','US/Florida','Italy','Spain','France','Germany'])]
dftr_d['Previsto']=np.where((dftr_d['Crescimento_2'])>(dftr_d['Crescimento_1']),dftr_d['Previsto'],dftr_d['Previsto']/7)

copy_dftr=dftr_d.copy()

dftr_d=df_f.pivot_table(index='Local',columns='Date',values='Mortalidade').reset_index()



dftr_copy=dftr_d.copy()

#dft.columns=Lista_colunas

C1=np.where(

        (dftr_d.iloc[: , -15].values)==0,

    (np.power(dftr_d.iloc[: , -8].values/((dftr_d.iloc[: , -15].values)+1),1/7)) -(1)

    ,(np.power(dftr_d.iloc[: , -8].values/((dftr_d.iloc[: , -15].values)),1/7)) -(1)

    )



C1=np.where(C1<0,0,C1)



C2=np.where(

        (dftr_d.iloc[: , -8].values)==0,

    (np.power(dftr_d.iloc[: , -1].values/((dftr_d.iloc[: , -8].values)+1),1/7)) -(1)

    ,(np.power(dftr_d.iloc[: , -1].values/((dftr_d.iloc[: , -8].values)),1/7)) -(1)

    )



C2=np.where(C2<0,0,C2)



dftr_d['Crescimento_1']=C1

dftr_d['Crescimento_2']=C2









Beta0=0.25

#Beta1=-0.1692



Beta1=copy_dftr['Previsto']







#dft['Cres_2020-04-01']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-02']=dft['Cres_2020-04-01']*((dft['Cres_2020-04-01']*(Beta1)+Beta0))

#dft['Cres_2020-04-03']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-04']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-05']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-06']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-07']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-08']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-09']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

#dft['Cres_2020-04-10']=dft['Cres_2020-04-09']*((dft['Cres_2020-04-09']*(Beta1)+Beta0))

#dft['Cres_2020-04-11']=dft['Cres_2020-04-10']*((dft['Cres_2020-04-10']*(Beta1)+Beta0))

#dft['Cres_2020-04-12']=dft['Cres_2020-04-11']*((dft['Cres_2020-04-11']*(Beta1)+Beta0))

#dft['Cres_2020-04-13']=dft['Cres_2020-04-12']*((dft['Cres_2020-04-12']*(Beta1)+Beta0))

#dft['Cres_2020-04-14']=dft['Crescimento_2']*((dft['Crescimento_2']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-15']=dftr_d['Crescimento_2']*((dftr_d['Crescimento_2']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-16']=dftr_d['Cres_2020-04-15']*((dftr_d['Cres_2020-04-15']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-17']=dftr_d['Cres_2020-04-16']*((dftr_d['Cres_2020-04-16']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-18']=dftr_d['Cres_2020-04-17']*((dftr_d['Cres_2020-04-17']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-19']=dftr_d['Cres_2020-04-18']*((dftr_d['Cres_2020-04-18']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-20']=dftr_d['Cres_2020-04-19']*((dftr_d['Cres_2020-04-19']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-21']=dftr_d['Cres_2020-04-20']*((dftr_d['Cres_2020-04-20']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-22']=dftr_d['Cres_2020-04-21']*((dftr_d['Cres_2020-04-21']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-23']=dftr_d['Cres_2020-04-22']*((dftr_d['Cres_2020-04-22']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-24']=dftr_d['Cres_2020-04-23']*((dftr_d['Cres_2020-04-23']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-25']=dftr_d['Cres_2020-04-24']*((dftr_d['Cres_2020-04-24']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-26']=dftr_d['Cres_2020-04-25']*((dftr_d['Cres_2020-04-25']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-27']=dftr_d['Cres_2020-04-26']*((dftr_d['Cres_2020-04-26']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-28']=dftr_d['Cres_2020-04-27']*((dftr_d['Cres_2020-04-27']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-29']=dftr_d['Cres_2020-04-28']*((dftr_d['Cres_2020-04-28']*(Beta1)+Beta0))

dftr_d['Cres_2020-04-30']=dftr_d['Cres_2020-04-29']*((dftr_d['Cres_2020-04-29']*(Beta1)+Beta0))



dftr_d['Cres_2020-05-01']=dftr_d['Cres_2020-04-30']*((dftr_d['Cres_2020-04-30']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-02']=dftr_d['Cres_2020-05-01']*((dftr_d['Cres_2020-05-01']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-03']=dftr_d['Cres_2020-05-02']*((dftr_d['Cres_2020-05-02']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-04']=dftr_d['Cres_2020-05-03']*((dftr_d['Cres_2020-05-03']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-05']=dftr_d['Cres_2020-05-04']*((dftr_d['Cres_2020-05-04']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-06']=dftr_d['Cres_2020-05-05']*((dftr_d['Cres_2020-05-05']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-07']=dftr_d['Cres_2020-05-06']*((dftr_d['Cres_2020-05-06']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-08']=dftr_d['Cres_2020-05-07']*((dftr_d['Cres_2020-05-07']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-09']=dftr_d['Cres_2020-05-08']*((dftr_d['Cres_2020-05-08']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-10']=dftr_d['Cres_2020-05-09']*((dftr_d['Cres_2020-05-09']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-11']=dftr_d['Cres_2020-05-10']*((dftr_d['Cres_2020-05-10']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-12']=dftr_d['Cres_2020-05-11']*((dftr_d['Cres_2020-05-11']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-13']=dftr_d['Cres_2020-05-12']*((dftr_d['Cres_2020-05-12']*(Beta1)+Beta0))

dftr_d['Cres_2020-05-14']=dftr_d['Cres_2020-05-13']*((dftr_d['Cres_2020-05-13']*(Beta1)+Beta0))
dftr_d['2020-04-15']=(1+dftr_d['Cres_2020-04-15'])*dftr_d['2020-04-14']

dftr_d['2020-04-16']=(1+dftr_d['Cres_2020-04-16'])*dftr_d['2020-04-15']

dftr_d['2020-04-17']=(1+dftr_d['Cres_2020-04-17'])*dftr_d['2020-04-16']

dftr_d['2020-04-18']=(1+dftr_d['Cres_2020-04-18'])*dftr_d['2020-04-17']

dftr_d['2020-04-19']=(1+dftr_d['Cres_2020-04-19'])*dftr_d['2020-04-18']

dftr_d['2020-04-20']=(1+dftr_d['Cres_2020-04-20'])*dftr_d['2020-04-19']

dftr_d['2020-04-21']=(1+dftr_d['Cres_2020-04-21'])*dftr_d['2020-04-20']

dftr_d['2020-04-22']=(1+dftr_d['Cres_2020-04-22'])*dftr_d['2020-04-21']

dftr_d['2020-04-23']=(1+dftr_d['Cres_2020-04-23'])*dftr_d['2020-04-22']

dftr_d['2020-04-24']=(1+dftr_d['Cres_2020-04-24'])*dftr_d['2020-04-23']

dftr_d['2020-04-25']=(1+dftr_d['Cres_2020-04-25'])*dftr_d['2020-04-24']

dftr_d['2020-04-26']=(1+dftr_d['Cres_2020-04-26'])*dftr_d['2020-04-25']

dftr_d['2020-04-27']=(1+dftr_d['Cres_2020-04-27'])*dftr_d['2020-04-26']

dftr_d['2020-04-28']=(1+dftr_d['Cres_2020-04-28'])*dftr_d['2020-04-27']

dftr_d['2020-04-29']=(1+dftr_d['Cres_2020-04-29'])*dftr_d['2020-04-28']

dftr_d['2020-04-30']=(1+dftr_d['Cres_2020-04-30'])*dftr_d['2020-04-29']



dftr_d['2020-05-01']=(1+dftr_d['Cres_2020-05-01'])*dftr_d['2020-04-30']

dftr_d['2020-05-02']=(1+dftr_d['Cres_2020-05-02'])*dftr_d['2020-05-01']

dftr_d['2020-05-03']=(1+dftr_d['Cres_2020-05-03'])*dftr_d['2020-05-02']

dftr_d['2020-05-04']=(1+dftr_d['Cres_2020-05-04'])*dftr_d['2020-05-03']

dftr_d['2020-05-05']=(1+dftr_d['Cres_2020-05-05'])*dftr_d['2020-05-04']

dftr_d['2020-05-06']=(1+dftr_d['Cres_2020-05-06'])*dftr_d['2020-05-05']

dftr_d['2020-05-07']=(1+dftr_d['Cres_2020-05-07'])*dftr_d['2020-05-06']

dftr_d['2020-05-08']=(1+dftr_d['Cres_2020-05-08'])*dftr_d['2020-05-07']

dftr_d['2020-05-09']=(1+dftr_d['Cres_2020-05-09'])*dftr_d['2020-05-08']

dftr_d['2020-05-10']=(1+dftr_d['Cres_2020-05-10'])*dftr_d['2020-05-09']

dftr_d['2020-05-11']=(1+dftr_d['Cres_2020-05-11'])*dftr_d['2020-05-10']

dftr_d['2020-05-12']=(1+dftr_d['Cres_2020-05-12'])*dftr_d['2020-05-11']

dftr_d['2020-05-13']=(1+dftr_d['Cres_2020-05-13'])*dftr_d['2020-05-12']

dftr_d['2020-05-14']=(1+dftr_d['Cres_2020-05-14'])*dftr_d['2020-05-13']
dftr_d[dftr_d['Local'].isin(['Brazil','US/New York','US/New Jersey','US/Florida','Italy','Spain','France','Germany'])]
dfm=df_f.pivot_table(index='Local',columns='Date',values='Fatalities').reset_index()

#dft.iloc[: , -8].values/((dft.iloc[: , -15].values)

mortes_adj=dfm.iloc[: , -1].values.sum() / dft_copy.iloc[: , -1].values.sum()

dft['mortes']=dfm.iloc[: , -1].values / dft_copy.iloc[: , -1].values

 

print(mortes_adj)

dft.head()
dft[dft['Local'].isin(['Brazil','US/New York','US/New Jersey','US/Florida','Italy','Spain','France','Germany'])]
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

       '2020-04-29', '2020-04-30','2020-05-01','2020-05-02','2020-05-03','2020-05-04','2020-05-05','2020-05-06','2020-05-07'

      ,'2020-05-08','2020-05-09','2020-05-10','2020-05-11','2020-05-12','2020-05-13','2020-05-14']]

dfi.head()
dfi['2020-05-07'].sum()
df = dfi.melt('Local', var_name='Date', value_name='ConfirmedCases')





df=pd.merge(df,dft[['Local','mortes']],on='Local',how='left')

df['Fatalities']=df['ConfirmedCases']*df['mortes']

df[df['Local']=='Brazil']
dfif=dftr_d[['Local', '2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04',

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

       '2020-04-29', '2020-04-30','2020-05-01','2020-05-02','2020-05-03','2020-05-04','2020-05-05','2020-05-06','2020-05-07'

      ,'2020-05-08','2020-05-09','2020-05-10','2020-05-11','2020-05-12','2020-05-13','2020-05-14']]

dffat = dfif.melt('Local', var_name='Date', value_name='Mortalidade')



dffat.tail()


df_test.head()

df_test=df_test.drop(columns=['ConfirmedCases','Fatalities'])

df_test['Date']=df_test['Date'].astype('str')

df_test=pd.merge(df_test,df[['Local','Date','ConfirmedCases','Fatalities']],on=['Local','Date'],how='left')

df_test.drop(columns='Mortalidade',inplace=True)

df_test.tail()
dftestefinal=pd.merge(df_test,dffat,on=['Local','Date'],how='left')

dftestefinal['Fatalities']=dftestefinal['ConfirmedCases']*dftestefinal['Mortalidade']

dftestefinal.tail()
dff=dftestefinal.sort_values(by='ConfirmedCases',ascending=False)

dff.head(20)


dfm = dfm.melt('Local', var_name='Date', value_name='Fatalities')



dfat=pd.merge(dftestefinal,dfm[['Local','Fatalities','Date']],on=['Local','Date'],how='left',suffixes=('_predicted','_real'))

dfat['Fatalities_real'].fillna('Vazio',inplace=True)

dfat['Fatalities']=np.where(dfat['Fatalities_real']=='Vazio',dfat['Fatalities_predicted'],dfat['Fatalities_real'])

dfat[dfat['Local']=='Brazil']
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