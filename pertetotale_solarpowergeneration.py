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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt   

import seaborn as sns   # more plots

sns.set()



from dateutil.relativedelta import relativedelta #  dates with style

from scipy.optimize import minimize    # for function minimization



import statsmodels.formula.api as smf  # statistics and 

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



from itertools import product# some useful functions



import warnings    # `do not disturbe` mode

warnings.filterwarnings('ignore')



%matplotlib inline

%config InlineBackend.figure_format = 'retina' 

import plotly.express as px

import plotly.graph_objects as go

sns.set_context('notebook')

pd.options.display.max_rows = 102
solar1GEN = pd.read_csv(r'/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True )

#index_col=['Time'], 

solar1GEN.sample(12)
solar1Sensor  = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True )

solar2Sensor = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True )

solar2GEN = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True )

solar2GEN["TOTAL_YIELDkWh"] =solar2GEN.TOTAL_YIELD*0.0036 # to kWh

solar2GEN.head(6)
print("Mean yield in kWh plant 2:",round(solar2GEN["TOTAL_YIELDkWh"].mean()),"   Mean yield in kWh plant 1:", round(solar1GEN["TOTAL_YIELD"].mean()))
solar1GEN['DATE'] = solar1GEN['DATE_TIME'].apply(lambda x:x.date())

solar2GEN['DATE'] = solar2GEN['DATE_TIME'].apply(lambda x:x.date())

solar1GEN['TIME'] = solar1GEN['DATE_TIME'].apply(lambda x:x.time())

solar2GEN['TIME'] = solar2GEN['DATE_TIME'].apply(lambda x:x.time())
solar1GEN= solar1GEN.drop("PLANT_ID", axis=1); 

solar2GEN= solar2GEN.drop("PLANT_ID", axis=1)



solar1GENdt= solar1GEN.set_index("DATE_TIME",  ) #inplace=True

solar2GENdt= solar2GEN.set_index("DATE_TIME",  )



solar1Sensor= solar1Sensor.drop("PLANT_ID", axis=1); 

solar2Sensor= solar2Sensor.drop("PLANT_ID", axis=1); 



solar1Sensordt= solar1Sensor.set_index("DATE_TIME",  )

solar2Sensordt= solar2Sensor.set_index("DATE_TIME",  )



solar1GEN= solar1GEN.sort_values(["SOURCE_KEY", "DATE_TIME"]) #sort_index()
solar1GEN= solar1GEN.replace({'1BY6WEcLGh8j5v7':"A1", '1IF53ai7Xc0U56Y':"A2", '3PZuoBAID5Wc2HD':"A3",

   '7JYdWkrLSPkdwr4':"A4", 'McdE0feGgRqW7Ca':"A5", 'VHMLBKoKgIrUVDU':"A6",

   'WRmjgnKYAwPKWDb':"A10", 'ZnxXDlPa8U1GXgE':"A11", 'ZoEaEvLYb1n2sOq':"A12",

   'adLQvlD726eNBSB':"A13", 'bvBOhCH3iADSZry':"A14", 'iCRJl6heRkivqQ3':"A15",

   'ih0vzX44oOqAx2f':"A16", 'pkci93gMrogZuBj':"A17", 'rGa61gmuvPhdLxV':"A18",

   'sjndEbLyjtCKgGv':"A19", 'uHbuxQJl8lW7ozc':"A20", 'wCURE6d3bPkepu2':"A21",

   'z9Y9gH1T5YWrNuG':"A22", 'zBIq5rxdHJRwDNY':"A23", 'zVJPv84UY57bAof':"A24", 'YxYtjZvoooNbGkE':"A25"})
code = {'1BY6WEcLGh8j5v7':"A1", '1IF53ai7Xc0U56Y':"A2", '3PZuoBAID5Wc2HD':"A3",'7JYdWkrLSPkdwr4':"A4", 'McdE0feGgRqW7Ca':"A5", 'VHMLBKoKgIrUVDU':"A6",

   'WRmjgnKYAwPKWDb':"A10", 'ZnxXDlPa8U1GXgE':"A11", 'ZoEaEvLYb1n2sOq':"A12", 'adLQvlD726eNBSB':"A13", 'bvBOhCH3iADSZry':"A14", 'iCRJl6heRkivqQ3':"A15",'ih0vzX44oOqAx2f':"A16", 'pkci93gMrogZuBj':"A17", 'rGa61gmuvPhdLxV':"A18",

   'sjndEbLyjtCKgGv':"A19", 'uHbuxQJl8lW7ozc':"A20", 'wCURE6d3bPkepu2':"A21", 'z9Y9gH1T5YWrNuG':"A22", 'zBIq5rxdHJRwDNY':"A23", 'zVJPv84UY57bAof':"A24", 'YxYtjZvoooNbGkE':"A25"}

label_df = pd.DataFrame.from_dict(code, orient='index', columns=['New_Code'])
solar2GEN= solar2GEN.replace({'4UPUqMRk7TRMgml':"B1", '81aHJ1q11NBPMrL':"B2", '9kRcWv60rDACzjR':"B3",'Et9kgGMDl729KT4':"B4", 'IQ2d7wF4YD8zU1Q':"B5", 'LYwnQax7tkwH5Cb':"B6",

       'LlT2YUhhzqhg5Sw':"B7", 'Mx2yZCDsyf6DPfv':"B8", 'NgDl19wMapZy17u':"B9",'PeE6FRyGXUgsRhN':"B10", 'Qf4GUc1pJu5T6c6':"B11", 'Quc1TzYxW2pYoWX':"B12",

       'V94E5Ben1TlhnDV':"B13", 'WcxssY2VbP4hApt':"B14", 'mqwcsP2rE7J0TFp':"B15",'oZ35aAeoifZaQzV':"B16", 'oZZkBaNadn6DNKz':"B17", 'q49J1IKaHRwDQnt':"B18",

       'rrq4fwE8jgrTyWY':"B19", 'vOuJvMaM2sgwLmb':"B20", 'xMbIugepa2P7lBB':"B21",'xoJJ8DcxJEcupym':"B22"})
sources1 =solar1GEN.SOURCE_KEY.unique(); sources2 =solar2GEN.SOURCE_KEY.unique()
def roundingP1(df):

    df['DC_POWER'] = round(df['DC_POWER'],2)

    df['AC_POWER'] = round(df['AC_POWER'],2)

    df['DAILY_YIELD'] = round(df['DAILY_YIELD'],2)

    df['TOTAL_YIELD'] = round(df['TOTAL_YIELD'],2)
def roundingP2(df):

    df['AMBIENT_TEMP'] = round(df['AMBIENT_TEMP'],2)

    df['MODULE_TEMP'] = round(df['MODULE_TEMP'],2)

    df['IRRADIATION'] = round(df['IRRADIATION'],4)
gen1_dailymax=solar1GENdt.resample("D").max()  # better when 1 selected

gen1_dailymax.head(15)
gen2_dailymax=solar2GENdt.resample("D").max()  # better when 1 selected

gen2_dailymax.head(5)
def Inverterextract(df):

    global plant1

    plant1= pd.DataFrame(index=gen1_dailymax.index, data=gen1_dailymax.iloc[:,0])  #

    

    for b in sources1:

        dfs= df[df.SOURCE_KEY== b] #

        dfsi=dfs.set_index("DATE_TIME")



        invert =dfsi.resample("D").agg({'AC_POWER':np.sum, "DAILY_YIELD":np.max}) #'DC_POWER': np.sum, "DAILY_YIELD":np.max,"TOTAL_YIELD":"max"  lambda x: np.std(x, ddof=1)

        invert.columns = invert.columns+'_'+str(b)

        plant1= pd.concat([plant1, invert], axis=1)

        next   #
def Inverterextract_2(df):

    global plant2

    plant2= pd.DataFrame( index=gen1_dailymax.index, data=gen1_dailymax.iloc[:,0]) # [2]

    

    for b in sources2:

        dfs= df[df.SOURCE_KEY== b] #

        dfsi=dfs.set_index("DATE_TIME")



        invert =dfsi.resample("D").agg({'AC_POWER':np.sum, "DAILY_YIELD":np.max}) #'DC_POWER': np.sum,"TOTAL_YIELD":"max"  lambda x: np.std(x, ddof=1)

        invert.columns = invert.columns+'_'+str(b)

        plant2= pd.concat([plant2, invert], axis=1)#

        next #
Inverterextract(solar1GEN)

#if isinstance(gen1_dailymax):

roundingP1(gen1_dailymax); gen1_dailymax.head()
Inverterextract_2(solar2GEN)
plant1.head(15)
Invertors1=plant1[["AC_POWER_A1","DAILY_YIELD_A1","AC_POWER_A2","DAILY_YIELD_A2","AC_POWER_A3","DAILY_YIELD_A3","AC_POWER_A4","DAILY_YIELD_A4",

                   "AC_POWER_A5","AC_POWER_A6","DAILY_YIELD_A6",#"AC_POWER_A7",

                   "AC_POWER_A10","DAILY_YIELD_A10","AC_POWER_A11","DAILY_YIELD_A11","AC_POWER_A12","DAILY_YIELD_A12","AC_POWER_A13","DAILY_YIELD_A13",

                   "AC_POWER_A14","DAILY_YIELD_A14","AC_POWER_A15","AC_POWER_A16","DAILY_YIELD_A16","AC_POWER_A17","DAILY_YIELD_A17","AC_POWER_A18","DAILY_YIELD_A18",

                   "AC_POWER_A19","DAILY_YIELD_A19","AC_POWER_A20","DAILY_YIELD_A20","AC_POWER_A21","DAILY_YIELD_A21","AC_POWER_A22","DAILY_YIELD_A22",

                   "AC_POWER_A23","DAILY_YIELD_A23","AC_POWER_A24","DAILY_YIELD_A24","AC_POWER_A25","DAILY_YIELD_A25"]]
filter_col = [col for col in Invertors1 if col.startswith('A')]

#filter_col

Invertors1_AC=Invertors1[filter_col]

Invertors1_AC.head()
plt.figure(figsize=(16, 6))

sns.lineplot(gen1_dailymax.index, data=plant1, y="AC_POWER_A1",label="AC_POWER_A1",)

sns.lineplot( gen1_dailymax.index, data=plant1, y="DAILY_YIELD_A1",label="DAILY_YIELD_A1", ); 

sns.lineplot(gen1_dailymax.index, data=plant2, y="AC_POWER_B1",label="AC_POWER_B1",)

sns.lineplot( gen1_dailymax.index, data=plant2, y="DAILY_YIELD_B1", label="DAILY_YIELD_B1",); 
plt.figure(figsize=(16, 6))

sns.lineplot(gen1_dailymax.index, data=plant1, y="AC_POWER_A19",label="AC plant1 A19",)

sns.lineplot( gen1_dailymax.index, data=plant1, y="DAILY_YIELD_A19",label="DAILY YIELD A19", ); 

sns.lineplot(gen1_dailymax.index, data=plant2, y="AC_POWER_B19",label="AC plant2 B19",)

sns.lineplot( gen1_dailymax.index, data=plant2, y="DAILY_YIELD_B19", label="DAILY YIELD plant2 B19",);
plt.figure(figsize=(16, 6))



sns.lineplot(x=solar1GEN.index, y="DAILY_YIELD", data=solar1GEN,color="forestgreen"); # kind="marker"

#plt.plot(ads.Ads, )

plt.title('E generated (15 min. data)')

plt.grid(True)

plt.show()
solar1GEN_A1= solar1GEN[solar1GEN.SOURCE_KEY =="A1"]

solar1GEN_A1
import datetime 

from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY, MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY, SECONDLY)

solar1GEN_A1.set_index("DATE_TIME", inplace=True)

plt.figure(figsize=(18, 5)); solar1GEN_A1.DAILY_YIELD.plot(); 
solar1GEN_A10= solar1GEN[solar1GEN.SOURCE_KEY =="A10"]

#solar1GEN_A10.sample(10)

solar1GEN_A10.set_index("DATE_TIME", inplace=True)

plt.figure(figsize=(18, 5)); solar1GEN_A1.DAILY_YIELD.plot(); 

solar1GEN_A10.DAILY_YIELD.plot( c='indigo'); 
print(solar1GEN.TOTAL_YIELD.max(), solar2GEN.TOTAL_YIELD.max(),"Ratio:" ,round(solar1GEN.TOTAL_YIELD.max()/solar2GEN.TOTAL_YIELD.max(),3))
print(solar1GEN.TOTAL_YIELD.max(), solar2GEN.TOTAL_YIELDkWh.max(),"Ratio:" ,round(solar1GEN.TOTAL_YIELD.max()/solar2GEN.TOTAL_YIELDkWh.max(),3))
solar1GEN["Age"] =solar1GEN.TOTAL_YIELD/solar1GEN.TOTAL_YIELD.max()



solar2GEN["Age"] =solar2GEN.TOTAL_YIELD/solar2GEN.TOTAL_YIELD.max()



solar2GEN["Age_kWh"] =solar2GEN.TOTAL_YIELD/solar2GEN.TOTAL_YIELDkWh.max()
sns.distplot(solar1GEN["Age"], bins=50); #  module 'seaborn' has no attribute 'displot'
sns.distplot(solar2GEN["Age"], bins=50); 
dta= pd.to_datetime("2020-06-17 09:30:00"); #dtb= pd.to_datetime("2020-05-23 19:00:00")

solar1GEN_17june= solar1GEN[solar1GEN.DATE_TIME== dta] #

solar1GEN_17june["Age"] =solar1GEN_17june.TOTAL_YIELD/solar1GEN_17june.TOTAL_YIELD.max()

sns.distplot(solar1GEN_17june["Age"], bins=50); 
dta= pd.to_datetime("2020-06-17 05:30:00"); dtz= pd.to_datetime("2020-06-17 19:30:00"); #

solar2GEN_17june= solar2GEN[solar2GEN.DATE_TIME>= dta] ; solar2GEN_17june= solar2GEN_17june[solar2GEN_17june.DATE_TIME< dtz] 

# no 0 values

solar2GEN_17june= solar2GEN_17june[solar2GEN_17june.DAILY_YIELD !=0]

solar2GEN_17june= solar2GEN_17june[solar2GEN_17june.AC_POWER !=0]

billion= solar2GEN_17june#[solar2GEN_17june.TOTAL_YIELD>1E9] # 2080.43 TOTAL_YIELD>1E9

billion.DAILY_YIELD.mean()
dta= pd.to_datetime("2020-06-17 05:30:00"); dtz= pd.to_datetime("2020-06-17 19:30:00"); #

solar1GEN_17june= solar1GEN[solar1GEN.DATE_TIME>= dta] ; solar1GEN_17june= solar1GEN_17june[solar1GEN_17june.DATE_TIME< dtz] 

# no 0 values

solar1GEN= solar1GEN[solar1GEN.DAILY_YIELD !=0]

solar1GEN= solar1GEN[solar1GEN.AC_POWER !=0]

million= solar1GEN_17june#[solar1GEN_17june.TOTAL_YIELD>1E6] # 3452.55 TOTAL_YIELD>1E6

million.DAILY_YIELD.mean()
lastyield2 =solar2GEN.groupby(["DATE","SOURCE_KEY"])[["DAILY_YIELD"]].last() # [["DAILY_YIELD"]]

lastyield2 = lastyield2.reset_index( level="SOURCE_KEY")

lastyield2.index = pd.to_datetime( lastyield2.index)

#lastyield2     average=  maximum= ,"SOURCE_KEY"
lastyield1 =solar1GEN.groupby(["DATE","SOURCE_KEY"])[["DAILY_YIELD"]].last() # [["DAILY_YIELD"]]

lastyield1 = lastyield1.reset_index( level="SOURCE_KEY")

lastyield1.index = pd.to_datetime( lastyield1.index)
#sns.set_theme()

cmap = sns.palplot(sns.diverging_palette(250.0/250, 145.0/250, s=200/250, l=9, n=22))
fig, axe= plt.subplots(2,1, figsize=(20, 12)) #

sns.lineplot(x=lastyield1.index, y="DAILY_YIELD", data=lastyield1,  ax=axe[0],lw=0.7, hue="SOURCE_KEY"); #color=cmap,

sns.lineplot(x=lastyield2.index, y="DAILY_YIELD", data=lastyield2,  ax=axe[1],lw=0.7, hue="SOURCE_KEY");

#plt.title('Daily yields generated (15 min. data)')

plt.grid(True); plt.show()
KWHyield2 =solar2GEN.groupby(["DATE","SOURCE_KEY"])[["TOTAL_YIELDkWh"]].last() # [["DAILY_YIELD"]]

KWHyield2 = KWHyield2.reset_index( level="SOURCE_KEY")



KWHyield2.index = pd.to_datetime( KWHyield2.index)
KWHyield1 =solar1GEN.groupby(["DATE","SOURCE_KEY"])[["TOTAL_YIELD"]].last() # max

KWHyield1 = KWHyield1.reset_index( level="SOURCE_KEY")

KWHyield1.index = pd.to_datetime( KWHyield1.index)
KWHyield1 =solar1GEN.groupby(["DATE_TIME","SOURCE_KEY"])[["TOTAL_YIELD"]].last() # max

KWHyield1 = KWHyield1.reset_index( level="SOURCE_KEY")

KWHyield1.index = pd.to_datetime( KWHyield1.index)
fig, axe= plt.subplots(1,1, figsize=(12, 8)) #p

sns.lineplot(x=KWHyield1.index, y="TOTAL_YIELD", data=KWHyield1, color=cmap, lw=0.7, hue="SOURCE_KEY"); #ax=axe[0]
fig, axe= plt.subplots(1,1, figsize=(12, 8)) #p

sns.lineplot(x=KWHyield2.index, y="TOTAL_YIELDkWh", data=KWHyield2, color=cmap, lw=0.7, hue="SOURCE_KEY"); #ax=axe[0]
KWHyield2 =solar2GEN.groupby(["DATE_TIME","SOURCE_KEY"])[["TOTAL_YIELDkWh"]].last() # [["DAILY_YIELD"]]

KWHyield2 = KWHyield2.reset_index( level="SOURCE_KEY")

KWHyield2.index = pd.to_datetime( KWHyield2.index)

KWHyield2    # average=  maximum= ,"SOURCE_KEY"
KWHyield2.to_csv("/kaggle/working/KWHyield2.csv")
KWHyield2 =solar2GEN.groupby(["DATE_TIME","SOURCE_KEY"])[["TOTAL_YIELDkWh"]].max() # [["DAILY_YIELD"]]

KWHyield2 = KWHyield2.reset_index( level="SOURCE_KEY")

KWHyield2.index = pd.to_datetime( KWHyield2.index)
fig, axe= plt.subplots(1,1, figsize=(20, 12)) #

plt.get_cmap('jet')

sns.lineplot(x=KWHyield2.index, y="TOTAL_YIELDkWh", data=KWHyield2, color=cmap, lw=0.7, hue="SOURCE_KEY"); #ax=axe[0]
B2 =KWHyield2[KWHyield2.SOURCE_KEY =="B2"]; 

fig, axe= plt.subplots(1,1, figsize=(20, 4))

sns.lineplot(x=B2.index, y="TOTAL_YIELDkWh", data=B2); 
B2.head()
M24mayblib= B2[(B2.index>="2020-05-24 06:00:00") &(B2.index<"2020-05-24 23:00:00")]

M24may_2sensor = solar2Sensor[(solar2Sensor.DATE_TIME>="2020-05-24 06:00:00") &(solar2Sensor.DATE_TIME<"2020-05-24 23:00:00")]
fig, axe= plt.subplots(1,1, figsize=(12, 5))

sns.lineplot( x=M24may_2sensor.index ,y=M24may_2sensor.MODULE_TEMPERATURE, data=M24may_2sensor)#"DATE_TIME"

sns.lineplot(  x=M24may_2sensor.index,y=M24mayblib.TOTAL_YIELDkWh/50000, data=M24mayblib); 

plt.grid(b=True,which='minor', axis="both"); plt.ylim=(25,90)

axe.set( ylim=(25,70));  
fig, axe= plt.subplots(1,1, figsize=(12, 5))

sns.lineplot( x=M24may_2sensor.index,y=M24may_2sensor.MODULE_TEMPERATURE, data=M24may_2sensor)

sns.lineplot(  x=M24may_2sensor.index,y=M24mayblib.TOTAL_YIELDkWh/50000, data=M24mayblib); 

plt.grid(b=True,which='minor', axis="both"); 

axe.set( ylim=(5,90));  
B3 =KWHyield2[KWHyield2.SOURCE_KEY =="B3"]; 

fig, axe= plt.subplots(1,1, figsize=(18, 4))

sns.lineplot(  x=B3.index,y=B3.TOTAL_YIELDkWh, data=B3); 
M24maybliB3= B3[(B3.index>="2020-06-02 06:00:00") &(B3.index<"2020-06-04 23:00:00")]

M24may_2sensorB3 = solar2Sensor[(solar2Sensor.DATE_TIME>="2020-06-02 06:00:00") &(solar2Sensor.DATE_TIME<"2020-06-04 23:00:00")]

fig, axe= plt.subplots(1,1, figsize=(12, 5))

sns.lineplot( x=M24may_2sensorB3.index,y=M24may_2sensorB3.MODULE_TEMPERATURE, data=M24may_2sensorB3)

sns.lineplot(  x=M24may_2sensorB3.index,y=M24maybliB3.TOTAL_YIELDkWh/50000, data=M24maybliB3); 

plt.grid(b=True,which='minor', axis="both"); plt.title("Inverter B3 ECG")

axe.set( ylim=(5,90)); 
billion.DAILY_YIELD.mean()/billion.DAILY_YIELD.max() #0.4534437633686215
million.DAILY_YIELD.mean()/million.DAILY_YIELD.max()#0.5236844125145991
dta= pd.to_datetime("2020-05-23 05:30:00"); dtb= pd.to_datetime("2020-05-23 19:30:00")

Sensor2dt_mei23_= solar2Sensordt[solar2Sensordt.index>= dta] #"2020-5-23 05:30:00"

Sensor2dt_mei23_= Sensor2dt_mei23_[Sensor2dt_mei23_.index <=dtb]; #
dta= pd.to_datetime("2020-05-01 05:30:00"); dtb= pd.to_datetime("2020-05-31 19:30:00")

Sensor2dt_mei= solar2Sensordt[solar2Sensordt.index>= dta] #"2020-5-23 05:30:00"

Sensor2dt_mei= Sensor2dt_mei[Sensor2dt_mei.index <=dtb]; #
mei23B_merg = Sensor2dt_mei23_.merge(solar2GEN,left_index=True, right_on="DATE_TIME" )
mei23B_merg= mei23B_merg.loc[(mei23B_merg.AC_POWER !=0)] # weed out 0

TOTAL_YIELDkWh_avg= np.mean(mei23B_merg.TOTAL_YIELDkWh); TOTAL_YIELDkWh_avg #
mei23B_merg.sample()
DAILY_YIELD_avg= np.mean(mei23B_merg.DAILY_YIELD); DAILY_YIELD_avg #DAILY_YIELD
mei23B_merg= mei23B_merg.rename(columns={"SOURCE_KEY_y": "SOURCE_KEY"})



bagger= mei23B_merg.loc[(mei23B_merg.MODULE_TEMPERATURE >50)& ( mei23B_merg.DAILY_YIELD < DAILY_YIELD_avg)&(mei23B_merg.AC_POWER !=0)]

baggermean= bagger.groupby("SOURCE_KEY")["MODULE_TEMPERATURE"].agg( size= np.size)



baggermean.tail()
bagger_max= bagger.groupby("SOURCE_KEY")["MODULE_TEMPERATURE"].agg( Min= min, mean= np.mean,Max= max, size= np.size)
fig, ax= plt.subplots(1,1, figsize=(12, 5) ); bagger_maxx=bagger_max.loc[:,["Min","mean","Max"]]



sns.scatterplot(  data=bagger_maxx, ax=ax);
mei_B_merg = Sensor2dt_mei.merge(solar2GEN,left_index=True, right_on="DATE_TIME" )



mei_B_merg = mei_B_merg.loc[(mei_B_merg.AC_POWER !=0)]

mei_B_merg.tail()
DAILY_YIELD_avg1= np.mean(mei_B_merg.DAILY_YIELD); DAILY_YIELD_avg1 #
mei_B_merg= mei_B_merg.rename(columns={"SOURCE_KEY_y": "SOURCE_KEY"})



bagB= mei_B_merg.loc[(mei_B_merg.MODULE_TEMPERATURE >50)& ( mei_B_merg.DAILY_YIELD < DAILY_YIELD_avg)& (mei_B_merg.AC_POWER !=0)]

bagB_mean= bagB.groupby("SOURCE_KEY")["MODULE_TEMPERATURE"].agg( Min= min, mean= np.mean, Max= max, size= np.size) #



bagB_mean.head()
fig, ax= plt.subplots(1,1, figsize=(15, 7) ); bagB_meanx=bagB_mean.loc[:,["Min","mean","Max"]]



sns.scatterplot(  data=bagB_meanx, ax=ax); #

ax2 = plt.twinx()

sns.scatterplot(x=bagB_mean.index, y="size",data=bagB_mean, color="red",marker="^", label="Size",ax=ax2); #"SOURCE_KEY"

ax2.figure.legend();
dta= pd.to_datetime("2020-05-23 05:30:00"); dtb= pd.to_datetime("2020-05-23 19:30:00")

Sensor1dt_mei23_= solar1Sensordt[solar1Sensordt.index>= dta] #"2020-5-23 05:30:00"

Sensor1dt_mei23_= Sensor1dt_mei23_[Sensor1dt_mei23_.index <=dtb]; #
mei23A_merg = Sensor1dt_mei23_.merge(solar1GEN,left_index=True, right_on="DATE_TIME" )
DAILY_YIELD_avg1= np.mean(mei23A_merg.DAILY_YIELD); DAILY_YIELD_avg1 #DAILY_YIELD
bagA= mei23A_merg.loc[(mei23A_merg.MODULE_TEMPERATURE >50)& ( mei23A_merg.DAILY_YIELD < DAILY_YIELD_avg)& (mei23A_merg.AC_POWER !=0)]

bagA_mean= bagA.groupby("SOURCE_KEY_y")["MODULE_TEMPERATURE"].agg( Min= min, mean= np.mean, Max= max, ) # size= np.size



bagA_mean.head()
fig, ax= plt.subplots(1,1, figsize=(12, 5) ); #bagA_meanx=bagA_mean.loc[:,["Min","mean","Max"]]



sns.scatterplot(  data=bagA_mean, ax=ax); # hue="SOURCE_KEY"
solar1GEN["DC_POWER"] = solar1GEN.DC_POWER/10

solar1GENdt["DC_POWER"]= solar1GENdt.DC_POWER/10

df1_DC = solar1GENdt.DC_POWER.resample('D').sum()

df1_AC = solar1GENdt.AC_POWER.resample('D').sum()

df1_DY = solar1GENdt.DAILY_YIELD.resample('D').last() #.max()

df1_TY = solar1GENdt.TOTAL_YIELD.resample('D').last() #.max()

df1_DC.tail()
df = pd.merge( df1_DC, df1_AC, left_index=True, right_index=True) 

df_ = pd.merge( df1_DY ,df1_TY, left_index=True, right_index=True) 

df_1 = pd.merge( df ,df_, left_index=True, right_index=True) 



df_1["Effic"]=df_1.AC_POWER /df_1.DC_POWER 

roundingP1(df_1)

df_1.tail(15)
#df_1.info()
df2_DC = solar2GENdt.DC_POWER.resample('D').sum()

df2_AC = solar2GENdt.AC_POWER.resample('D').sum()

df2_DY = solar2GENdt.DAILY_YIELD.resample('D').last()

df2_TY = solar2GENdt.TOTAL_YIELD.resample('D').last()

df2_DC.tail()
df = pd.merge( df2_DC, df2_AC, left_index=True, right_index=True) 

df_ = pd.merge( df2_DY ,df2_TY, left_index=True, right_index=True) 



df_2 = pd.merge( df ,df_, left_index=True, right_index=True) 
#df_2.info()
fig, ax= plt.subplots(2,2, figsize=(16, 10), sharex=True, sharey=True)

sns.pointplot(df_1.index, data=df_1, y=df_1.AC_POWER/85,label="AC POWER 1", join=False, ci="sd", ax=ax[0,0], ); #units="AC POWER 1"

sns.pointplot( df_1.index, data=df_1, y="DAILY_YIELD",label="DAILY YIELD 1", join=False, ci="sd", ax=ax[0,1]); plt.xticks([])

sns.pointplot(df_1.index, data=df_2, y=df_2.AC_POWER/85,label="AC POWER 2", join=False, ci="sd", ax=ax[1,0]); plt.xticks([])

sns.pointplot( df_1.index, data=df_2, y="DAILY_YIELD", label="DAILY_YIELD 2",join=False, ci="sd", ax=ax[1,1]);  # 

plt.xticks([]);
dta= pd.to_datetime("2020-05-23 05:30:00"); dtb= pd.to_datetime("2020-05-23 19:00:00")

solar1GEN23MAY= solar1GEN[solar1GEN.DATE_TIME>= dta] #"2020-5-23 05:30:00"

solar1GEN23MAY= solar1GEN23MAY[solar1GEN23MAY.DATE_TIME <dtb]; #"2020-5-23 19:30:00"
plt.figure(figsize=(16, 9))

ax= sns.lineplot(x=solar1GEN23MAY.DATE_TIME, y="DC_POWER", data=solar1GEN23MAY,color="forestgreen"); # kind="marker"

sns.lineplot(x=solar1GEN23MAY.DATE_TIME, y="AC_POWER", data=solar1GEN23MAY,color="indigo", ax=ax); 

plt.title('E generated (23 MAY)')

plt.grid("True"); 

plt.ylim=(250,1400); #plt.show()
dta= pd.to_datetime("2020-05-23 05:30:00"); dtb= pd.to_datetime("2020-05-23 19:00:00")

solar1GEN_mei23_= solar1GEN[solar1GEN.DATE_TIME>= dta] # 2nd version !

solar1GEN_mei23_= solar1GEN_mei23_[solar1GEN_mei23_.DATE_TIME <dtb]; #"2020-5-23 19:30:00"
solar1GEN_mei23_piv= solar1GEN_mei23_.pivot(index="SOURCE_KEY" , columns="DATE_TIME", values= "AC_POWER")
solar1GEN_mei23_piv2= solar1GEN_mei23_.pivot(index="DATE_TIME", columns="SOURCE_KEY", values= "AC_POWER")

solar1GEN_mei23_piv2.head()
plt.figure(figsize=(18, 15))

sns.heatmap( data=solar1GEN_mei23_piv);
#solar1GEN.head()
solar1GEN_piv= solar1GEN[(solar1GEN.TIME>=  pd.to_datetime("06:00:00").time()) & (solar1GEN.TIME<=  pd.to_datetime("22:00:00").time())]

solar1GEN_piv= solar1GEN_piv.pivot_table( index="DATE_TIME", columns="SOURCE_KEY", values= "AC_POWER")#.max(

solar1GEN_piv
solar1GEN_piv= solar1GEN_piv.dropna()

#solar1GEN_piv['TIME'] = solar1GEN_piv.index.apply(lambda x:x.time()) # ['DATE_TIME']

#solar1GEN_piv = solar1GEN_piv[(solar1GEN_piv.TIME >  pd.to_datetime("06:00:00")) & (solar1GEN_piv.TIME <=  pd.to_datetime("22:00:00"))]
plt.figure(figsize=(18, 14))

sns.heatmap(data=solar1GEN_piv, ); #linewidths=0.1,  
groupfallout= ["2020-05-15","2020-05-19","2020-05-21","2020-05-22","2020-05-27", "2020-05-29", 

               "2020-06-02", "2020-06-04", "2020-06-05", "2020-06-06"]

dayends=["2020-05-15 23:00:00","2020-05-19 23:00:00","2020-05-21 23:00:00","2020-05-22 23:00:00","2020-05-27 23:00:00","2020-05-29 23:00:00",

         "2020-06-02 23:00:00", "2020-06-04 23:00:00","2020-06-05 23:00:00","2020-06-06 23:00:00"]

dfx=pd.DataFrame()



df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-05-15 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-05-15 23:00:00"))]

dfx=dfx.append(df)

df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-05-19 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-05-19 23:00:00"))]

dfx=dfx.append(df)

df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-05-21 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-05-21 23:00:00"))]

dfx=dfx.append(df)

df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-05-22 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-05-22 23:00:00"))]

dfx=dfx.append(df)

df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-05-27 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-05-27 23:00:00"))]

dfx=dfx.append(df)

df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-05-29 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-05-29 23:00:00"))]

dfx=dfx.append(df)

df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-06-02 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-06-02 23:00:00"))]

dfx=dfx.append(df)

df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-06-04 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-06-04 23:00:00"))]

dfx=dfx.append(df)

df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-06-05 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-06-05 23:00:00"))]

dfx=dfx.append(df)

df = solar1GEN[(solar1GEN.DATE_TIME >  pd.to_datetime("2020-06-06 06:00:00")) & (solar1GEN.DATE_TIME <=  pd.to_datetime("2020-06-06 23:00:00"))]

dfx=dfx.append(df)
outgroup=["A5","A11", "A12","A14","A16", "A18","A19","A21", "A22","A25"]

solar1GEN_out= dfx.loc[dfx.SOURCE_KEY.isin(outgroup) ]

solar1GEN_out= solar1GEN_out.set_index("DATE_TIME")

solar1GEN_out_merg = solar1Sensordt.merge(solar1GEN_out, left_index=True, right_index=True ) # solar1GEN_A12
def Rollingsum(df,t):

    df["AMB_rol"+str(t)] =df.AMBIENT_TEMPERATURE.rolling(t).sum()

    df["MOD_rol"+str(t)] =df.MODULE_TEMPERATURE.rolling(t).sum()

    df["D_Y_dif"+str(t)] =df.DAILY_YIELD.diff(1)  #rolling(t)

    df["D_Y_rol"+str(t)] =df["D_Y_dif"+str(t)].rolling(t).sum()

    df["AC_rol"+str(t)] =df.AC_POWER.rolling(t).sum()

    df["DC_rol"+str(t)] =df.DC_POWER.rolling(t).sum()
Rollingsum(solar1GEN_out_merg,4)
#solar1GEN_out_merg.sample(6)
solar1GEN_out_merghigh =solar1GEN_out_merg.loc[solar1GEN_out_merg.MODULE_TEMPERATURE> 30] 

solar1GEN_out_merghigh.head()
plt.figure(figsize=(20, 10))

sns.lineplot(x=solar1GEN_out_merghigh.index, y="MODULE_TEMPERATURE", data=solar1GEN_out_merghigh, lw=0.85); # "DATE_TIME"

sns.lineplot(x=solar1GEN_out_merghigh.index, y="AMBIENT_TEMPERATURE", data=solar1GEN_out_merghigh, lw=0.95); 

plt.xlim( pd.to_datetime("2020-05-29"), pd.to_datetime("2020-06-06")); 

plt.ylim=(25,65); 
sns.scatterplot(x="MODULE_TEMPERATURE" ,y='AMBIENT_TEMPERATURE', data=solar1GEN_out_merghigh, hue="AC_POWER",s=7); 
sns.scatterplot(x="MODULE_TEMPERATURE" ,y='AMBIENT_TEMPERATURE', data=solar1GEN_out_merghigh, hue="DAILY_YIELD",s=8);
dfx=pd.DataFrame()



df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-05-15 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-05-15 23:00:00"))]

dfx=dfx.append(df)

df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-05-19 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-05-19 23:00:00"))]

dfx=dfx.append(df)

df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-05-21 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-05-21 23:00:00"))]

dfx=dfx.append(df)

df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-05-22 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-05-22 23:00:00"))]

dfx=dfx.append(df)

df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-05-27 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-05-27 23:00:00"))]

dfx=dfx.append(df)

df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-05-29 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-05-29 23:00:00"))]

dfx=dfx.append(df)

df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-06-02 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-06-02 23:00:00"))]

dfx=dfx.append(df)

df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-06-04 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-06-04 23:00:00"))]

dfx=dfx.append(df)

df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-06-05 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-06-05 23:00:00"))]

dfx=dfx.append(df)

df = solar2GEN[(solar2GEN.DATE_TIME >  pd.to_datetime("2020-06-06 06:00:00")) & (solar2GEN.DATE_TIME <=  pd.to_datetime("2020-06-06 23:00:00"))]

dfx=dfx.append(df)
solar2GEN_out= dfx#.loc[dfx.SOURCE_KEY.isin(outgroup) ]

solar2GEN_out= solar2GEN_out.set_index("DATE_TIME")

solar2GEN_out_merg = solar1Sensordt.merge(solar2GEN_out, left_index=True, right_index=True ) # solar2GEN_A12
solar2GEN_out_merghigh =solar2GEN_out_merg.loc[solar2GEN_out_merg.MODULE_TEMPERATURE> 30] 
plt.figure(figsize=(20, 12))

sns.lineplot(x=solar2GEN_out_merghigh.index , y="MODULE_TEMPERATURE", data=solar2GEN_out_merghigh, lw=0.85); 

sns.lineplot(x=solar2GEN_out_merghigh.index , y="AMBIENT_TEMPERATURE", data=solar2GEN_out_merghigh, lw=0.95); 

plt.xlim(pd.to_datetime("2020-05-21") ,pd.to_datetime("2020-05-30")); plt.ylim=(30,63); 
sns.scatterplot(x="MODULE_TEMPERATURE" ,y='AMBIENT_TEMPERATURE', data=solar2GEN_out_merghigh, hue="AC_POWER",s=9); 
def Rollingsum(df,t):

    df["AMB_rol"+str(t)] =df.AMBIENT_TEMPERATURE.rolling(t).sum()

    df["MOD_rol"+str(t)] =df.MODULE_TEMPERATURE.rolling(t).sum()

    df["D_Y_dif"+str(t)] =df.DAILY_YIELD.diff(1)  #rolling(t)

    df["D_Y_rol"+str(t)] =df["D_Y_dif"+str(t)].rolling(t).sum()

    df["AC_rol"+str(t)] =df.AC_POWER.rolling(t).sum()

    df["DC_rol"+str(t)] =df.DC_POWER.rolling(t).sum()
Rollingsum(solar1GEN_out_merg,4)
solar1GEN_out_merg.sample(1)
rol4 =solar1GEN_out_merg[["AMB_rol4","MOD_rol4","D_Y_rol4","AC_rol4", "DC_rol4"]] # "DATE_TIME",

rol4= rol4[(rol4.index >"2020-06-04 05:00:00") & (rol4.index <"2020-06-05")]

rol4= rol4[rol4.D_Y_rol4 > -2000]



plt.style.use("ggplot")

fig, ax= plt.subplots(1,1, figsize=(16, 10))

sns.lineplot(x=rol4.index, y=rol4.MOD_rol4, data=rol4, label="MOD_rol4", lw=0.95); 

sns.lineplot(x=rol4.index , y=rol4.D_Y_rol4*2, data=rol4, label="D_Y_rol4 $*2$", lw=0.95); 

sns.lineplot(x=rol4.index, y= rol4.DC_rol4/18, data=rol4, label="DC_rol4 $/18$", lw=0.95); 

plt.axhline(y=180, xmin=0, xmax=1, ls=":", color="violet") # 45°C

#plt.gca()

dt04 =pd.to_datetime("2020-06-04 05:00:00"); dt05 =pd.to_datetime("2020-06-04 21:00:00"); 

ax.set_xlim( dt04, dt05) #plt.xlim( dt04, dt05)

ax.set_ylim(75,275)

plt.grid(True)

plt.title("Produced current and yield for plant 1, 2020-06-04"); 
Rollingsum(solar2GEN_out_merg,4)



rol4 =solar2GEN_out_merg[["AMB_rol4","MOD_rol4","D_Y_rol4","AC_rol4", "DC_rol4"]] # "DATE_TIME",

rol4= rol4[(rol4.index >="2020-06-04 05:00:00") & (rol4.index <"2020-06-05")]

rol4= rol4[rol4.D_Y_rol4 > -2000]



plt.style.use("ggplot")

fig, ax= plt.subplots(1,1, figsize=(16, 10))

sns.lineplot(x=rol4.index, y=rol4.MOD_rol4, data=rol4,label="MOD_rol4", lw=0.95);  #DATE_TIME

sns.lineplot(x=rol4.index, y=rol4.D_Y_rol4*4, data=rol4,label="D_Y_rol4 $*4$", lw=0.95); 

sns.lineplot(x=rol4.index, y= rol4.DC_rol4/18, data=rol4,label="DC_rol4 $/18$", lw=0.95); 

ax.set_xlim(pd.to_datetime("2020-06-04 06:00:00"),pd.to_datetime("2020-06-04 21:00:00"))

ax.set_ylim(50,250)

plt.axhline(y=180, xmin=0, xmax=1, ls=":") # 45°C

plt.title("Produced current and yield for plant2, 2020-06-04"); 
sensors1_list =solar1Sensor.SOURCE_KEY.unique()

solar1Sensor= solar1Sensor.drop(["SOURCE_KEY"], axis=1) # "PLANT_ID",

solar1Sensordt= solar1Sensor.set_index("DATE_TIME",  ) #inplace=True

solar1Sensordt.head()
sensors2_list =solar2Sensor.SOURCE_KEY.unique()

solar2Sensor= solar2Sensor.drop(["SOURCE_KEY"], axis=1) # "PLANT_ID",

solar2Sensordt= solar2Sensor.set_index("DATE_TIME",  ) #inplace=True
fig, ax=  plt.subplots(1,1, figsize=(18, 7.6))  

plt.gca()  

sns.lineplot( data=solar1Sensordt);  
dta= pd.to_datetime("2020-05-23 05:30:00"); dtb= pd.to_datetime("2020-05-23 19:30:00")

Sensor1dt_mei23_= solar1Sensordt[solar1Sensordt.index>= dta] #"2020-5-23 05:30:00"

Sensor1dt_mei23_= Sensor1dt_mei23_[Sensor1dt_mei23_.index <=dtb]; #
dta= pd.to_datetime("2020-05-01 05:30:00"); dtb= pd.to_datetime("2020-05-31 19:30:00")

Sensor2dt_mei= solar2Sensordt[solar2Sensordt.index>= dta] #"2020-5-23 05:30:00"

Sensor2dt_mei= Sensor2dt_mei[Sensor2dt_mei.index <=dtb]; #
solar1GEN_mei23_.head()
mei23_merg =solar1GEN_mei23_.merge( solar1Sensordt, right_index=True, left_on="DATE_TIME")
mei23B_merg = Sensor2dt_mei23_.merge(solar2GEN,left_index=True, right_on="DATE_TIME" )
mei_B_merg = Sensor2dt_mei.merge(solar2GEN,left_index=True, right_on="DATE_TIME" )