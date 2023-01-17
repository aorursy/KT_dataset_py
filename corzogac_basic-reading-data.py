!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
import os
import pandas as pd
os.chdir("/kaggle")
os.chdir("input")
os.chdir("magdalena-colombia-data")
import FunctionsTools as FT
os.chdir("Magdalena_Data")
os.chdir("Magdalena_Data")

os.chdir("Inputs")


Start='2000-06-04'
End='2010-12-31'
L=os.listdir()
L
  
def Direct(i):
    Root='/kaggle/input/magdalena-colombia-data/Magdalena_Data/'
    if(i==0):
        os.chdir(Root)
    if(i==1):
        os.chdir( Root+'Magdalena_Data/Inputs/Clima')
    if(i==2):
        os.chdir( Root+'Magdalena_Data/Inputs/Hidrologia Embalses')
    if(i==3):
        os.chdir( Root+'Magdalena_Data/Outputs/Calibration dataset/Caudales')
    if(i==4):
        os.chdir( Root+'Magdalena_Data/Outputs/Calibration dataset/Energia')
    if(i==5):
        os.chdir( Root+'Magdalena_Data/Outputs/Calibration dataset/Pesquerias')
    #print(os.getcwd())
    #print(os.listdir())
    return os.listdir()


def Pesca():
    L=Direct(5,)
    R=pd.read_excel(L[0],sheet_name=[0],index_col=0,parse_dates=True)
    P=R[0]
    P.index = pd.to_datetime(P.index)

    #L=P['localidad'].unique()
    #P['localidad']=P['localidad'].astype("category")
    #P['localidad'].cat.set_categories(L,inplace=True)
    
    L=P['especie'].unique()
    P['especie']=P['especie'].astype("category")
    P['especie'].cat.set_categories(L,inplace=True)
    
    
    PvT1=pd.pivot_table(P,values=['biomasa (Toneladas)'],columns=['especie'],index=['año'])
    
    PvT1.dropna(axis=1,thresh=15,inplace=True)

    Te=P[(P['especie'] =='bocachico') | (P['especie'] =='bagre rayado')| (P['especie'] =='nicuro')]
    
    PvT=pd.pivot_table(Te,values=['biomasa (Toneladas)'],columns=['especie'],index=['año'])
    

    Direct(0,)
    return PvT,PvT1

Pes1,Pes2=Pesca()
Pes1.plot()
Pes2.plot()
def Caudales():
    L=Direct(3)
    R=pd.read_excel(L[0],sheet_name=[0],index_col=0,parse_dates=True)
    P=R[0]
    P.index = pd.to_datetime(P.index)
    Direct(0)
    return P
C=Caudales()
C=C[C.index>=Start]
C.plot()
def Precipitation():
    L=Direct(1)
    R=pd.read_excel(L[0],sheet_name=[0],index_col=0,parse_dates=True)
    P=R[0]
    del R[0]
    P.index = pd.to_datetime(P.index)
    Direct(0)
    return P
def CleanData(P,Start, End):    
    P=P[P.index>=Start]
    max_perc_nan =10
    p_m = P.isnull().sum() * 100 / len(P)
    P = P.loc[:, (p_m<max_perc_nan)]
    df1=P[P.index>End]
    df2=P[P.index<=End]

    return df2,df1
P=Precipitation()
PT,PV=CleanData(P,Start,End)

P=pd.concat([PT,PV])
P=P.dropna(axis=1)

P=pd.concat([PT,PV])
P=P.dropna(axis=1)
PT=P.iloc[1:3863,:]
PV=P.iloc[3864:-1,:]
def Energia(T,Start,End):
    L=Direct(2)
    if(T=='Aportes'):
        R=pd.read_excel(L[0],sheet_name='Aportes',index_col=2,parse_dates=True)
    if(T=='Reservas'):
        R=pd.read_excel(L[0],sheet_name='Reservas',index_col=2,parse_dates=True)
  
    #E.R=pd.read_excel(L[0],sheet_name='Reservas',index_col=0,parse_dates=True)
    R.index = pd.to_datetime(R.index)
    R=R[R.index>=Start]

    #extract regions and define them as category
    L=R['Region Hidrologica'].unique()
    R['Region Hidrologica']=R['Region Hidrologica'].astype("category")
    R['Region Hidrologica'].cat.set_categories(L,inplace=True)
    
    #create a pivot table
    if(T=='Aportes'):
        PvT=pd.pivot_table(R,values=['Aportes Energía kWh'],columns=['Region Hidrologica'],index=['Fecha'])
        
    if(T=='Reservas'):
        PvT=pd.pivot_table(R,values=['Volumen Útil Diario m3'],columns=['Region Hidrologica'],index=['Fecha'])
        
    
    #Clear the memory and return to working directory
    del R
    Direct(0)
    #Remove more that 10%
    p_m = PvT.isnull().sum() * 100 / len(PvT)
    PvT=PvT.loc[:,p_m<10]
    
    PvT1=PvT[PvT.index<=End]
    
    PvT2=PvT[PvT.index>End]
    PvT2=PvT2[PvT2.index<='31-12-2015']
    
    return PvT1,PvT2

EaT,EaV=FT.Energia('Aportes',Start,End)
ErT,ErV=FT.Energia('Reservas',Start,End)

EaV=EaV[EaV.index>'31-12-2010']
ErV=ErV[ErV.index>'31-12-2010']

def ReadSample():
    L=Direct(0)
    os.chdir("..")
    Sf=pd.read_excel('SolutionFile_ID_Maps.xlsx',sheet_name=0,index_col=3,parse_dates=True)
    return Sf 
Sf=ReadSample()
import numpy as np
import matplotlib.pyplot as plt
#%% Creating submission file
#array([21137010, 21137050, 23097030, 25027330, 29037020, 25027270,
#       'BETANIA', 'MIEL I', 'PORCE III', 'SOGAMOSO', 'bagre rayado',
#       'bocachico', 'capaz', 'nicuro', 'otras especies', 'Total general'],
#      dtype=object)


Cat=Sf[Sf.columns[1]].unique()
Sf[Sf.columns[1]]=Sf[Sf.columns[1]].astype("category")
Sf[Sf.columns[1]].cat.set_categories(Cat,inplace=True)

CR1=Sf[Sf[Sf.columns[1]]==21137010]
CR2=Sf[Sf[Sf.columns[1]]==21137050]
CR3=Sf[Sf[Sf.columns[1]]==23097030]
CR4=Sf[Sf[Sf.columns[1]]==25027330]
CR5=Sf[Sf[Sf.columns[1]]==29037020]
CR6=Sf[Sf[Sf.columns[1]]==25027270]
#len(CR1)+len(CR2)+len(CR3)+len(CR4)+len(CR5)+len(CR6) ->8746

#%%
#Energia producida
#betania
ER1=Sf[Sf[Sf.columns[1]]=='BETANIA']
ER2=Sf[Sf[Sf.columns[1]]=='MIEL I']
ER3=Sf[Sf[Sf.columns[1]]=='PORCE III']
#*Solo disponible el ultimo ano
ER4=Sf[Sf[Sf.columns[1]]=='SOGAMOSO']

#%%
PR1=Sf[Sf[Sf.columns[1]]=='bagre rayado']
PR2=Sf[Sf[Sf.columns[1]]=='bocachico']
PR3=Sf[Sf[Sf.columns[1]]=='capaz']
PR4=Sf[Sf[Sf.columns[1]]=='nicuro']
PR5=Sf[Sf[Sf.columns[1]]=='otras especies']
PR6=Sf[Sf[Sf.columns[1]]=='Total general']

Forms=[CR1,CR2,CR3,CR4,CR5,CR6,ER1,ER2,ER3,ER4,PR1,PR2,PR3,PR4,PR5,PR6]
R2=pd.concat(Forms)



Id=R2['Numero'].to_numpy()
Value=np.random.rand(len(Id))*Id
plt.scatter(Id,Value)

#Data=np.array([Id,Value])

df = pd.DataFrame({'Id':Id,'Expected':Value}, index=Value)
df.to_csv('Result_Example.csv')

Input=pd.merge(PT,ET, how='inner', left_index=True, right_index=True)

Mat1=pd.merge(Input,C.iloc[:,1], how='inner', left_index=True, right_index=True)
#Mat1=FT.Mapdates(Input,C.iloc[:,1])
#Mat2=FT.Mapdates(Input,C.iloc[:,2])
#Mat3=FT.Mapdates(Input,C.iloc[:,3])
#Mat4=FT.Mapdates(Input,C.iloc[:,4])

#%%
#Convert data frame into a numpy
Mat1=Mat1.dropna()
dataset=Mat1.to_numpy()
plt.plot(dataset)

#get number of rows  80% training
training_data_len=math.ceil(len(dataset)*.8)
training_data_len



#%%  Normalizing
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
plt.plot(scaled_data)



