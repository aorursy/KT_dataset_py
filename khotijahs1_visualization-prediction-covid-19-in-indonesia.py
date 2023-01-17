import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
content = """Date

Location_ISO_Code

Location

New_Cases

New_Deaths

New_Recovered

New_Active_Cases

Total_Cases

Total_Deaths

Total_Recovered

Total_Active_Cases

Location_Level

City_or_Regency

Province

Country

Continent

Island

Time_Zone

Special_Status

Total_Regencies

Total_Cities

Total_Districts

Total_Urban_Villages

Total_Rural_Villages

Area_(km2)

Population

Population_Density

Longitude

Latitude

New_Cases_per_Million

Total_Cases_per_Million

New_Deaths_per_Million

Total_Deaths_per_Million

Case_Fatality_Rate

Case_Recovered_Rate

Growth_Factor_of_New_Cases

Growth_Factor_of_New_Deaths"""

columns_list = content.split("\n")

# for i in range(len(columns_list)):

#   columns_list[i] = columns_list[i].strip()
data = pd.read_csv("../input/covid19-indonesia/covid_19_indonesia_time_series_all.csv",header=0,names = columns_list,index_col=False)

data = data.set_index('Location')

data.head()



data.info()

data[0:10]
data = data[['Date','Location_ISO_Code','New_Cases','New_Deaths','Total_Cases','Total_Deaths','Total_Recovered','New_Active_Cases','Total_Active_Cases','Longitude','Latitude']]

data.head(1000)
#IDN

ConfirmedCases_date_IDN= data[data['Location_ISO_Code']=='IDN'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_IDN = data[data['Location_ISO_Code']=='IDN'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_IDN= ConfirmedCases_date_IDN.join(fatalities_date_IDN)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_IDN.plot(ax=plt.gca(), title='Indonesia')
#ID-JK

ConfirmedCases_date_JK= data[data['Location_ISO_Code']=='ID-JK'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_JK = data[data['Location_ISO_Code']=='ID-JK'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JK = ConfirmedCases_date_JK.join(fatalities_date_JK)





#ID-BT

ConfirmedCases_date_BT= data[data['Location_ISO_Code']=='ID-BT'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_BT = data[data['Location_ISO_Code']=='ID-BT'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_BT = ConfirmedCases_date_BT.join(fatalities_date_BT)





#ID-GO

ConfirmedCases_date_GO= data[data['Location_ISO_Code']=='ID-GO'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_GO = data[data['Location_ISO_Code']=='ID-GO'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_GO = ConfirmedCases_date_GO.join(fatalities_date_GO)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_JK.plot(ax=plt.gca(), title='JAKARTA')

plt.ylabel("Confirmed  cases", size=13)





plt.subplot(2, 2, 2)

total_date_BT.plot(ax=plt.gca(), title='BANTEN')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 3)

total_date_GO.plot(ax=plt.gca(), title='Gorontalo')

plt.ylabel("Confirmed cases", size=13)
#ID-JI

ConfirmedCases_date_JI= data[data['Location_ISO_Code']=='ID-JI'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_JI = data[data['Location_ISO_Code']=='ID-JI'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JI = ConfirmedCases_date_JI.join(fatalities_date_JI)





#ID-JB

ConfirmedCases_date_JB= data[data['Location_ISO_Code']=='ID-JB'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_JB = data[data['Location_ISO_Code']=='ID-JB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JB = ConfirmedCases_date_JB.join(fatalities_date_JB)





#ID-JT

ConfirmedCases_date_JT= data[data['Location_ISO_Code']=='ID-JT'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_JT = data[data['Location_ISO_Code']=='ID-JT'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JT = ConfirmedCases_date_JT.join(fatalities_date_JT)





#ID-YO

ConfirmedCases_date_YO= data[data['Location_ISO_Code']=='ID-YO'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_YO = data[data['Location_ISO_Code']=='ID-YO'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_YO = ConfirmedCases_date_YO.join(fatalities_date_YO)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_JI.plot(ax=plt.gca(), title='JAWA TIMUR')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_JB.plot(ax=plt.gca(), title='JAWA BARAT')





plt.subplot(2, 2, 3)

total_date_JT.plot(ax=plt.gca(), title='JAWA TENGAH')



plt.subplot(2, 2, 4)

total_date_YO.plot(ax=plt.gca(), title='Daerah Istimewa Yogyakarta')

#ID-AC

ConfirmedCases_date_AC= data[data['Location_ISO_Code']=='ID-AC'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_AC = data[data['Location_ISO_Code']=='ID-AC'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_AC = ConfirmedCases_date_AC.join(fatalities_date_AC)





#ID-BA

ConfirmedCases_date_BA= data[data['Location_ISO_Code']=='ID-BA'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_BA = data[data['Location_ISO_Code']=='ID-BA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_BA = ConfirmedCases_date_BA.join(fatalities_date_BA)





#ID-BB

ConfirmedCases_date_BB= data[data['Location_ISO_Code']=='ID-BB'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_BB = data[data['Location_ISO_Code']=='ID-BB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_BB = ConfirmedCases_date_BB.join(fatalities_date_BB)





#ID-BE

ConfirmedCases_date_BE= data[data['Location_ISO_Code']=='ID-BE'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_BE = data[data['Location_ISO_Code']=='ID-BE'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_BE = ConfirmedCases_date_BE.join(fatalities_date_BE)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_AC.plot(ax=plt.gca(), title='Aceh')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_BA.plot(ax=plt.gca(), title='Bali')





plt.subplot(2, 2, 3)

total_date_BB.plot(ax=plt.gca(), title='Bangka Belitung')



plt.subplot(2, 2, 4)

total_date_BE.plot(ax=plt.gca(), title='Bengkulu')
#ID-KI

ConfirmedCases_date_KI= data[data['Location_ISO_Code']=='ID-KI'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_KI = data[data['Location_ISO_Code']=='ID-KI'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KI = ConfirmedCases_date_KI.join(fatalities_date_BT)





#ID-KB

ConfirmedCases_date_KB= data[data['Location_ISO_Code']=='ID-KB'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_KB = data[data['Location_ISO_Code']=='ID-KB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KB = ConfirmedCases_date_KB.join(fatalities_date_KB)







#ID-KT

ConfirmedCases_date_KT= data[data['Location_ISO_Code']=='ID-KT'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_KT = data[data['Location_ISO_Code']=='ID-KT'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KT = ConfirmedCases_date_KT.join(fatalities_date_KT)



#ID-KU

ConfirmedCases_date_KU= data[data['Location_ISO_Code']=='ID-KU'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_KU = data[data['Location_ISO_Code']=='ID-KU'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KU = ConfirmedCases_date_KU.join(fatalities_date_KU)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_KI.plot(ax=plt.gca(), title='Kalimantan Timur')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(2, 2, 2)

total_date_KB.plot(ax=plt.gca(), title='Kalimantan Barat')



plt.subplot(2, 2, 3)

total_date_KT.plot(ax=plt.gca(), title='Kalimantan Tengah')



plt.subplot(2, 2, 4)

total_date_KU.plot(ax=plt.gca(), title='Kalimantan Utara')



#ID-KS

ConfirmedCases_date_KS= data[data['Location_ISO_Code']=='ID-KS'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_KS = data[data['Location_ISO_Code']=='ID-KS'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KS = ConfirmedCases_date_KS.join(fatalities_date_KS)



#ID-KR

ConfirmedCases_date_KR= data[data['Location_ISO_Code']=='ID-KR'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_KR = data[data['Location_ISO_Code']=='ID-KR'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KR = ConfirmedCases_date_KR.join(fatalities_date_KR)













plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_KS.plot(ax=plt.gca(), title='Kalimantan Selatan')

plt.ylabel("Confirmed cases", size=13)







plt.subplot(2, 2, 2)

total_date_KR.plot(ax=plt.gca(), title='Kepulauan Riau')

#ID-LA

ConfirmedCases_date_LA= data[data['Location_ISO_Code']=='ID-LA'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_LA = data[data['Location_ISO_Code']=='ID-LA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_LA = ConfirmedCases_date_LA.join(fatalities_date_LA)





#ID-MA

ConfirmedCases_date_MA= data[data['Location_ISO_Code']=='ID-MA'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_MA = data[data['Location_ISO_Code']=='ID-MA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_MA = ConfirmedCases_date_MA.join(fatalities_date_MA)







#ID-MU

ConfirmedCases_date_MU= data[data['Location_ISO_Code']=='ID-MU'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_MU = data[data['Location_ISO_Code']=='ID-MU'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_MU = ConfirmedCases_date_MU.join(fatalities_date_MU)



#ID-NB

ConfirmedCases_date_NB= data[data['Location_ISO_Code']=='ID-NB'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_NB = data[data['Location_ISO_Code']=='ID-NB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_NB = ConfirmedCases_date_NB.join(fatalities_date_NB)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_LA.plot(ax=plt.gca(), title='Lampung')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_MA.plot(ax=plt.gca(), title='Maluku')





plt.subplot(2, 2, 3)

total_date_MU.plot(ax=plt.gca(), title='Maluku Utara')



plt.subplot(2, 2, 4)

total_date_NB.plot(ax=plt.gca(), title='Nusa Tenggara Barat')
#ID-NT

ConfirmedCases_date_NT= data[data['Location_ISO_Code']=='ID-NT'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_NT = data[data['Location_ISO_Code']=='ID-NT'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_NT = ConfirmedCases_date_NT.join(fatalities_date_NT)





#ID-PA

ConfirmedCases_date_PA= data[data['Location_ISO_Code']=='ID-PA'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_PA = data[data['Location_ISO_Code']=='ID-PA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_PA = ConfirmedCases_date_PA.join(fatalities_date_PA)



#ID-PB

ConfirmedCases_date_PB= data[data['Location_ISO_Code']=='ID-PB'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_PB = data[data['Location_ISO_Code']=='ID-PB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_PB = ConfirmedCases_date_PB.join(fatalities_date_PB)



#ID-RI

ConfirmedCases_date_RI= data[data['Location_ISO_Code']=='ID-RI'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_RI = data[data['Location_ISO_Code']=='ID-RI'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_RI= ConfirmedCases_date_RI.join(fatalities_date_RI)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_NT.plot(ax=plt.gca(), title='Nusa Tenggara Timur')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_PA.plot(ax=plt.gca(), title='Papua')





plt.subplot(2, 2, 3)

total_date_PB.plot(ax=plt.gca(), title='Papua Barat')



plt.subplot(2, 2, 4)

total_date_RI.plot(ax=plt.gca(), title='Riau')
#ID-SA

ConfirmedCases_date_SA= data[data['Location_ISO_Code']=='ID-SA'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_SA = data[data['Location_ISO_Code']=='ID-SA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SA = ConfirmedCases_date_SA.join(fatalities_date_SA)





#ID-SN

ConfirmedCases_date_SN= data[data['Location_ISO_Code']=='ID-SN'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_SN = data[data['Location_ISO_Code']=='ID-SN'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SN = ConfirmedCases_date_SN.join(fatalities_date_SN)



#ID-SG

ConfirmedCases_date_SG= data[data['Location_ISO_Code']=='ID-SG'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_SG = data[data['Location_ISO_Code']=='ID-SG'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SG = ConfirmedCases_date_SG.join(fatalities_date_SG)



#ID-ST

ConfirmedCases_date_ST= data[data['Location_ISO_Code']=='ID-ST'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_ST = data[data['Location_ISO_Code']=='ID-ST'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_ST = ConfirmedCases_date_ST.join(fatalities_date_ST)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_SA.plot(ax=plt.gca(), title='Sulawesi Utara')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_SN.plot(ax=plt.gca(), title='Sulawesi Selatan')



plt.subplot(2, 2, 3)

total_date_SG.plot(ax=plt.gca(), title='Sulawesi Tenggara')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 4)

total_date_ST.plot(ax=plt.gca(), title='Sulawesi Tengah')
#ID-SS

ConfirmedCases_date_SS= data[data['Location_ISO_Code']=='ID-SS'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_SS = data[data['Location_ISO_Code']=='ID-SS'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SS= ConfirmedCases_date_SS.join(fatalities_date_SS)



#ID-SB

ConfirmedCases_date_SB= data[data['Location_ISO_Code']=='ID-SB'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_SB = data[data['Location_ISO_Code']=='ID-SB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SB = ConfirmedCases_date_SB.join(fatalities_date_SB)





#ID-JA

ConfirmedCases_date_JA= data[data['Location_ISO_Code']=='ID-JA'].groupby(['Date']).agg({'Total_Cases':['sum']})

fatalities_date_JA = data[data['Location_ISO_Code']=='ID-JA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JA = ConfirmedCases_date_JA.join(fatalities_date_JA)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_SS.plot(ax=plt.gca(), title='Sumatera Selatan')





plt.subplot(2, 2, 2)

total_date_SB.plot(ax=plt.gca(), title='Sumatera Barat')





plt.subplot(2, 2, 3)

total_date_JA.plot(ax=plt.gca(), title='Jambi')

#IDN

Total_Recovered_date_IDN= data[data['Location_ISO_Code']=='IDN'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_IDN = data[data['Location_ISO_Code']=='IDN'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_IDN= Total_Recovered_date_IDN.join(Total_Deaths_date_IDN)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_IDN.plot(ax=plt.gca(), title='Indonesia')
#ID-JK

Total_Recovered_date_JK= data[data['Location_ISO_Code']=='ID-JK'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_JK = data[data['Location_ISO_Code']=='ID-JK'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JK = Total_Recovered_date_JK.join(Total_Deaths_date_JK)





#ID-BT

Total_Recovered_date_BT= data[data['Location_ISO_Code']=='ID-BT'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_BT = data[data['Location_ISO_Code']=='ID-BT'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_BT = Total_Recovered_date_BT.join(Total_Deaths_date_BT)





#ID-GO

Total_Recovered_date_GO= data[data['Location_ISO_Code']=='ID-GO'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_GO = data[data['Location_ISO_Code']=='ID-GO'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_GO = Total_Recovered_date_GO.join(Total_Deaths_date_GO)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_JK.plot(ax=plt.gca(), title='JAKARTA')

plt.ylabel("Confirmed  cases", size=13)





plt.subplot(2, 2, 2)

total_date_BT.plot(ax=plt.gca(), title='BANTEN')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 3)

total_date_GO.plot(ax=plt.gca(), title='Gorontalo')

plt.ylabel("Confirmed cases", size=13)
#ID-JI

Total_Recovered_date_JI= data[data['Location_ISO_Code']=='ID-JI'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_JI = data[data['Location_ISO_Code']=='ID-JI'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JI = Total_Recovered_date_JI.join(Total_Deaths_date_JI)





#ID-JB

Total_Recovered_date_JB= data[data['Location_ISO_Code']=='ID-JB'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_JB = data[data['Location_ISO_Code']=='ID-JB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JB = Total_Recovered_date_JB.join(Total_Deaths_date_JB)





#ID-JT

Total_Recovered_date_JT= data[data['Location_ISO_Code']=='ID-JT'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_JT = data[data['Location_ISO_Code']=='ID-JT'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JT = Total_Recovered_date_JT.join(Total_Deaths_date_JT)





#ID-YO

Total_Recovered_date_YO= data[data['Location_ISO_Code']=='ID-YO'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_YO = data[data['Location_ISO_Code']=='ID-YO'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_YO = Total_Recovered_date_YO.join(Total_Deaths_date_YO)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_JI.plot(ax=plt.gca(), title='JAWA TIMUR')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_JB.plot(ax=plt.gca(), title='JAWA BARAT')





plt.subplot(2, 2, 3)

total_date_JT.plot(ax=plt.gca(), title='JAWA TENGAH')



plt.subplot(2, 2, 4)

total_date_YO.plot(ax=plt.gca(), title='Daerah Istimewa Yogyakarta')

#ID-AC

ConfirmedCases_date_AC= data[data['Location_ISO_Code']=='ID-AC'].groupby(['Date']).agg({'Total_Recovered':['sum']})

fatalities_date_AC = data[data['Location_ISO_Code']=='ID-AC'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_AC = ConfirmedCases_date_AC.join(fatalities_date_AC)





#ID-BA

ConfirmedCases_date_BA= data[data['Location_ISO_Code']=='ID-BA'].groupby(['Date']).agg({'Total_Recovered':['sum']})

fatalities_date_BA = data[data['Location_ISO_Code']=='ID-BA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_BA = ConfirmedCases_date_BA.join(fatalities_date_BA)





#ID-BB

ConfirmedCases_date_BB= data[data['Location_ISO_Code']=='ID-BB'].groupby(['Date']).agg({'Total_Recovered':['sum']})

fatalities_date_BB = data[data['Location_ISO_Code']=='ID-BB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_BB = ConfirmedCases_date_BB.join(fatalities_date_BB)





#ID-BE

ConfirmedCases_date_BE= data[data['Location_ISO_Code']=='ID-BE'].groupby(['Date']).agg({'Total_Recovered':['sum']})

fatalities_date_BE = data[data['Location_ISO_Code']=='ID-BE'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_BE = ConfirmedCases_date_BE.join(fatalities_date_BE)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_AC.plot(ax=plt.gca(), title='Aceh')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_BA.plot(ax=plt.gca(), title='Bali')





plt.subplot(2, 2, 3)

total_date_BB.plot(ax=plt.gca(), title='Bangka Belitung')



plt.subplot(2, 2, 4)

total_date_BE.plot(ax=plt.gca(), title='Bengkulu')
#ID-KI

Total_Recovered_date_KI= data[data['Location_ISO_Code']=='ID-KI'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_KI = data[data['Location_ISO_Code']=='ID-KI'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KI = Total_Recovered_date_KI.join(Total_Deaths_date_KI)





#ID-KB

Total_Recovered_date_KB= data[data['Location_ISO_Code']=='ID-KB'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_KB = data[data['Location_ISO_Code']=='ID-KB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KB = Total_Recovered_date_KB.join(Total_Deaths_date_KB)



#ID-KT

Total_Recovered_date_KT= data[data['Location_ISO_Code']=='ID-KT'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_KT = data[data['Location_ISO_Code']=='ID-KT'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KT = Total_Recovered_date_KT.join(Total_Deaths_date_KT)



#ID-KU

Total_Recovered_date_KU= data[data['Location_ISO_Code']=='ID-KU'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_KU = data[data['Location_ISO_Code']=='ID-KU'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KU = Total_Recovered_date_KU.join(Total_Deaths_date_KU)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_KI.plot(ax=plt.gca(), title='Kalimantan Timur')

plt.ylabel("Confirmed cases", size=13)



plt.subplot(2, 2, 2)

total_date_KB.plot(ax=plt.gca(), title='Kalimantan Barat')





plt.subplot(2, 2, 3)

total_date_KT.plot(ax=plt.gca(), title='Kalimantan Tengah')



plt.subplot(2, 2, 4)

total_date_KU.plot(ax=plt.gca(), title='Kalimantan Utara')

#ID-KS

Total_Recovered_date_KS= data[data['Location_ISO_Code']=='ID-KS'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_KS = data[data['Location_ISO_Code']=='ID-KS'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KS = Total_Recovered_date_KS.join(Total_Deaths_date_KS)



#ID-KR

Total_Recovered_date_KR= data[data['Location_ISO_Code']=='ID-KR'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_KR = data[data['Location_ISO_Code']=='ID-KR'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_KR = Total_Recovered_date_KR.join(Total_Deaths_date_KR)











plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_KS.plot(ax=plt.gca(), title='Kalimantan Selatan')



plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_KR.plot(ax=plt.gca(), title='Kepulauan Riau')



#ID-LA

Total_Recovered_date_LA= data[data['Location_ISO_Code']=='ID-LA'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_LA = data[data['Location_ISO_Code']=='ID-LA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_LA = Total_Recovered_date_LA.join(Total_Deaths_date_LA)





#ID-MA

Total_Recovered_date_MA= data[data['Location_ISO_Code']=='ID-MA'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_MA = data[data['Location_ISO_Code']=='ID-MA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_MA = Total_Recovered_date_MA.join(Total_Deaths_date_MA)







#ID-MU

Total_Recovered_date_MU= data[data['Location_ISO_Code']=='ID-MU'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_MU = data[data['Location_ISO_Code']=='ID-MU'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_MU = Total_Recovered_date_MU.join(Total_Deaths_date_MU)



#ID-NB

Total_Recovered_date_NB= data[data['Location_ISO_Code']=='ID-NB'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_NB = data[data['Location_ISO_Code']=='ID-NB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_NB = Total_Recovered_date_NB.join(Total_Deaths_date_NB)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_LA.plot(ax=plt.gca(), title='Lampung')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_MA.plot(ax=plt.gca(), title='Maluku')





plt.subplot(2, 2, 3)

total_date_MU.plot(ax=plt.gca(), title='Maluku Utara')



plt.subplot(2, 2, 4)

total_date_NB.plot(ax=plt.gca(), title='Nusa Tenggara Barat')
#ID-NT

Total_Recovered_date_NT= data[data['Location_ISO_Code']=='ID-NT'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_NT = data[data['Location_ISO_Code']=='ID-NT'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_NT = Total_Recovered_date_NT.join(Total_Deaths_date_NT)





#ID-PA

Total_Recovered_date_PA= data[data['Location_ISO_Code']=='ID-PA'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_PA = data[data['Location_ISO_Code']=='ID-PA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_PA = Total_Recovered_date_PA.join(Total_Deaths_date_PA)



#ID-PB

Total_Recovered_date_PB= data[data['Location_ISO_Code']=='ID-PB'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_PB = data[data['Location_ISO_Code']=='ID-PB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_PB = Total_Recovered_date_PB.join(Total_Deaths_date_PB)



#ID-RI

Total_Recovered_date_RI= data[data['Location_ISO_Code']=='ID-RI'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_RI = data[data['Location_ISO_Code']=='ID-RI'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_RI= Total_Recovered_date_RI.join(Total_Deaths_date_RI)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_NT.plot(ax=plt.gca(), title='Nusa Tenggara Timur')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_PA.plot(ax=plt.gca(), title='Papua')





plt.subplot(2, 2, 3)

total_date_PB.plot(ax=plt.gca(), title='Papua Barat')



plt.subplot(2, 2, 4)

total_date_RI.plot(ax=plt.gca(), title='Riau')
#ID-SA

Total_Recovered_date_SA= data[data['Location_ISO_Code']=='ID-SA'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_SA = data[data['Location_ISO_Code']=='ID-SA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SA = Total_Recovered_date_SA.join(Total_Deaths_date_SA)





#ID-SN

Total_Recovered_date_SN= data[data['Location_ISO_Code']=='ID-SN'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_SN = data[data['Location_ISO_Code']=='ID-SN'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SN = Total_Recovered_date_SN.join(Total_Deaths_date_SN)



#ID-SG

Total_Recovered_date_SG= data[data['Location_ISO_Code']=='ID-SG'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_SG = data[data['Location_ISO_Code']=='ID-SG'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SG = Total_Recovered_date_SG.join(Total_Deaths_date_SG)



#ID-ST

Total_Recovered_date_ST= data[data['Location_ISO_Code']=='ID-ST'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_ST = data[data['Location_ISO_Code']=='ID-ST'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_ST = Total_Recovered_date_ST.join(Total_Deaths_date_ST)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_SA.plot(ax=plt.gca(), title='Sulawesi Utara')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 2)

total_date_SN.plot(ax=plt.gca(), title='Sulawesi Selatan')



plt.subplot(2, 2, 3)

total_date_SG.plot(ax=plt.gca(), title='Sulawesi Tenggara')

plt.ylabel("Confirmed cases", size=13)





plt.subplot(2, 2, 4)

total_date_ST.plot(ax=plt.gca(), title='Sulawesi Tengah')
#ID-SS

Total_Recovered_date_SS= data[data['Location_ISO_Code']=='ID-SS'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_SS = data[data['Location_ISO_Code']=='ID-SS'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SS= Total_Recovered_date_SS.join(Total_Deaths_date_SS)



#ID-SB

Total_Recovered_date_SB= data[data['Location_ISO_Code']=='ID-SB'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_SB = data[data['Location_ISO_Code']=='ID-SB'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_SB = Total_Recovered_date_SB.join(Total_Deaths_date_SB)





#ID-JA

Total_Recovered_date_JA= data[data['Location_ISO_Code']=='ID-JA'].groupby(['Date']).agg({'Total_Recovered':['sum']})

Total_Deaths_date_JA = data[data['Location_ISO_Code']=='ID-JA'].groupby(['Date']).agg({'Total_Deaths':['sum']})

total_date_JA = Total_Recovered_date_JA.join(Total_Deaths_date_JA)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_SS.plot(ax=plt.gca(), title='Sumatera Selatan')





plt.subplot(2, 2, 2)

total_date_SB.plot(ax=plt.gca(), title='Sumatera Barat')





plt.subplot(2, 2, 3)

total_date_JA.plot(ax=plt.gca(), title='Jambi')

# All Cases in Indonesia

#IDN

TotalCases_date_IDN= data[data['Location_ISO_Code']=='IDN'].groupby(['Date']).agg({'Total_Cases':['sum']})

New_Active_Cases_date_IDN = data[data['Location_ISO_Code']=='IDN'].groupby(['Date']).agg({'New_Active_Cases':['sum']})

total_date_IDN= TotalCases_date_IDN.join(New_Active_Cases_date_IDN)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_IDN.plot(ax=plt.gca(), title='Indonesia')
# All Cases in Indonesia

#IDN

Total_Deaths_date_IDN= data[data['Location_ISO_Code']=='IDN'].groupby(['Date']).agg({'Total_Deaths':['sum']})

New_Deaths_date_IDN = data[data['Location_ISO_Code']=='IDN'].groupby(['Date']).agg({'New_Deaths':['sum']})

total_date_IDN= Total_Deaths_date_IDN.join(New_Deaths_date_IDN)





plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_IDN.plot(ax=plt.gca(), title='Indonesia')
##Convert sting to numeric

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def FunLabelEncoder(df):

    for c in df.columns:

        if df.dtypes[c] == object:

            le.fit(df[c].astype(str))

            df[c] = le.transform(df[c].astype(str))

    return df
data = FunLabelEncoder(data)

data.info()

data.iloc[235:300,:]
from sklearn.model_selection import train_test_split

Y = data['New_Cases']

X = data.drop(columns=['New_Cases'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=9)
print('X train shape: ', X_train.shape)

print('Y train shape: ', Y_train.shape)

print('X test shape: ', X_test.shape)

print('Y test shape: ', Y_test.shape)
from sklearn.tree import DecisionTreeClassifier



# We define the model

dtcla = DecisionTreeClassifier(random_state=None)



# We train model

dtcla.fit(X_train, Y_train)



# We predict target values

Y_predict = dtcla.predict(X_test)
#Test

X_test
from sklearn.model_selection import train_test_split

Y1 = data['New_Deaths']

X1 = data.drop(columns=['New_Deaths'])

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.1, random_state=9)
print('X1 train shape: ', X1_train.shape)

print('Y1 train shape: ', Y1_train.shape)

print('X1 test shape: ', X1_test.shape)

print('Y1 test shape: ', Y1_test.shape)
from sklearn.tree import DecisionTreeClassifier



# We define the model

dtcla = DecisionTreeClassifier(random_state=None)



# We train model

dtcla.fit(X1_train, Y1_train)



# We predict target values

Y1_predict = dtcla.predict(X1_test)
#Test

X1_test
#Create a  DataFrame

submission = pd.DataFrame({'New_Cases':Y_predict,'New_Deaths':Y1_predict})

                        



#Visualize the first 100 rows

submission.head(100)
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)