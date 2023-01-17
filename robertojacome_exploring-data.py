# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2016_Vehculos Matriculados_BDD.csv")

print(data)
#Which is the province with more registered vehicles?
step = 0.5
bin_range = np.arange(0.5, 24.5, 1)
out, bins  = pd.cut(data['PROVINCIA'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()

#results show 1. Pichincha
#             2. Guayas

#list of codes of all provinces
#1=AZUAY
#2=BOLIVAR
#3=CAÑAR
#4=CARCHI
#5=COTOPAXI
#6=CHIMBORAZO
#7=EL ORO
#8=ESMERALDAS
#9=GUAYAS
#10=IMBABURA
#11=LOJA
#12=LOS RIOS
#13=MANABI
#14=MORONA SANTIAGO
#15=NAPO
#16=PASTAZA
#17=PICHINCHA
#18=TUNGURAHUA
#19=ZAMORA CHINCHIPE
#20=GALAPAGOS
#21=SUCUMBIOS
#22=ORELLANA
#23=SANTO DOMINGO DE LOS TSACHILAS
#24=SANTA ELENA

#Let's see the uses of vehicles
step = 0.5
bin_range = np.arange(0.5, 6.5, 1)
out, bins  = pd.cut(data['USO'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()

#1=ESTADO
#2=GOBIERNOS SECCIONALES
#3=MUNICIPIO
#4=ALQUILER
#5=PARTICULAR
#6=OTROS
#Let's see the uses of vehicles
step = 0.5
bin_range = np.arange(0.5, 5.5, 1)
out, bins  = pd.cut(data['COMBUSTIBLE'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()

#The use of electric vehicules is almost zero

#1=DIESEL
#2=GASOLINA
#3=HÍBRIDO
#4=ELÉCTRICO
#5=GAS LICUADO DE PETROLEO
#Let's see the uses of vehicles
step = 0.5
bin_range = np.arange(0.5, 12.5, 1)
out, bins  = pd.cut(data['CLASE'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()

#1=AUTOMÓVIL
#2=AUTOBÚS
#3=CAMIÓN
#4=CAMIONETA
#5=FURGONETA C
#6=FURGONETA P
#7=JEEP
#8=MOTOCICLETA
#9=TANQUERO
#10=TRAILER
#11=VOLQUETA
#12=OTRA CLASE
#Let's see the brands of vehicles
step = 0.5
bin_range = np.arange(0.5, 10.5, 1)
out, bins  = pd.cut(data['CLASE'], bins=bin_range, include_lowest=True, right=False, retbins=True)
out.value_counts(sort=False).plot.bar()


#1=CHEVROLET
#2=SUZUKI
#3=TOYOTA
#4=HYUNDAI
#5=MAZDA
#6=NISSAN
#7=KIA
#8=FORD
#9=VOLKSWAGEN
#10=HINO
#11=SHINERAY
#12=HONDA
#13=MITSUBISHI
#14=MOTOR UNO
#15=RENAULT
#16=YAMAHA
#17=BAJAJ
#18=DAYTONA
#19=SUKIDA
#20=RANGER
#21=TUNDRA
#22=FIAT
#23=QMC
#24=MERCEDES BENZ
#25=DUKARE
#26=GREAT WALL
#27=DAEWOO
#28=SKODA
#29=DAIHATSU
#30=TRAXX
#31=LONCIN
#32=THUNDER
#33=PEUGEOT
#34=LADA
#35=TUKO
#36=DAYANG
#37=PEGASSO
#38=DATSUN
#39=CHERY
#40=ICS
#41=OROMOTO
#42=QINGQI
#43=JAC
#44=JEEP
#45=TEKNO
#46=BMW
#47=UM
#48=AXXO
#49=JIALING
#50=MACK
#51=INTERNATIONAL
#52=FORMOSA
#53=ISUZU
#54=SANYA
#55=BULTACO
#56=CITROEN
#57=LINGKEN
#58=Z1
#59=KEEWAY
#60=KENWORTH
#61=LIFAN
#62=SKYGO
#63=GALARDI
#64=KAWASAKI
#65=JIANSHE
#66=FREIGHTLINER
#67=VOLVO
#68=LAMBORBINI
#69=AUDI
#70=LAND ROVER
#71=CHANGHE
#72=DODGE
#73=DONGFENG
#74=ZOTYE
#75=DAYUN
#76=GMC
#77=KTM
#78=NIMBUS
#79=DFSK
#80=VYCAST
#81=KINGDOM
#82=JMC
#83=HERO
#84=MICARGI
#85=ORION
#86=TVS
#87=LEXUS
#88=BYD
#89=AUSTIN
#90=SSANGYONG
#91=HUSSAR
#92=MAHINDRA
#93=SAIC WULING
#94=FACTORY
#95=SCANIA
#96=LML
#97=DUCAR
#98=KINGTON
#99=PORSCHE
#100=FENGCHI
#101=ZANYA
#102=JINCHENG
#103=MAN
#104=SINOTRUK
#105=VESPA
#106=JRI
#107=SUBARU
#108=FOTON
#109=BMA
#110=KOSHIN MOTOR
#111=FAW
#112=KENBO
#113=TATA
#114=YASAKI
#115=TIANYE
#116=KINLON
#117=OKAZAKI
#118=ASIA
#119=JINBEI HAISE
#120=HARLEY DAVIDSON
#121=SAEHAN
#122=WILLYS
#123=YUTONG
#124=LONGJIA
#125=DACIA
#126=AMAZON
#127=ZX AUTO
#128=MOTO PLUS
#129=ZONGSHEN
#130=CHRYSLER
#131=SRM
#132=JIEDA
#133=AKT
#134=UD TRUCKS
#135=GOLDEN DRAGON
#136=CHANGAN
#137=SINSKI
#138=YAMOTO
#139=PIAGGIO
#140=MACAT
#141=DAKAR
#142=DUCATI
#143=IVECO
#144=OSAKA
#145=PETERBILT
#146=ROYAL ENFIELD
#147=EXPLORER
#148=BEIBEN
#149=HUSQVARNA
#150=ALFA ROMEO
#151=WESTERN STAR
#152=OPEL
#153=SPEED FIRE
#154=SEAT
#155=NANFANG
#156=HARLEM DAYTON
#157=JET MOTO
#158=IGM
#159=YUTONG
#160=ACCONA
#161=SINAI