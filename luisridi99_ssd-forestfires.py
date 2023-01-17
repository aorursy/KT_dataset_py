# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importamos el dataset de kaggle



df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding = 'latin1')
# Vemos que datos hay en la tabla



df.head(10)
# En primer lugar, cambiaremos los meses a espanol



meses_port = list(pd.unique(df['month']))

meses_esp = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Setiembre',

             'Octubre','Noviembre','Diciembre']

dict_meses = dict(zip(meses_port,meses_esp))



df.month = df['month'].map(dict_meses)
# Verificamos

df.head(10)
# Ahora convertiremos el estado en latitud y longitud para efectos de variables numericas



latitud={'Acre':-9.02,'Alagoas':-9.57,'Amapa':02.05,'Amazonas':-5.00,'Bahia':-12.00,'Ceara':-5.00,

          

          'Distrito Federal':-15.45,'Espirito Santo':-20.00,'Goias':-15.55,'Maranhao':-5.00,'Mato Grosso':-14.00

          

          ,'Minas Gerais':-18.50,'Pará':-3.20,'Paraiba':-7.00,'Pernambuco':-8.00,'Piau':-7.00,'Rio':-22.90,

          

          'Rondonia':-11.00,'Roraima':-2.00,'Santa Catarina':-27.25,'Sao Paulo':-23.32,'Sergipe':-10.30,

         

         'Tocantins':-10.00

         }





longitud={

    'Acre':-70.8120,'Alagoas':-36.7820,'Amapa':-50.50,'Amazonas':-65.00,'Bahia':-42.00,'Ceara':-40.00,

    

    'Distrito Federal':-47.45,'Espirito Santo':-40.45,'Goias':-50.10,'Maranhao':-46.00,'Mato Grosso':-55.00,

    

    'Minas Gerais':-46.00,'Pará':-52.00,'Paraiba':-36.00,'Pernambuco':-37.00,'Piau':-73.00, 'Rio':-43.17,

    

    'Rondonia':-63.00,'Roraima':-61.30,'Santa Catarina':-48.30,'Sao Paulo':-46.37,'Sergipe':-37.30,

    

    'Tocantins':-48.00

}



df['latitud'] = df['state'].map(latitud)

df['longitud'] = df['state'].map(longitud)
# Verificamos



df.head()
#df_new = df.groupby(['state','month']).mean()

df_new = df
# Creamos la columna col_ref donde concatenaremos la columna state y month para hacer mas facil

# la creacion de diccionarios

df_new['col_ref'] = df_new['state']+df_new['month']
# Como los meses no son datos numericos, crearesmos diccionarios correspondientes a cada uno: humedad, temperatura y velocidad del viento



humedad = {'RioEnero': 0.79,'RioFebrero': 0.79,'RioMarzo':0.80,'RioAbril':0.80,'RioMayo':0.80,'RioJunio':0.79,

          'RioJulio':0.77,'RioAgosto':0.77,'RioSetiembre':0.79,'RioOctubre':0.80,'RioNoviembre':0.79,'RioDiciembre':0.80,

          'ParaibaEnero': 1.00,'ParaibaFebrero': 1.00,'ParaibaMarzo':1.00,'ParaibaAbril':1.00,'ParaibaMayo':1.00,'ParaibaJunio':1.00,

          'ParaibaJulio':1.00,'ParaibaAgosto':1.00,'ParaibaSetiembre':0.995,'ParaibaOctubre':1.00,'ParaibaNoviembre':1.00,'ParaibaDiciembre':1.00,

          'Mato GrossoEnero': 0.9925,'Mato GrossoFebrero': 1.00,'Mato GrossoMarzo':0.9875,'Mato GrossoAbril':0.93,'Mato GrossoMayo':0.7333,'Mato GrossoJunio':0.42,

          'Mato GrossoJulio':0.1825,'Mato GrossoAgosto':0.2133,'Mato GrossoSetiembre':0.4333,'Mato GrossoOctubre':0.7975,'Mato GrossoNoviembre':0.9367,'Mato GrossoDiciembre':0.98,

          'AlagoasEnero': 1.00,'AlagoasFebrero': 1.00,'AlagoasMarzo':1.00,'AlagoasAbril':1.00,'AlagoasMayo':1.00,'AlagoasJunio':0.99,

          'AlagoasJulio':0.965,'AlagoasAgosto':0.9625,'AlagoasSetiembre':0.9833,'AlagoasOctubre':0.9925,'AlagoasNoviembre':1.00,'AlagoasDiciembre':1.00,

          'PiauEnero': 0.9875,'PiauFebrero': 1.00,'PiauMarzo':1.00,'PiauAbril':1.00,'PiauMayo':0.9967,'PiauJunio':0.9425,

          'PiauJulio':0.805,'PiauAgosto':0.69,'PiauSetiembre':0.7267,'PiauOctubre':0.8025,'PiauNoviembre':0.8667,'PiauDiciembre':0.9375,

          'Espirito SantoEnero': 0.9925,'Espirito SantoFebrero': 1.00,'Espirito SantoMarzo':0.9925,'Espirito SantoAbril':0.9333,'Espirito SantoMayo':0.81,'Espirito SantoJunio':0.67,

          'Espirito SantoJulio':0.555,'Espirito SantoAgosto':0.545,'Espirito SantoSetiembre':0.65,'Espirito SantoOctubre':0.80,'Espirito SantoNoviembre':0.9133,'Espirito SantoDiciembre':0.9725,

          'CearaEnero': 1.00,'CearaFebrero': 1.00,'CearaMarzo':1.00,'CearaAbril':1.00,'CearaMayo':1.00,'CearaJunio':0.995,

          'CearaJulio':0.97,'CearaAgosto':0.9475,'CearaSetiembre':0.96,'CearaOctubre':0.98,'CearaNoviembre':0.9967,'CearaDiciembre':1.00,

          'MaranhaoEnero': 1.00,'MaranhaoFebrero': 1.00,'MaranhaoMarzo':1.00,'MaranhaoAbril':1.00,'MaranhaoMayo':1.00,'MaranhaoJunio':1.00,

          'MaranhaoJulio':0.9975,'MaranhaoAgosto':0.99,'MaranhaoSetiembre':0.99,'MaranhaoOctubre':0.9925,'MaranhaoNoviembre':1.00,'MaranhaoDiciembre':1.00,

          'SergipeEnero': 1.00,'SergipeFebrero': 1.00,'SergipeMarzo':1.00,'SergipeAbril':1.00,'SergipeMayo':1.00,'SergipeJunio':0.995,

          'SergipeJulio':0.9675,'SergipeAgosto':0.9525,'SergipeSetiembre':0.9667,'SergipeOctubre':0.99,'SergipeNoviembre':1.00,'SergipeDiciembre':1.00,

          'Distrito FederalEnero': 0.3233,'Distrito FederalFebrero': 0.3033,'Distrito FederalMarzo':0.32,'Distrito FederalAbril':0.2033,'Distrito FederalMayo':0.467,'Distrito FederalJunio':0.0025,

          'Distrito FederalJulio':0,'Distrito FederalAgosto':0,'Distrito FederalSetiembre':0.02,'Distrito FederalOctubre':0.0925,'Distrito FederalNoviembre':0.233,'Distrito FederalDiciembre':0.325,

          'AmapaEnero': 1.00,'AmapaFebrero': 1.00,'AmapaMarzo':1.00,'AmapaAbril':1.00,'AmapaMayo':1.00,'AmapaJunio':1.00,

          'AmapaJulio':1.00,'AmapaAgosto':1.00,'AmapaSetiembre':1.00,'AmapaOctubre':1.00,'AmapaNoviembre':1.00,'AmapaDiciembre':1.00,

          'RoraimaEnero': 0.88,'RoraimaFebrero': 0.83,'RoraimaMarzo':0.8425,'RoraimaAbril':0.9133,'RoraimaMayo':0.99,'RoraimaJunio':1.00,

          'RoraimaJulio':1.00,'RoraimaAgosto':0.9975,'RoraimaSetiembre':0.9867,'RoraimaOctubre':0.97,'RoraimaNoviembre':0.97,'RoraimaDiciembre':0.955,

          'Santa CatarinaEnero': 0.89,'Santa CatarinaFebrero': 0.9167,'Santa CatarinaMarzo':0.85,'Santa CatarinaAbril':0.6433,'Santa CatarinaMayo':0.3367,'Santa CatarinaJunio':0.125,

          'Santa CatarinaJulio':0.05,'Santa CatarinaAgosto':0.075,'Santa CatarinaSetiembre':0.1667,'Santa CatarinaOctubre':0.3475,'Santa CatarinaNoviembre':0.5433,'Santa CatarinaDiciembre':0.7425,

          'ParáEnero': 1.00,'ParáFebrero': 1.00,'ParáMarzo':1.00,'ParáAbril':1.00,'ParáMayo':1.00,'ParáJunio':1.00,

          'ParáJulio':1.00,'ParáAgosto':1.00,'ParáSetiembre':1.00,'ParáOctubre':1.00,'ParáNoviembre':1.00,'ParáDiciembre':1.00,

          'Sao PauloEnero': 0.5725,'Sao PauloFebrero': 0.63,'Sao PauloMarzo':0.505,'Sao PauloAbril':0.2567,'Sao PauloMayo':0.0333,'Sao PauloJunio':0,

          'Sao PauloJulio':0,'Sao PauloAgosto':0,'Sao PauloSetiembre':0.02,'Sao PauloOctubre':0.09,'Sao PauloNoviembre':0.2167,'Sao PauloDiciembre':0.40,

          'RondoniaEnero': 1.00,'RondoniaFebrero': 1.00,'RondoniaMarzo':1.00,'RondoniaAbril':0.9967,'RondoniaMayo':0.9833,'RondoniaJunio':0.9575,

          'RondoniaJulio':0.875,'RondoniaAgosto':0.8733,'RondoniaSetiembre':0.9467,'RondoniaOctubre':0.995,'RondoniaNoviembre':1.00,'RondoniaDiciembre':1.00,

          'PernambucoEnero': 1.00,'PernambucoFebrero': 1.00,'PernambucoMarzo':1.00,'PernambucoAbril':1.00,'PernambucoMayo':1.00,'PernambucoJunio':0.995,

          'PernambucoJulio':0.9825,'PernambucoAgosto':0.97,'PernambucoSetiembre':0.975,'PernambucoOctubre':0.99,'PernambucoNoviembre':1.00,'PernambucoDiciembre':1.00,

          'Minas GeraisEnero': 0.50,'Minas GeraisFebrero': 0.4567,'Minas GeraisMarzo':0.4467,'Minas GeraisAbril':0.28,'Minas GeraisMayo':0.0767,'Minas GeraisJunio':0.02,

          'Minas GeraisJulio':0,'Minas GeraisAgosto':0.01,'Minas GeraisSetiembre':0.03,'Minas GeraisOctubre':0.115,'Minas GeraisNoviembre':0.2867,'Minas GeraisDiciembre':0.4825,

          'BahiaEnero': 1.00,'BahiaFebrero': 1.00,'BahiaMarzo':1.00,'BahiaAbril':1.00,'BahiaMayo':0.9967,'BahiaJunio':0.97,

          'BahiaJulio':0.925,'BahiaAgosto':0.915,'BahiaSetiembre':0.95,'BahiaOctubre':0.985,'BahiaNoviembre':0.9967,'BahiaDiciembre':1.00,

          'GoiasEnero': 0.85,'GoiasFebrero': 0.8433,'GoiasMarzo':0.8533,'GoiasAbril':0.6225,'GoiasMayo':0.23,'GoiasJunio':0.0275,

          'GoiasJulio':0.0025,'GoiasAgosto':0.0175,'GoiasSetiembre':0.1333,'GoiasOctubre':0.4425,'GoiasNoviembre':0.72,'GoiasDiciembre':0.8567,

          'AcreEnero': 1.00,'AcreFebrero': 1.00,'AcreMarzo':1.00,'AcreAbril':0.98,'AcreMayo':0.9367,'AcreJunio':0.8875,

          'AcreJulio':0.79,'AcreAgosto':0.7725,'AcreSetiembre':0.8633,'AcreOctubre':0.97,'AcreNoviembre':0.9933,'AcreDiciembre':1.00,

          'TocantinsEnero': 1.00,'TocantinsFebrero': 1.00,'TocantinsMarzo':1.00,'TocantinsAbril':0.9867,'TocantinsMayo':0.84,'TocantinsJunio':0.355,

          'TocantinsJulio':0.07,'TocantinsAgosto':0.0675,'TocantinsSetiembre':0.333,'TocantinsOctubre':0.7575,'TocantinsNoviembre':0.9367,'TocantinsDiciembre':0.995,

          'AmazonasEnero': 1.00,'AmazonasFebrero': 1.00,'AmazonasMarzo':1.00,'AmazonasAbril':1.00,'AmazonasMayo':1.00,'AmazonasJunio':1.00,

          'AmazonasJulio':0.995,'AmazonasAgosto':0.997,'AmazonasSetiembre':1.00,'AmazonasOctubre':1.00,'AmazonasNoviembre':1.00,'AmazonasDiciembre':1.00}
temperatura = {'RioEnero': 26.75,'RioFebrero': 26.85,'RioMarzo':26.35,'RioAbril':24.85,'RioMayo':23.4,'RioJunio':21.95,

          'RioJulio':21.7,'RioAgosto':22.2,'RioSetiembre':22.3,'RioOctubre':23.1,'RioNoviembre':24.4,'RioDiciembre':25.5,

          'ParaibaEnero': 28.5,'ParaibaFebrero': 28.5,'ParaibaMarzo':28.5,'ParaibaAbril':28,'ParaibaMayo':27.25,'ParaibaJunio':26.38,

          'ParaibaJulio':25.88,'ParaibaAgosto':25.88,'ParaibaSetiembre':26.50,'ParaibaOctubre':27.25,'ParaibaNoviembre':27.83,'ParaibaDiciembre':28.16,

          'Mato GrossoEnero': 28,'Mato GrossoFebrero': 28,'Mato GrossoMarzo':28,'Mato GrossoAbril':27.67,'Mato GrossoMayo':26.17,'Mato GrossoJunio':25.17,

          'Mato GrossoJulio':25.25,'Mato GrossoAgosto':27.25,'Mato GrossoSetiembre':28.5,'Mato GrossoOctubre':28.88,'Mato GrossoNoviembre':28.5,'Mato GrossoDiciembre':28,

          'AlagoasEnero': 27.38,'AlagoasFebrero': 27.5,'AlagoasMarzo':27.5,'AlagoasAbril':27.17,'AlagoasMayo':26.5,'AlagoasJunio':25.38,

          'AlagoasJulio':24.83,'AlagoasAgosto':24.88,'AlagoasSetiembre':25.17,'AlagoasOctubre':26,'AlagoasNoviembre':26.67,'AlagoasDiciembre':27.13,

          'PiauEnero': 28.63,'PiauFebrero': 28.33,'PiauMarzo':28.17,'PiauAbril':28.17,'PiauMayo':28.5,'PiauJunio':28.75,

          'PiauJulio':28.75,'PiauAgosto':29.5,'PiauSetiembre':30.5,'PiauOctubre':31,'PiauNoviembre':30.67,'PiauDiciembre':29.88,

          'Espirito SantoEnero': 27.75,'Espirito SantoFebrero': 28,'Espirito SantoMarzo':27.63,'Espirito SantoAbril':26.33,'Espirito SantoMayo':24.67,'Espirito SantoJunio':23.38,

          'Espirito SantoJulio':23,'Espirito SantoAgosto':23.17,'Espirito SantoSetiembre':23.83,'Espirito SantoOctubre':24.88,'Espirito SantoNoviembre':25.83,'Espirito SantoDiciembre':26.75,

          'CearaEnero': 28,'CearaFebrero': 27.5,'CearaMarzo':27.5,'CearaAbril':27.5,'CearaMayo':27.5,'CearaJunio':27,

          'CearaJulio':27,'CearaAgosto':27,'CearaSetiembre':27.5,'CearaOctubre':28,'CearaNoviembre':28.17,'CearaDiciembre':28.5,

          'MaranhaoEnero': 28.13,'MaranhaoFebrero': 27.83,'MaranhaoMarzo':27.5,'MaranhaoAbril':27.67,'MaranhaoMayo':28,'MaranhaoJunio':28,

          'MaranhaoJulio':28.17,'MaranhaoAgosto':28.83,'MaranhaoSetiembre':29,'MaranhaoOctubre':29,'MaranhaoNoviembre':28.83,'MaranhaoDiciembre':28.8,

          'SergipeEnero': 28.38,'SergipeFebrero': 28.5,'SergipeMarzo':28.25,'SergipeAbril':27.83,'SergipeMayo':27.01,'SergipeJunio':26,

          'SergipeJulio':25.38,'SergipeAgosto':25.38,'SergipeSetiembre':25.83,'SergipeOctubre':26.75,'SergipeNoviembre':27.5,'SergipeDiciembre':27.63,

          'Distrito FederalEnero': 22.5,'Distrito FederalFebrero': 22.5,'Distrito FederalMarzo':22.5,'Distrito FederalAbril':22,'Distrito FederalMayo':20.83,'Distrito FederalJunio':19.5,

          'Distrito FederalJulio':19.13,'Distrito FederalAgosto':20.5,'Distrito FederalSetiembre':22.33,'Distrito FederalOctubre':23.13,'Distrito FederalNoviembre':22.67,'Distrito FederalDiciembre':22.5,

          'AmapaEnero': 27.38,'AmapaFebrero': 27,'AmapaMarzo':27,'AmapaAbril':27.33,'AmapaMayo':27.67,'AmapaJunio':28,

          'AmapaJulio':28.13,'AmapaAgosto':28.63,'AmapaSetiembre':29,'AmapaOctubre':29,'AmapaNoviembre':29,'AmapaDiciembre':28.5,

          'RoraimaEnero': 28.5,'RoraimaFebrero': 28.83,'RoraimaMarzo':29.17,'RoraimaAbril':29,'RoraimaMayo':28,'RoraimaJunio':27.5,

          'RoraimaJulio':27.5,'RoraimaAgosto':28,'RoraimaSetiembre':28.99,'RoraimaOctubre':29.5,'RoraimaNoviembre':29.33,'RoraimaDiciembre':28.88,

          'Santa CatarinaEnero': 25.38,'Santa CatarinaFebrero': 25.67,'Santa CatarinaMarzo':24.75,'Santa CatarinaAbril':22.99,'Santa CatarinaMayo':20.17,'Santa CatarinaJunio':17.88,

          'Santa CatarinaJulio':17.25,'Santa CatarinaAgosto':18,'Santa CatarinaSetiembre':18.83,'Santa CatarinaOctubre':20.75,'Santa CatarinaNoviembre':22.50,'Santa CatarinaDiciembre':24.25,

          'ParáEnero': 27.75,'ParáFebrero': 28,'ParáMarzo':28,'ParáAbril':27.17,'ParáMayo':27.67,'ParáJunio':28,

          'ParáJulio':28,'ParáAgosto':28,'ParáSetiembre':28,'ParáOctubre':28,'ParáNoviembre':28,'ParáDiciembre':27.75,

          'Sao PauloEnero': 23.88,'Sao PauloFebrero': 24.17,'Sao PauloMarzo':23.5,'Sao PauloAbril':22,'Sao PauloMayo':19.5,'Sao PauloJunio':17.88,

          'Sao PauloJulio':17.63,'Sao PauloAgosto':18.75,'Sao PauloSetiembre':19.67,'Sao PauloOctubre':21,'Sao PauloNoviembre':22.17,'Sao PauloDiciembre':23.25,

          'RondoniaEnero': 26.5,'RondoniaFebrero': 26.67,'RondoniaMarzo':27.38,'RondoniaAbril':27.33,'RondoniaMayo':27,'RondoniaJunio':26.75,

          'RondoniaJulio':26.75,'RondoniaAgosto':27.67,'RondoniaSetiembre':28,'RondoniaOctubre':27.75,'RondoniaNoviembre':27.33,'RondoniaDiciembre':26.88,

          'PernambucoEnero': 28.5,'PernambucoFebrero': 28.5,'PernambucoMarzo':28.5,'PernambucoAbril':28,'PernambucoMayo':27.49,'PernambucoJunio':26.38,

          'PernambucoJulio':25.75,'PernambucoAgosto':25.88,'PernambucoSetiembre':26.67,'PernambucoOctubre':27.5,'PernambucoNoviembre':28.17,'PernambucoDiciembre':28.5,

          'Minas GeraisEnero': 24,'Minas GeraisFebrero': 24.5,'Minas GeraisMarzo':23.88,'Minas GeraisAbril':22.83,'Minas GeraisMayo':20.67,'Minas GeraisJunio':19.17,

          'Minas GeraisJulio':18.88,'Minas GeraisAgosto':20,'Minas GeraisSetiembre':21.67,'Minas GeraisOctubre':23,'Minas GeraisNoviembre':23.33,'Minas GeraisDiciembre':23.33,

          'BahiaEnero': 28,'BahiaFebrero': 28,'BahiaMarzo':28,'BahiaAbril':27.5,'BahiaMayo':26.5,'BahiaJunio':25.63,

          'BahiaJulio':24.75,'BahiaAgosto':24.83,'BahiaSetiembre':25.5,'BahiaOctubre':26.25,'BahiaNoviembre':26.83,'BahiaDiciembre':27.25,

          'GoiasEnero': 25.25,'GoiasFebrero': 25.5,'GoiasMarzo':25.38,'GoiasAbril':24.67,'GoiasMayo':23.33,'GoiasJunio':22.17,

          'GoiasJulio':22.25,'GoiasAgosto':23.88,'GoiasSetiembre':25.67,'GoiasOctubre':26.13,'GoiasNoviembre':25.67,'GoiasDiciembre':21.17,

          'AcreEnero': 26.5,'AcreFebrero': 26.67,'AcreMarzo':27,'AcreAbril':26.83,'AcreMayo':25.99,'AcreJunio':25.5,

          'AcreJulio':25.38,'AcreAgosto':26.63,'AcreSetiembre':27.17,'AcreOctubre':27.25,'AcreNoviembre':27.33,'AcreDiciembre':26.88,

          'TocantinsEnero': 26.5,'TocantinsFebrero': 26.5,'TocantinsMarzo':26.5,'TocantinsAbril':26.83,'TocantinsMayo':26.83,'TocantinsJunio':26.25,

          'TocantinsJulio':26.38,'TocantinsAgosto':27.75,'TocantinsSetiembre':29,'TocantinsOctubre':28.25,'TocantinsNoviembre':27,'TocantinsDiciembre':26.5,

          'AmazonasEnero': 27,'AmazonasFebrero': 27,'AmazonasMarzo':27.5,'AmazonasAbril':27,'AmazonasMayo':27.5,'AmazonasJunio':28,

          'AmazonasJulio':28.5,'AmazonasAgosto':29,'AmazonasSetiembre':29,'AmazonasOctubre':29,'AmazonasNoviembre':28.5,'AmazonasDiciembre':27}
vel_viento = {'RioEnero': 10.43,'RioFebrero': 10.3,'RioMarzo':10.2,'RioAbril':10.73,'RioMayo':11.27,'RioJunio':11.37,

          'RioJulio':12,'RioAgosto':12.7,'RioSetiembre':13.4,'RioOctubre':12.97,'RioNoviembre':12.07,'RioDiciembre':11.07,

          'ParaibaEnero': 19.23,'ParaibaFebrero': 18.35,'ParaibaMarzo':17.35,'ParaibaAbril':17.13,'ParaibaMayo':18.37,'ParaibaJunio':19.9,

          'ParaibaJulio':21.1,'ParaibaAgosto':21.8,'ParaibaSetiembre':21.43,'ParaibaOctubre':21.1,'ParaibaNoviembre':20.73,'ParaibaDiciembre':20.17,

          'Mato GrossoEnero': 9.3,'Mato GrossoFebrero': 8.53,'Mato GrossoMarzo':7.9,'Mato GrossoAbril':8.17,'Mato GrossoMayo':9.2,'Mato GrossoJunio':10.2,

          'Mato GrossoJulio':11.4,'Mato GrossoAgosto':12.23,'Mato GrossoSetiembre':11.87,'Mato GrossoOctubre':10.33,'Mato GrossoNoviembre':9.6,'Mato GrossoDiciembre':9.63,

          'AlagoasEnero': 19.9,'AlagoasFebrero': 19.1,'AlagoasMarzo':18.07,'AlagoasAbril':17.33,'AlagoasMayo':17.8,'AlagoasJunio':18.97,

          'AlagoasJulio':19.7,'AlagoasAgosto':19.8,'AlagoasSetiembre':19.45,'AlagoasOctubre':20.2,'AlagoasNoviembre':21.1,'AlagoasDiciembre':20.9,

          'PiauEnero': 4.77,'PiauFebrero': 4.5,'PiauMarzo':4.27,'PiauAbril':4.17,'PiauMayo':4.7,'PiauJunio':5.43,

          'PiauJulio':5.9,'PiauAgosto':6.3,'PiauSetiembre':6.73,'PiauOctubre':7.05,'PiauNoviembre':6.43,'PiauDiciembre':5.63,

          'Espirito SantoEnero': 16.97,'Espirito SantoFebrero': 15.7,'Espirito SantoMarzo':14.63,'Espirito SantoAbril':14.05,'Espirito SantoMayo':14.3,'Espirito SantoJunio':14.25,

          'Espirito SantoJulio':15.5,'Espirito SantoAgosto':16.93,'Espirito SantoSetiembre':18.27,'Espirito SantoOctubre':18.6,'Espirito SantoNoviembre':18.13,'Espirito SantoDiciembre':17.6,

          'CearaEnero': 19.23,'CearaFebrero': 17.4,'CearaMarzo':15.67,'CearaAbril':15.25,'CearaMayo':17.43,'CearaJunio':20.2,

          'CearaJulio':22.67,'CearaAgosto':25.07,'CearaSetiembre':26.07,'CearaOctubre':25.63,'CearaNoviembre':24.43,'CearaDiciembre':22.2,

          'MaranhaoEnero': 9.3,'MaranhaoFebrero': 8.7,'MaranhaoMarzo':8.07,'MaranhaoAbril':7.27,'MaranhaoMayo':7.1,'MaranhaoJunio':7.43,

          'MaranhaoJulio':8,'MaranhaoAgosto':9.2,'MaranhaoSetiembre':10.73,'MaranhaoOctubre':11.4,'MaranhaoNoviembre':11.33,'MaranhaoDiciembre':10.6,

          'SergipeEnero': 19.13,'SergipeFebrero': 18.5,'SergipeMarzo':17.67,'SergipeAbril':16.93,'SergipeMayo':17.27,'SergipeJunio':17.9,

          'SergipeJulio':18.47,'SergipeAgosto':18.63,'SergipeSetiembre':18.8,'SergipeOctubre':19.5,'SergipeNoviembre':20.03,'SergipeDiciembre':19.67,

          'Distrito FederalEnero': 9.63,'Distrito FederalFebrero': 9.23,'Distrito FederalMarzo':9.07,'Distrito FederalAbril':9.8,'Distrito FederalMayo':10.57,'Distrito FederalJunio':11.67,

          'Distrito FederalJulio':12.83,'Distrito FederalAgosto':13.7,'Distrito FederalSetiembre':12.87,'Distrito FederalOctubre':11,'Distrito FederalNoviembre':9.77,'Distrito FederalDiciembre':9.73,

          'AmapaEnero': 5.5,'AmapaFebrero': 5.4,'AmapaMarzo':5.07,'AmapaAbril':4.63,'AmapaMayo':4.37,'AmapaJunio':4.57,

          'AmapaJulio':4.77,'AmapaAgosto':5.1,'AmapaSetiembre':5.6,'AmapaOctubre':6.07,'AmapaNoviembre':6.25,'AmapaDiciembre':6,

          'RoraimaEnero': 11.9,'RoraimaFebrero': 12.5,'RoraimaMarzo':11.47,'RoraimaAbril':9.4,'RoraimaMayo':7.73,'RoraimaJunio':7.17,

          'RoraimaJulio':6.7,'RoraimaAgosto':6.55,'RoraimaSetiembre':6.73,'RoraimaOctubre':6.83,'RoraimaNoviembre':7.5,'RoraimaDiciembre':9.8,

          'Santa CatarinaEnero': 17.1,'Santa CatarinaFebrero': 16.35,'Santa CatarinaMarzo':15.65,'Santa CatarinaAbril':16,'Santa CatarinaMayo':15.77,'Santa CatarinaJunio':15.55,

          'Santa CatarinaJulio':15.97,'Santa CatarinaAgosto':16.87,'Santa CatarinaSetiembre':18.1,'Santa CatarinaOctubre':18.6,'Santa CatarinaNoviembre':18.53,'Santa CatarinaDiciembre':17.83,

          'ParáEnero': 5.4,'ParáFebrero': 5.13,'ParáMarzo':4.87,'ParáAbril':4.4,'ParáMayo':4.25,'ParáJunio':4.47,

          'ParáJulio':4.67,'ParáAgosto':5.1,'ParáSetiembre':5.9,'ParáOctubre':6.4,'ParáNoviembre':6.4,'ParáDiciembre':6.13,

          'Sao PauloEnero': 11.77,'Sao PauloFebrero': 11,'Sao PauloMarzo':10.93,'Sao PauloAbril':11.13,'Sao PauloMayo':11,'Sao PauloJunio':11.23,

          'Sao PauloJulio':11.8,'Sao PauloAgosto':12.27,'Sao PauloSetiembre':13.53,'Sao PauloOctubre':13.87,'Sao PauloNoviembre':13.6,'Sao PauloDiciembre':12.73,

          'RondoniaEnero': 2.53,'RondoniaFebrero': 2.53,'RondoniaMarzo':2.5,'RondoniaAbril':2.6,'RondoniaMayo':2.83,'RondoniaJunio':3.07,

          'RondoniaJulio':3.1,'RondoniaAgosto':2.9,'RondoniaSetiembre':2.73,'RondoniaOctubre':2.53,'RondoniaNoviembre':2.5,'RondoniaDiciembre':2.5,

          'PernambucoEnero': 18.27,'PernambucoFebrero': 17.75,'PernambucoMarzo':16.95,'PernambucoAbril':16.83,'PernambucoMayo':17.93,'PernambucoJunio':19.3,

          'PernambucoJulio':20.23,'PernambucoAgosto':20.55,'PernambucoSetiembre':19.9,'PernambucoOctubre':19.53,'PernambucoNoviembre':19.43,'PernambucoDiciembre':19.07,

          'Minas GeraisEnero': 11.87,'Minas GeraisFebrero': 11.45,'Minas GeraisMarzo':11.23,'Minas GeraisAbril':11.15,'Minas GeraisMayo':11.3,'Minas GeraisJunio':11.5,

          'Minas GeraisJulio':12.53,'Minas GeraisAgosto':13.97,'Minas GeraisSetiembre':15,'Minas GeraisOctubre':14.73,'Minas GeraisNoviembre':13.4,'Minas GeraisDiciembre':12.47,

          'BahiaEnero': 13.67,'BahiaFebrero': 13.45,'BahiaMarzo':13.03,'BahiaAbril':13.13,'BahiaMayo':13.77,'BahiaJunio':14.23,

          'BahiaJulio':14.53,'BahiaAgosto':14.75,'BahiaSetiembre':14.4,'BahiaOctubre':14.27,'BahiaNoviembre':14.1,'BahiaDiciembre':13.7,

          'GoiasEnero': 10.23,'GoiasFebrero': 9.47,'GoiasMarzo':9.17,'GoiasAbril':9.87,'GoiasMayo':10.7,'GoiasJunio':11.73,

          'GoiasJulio':13.1,'GoiasAgosto':14.05,'GoiasSetiembre':13.13,'GoiasOctubre':11.07,'GoiasNoviembre':10,'GoiasDiciembre':10.27,

          'AcreEnero': 2.53,'AcreFebrero': 2.5,'AcreMarzo':2.5,'AcreAbril':2.6,'AcreMayo':2.8,'AcreJunio':3,

          'AcreJulio':3.15,'AcreAgosto':3.1,'AcreSetiembre':2.93,'AcreOctubre':2.67,'AcreNoviembre':2.53,'AcreDiciembre':2.53,

          'TocantinsEnero': 7.3,'TocantinsFebrero': 7.2,'TocantinsMarzo':7,'TocantinsAbril':7.37,'TocantinsMayo':8.63,'TocantinsJunio':10.1,

          'TocantinsJulio':11.2,'TocantinsAgosto':12,'TocantinsSetiembre':11.53,'TocantinsOctubre':9.43,'TocantinsNoviembre':7.93,'TocantinsDiciembre':7.4,

          'AmazonasEnero': 4.3,'AmazonasFebrero': 4.33,'AmazonasMarzo':4.23,'AmazonasAbril':4.15,'AmazonasMayo':4.17,'AmazonasJunio':4.5,

          'AmazonasJulio':4.7,'AmazonasAgosto':4.6,'AmazonasSetiembre':4.4,'AmazonasOctubre':4.03,'AmazonasNoviembre':3.85,'AmazonasDiciembre':4}
# Ahora crearemos las columnas de humedad, temperatura y velocidad del viento

df_new['humedad'] = df_new['col_ref'].map(humedad)

df_new['temperatura'] = df_new['col_ref'].map(temperatura)

df_new['vel_viento'] = df_new['col_ref'].map(vel_viento)
# Array de los valores

Y = df_new.groupby(['state','month']).mean().loc[:,'number':].values[:,0]

X = df_new.groupby(['state','month']).mean().loc[:,'number':].values[:,1:]



# Convertimos todos los numeros en porcentaje en base al mayor numero promedio de incendios

num_max = df_new.groupby(['state','month']).mean().max().number

aux = map(lambda x: x*100/num_max,Y)



# Y to converted percents

Y = np.array(list(aux))

# Vamos a estandarizar los datos primero

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X_s = scaler.transform(X)
# Aplicaremos el modelo lineal: Polynomial Regression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures



polynomial_features = PolynomialFeatures(degree=5)

x_poly = polynomial_features.fit_transform(X_s)
model = LinearRegression()

model.fit(x_poly,Y)

y_poly_pred = model.predict(x_poly)
model.predict([x_poly[0],x_poly[12],x_poly[24]])
rmse = np.sqrt(mean_squared_error(Y,y_poly_pred))

r2 = r2_score(Y,y_poly_pred)
print('RMSE: ', rmse)

print('r2_score: ', r2)
df_new.groupby(['state','month']).mean().loc[:,'number':]
# Ya definido y entrenado el modelo polinomial, pasaremos a crear el modelo binario

# Instalamos la libreria dependencia

!pip install cvxpy

!pip install cvxopt
# Importamos la libreria

import cvxpy as cp
# Datos y parametros para el problema de asignacion de recursos para...

# incendios forestales

# Indices i:estados -> de 1 a 23, j:cat. helicopteros -> de 1 a 5

# Para efectos de sencillez de primera muestra del modelo solo usaremos

# 3 estados



# Parametros

# Superficie forestal del estado i en millones de hectareas

sup_for = np.array([15.3, 2.8, 14.2])

# Nivel de deforestacion del estado i : 1-> bajo, 2-> medio, 3-> alto

niv_def = np.array([2.,1.,1.])



# Datos

# Probabilidad de que ocurra un incendio forestal en el estado i

# (Obtenido del modelo polinomial): Acre, Alagoas y Amapa.

prob_inc_for = model.predict([x_poly[0],x_poly[12],x_poly[24]])

# Cantidad de helicopteros disponibles en la categoria j

# entrada por el usuario, se simula en este caso con 1 helicoptero por categoria

# desde la categoria 3 a la 5

cant_h_cat = np.array([0,0,1,1,1])

# Variable de decision

# Booleana para asignar el helicoptero de la categoria j al estado i

# 1 -> si asigna

# 0 -> no asigna

# matriz jxi

assign = cp.Variable((5,3),boolean=True)
# Restricciones

# No asignar mas de un helicoptero al estado i

h_to_s_const = cp.sum(assign,axis=0) <= 1

# No sobrepasar la disponibilidad de helicopteros en la categoria j

disp_cat_const = cp.sum(assign,axis=1)<= cant_h_cat

# Restriccion de categorias: si hay helicopteros de una categoria mas alta

# se debera asignar a los estados con superficie forestal mas alta

cat1 = cant_h_cat[0]

cat2 = cant_h_cat[1]

cat3 = cant_h_cat[2]

cat4 = cant_h_cat[3]

cat5 = cant_h_cat[4]



if cat5>0:

    cat5_const = cp.sum(assign[4]*sup_for) >= cp.sum(assign[3]*sup_for)

else:

    cat5_const = cp.sum(assign[4]*sup_for) == 0

    

if cat4>0:

    cat4_const = cp.sum(assign[3]*sup_for) >= cp.sum(assign[2]*sup_for)

else:

    cat4_const = cp.sum(assign[3]*sup_for) == 0

    

if cat3>0:

    cat3_const = cp.sum(assign[2]*sup_for) >= cp.sum(assign[1]*sup_for)

else:

    cat3_const = cp.sum(assign[2]*sup_for) == 0

    

if cat2>0:

    cat2_const = cp.sum(assign[1]*sup_for) >= cp.sum(assign[0]*sup_for)

else:

    cat2_const = cp.sum(assign[1]*sup_for) == 0

    

constraints = [h_to_s_const,disp_cat_const,cat5_const,cat4_const,cat3_const,

              cat2_const]
# Funcion objetivo

FO = cp.sum(cp.sum(assign*(prob_inc_for*(sup_for/niv_def))))

problem = cp.Problem(cp.Maximize(FO),constraints = constraints)

problem.solve(solver=cp.GLPK_MI)
# Valores optimos de la variable de decision

assign.value