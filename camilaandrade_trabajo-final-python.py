import matplotlib.pyplot as plt 

import numpy as np 

import os 

import seaborn as sns

import pandas as pd

World_Bank_Data_India_Definitions = pd.read_csv("../input/world-bank-data-of-indian-economy-since-1991/World_Bank_Data_India_Definitions.csv")

WorldBank_Data_India = pd.read_csv("../input/world-bank-data-of-indian-economy-since-1991/World_Bank_Data_India.csv")
WorldBank_Data_India.info()
WorldBank_Data_India.head()
WorldBank_Data_India.columns
NAValues = WorldBank_Data_India.loc[:, WorldBank_Data_India.isnull().any()].isnull().sum().sort_values(ascending=False)



print(NAValues)
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
IPLOT = plt.figure(figsize=(18,10))

IPLOT.suptitle('Desempeño Económico de India', fontsize= 25)



ax1 = IPLOT.add_subplot(231)

ax1.set_title('Población Total')

ax1.plot(WorldBank_Data_India['Years'], WorldBank_Data_India['POP_TOTL'], color = 'green')

ax1.grid(True)



ax2 = IPLOT.add_subplot(232)

ax2.set_title('Fuerza Laboral')

ax2.plot(WorldBank_Data_India['Years'], WorldBank_Data_India['LF_TOTL'], color = 'magenta')

ax2.grid(True)



ax3 = IPLOT.add_subplot(233)

ax3.set_title('Trabajadores Independientes')

ax3.plot(WorldBank_Data_India['Years'], WorldBank_Data_India['EMP_SELF'], color = 'orange')

ax3.grid(True)



ax4 = IPLOT.add_subplot(234)

ax4.set_title('PIB de India')

ax4.plot(WorldBank_Data_India['Years'], WorldBank_Data_India['GDP_IND'], color = 'chocolate')

ax4.grid(True)



ax5 = IPLOT.add_subplot(235)

ax5.set_title('Desempleo Total')

ax5.plot(WorldBank_Data_India['Years'], WorldBank_Data_India['UEM_TOTL'], color = 'red')

ax5.grid(True)





plt.show()


