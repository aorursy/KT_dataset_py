import pandas as pd 

import numpy as np 

import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns 

color = sns.color_palette()

import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from matplotlib.ticker import PercentFormatter

import plotly.plotly as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import os

print(os.listdir("../input"))



# import cufflinks and offline mode

import cufflinks as cf

cf.go_offline()
mn = pd.read_csv("../input/GreenHouseGasEmission.csv")
mn.head()
gases = mn.GHGs.value_counts() / len(mn.GHGs)

ax = gases.plot.bar()



# manipulate

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
sector = mn['Sector'].value_counts() / len(mn.Sector)

ax1 = sector.plot.bar()



# manipulate

vals1 = ax1.get_yticks()

ax1.set_yticklabels(['{:,.2%}'.format(x) for x in vals1])
source_group = mn['Source_Group'].value_counts() / len(mn.Source_Group)

ax3 = source_group.plot.bar()

# manipulate

vals3 = ax3.get_yticks()

ax3.set_yticklabels(['{:,.2%}'.format(x) for x in vals3])
source = mn['Source'].value_counts() 

source = source[source >= 76] / 1000

source.plot.bar()

ax4 = source.plot.bar()

# manipulate

vals4 = ax4.get_yticks()

ax4.set_yticklabels(['{:,.2%}'.format(x) for x in vals4])
source_categories = mn['Source_Categories'].value_counts()

source_categories = source_categories[source_categories >= 76] / 1000

ax2 = source_categories.plot.bar()



# manipulate

vals2 = ax2.get_yticks()

ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals2])
#mn.Year.unique()

#mn.GHGs.unique()

#CH4 = mn[mn.GHGs == 'CH4']

#CH4.GHGs.value_counts()

#CH4_year = CH4["Year"].value_counts()
table = pd.pivot_table(mn, values='Emissions', index=['Sector', 'Year'], columns=['GHGs'])

table.head()
gasnames = table.columns.values

gasnames
def sector_plot(nameOfSector):

    data = table.loc[nameOfSector]

    plt.plot(data)

    plt.legend(gasnames)

    plt.title(nameOfSector)
sector_plot('Agriculture')
sector_plot('Waste')
sector_plot('Transportation')
sector_plot('Industrial')
sector_plot('Electricity Generation')
sector_plot('Residential')
sector_plot('Commercial')
#def gas_plot(nameOfSector): 

#   table.plot(y = nameOfSector) 

    

#gas_plot(gasnames[4])
table2 = pd.pivot_table(mn, values='Emissions', index=['GHGs', 'Year'], columns=['Sector'])

table2.head()
sectornames = table2.columns.values

sectornames
def sector_plot2(nameOfGas):

    data = table2.loc[nameOfGas]

    plt.plot(data)

    plt.legend(sectornames)

    plt.title(nameOfGas)
sector_plot2(gasnames[0])
sector_plot2(gasnames[1])
sector_plot2(gasnames[3])
sector_plot2(gasnames[4])
sector_plot2(gasnames[5])
def gas_sector_plot3(nameOfGas, namesOfSect):

    data = table2.loc[nameOfGas]

    data.plot( y = namesOfSect)

    plt.legend(namesOfSect)

    plt.title(nameOfGas)
gas_sector_plot3(gasnames[1],sectornames[:4])
table3 = pd.pivot_table(mn, values='Emissions', index=['GHGs', 'Year'], columns=['Source_Categories'])

table3.head()
scnames = table3.columns.values
def sc_plot(nameOfGas):

    data = table2.loc[nameOfGas]

    plt.plot(data)

    plt.legend(scnames)

    plt.title(nameOfGas)
sc_plot(gasnames[0])
sc_plot(gasnames[1])
sc_plot(gasnames[2])
sc_plot(gasnames[4])
sc_plot(gasnames[5])
table4 = pd.pivot_table(mn, values='Emissions', index=['Source_Categories', 'Year'], columns=['GHGs'])

table4.head()
scnames1 = table4.columns.values

scnames1
def sc_plot1(nameofSC):

    data = table4.loc[nameofSC]

    plt.plot(data)

    plt.legend(scnames1)

    plt.title(nameofSC)
sc_plot1('Petroleum')
sc_plot1('Waste processing')
sc_plot1('Crops')
sc_plot1('Natural gas')
sc_plot1('Coal')
sc_plot1('Process')
sc_plot1('Other')
SCNames = table3.columns.values

SCNames
def gas_sc_plot(nameOfGas, nameofSC):

    data = table3.loc[nameOfGas]

    data.plot( y = nameofSC)

    plt.legend(nameofSC)

    plt.title(nameOfGas)
gas_sc_plot(scnames1[0], SCNames[:10])
gas_sc_plot(scnames1[1], SCNames[:10])
gas_sc_plot(scnames1[2], SCNames[:10])
gas_sc_plot(scnames1[3], SCNames[:10])
gas_sc_plot(scnames1[5], SCNames[:10])
table5 = pd.pivot_table(mn, values='Emissions', index=['Source_Group', 'Year'], columns=['GHGs'])

table5.head()
sgnames1 = table5.columns.values

sgnames1
def sg_plot1(nameofSG):

    data = table5.loc[nameofSG]

    plt.plot(data)

    plt.legend(sgnames1)

    plt.title(nameofSG)
sg_plot1('Fossil fuel')
sg_plot1('Processing')
sg_plot1('Process')
sg_plot1('Crop agriculture')
sg_plot1('Incineration')
sg_plot1('Fire')
table6 = pd.pivot_table(mn, values='Emissions', index=['GHGs', 'Year'], columns=['Source_Categories'])

table6.head()
SGNames = table6.columns.values

SGNames
def gas_sg_plot(nameOfGas, nameofSG):

    data = table6.loc[nameOfGas]

    data.plot( y = nameofSG)

    plt.legend(nameofSG)

    plt.title(nameOfGas)
gas_sg_plot(scnames1[0], SGNames[:10])
gas_sg_plot(scnames1[1], SGNames[:10])
gas_sg_plot(scnames1[2], SGNames[:10])
gas_sg_plot(scnames1[3], SGNames[:10])
gas_sg_plot(scnames1[5], SGNames[:10])
table7 = pd.pivot_table(mn, values='Emissions', index=['Source', 'Year'], columns=['Sector'])

table7.head()
snames1 = table7.columns.values

snames1
def s_plot1(nameofS):

    data = table7.loc[nameofS]

    plt.plot(data)

    plt.legend(snames1)

    plt.title(nameofS)
s_plot1('Oil')
s_plot1('Natural gas')
s_plot1('Other')
s_plot1('Coal')
s_plot1('Bus')
s_plot1('Net electricity imports')
table8 = pd.pivot_table(mn, values='Emissions', index=['Sector', 'Year'], columns=['Source'])

table8.head()
SNames = table8.columns.values

SNames
def gas_s_plot(nameOfGas, nameofS):

    data = table8.loc[nameOfGas]

    data.plot( y = nameofS)

    plt.legend(nameofS)

    plt.title(nameOfGas)
gas_s_plot(snames1[0], SNames[:10])
gas_s_plot(snames1[1], SNames[:10])
gas_s_plot(snames1[4], SNames[:10])
gas_s_plot(snames1[3], SNames[:10])