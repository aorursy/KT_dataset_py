import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
data = pd.read_excel('../input/covid19-india/Complete COVID-19_Report in India.xlsx')

data.head()
def data_info():

    print(data.shape)

    print("****************************************##########*************************************")

    print(data.info())
data_info()
data['Detected State'].value_counts()
viz = pd.value_counts(data['Detected State'], ascending=True).plot(kind='barh',fontsize='40',title='STATE-WISE Patient Distribution', figsize=(50,100))

viz.set(xlabel='Affected_Patient_Count',ylabel='States')

viz.xaxis.label.set_size(50)

viz.yaxis.label.set_size(60)

viz.title.set_size(50)

plt.show()
data['Current Status'].value_counts()