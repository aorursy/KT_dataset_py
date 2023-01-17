import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df =pd.read_csv('../input/imports_85_data.csv',header=None,encoding='utf-8')
header=['Symbolling','Normalized Losses','Make','Fuel Type','Aspiration','Num_Doors','Body-Style','Drive-Wheels',

       'Engine Location','Wheel Base','Length','Width','Height','Curb-Weight','Engine-Type',

       'Num Of Cylinders', 'Engine - Size' , 'Fuel System' ,'Bore','Stroke','Compression-Ratio','Horse-Power','Peak-RPM','City-MPG',"Price"]
header.append('Highway-MPG')

df.columns=header
df.dtypes
df =df.drop([0])

df.head()