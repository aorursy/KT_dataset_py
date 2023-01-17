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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

import scipy.cluster.hierarchy as shc

from sklearn.cluster import AgglomerativeClustering
df_airbnb = pd.read_csv ( "/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv" ) 
df_airbnb.info ( )
# ใช้คอลัมน์ latitude และ longitude ในการทำ clustering

df_airbnb = df_airbnb [ [ "latitude", "longitude" ] ]



# เนื่องจากข้อมูลมีปริมาณ 48895 ทำให้ประมวลผลไม่ไหว จึงลองทำแค่ 1000 ข้อมูลเท่านั้น

latlong_values = df_airbnb.loc [ 0 : 1000 ].values

latlong_values
# ลอง plot กราฟดู



latitude = latlong_values [ : , 0 ]

longitude = latlong_values [ : , 1 ]



plt.figure ( figsize = ( 25, 8 ) )

plt.scatter ( latitude, longitude, s = 15 )

plt.show ( )
# ทำ scaler เพื่อให้ข้อมูลมีขนาดเล็กลง



scaler = StandardScaler ( )

latlong_values = scaler.fit_transform ( latlong_values )

latlong_values
# ลอง plot กราฟดูอีกครั้ง เพื่อเปรียบเทียบ ก่อนและหลัง การทำ scaler



latitude = latlong_values [ : , 0 ]

longitude = latlong_values [ : , 1 ]



plt.figure ( figsize = ( 25, 8 ) )

plt.scatter ( latitude, longitude, s = 15 )

plt.show ( )
# ทำ Hierarchical Clustering



plt.figure ( figsize = ( 25, 12 ) )

dendrogram = shc.dendrogram ( shc.linkage ( latlong_values, method = "ward" ) )

plt.title ( "Latitude & Longitude" )

plt.show ( )