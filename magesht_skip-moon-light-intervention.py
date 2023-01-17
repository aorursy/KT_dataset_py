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
from skimage import color
from skimage import io

#Read CSV
oil=pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv')
oil.head()

# Read Numpy Files
usa_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/USA_0101-0629.npy')

#View Numpy format images
fig = plt.figure(figsize=(15, 40))  # width, height in inches
for i in range(174):
    sub = fig.add_subplot(22,8, i + 1)
    sub.imshow(usa_npy[i],interpolation='nearest')
stack=[]
i=0
while i<181:
    #print(usa_npy[i].mean())
    stack.append(usa_npy[i].mean())
    i=i+1
#print(stack)

#Graph Plot
plt.xlabel('Jan01 to June-29 ---->')
plt.ylabel('USA: Pixel Mean Value')
plt.plot(stack)
plt.show()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('../input/moon-phase-calendar/2020_moon_phase.PNG')
plt.figure(figsize=(18, 10))
imgplot = plt.imshow(img)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import io

#Read CSV
oil=pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv')
oil.head()

# Read Numpy Files
usa_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/USA_0101-0629.npy')
#print(usa_npy.shape)

Moon_phase1_till_jan25=usa_npy[0:24].mean()
Moon_phase2_till_feb23=usa_npy[25:53].mean()
Moon_phase3_till_mar24=usa_npy[54:83].mean()
Moon_phase4_till_apr23=usa_npy[84:113].mean()
Moon_phase5_till_may22=usa_npy[114:142].mean()
Moon_phase6_till_jun21=usa_npy[143:173].mean()
#Moon_phase7_till_jul20=usa_npy[174:203].mean()
me=[Moon_phase1_till_jan25,Moon_phase2_till_feb23,Moon_phase3_till_mar24,Moon_phase4_till_apr23,Moon_phase5_till_may22,Moon_phase6_till_jun21]
x=['phase1(till jan25)','phase2(till feb23)','phase3(till mar24)','phase4(till apr23)','phase5(till may22)','phase6(till june22)']
plt.xticks(rotation=45)
plt.plot(x,me)
plt.xlabel('Moon_phase')
plt.ylabel('Pixel mean value')
usa_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/USA_0101-0629.npy')
China_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/China_0101-0629.npy')
France_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/France_0101-0629.npy')
Italy_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/Italy_0101-0629.npy')

#USA
Moon_phase1_till_jan25=usa_npy[0:24].mean()
Moon_phase2_till_feb23=usa_npy[25:53].mean()
Moon_phase3_till_mar24=usa_npy[54:83].mean()
Moon_phase4_till_apr23=usa_npy[84:113].mean()
Moon_phase5_till_may22=usa_npy[114:142].mean()
Moon_phase6_till_jun21=usa_npy[143:173].mean()
usa=[Moon_phase1_till_jan25,Moon_phase2_till_feb23,Moon_phase3_till_mar24,Moon_phase4_till_apr23,Moon_phase5_till_may22]
x=['phase1(till jan25)','phase2(till feb23)','phase3(till mar24)','phase4(till apr23)','phase5(till may22)']
plt.figure(figsize=(25, 10))
plt.xticks(rotation=45,fontsize=20)
plt.yticks(fontsize=20)
plt.plot(x,usa,label='USA',color='black')
plt.ylabel('Pixel mean value',fontsize=26)
plt.legend(fontsize=26)

#CHINA
Moon_phase1_till_jan25=China_npy[0:24].mean()
Moon_phase2_till_feb23=China_npy[25:53].mean()
Moon_phase3_till_mar24=China_npy[54:83].mean()
Moon_phase4_till_apr23=China_npy[84:113].mean()
Moon_phase5_till_may22=China_npy[114:142].mean()
Moon_phase6_till_jun21=China_npy[143:173].mean()
china=[Moon_phase1_till_jan25,Moon_phase2_till_feb23,Moon_phase3_till_mar24,Moon_phase4_till_apr23,Moon_phase5_till_may22]
x=['phase1(till jan25)','phase2(till feb23)','phase3(till mar24)','phase4(till apr23)','phase5(till may22)']
plt.xticks(rotation=45)
plt.plot(x,china,label='China',color='red')
plt.legend(fontsize=26)

#FRANCE
Moon_phase1_till_jan25=France_npy[0:24].mean()
Moon_phase2_till_feb23=France_npy[25:53].mean()
Moon_phase3_till_mar24=France_npy[54:83].mean()
Moon_phase4_till_apr23=France_npy[84:113].mean()
Moon_phase5_till_may22=France_npy[114:142].mean()
Moon_phase6_till_jun21=France_npy[143:173].mean()
france=[Moon_phase1_till_jan25,Moon_phase2_till_feb23,Moon_phase3_till_mar24,Moon_phase4_till_apr23,Moon_phase5_till_may22]
x=['phase1(till jan25)','phase2(till feb23)','phase3(till mar24)','phase4(till apr23)','phase5(till may22)']
plt.xticks(rotation=45)
plt.plot(x,france,label='France',color='yellow')
plt.legend(fontsize=26)

#ITALY
Moon_phase1_till_jan25=Italy_npy[0:24].mean()
Moon_phase2_till_feb23=Italy_npy[25:53].mean()
Moon_phase3_till_mar24=Italy_npy[54:83].mean()
Moon_phase4_till_apr23=Italy_npy[84:113].mean()
Moon_phase5_till_may22=Italy_npy[114:142].mean()
Moon_phase6_till_jun21=Italy_npy[143:173].mean()
italy=[Moon_phase1_till_jan25,Moon_phase2_till_feb23,Moon_phase3_till_mar24,Moon_phase4_till_apr23,Moon_phase5_till_may22]
x=['phase1(till jan25)','phase2(till feb23)','phase3(till mar24)','phase4(till apr23)','phase5(till may22)']
plt.plot(x,italy,label='Italy',color='green')
plt.legend(fontsize=26)

oil=pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv')
start_date='2020-01-01'
end_date='2020-06-29'
oil1=oil[oil.Date>'2020-01-01']
plt.figure(figsize=(25, 10))
plt.plot(oil1.Price,label='Oil price')
plt.legend(fontsize=26)
plt.xlabel('Date Jan01 to July 29 ----->',fontsize=26)
plt.yticks(fontsize=20)


