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
import matplotlib.pyplot as plt
import numpy as np
usa_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/USA_0101-0703.npy')
plt.imshow(usa_npy[187]) #July 6th USA satellite image

import matplotlib.pyplot as plt
import numpy as np
usa_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/USA_0101-0703.npy')
inp=usa_npy[187]
#count_img=len(usa_npy)
#height,width=usa_npy[187].shape

# Set missing pixel values as zero
inp[0:400,0:1500]=0
inp[0:260,2000:]=0
plt.imshow(usa_npy[187])

import matplotlib.pyplot as plt
import numpy as np
usa_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/USA_0101-0703.npy')
#plt.imshow(usa_npy[187])


#View Numpy format images
fig = plt.figure(figsize=(15, 40))  # width, height in inches
for i in range(188):
    sub = fig.add_subplot(24,8, i + 1)
    usa_npy[i,0:400,0:1500]=0
    usa_npy[i,0:260,2000:]=0
    sub.imshow(usa_npy[i],interpolation='nearest')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('../input/moon-phase-calendar-2020/moon.png')
plt.figure(figsize=(18, 10))
imgplot = plt.imshow(img)
import matplotlib.pyplot as plt
import numpy as np
usa_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/USA_0101-0703.npy')  #Till july6

# Setting missing pixel values as Zero

usa_npy[0:173,0:400,0:1500]=0
usa_npy[0:173,0:260,2000:]=0

#USA
Moon_phase1_till_jan25=usa_npy[0:24].mean()
Moon_phase2_till_feb23=usa_npy[25:53].mean()
Moon_phase3_till_mar24=usa_npy[54:83].mean()
Moon_phase4_till_apr23=usa_npy[84:113].mean()
Moon_phase5_till_may22=usa_npy[114:142].mean()
Moon_phase6_till_jun21=usa_npy[143:173].mean()


me=[Moon_phase1_till_jan25,Moon_phase2_till_feb23,Moon_phase3_till_mar24,Moon_phase4_till_apr23,Moon_phase5_till_may22,Moon_phase6_till_jun21]
x=['phase1(till jan25)','phase2(till feb23)','phase3(till mar24)','phase4(till apr23)','phase5(till may22)','phase6(till june21)']
plt.xticks(rotation=45)
plt.plot(x,me)
plt.xlabel('Moon_phase')
plt.ylabel('Pixel mean value')
