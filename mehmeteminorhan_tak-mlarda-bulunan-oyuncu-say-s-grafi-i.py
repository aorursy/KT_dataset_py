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
###TAKIMLARDA KAÇAR ADET FUTBOLCU OLDUĞUNUN GÖRÜNTÜLENMESİ



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



takım = pd.read_excel("/kaggle/input/sper-lig-20192020-players/Futbolcular.xlsx")

result = takım["Kulüp Adı"].value_counts(sort = False)

x_label = list(result.index)

x = range(0,18)

y = list(result)

fig, ax = plt.subplots(1,constrained_layout=True)

ax.plot(x,y,"o--b")

ax.set_xticks(x)

ax.set_xticklabels(x_label, rotation=25, fontsize=5)

ax.autoscale(enable=True)



for xs,ys in zip(x,y):

    plt.text(xs,ys,str(ys),color="red",fontsize = 12)



plt.title("Takımlarda Bulunan Oyuncu İstatistiği")

plt.xlabel("Takımlar")

plt.ylabel("Oyuncu Sayıları")

plt.show()