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
import matplotlib.pyplot as plt

import seaborn as sns



df_tst = pd.read_csv("../input/liverpool-ion-switching/test.csv")

df_trn = pd.read_csv("../input/liverpool-ion-switching/train.csv")
df_trn.head()
#df_trn['signal'].plot()

plt.plot(df_trn.time, df_trn.signal, color='blue')

plt.plot(df_trn.time, df_trn.open_channels, color='red')
# current가 -값도 있는데 offset을 잡아줘야 할 수도 있겠음

# current가 오르내리면서 (500개 배치긴 하지만) 채널이 열리고 닫히고 하는 것인가?

# noise를 low pass filter를 써야 하나?

# 이미지 패턴으로 접근해야 하나?
# 처음 500개만 다시 그려보기

# https://talkingaboutme.tistory.com/entry/PythonVisualization-matplotlib-multiple-plot

df_trn_500 = df_trn[df_trn.index <= 500]

#df_trn_500.head()

plt.plot(df_trn_500.time, df_trn_500.signal, color='blue')

plt.plot(df_trn_500.time, df_trn_500.open_channels, color='red')
df_trn_1000 = df_trn[500:1001]

#df_trn_1000.tail()

plt.plot(df_trn_500.time, df_trn_500.signal, color='blue')

plt.plot(df_trn_500.time, df_trn_500.open_channels, color='red')
# 2nd 배치까지 channel open이 없음

df_trn_1000[df_trn_1000['open_channels']>0].head()
df_trn[df_trn['open_channels']>0].head()
df_trn_7500 = df_trn[7000:7501]

plt.plot(df_trn_7500.time, df_trn_7500.signal, color='blue')

plt.plot(df_trn_7500.time, df_trn_7500.open_channels, color='red')



# channel이 하나 열렸다 닫히는 케이스로 보아야 하는 것인지?

# open_channels가 traget value라면... 열렸다 닫힌 것으로 센싱되는게 맞을듯하고

# 0.7231 근처의 데이터를 보고 어느 범주까지 들어오는 것이 맞는지

# 범위를 좁힐 수 있어야 open_channels 정확도가 올라갈 것으로 보임
df_trn[7225:7237]
# -2에서 -1 수준으로 확바뀔 때 채널이 한번 열렸다고 체크했는데

# 실제 데이터는 7230이 -2.0027 도 채널이 열린것으로 체크했음

# 근데 이정도 오차는 발생할텐데.. 이걸 정밀도를 높이면 overfit이 발생할듯함

# 7230이 답은 1이지만 정말 1인지도 알 수 없는 일이고..