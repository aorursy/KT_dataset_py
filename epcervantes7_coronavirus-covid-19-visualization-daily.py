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




import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



confirmed = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

recovered = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')



confirmed_sum = np.sum(confirmed.iloc[:,4:confirmed.shape[1]])

recovered_sum  = np.sum(recovered.iloc[:,4:recovered.shape[1]])

deaths_sum  = np.sum(deaths.iloc[:,4:deaths.shape[1]])
ind=[0]

for i in range(len(confirmed_sum)-1):

#     print(confirmed_sum[i+1],confirmed_sum[i],confirmed_sum[i+1]-confirmed_sum[i])

    ind.append(confirmed_sum[i+1]-confirmed_sum[i])

by_day=pd.DataFrame(ind)

by_day.index=confirmed_sum.index

plt.figure(figsize=(20,10))

plt.plot(by_day, color = 'red'

        , label = 'new cases'

        , marker = 'o')

plt.title('daily new cases of coronavirus',size=30)

plt.ylabel('new cases',size=20)

plt.xticks(rotation=90,size=20)

plt.grid(True)

plt.show()

# plt.yticks(size=15)
ind=[0]

for i in range(len(deaths_sum)-1):

#     print(confirmed_sum[i+1],confirmed_sum[i],confirmed_sum[i+1]-confirmed_sum[i])

    ind.append(deaths_sum[i+1]-deaths_sum[i])

by_day=pd.DataFrame(ind)

by_day.index=deaths_sum.index

plt.figure(figsize=(20,10))

plt.plot(by_day, color = 'red'

        , label = 'deaths'

        , marker = 'o')

plt.title('daily deaths by coronavirus',size=30)

plt.ylabel('deaths',size=20)

plt.xticks(rotation=90,size=20)

plt.grid(True)

plt.show()

# plt.yticks(size=15)
ind=[0]

for i in range(len(recovered_sum)-1):

    ind.append(((recovered_sum[i+1]-recovered_sum[i])/(confirmed_sum[i+1]-confirmed_sum[i])))

by_day=pd.DataFrame(ind)

by_day.index=recovered_sum.index

plt.figure(figsize=(20,10))

plt.plot(by_day

        , label = 'recovered/new cases'

        , marker = 'o')

plt.title('daily recovered/new cases',size=30)

plt.ylabel('recovered/new cases',size=20)

plt.xticks(rotation=90,size=20)

plt.grid(True)

plt.show()

# plt.yticks(size=15)