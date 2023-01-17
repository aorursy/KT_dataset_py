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

import matplotlib.pyplot as plt

import seaborn as sns



# 한글 표시를 위해서 해봄 (결국에는 처리가 안됬습니다 -> 캐글에는 한글을 지원해주는 글자형태가 없는것 같아서 결과적으로 이것은 의미 없었네요)

import matplotlib as mpl

import matplotlib.pyplot as plt



plt.rcParams['font.family'] = 'AppleGothic'

plt.rcParams['font.size'] = 12

plt.rcParams['figure.figsize'] = (14, 4)



mpl.rcParams['axes.unicode_minus'] = False
data = pd.read_csv('/kaggle/input/2019-covid19-ncov19-data-set-in-korean/COVID-19_Korean.csv')

data.head()
data.shape
data.tail()
data.describe()
# 결측치 확인

data.isnull().sum()
#몇개의 국가에 확진자가 있는지 확인

country = data['국가/지역'].unique()
country
# 각 국가별로 몇명의 총 확진자가 있는지 결과 확인

confirmed_case = {}

for con in country:

    confirmed_case[con] =  [data[data['국가/지역'] == con ]['확진자'].sum()]

confirmed_case = pd.DataFrame(confirmed_case).T

confirmed_case.reset_index(inplace = True)

confirmed_case.columns = ['country', 'Total Confirmed Count']

confirmed_case.sort_values(by = 'Total Confirmed Count',ascending=False, inplace=True )

confirmed_case.reset_index(drop = True, inplace = True)

confirmed_case
#barplot을 그리기에 너무 큰 값을 가진 중국과 국가에 따로 귀속되지 않은 기타운송수단을 빼고 9가지만 뽑기로 함

confirmed_case = confirmed_case.iloc[1:10, :]
confirmed_case
sns.barplot(data = confirmed_case, x = 'country', y = 'Total Confirmed Count')

plt.show()