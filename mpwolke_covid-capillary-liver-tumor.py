#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRSVjn-D0Tv9X1R97WskNQrVWcKLhZDRaRd6QuzdpquBbG3L7ib',width=400,height=400)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/corona-virus-capillary-and-liver-tumor-samples/both_clean_liver_capillary_CoV.csv")
df.shape
df.head().style.background_gradient(cmap='PuBuGn')
cat = []

num = []

for col in df.columns:

    if df[col].dtype=='O':

        cat.append(col)

    else:

        num.append(col)  

        

        

num 
plt.style.use('dark_background')

for col in df[num].drop(['LiverTumorSamples GSM2359851_CoV1'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Blue')

    plt.xlabel(col)

    plt.ylabel('LiverTumorSamples GSM2359851_CoV1')

    plt.tight_layout()

    plt.show()
plt.style.use('dark_background')

for col in df[num].drop(['LiverTumorSamples GSM2359853_CoV2'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Red')

    plt.xlabel(col)

    plt.ylabel('')

    plt.tight_layout()

    plt.show()
plt.style.use('dark_background')

for col in df[num].drop(['LiverTumorSamples GSM2359910_CoV3'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Green')

    plt.xlabel(col)

    plt.ylabel('LiverTumorSamples GSM2359910_CoV3')

    plt.tight_layout()

    plt.show()
plt.style.use('dark_background')

for col in df[num].drop(['LiverTumorSamples GSM2359913_CoV4'],axis=1):

    plt.figure(figsize=(8,5))

    plt.plot(df[col].value_counts(),color='Orange')

    plt.xlabel(col)

    plt.ylabel('LiverTumorSamples GSM2359913_CoV4')

    plt.tight_layout()

    plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQfmDSrKHUIv-FPuOvhvUZoyvuXjVjNkwgJtB2jo-azGzaibxiY',width=400,height=400)