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
!pip install ctgan
import pandas as pd

data = pd.read_csv("../input/iris/Iris.csv")

labels = pd.read_json("../input/label4ctgan/label.json")
data.head()
data.drop(["Id"], inplace=True, axis=1)

data.describe()
from ctgan import CTGANSynthesizer



ctgan = CTGANSynthesizer()



discrete_columns = ["Species"]

ctgan.fit(data, discrete_columns,  epochs=1000)
samples = ctgan.sample(1000)
type(samples.SepalLengthCm)
samples[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']] = samples[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']].astype(float)
samples.describe()
data.describe()