import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mp



# This is team 33 ship

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



dataframe = pd.read_csv("../input/train.csv")

dataframe.head(10)



# Any results you write to the current directory are saved as output.
gender = dataframe['Sex'].map({'female':1,'male':0})

aliveF = gender.dot(dataframe['Survived'])

percentFsurviv=aliveF/gender.sum()

percentFsurviv*100
gender = dataframe['Sex'].map({'female':0,'male':1})

aliveM = gender.dot(dataframe['Survived'])

percentMsurviv=aliveF/gender.sum()

percentMsurviv*100