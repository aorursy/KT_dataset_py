# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pyplot import pie
data = pd.read_csv("../input/xAPI-Edu-Data.csv")
countOfEachNationality = data.NationalITy.groupby(data.NationalITy).count()

Nationalities = countOfEachNationality.keys()
pie(countOfEachNationality,labels=Nationalities)
