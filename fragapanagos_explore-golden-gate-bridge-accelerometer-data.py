# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
pdf = pd.read_csv("../input/Golden Gate Bridge Accelerometer Data.csv")
# Any results you write to the current directory are saved as output.
fig, ax = plt.subplots(figsize=(16, 12))
pdf.plot(x="time", y="ax", ax=ax)
pdf.plot(x="time", y="ay", ax=ax)
pdf.plot(x="time", y="az", ax=ax)
pdf.plot(x="time", y="aT", ax=ax)

