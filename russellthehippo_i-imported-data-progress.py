# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly as py #data visualization, let's make this pretty

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/data"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# i'm interested in the alcohol consumption dataset

import zipfile

Dataset = "alcohol-consumption"



# unzip the files so that you can see them

with zipfile.ZipFile("../input/data/"+Dataset+".zip","r") as z:

    z.extractall(".")
from subprocess import check_output

print(check_output(["ls", "alcohol-consumption"]).decode("utf8"))
d = pd.read_csv(Dataset + "/drinks.csv")

d.head()
d.shape



# 193 countries, about the same as other lists; only 5 columns, thank gods
d.max()