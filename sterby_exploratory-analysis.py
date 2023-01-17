# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Read the data
catches = pd.read_csv("../input/fish_catches.csv")
country_codes = pd.read_csv("../input/country_codes.csv", sep=",")
species = pd.read_csv("../input/species.txt", sep="\t", encoding='cp1252')

# modify a bit to work with the data
catches=catches.dropna(axis=1,how='all')
data = catches.merge(country_codes, left_on="Country", right_on="Code", how="left")
data = data.merge(species, left_on="Species", right_on="3A_CODE", how="left")
data = data.drop("Code",axis=1)
data = data.drop("French_name",axis=1)
data = data.drop("Spanish_name",axis=1)
data = data.drop("3A_CODE",axis=1)
data["Country"] = data[" Description"]
data = data.drop(" Description",axis=1)

print(data.columns)
print(data.head(5))
