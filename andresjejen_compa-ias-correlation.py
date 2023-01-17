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
data  = pd.read_csv('/kaggle/input/investdatatest/INVEST.csv',delimiter=";")
data
data.info()
data["DESEMPLEO"] = [float(x.replace(',','.')) for x in data["DESEMPLEO"]]

data["INFLACION"] = [float(x.replace(',','.')) for x in data["INFLACION"]]

data["CONFIANZA_USA"] = [float(x.replace(',','.')) for x in data["CONFIANZA_USA"]]

data["CONFIANZA_OCDE"] = [float(x.replace(',','.')) for x in data["CONFIANZA_OCDE"]]

data["PRECIO_CIERRE"] = [float(x.replace(',','.')) for x in data["PRECIO_CIERRE"]]

data[["DESEMPLEO","INFLACION","CONFIANZA_USA","CONFIANZA_OCDE","PRECIO_CIERRE"]] = data[["DESEMPLEO","INFLACION","CONFIANZA_USA","CONFIANZA_OCDE","PRECIO_CIERRE"]].apply(pd.to_numeric)
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import pearsonr



def corrfunc(x,y, ax=None, **kws):

    """Plot the correlation coefficient in the top left hand corner of a plot."""

    r, _ = pearsonr(x, y)

    ax = ax or plt.gca()

    # Unicode for lowercase rho (ρ)

    rho = '\u03C1'

    ax.annotate(f'{rho} = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)
g = sns.pairplot(data,vars=["DESEMPLEO","INFLACION","CONFIANZA_USA","CONFIANZA_OCDE","PRECIO_CIERRE"])

g.map_lower(corrfunc)

plt.show()
g = sns.pairplot(data,vars=["DESEMPLEO","INFLACION","CONFIANZA_USA","CONFIANZA_OCDE","PRECIO_CIERRE"],diag_kind="kde")

g.map_lower(corrfunc)

plt.show()