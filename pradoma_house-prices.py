# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
%config InlineBeckend.format.figure = 'retina'
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from scipy.stats import chi2_contingency
from scipy import stats
# Permite multiplos outputs de uma mesma celula
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# price_train = pd.read_csv("train.csv")
# train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.isnull().sum().sort_values(ascending = False).head(20)
test.isnull().sum().sort_values(ascending = False).head(20)
