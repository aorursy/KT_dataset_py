# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
print(df.head())
print(df.describe())
sns.pairplot(df, diag_kind="kde",hue="Survived")
sns.pairplot(df.loc[df.Sex == 'male',], diag_kind="kde",hue="Survived")
sns.pairplot(df.loc[df.Sex == 'female',], diag_kind="kde",hue="Survived")

plt.bar(df.Survived.unique(),[len(df.loc[df.Sex == 'female' & df.Survived == 0,]),len(df.loc[df.Sex == 'female' & df.Survived == 1,])])
