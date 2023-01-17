# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
columns = ['Size', 'Grid', 'Block', 'Time(s)', 'Error']

gridblock_result = pd.read_csv('../input/gpu-programming-matrix-multiplication-results/result_varGridBlock.txt', header=None, names=columns, delim_whitespace=True)

df = gridblock_result

display(df)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



g =sns.lineplot(x="Size", y="Time(s)",

              hue="Block",

              marker="o",

              data=df);
columns = ['Size', 'Sequential', 'Multi-core', 'CUDA non-shared', 'CUDA shared', 'cuBLAS']

results = pd.read_csv('../input/gpu-programming-matrix-multiplication-results/compute_times.txt', header=None, names=columns, delim_whitespace=True)

df2 = results

display(df2)
ax = df2.plot.line(x="Size", marker="o");

ax.set_ylabel("Time");