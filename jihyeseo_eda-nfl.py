# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_json('../input/games_1512362753.8735218.json')
dg = pd.read_json('../input/profiles_1512362725.022629.json')
df.head()
dg.head()
df.dtypes
df.player_team_score.hist()
dg.current_salary.hist()
dg.current_salary.unique()
dg.current_salary.dtype
dg.dtypes
dg.weight.hist()
