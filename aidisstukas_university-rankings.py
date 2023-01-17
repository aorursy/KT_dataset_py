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
shanghai = pd.read_csv('../input/shanghaiData.csv')

times = pd.read_csv('../input/timesData.csv')

cwur = pd.read_csv('../input/cwurData.csv')
shanghai['rankings'] = 'shanghai'

shanghai.head(2)
times['rankings'] = 'times'

times.head(2)

cwur['rankings'] = 'cwur'

dynamics_of_world_rank = cwur.set_index('institution')[['world_rank','year']].pivot(columns='year')['world_rank'].sort_values(2015)
import seaborn as sns
data = dynamics_of_world_rank.iloc[300:600]
data.plot()
1
df = pd.concat([shanghai, times, cwur])
dfd('world_rank')