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

data = pd.read_csv("../input/metal_bands_2017.csv",encoding ='latin1')

data.describe()



data2 = pd.read_csv("../input/world_population_1960_2015.csv",encoding='latin1')

data2.describe()
import seaborn as sns
fans = data['fans'].astype(str).astype(int)

sns.distplot(fans,kde=False).set_title('Histogram of Fan numbers')

print('Mean number of fans: %s '% fans.mean())

print('Standard deviation: %s '% fans.std())