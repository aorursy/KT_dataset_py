# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #for visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



metal_bands = pd.read_csv("../input/metal_bands_2017.csv", encoding='latin-1'

                          ,usecols=['band_name','fans','formed','origin','split','style'])

world_population = pd.read_csv("../input/world_population_1960_2015.csv", encoding='latin-1')



metal_bands.describe()

world_population.describe()
metal_bands.info()
sns.distplot(metal_bands["fans"],kde = False).set_title("Fans")
