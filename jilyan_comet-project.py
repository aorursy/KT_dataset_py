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
data_set = pd.read_csv("../input/meteorite-landings.csv")

data_set
unique, counts = np.unique(data_set.year, return_counts=True)

dict(zip(unique, counts))
np.mean(counts)/365
unique, counts = np.unique(data_set.fall, return_counts=True)

dict(zip(unique, counts))
fell = counts[0]/(counts[0] + counts[1])

fell
latitude = np.std(data_set.reclat)

latitude
longitude = np.std(data_set.reclong)

longitude 