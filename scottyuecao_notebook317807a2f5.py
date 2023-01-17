# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



% matplotlib inline

import matplotlib.pyplot as plt





# Any results you write to the current directory are saved as output.
df = pd.DataFrame.from_csv('../input/presidents.csv')

df.keys()
plt.figure()

plt.clf()

plt.hist(df['Average Orders per Year'])

plt.show()