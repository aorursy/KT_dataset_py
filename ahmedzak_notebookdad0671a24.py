# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

dftechnique = pd.read_csv('../input/Taekwondo_Technique_Classification_Stats.csv', index_col=0)

dftb1 = pd.read_csv('../input/Table1.csv')

dftb1.info()

dftechnique.info()

dftb1.describe()

dftechnique.describe()



# Any results you write to the current directory are saved as output.
df.describe()
df.head(5)
df.info()