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
df=pd.read_csv('../input/juvinil_background.csv',index_col='Year')
df.describe()
df.head(10)
import matplotlib.pyplot as plt

%matplotlib inline
fig, ((ax1, ax2) ,(ax3,ax4)) = plt.subplots(nrows=2, ncols=2,figsize=(10,5))

ax1=df['Homeless'].plot(title="Count of Homeless Juveniles arrested",ax=ax1)

ax2=df['Illiterate'].plot(title="Count of Illiterate Juveniles arrested",ax=ax2)

ax3=df['Matric/Higher Sec. and above'].plot(title="Count of Matric/Higher Sec. and above Juveniles arrested",ax=ax3)

ax4=df['Above Primary but below Matric/Hr.Sec'].plot(title="Count of Above Primary but below Matric/Hr.Sec Juveniles arrested",ax=ax4)

plt.tight_layout()

plt.savefig("./viz2.png")