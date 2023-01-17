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
df = pd.read_csv('../input/complete.csv')
df.head()
df.columns.tolist()
Ronaldo=df[df['name'] == 'Cristiano Ronaldo' ]
Ronaldo.columns[(Ronaldo == True).iloc[0]]
Ronaldo['height_cm']
Ronaldo.loc[:, Ronaldo.dtypes != bool ]
Ronaldo.loc[:, Ronaldo.dtypes == np.float64]
Ronaldo.loc[:, Ronaldo.dtypes == np.float64]
d =Ronaldo.loc[:, Ronaldo.dtypes == np.float64]
d.iloc[:,5:-1]
Ronaldo[['height_cm','weight_kg']]
bmi = Ronaldo[['height_cm','weight_kg']]
BMI = bmi['weight_kg']/ (bmi['height_cm']*0.01)**2
BMI
{'less than 18.5':'underweight',
'18.5 - 24.9':'normal weight',
'25 - 29.9':'overweight',
'30 - 34.9':'class I obese',
'35 - 39.9':'class II obese',
'40 upwards':'class III obese'}
