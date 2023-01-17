# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import statistics as s
a="../input/kiva_loans.csv"
df=pd.read_csv(a)
#print(df)
#df.describe()
#df.boxplot('funded_amount','loan_amount')
#df.plot.bar('funded_amount','loan_amount')
#df.plot.scatter('funded_amount','loan_amount')
b=df["loan_amount"]
print("median :- ",s.median(b))
print("mean :- ",s.mean(b))
print("mode :- ",s.mode(b))


