# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from datetime import datetime
my_year=2020
my_month=1
my_day=2
my_hour=13
my_min=30
my_sec=15

my_date=datetime(my_year,my_month,my_day)
my_date
my_date_time=datetime(my_year,my_month,my_day,my_hour,my_min,my_sec)
my_date_time
my_date_time.hour
type(my_date_time)
import numpy as np
np.array(["2020-03-05","2020-08-05","2020-03-06"],dtype="datetime64[D]")
np.array(["2020-03-05","2020-08-05","2020-03-06"],dtype="datetime64[Y]")
np.array(["2020-03-05","2020-08-05","2020-03-06"],dtype="datetime64[m]")
np.arange("2020-06-01","2020-06-29",7,dtype="datetime64[D]")
np.arange("1978","1990",dtype="datetime64[Y]")
import pandas as pd
pd.date_range("2020-01-01",periods=7,freq="Y") #D-Day,Y-Year,M-Month,m_minute
pd.date_range("Jan 01, 2018",periods=7,freq="Y") #D-Day,Y-Year,M-Month,m_minute
pd.to_datetime(["1/2/2019","Jan 03, 2018"])
pd.to_datetime(["1/2/2019","3/2019"])
pd.to_datetime(["1/2/2019","2/3/2019"],format="%d/%m/%Y")
pd.to_datetime(["1++2++2019","2++3++2019"],format="%d++%m++%Y")
data=np.random.randn(3,2)
cols=["A","B"]
print(data)
idx=pd.date_range("2020-01-01",periods=3,freq="D")
df=pd.DataFrame(data,index=idx,columns=cols)
df.plot()
df.index
df.index.max()
df.index.argmax()
df.index.min()
import pandas as pd
df=pd.read_csv("../input/starbucks-data/starbucks.csv",index_col="Date",parse_dates=True)
df.head(
)
df.index
df.plot()
#Rule-A stands for Year-End data

#daily-->Yearly
df.resample(rule="A").mean().plot()
