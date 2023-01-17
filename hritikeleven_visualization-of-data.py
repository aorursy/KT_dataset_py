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
import pandas as pd
df=pd.read_csv("/kaggle/input/co2-emission/co2_emission.csv")
df
import pandas as pd
df=pd.read_csv("/kaggle/input/co2-emission/co2_emission.csv")
df
print(df.columns)
df.columns = ['Entity', 'Code', 'Year','Emissions']
print(df.columns)
import pandas as pd
df=pd.read_csv("/kaggle/input/co2-emission/co2_emission.csv")
df
hist=df.hist(bins=3)

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("/kaggle/input/co2-emission/co2_emission.csv")
df.columns = ['Entity', 'Code', 'Year','Emissions']
plt.plot(df['Year'], df['Emissions'])
plt.title('Annual CO₂ emissions (tonnes ) Vs Year')
plt.xlabel('Year')
plt.ylabel('Annual CO₂ emissions (tonnes )')
plt.show()
from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np 
df=pd.read_csv("/kaggle/input/co2-emission/co2_emission.csv")
df.columns = ['Entity', 'Code', 'Year','Emissions']  

plt.pie(df['Emissions'], labels = df['Entity']) 

plt.show()
import pandas as pd
from matplotlib import pyplot as plt 
df=pd.read_csv("/kaggle/input/co2-emission/co2_emission.csv")

df.columns = ['Entity', 'Code', 'Year','Emissions']  

plt.bar(df['Year'],df['Emissions'])
plt.show()

