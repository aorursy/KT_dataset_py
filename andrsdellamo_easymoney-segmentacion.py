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
import numpy as np 
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import gc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
clientes_actuales=df_sorted[(df_sorted['pk_partition']=='2019-05-28') &
          (df_sorted['isActive']==1)]
variable_segmentacion=['totalAssets','salary','age']
pipe = Pipeline(
        steps=[
            ('StandardScaler', StandardScaler()),
            ('KMeans', KMeans(n_clusters=4))
        ]
)
pipe.fit(clientes_actuales[variable_segmentacion])
clientes_actuales['Cluster'] = pipe.predict(clientes_actuales[variable_segmentacion])
clientes_actuales.groupby('Cluster').agg({
                                        'totalAssets':np.mean,
                                         'salary':np.mean,
                                         'age':[np.mean,np.max,np.min]
                                        })
# Para evoitar error Selected KDE bandwidth is 0. Cannot estimate density.
sns.distributions._has_statsmodels = False
sns.pairplot(clientes_actuales.sample(10000), vars=variable_segmentacion, hue='Cluster', aspect=1.5)
plt.show()
