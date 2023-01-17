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

import time
import datetime
from datetime import datetime
import calendar

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

sns.set_style('white')


pd.options.display.float_format = '{:,.2f}'.format
df_sorted = pd.read_pickle('/kaggle/input/easymoney/EasyMoney_base.pkl',compression='zip')
lista_mostrar=['pk_cid','pk_partition','isNewClient','isActive','totalAssets',
               'totalCuentas','totalAhorro','totalFinanciacion','totalIngresos',
               'totalBeneficio','hayAlta','diasDesdeUltimaAlta','diasDesdeUltimaAltaInt',
               'hayBaja','diasDesdeUltimaBaja','diasDesdeUltimaBajaInt']
df_sorted[df_sorted['pk_cid']==17457][lista_mostrar]
