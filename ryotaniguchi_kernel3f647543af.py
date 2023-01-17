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
raw = pd.read_csv("/kaggle/input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv",decimal=",")
oldname = ['% Iron Concentrate','% Iron Feed','% Silica Concentrate','% Silica Feed','Amina Flow','Flotation Column 01 Air Flow','Flotation Column 01 Level','Flotation Column 02 Air Flow','Flotation Column 02 Level','Flotation Column 03 Air Flow','Flotation Column 03 Level','Flotation Column 04 Air Flow','Flotation Column 04 Level','Flotation Column 05 Air Flow','Flotation Column 05 Level','Flotation Column 06 Air Flow','Flotation Column 06 Level','Flotation Column 07 Air Flow','Flotation Column 07 Level','Ore Pulp Density','Ore Pulp Flow','Ore Pulp pH','Starch Flow']
newname = ['IronConc','IronFeed','SiConc','SiFeed','AminaFlow','FC1_AirFlow','FC1_Lv','FC2_AirFlow','FC2_Lv','FC3_AirFlow','FC3_Lv','FC4_AirFlow','FC4_Lv','FC5_AirFlow','FC5_Lv','FC6_AirFlow','FC6_Lv','FC7_AirFlow','FC7_Lv','PulpDensity','PulpFlow','PulpPH','StarchFlow']
rename  = dict(zip(oldname,newname))

raw = raw.rename(columns=rename)
raw.head
raw.date.unique()
raw.date
def aggregateDescribe(df):
    res_data = df.describe().transpose()
    res_data.index.name = "attributes"
    return res_data


        
des = aggregateDescribe(raw)
des.sort_values(["attributes"]).round(2).to_csv("/kaggle/working/describe.csv")
des.sort_values(["attributes"]).round(2)
def compAttributes(df,col=[], x=""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set()
    
    # extarct data
#     name = data
#     tmp_df = df[(df['data'] == name)]
    
    ## fix graph order 
    attr_order = y
    
    fig = plt.figure(figsize=(15,2*len(attr_order)))

    
    ax_list = list(map(lambda i: fig.add_subplot(len(attr_order),1,i),range(1,len(attr_order))))
#     attrs = list(attr_order.values())
                  
    for ax, attr in zip(ax_list, attr_order):
#         attr_df = df[(df["attributes"] == attr)]
        ax.plot(df[x], df[attr])
        ax.set_ylabel(attr)
    
    ax_list[-1].set_xlabel(x)
#     ax_list[0].set_title(f'Data Name: {name}')

    plt.show()
import numpy as np
ts_uq = pd.to_datetime(raw["date"].unique(),format="%Y-%m-%d %H:%M:%S")
ts_uq_uxt = ts_uq.view("int64")
ts_uq_diff = np.diff(ts_uq_uxt)
def viewTsPeriod(x="",y=""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set()
    

    fig = plt.figure(figsize=(15,2))
    ax = fig.add_subplot()
    ax.plot(x,y,marker=".",markeredgecolor="blue")
    ax.set_ylabel("interval[sec]")
    ax.set_xlabel("timestamp")
    
    plt.show()
viewTsPeriod(x=ts_uq[1:], y = ts_uq_diff)
date = raw["date"]
count_ts_uq = date.groupby(date).count()
count_ts_uq.shape
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

fig =plt.figure(figsize=(15,2))
ax = fig.add_subplot(1,1,1)
ax.hist(count_ts_uq)
plt.show()
tmp  = count_ts_uq.unique()
count_ts_uq.groupby(count_ts_uq).count()
