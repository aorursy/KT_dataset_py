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
import seaborn as sns
import os
import re

W1= pd.read_csv('/kaggle/input/vega-shrinkwrapper-runtofailure-data/WornBlade001.csv')
W2= pd.read_csv('/kaggle/input/vega-shrinkwrapper-runtofailure-data/WornBlade002.csv')
W3= pd.read_csv('/kaggle/input/vega-shrinkwrapper-runtofailure-data/WornBlade003.csv')
N1= pd.read_csv('/kaggle/input/vega-shrinkwrapper-runtofailure-data/NewBlade001.csv')
N2= pd.read_csv('/kaggle/input/vega-shrinkwrapper-runtofailure-data/NewBlade002.csv')
N3= pd.read_csv('/kaggle/input/vega-shrinkwrapper-runtofailure-data/NewBlade003.csv')

data = {"W1": W1, "W2": W2, "W3": W3, "N1": N1, "N2": N2, "N3": N3}
   
def aggregateDescribe(df):
    res_data = df.describe().transpose()
    res_data.index.name = "attributes"
    return res_data

def makeStackedDescribes(data_dict):
    res_describes = {}    
    for name,data in data_dict.items():
        res_data = aggregateDescribe(data)
        res_data["data"] = name        
        res_describes[name] = res_data

    describes = pd.concat(res_describes.values())
    describes = describes.set_index("data", append = True)
 
    return describes
        
des = makeStackedDescribes(data)
des.sort_values(["attributes", "data"]).round(2).to_csv("/kaggle/working/bladedata_describe.csv")
des.sort_values(["attributes", "data"]).round(2)
# DataStacking
def stackData(data_dict):
    """
    stack data and add data name value  
    """
    stacked = {}
    i=1
    for name, data in data_dict.items():
        data["data"] = name # append data name value
        stacked[name] = data # add result dictionary

    stacked_data = pd.concat(stacked.values())
    
    stacked_data = stacked_data.set_index(["data","Timestamp"], append = True)
    stacked_data = stacked_data.stack().reset_index()
    stacked_data.columns = ["id","data","timestamp","attributes","value"]
    
    return stacked_data

stacked_data = stackData(data)

# Rename attributes(too long!!)
old_attr = stacked_data["attributes"].unique()
# print(old_attributes)

import re
replace = {'attributes':{}}
for attr in old_attr:
#     new_attr = 
    new_attr = re.sub(r'^ ','',attr)
    new_attr = re.sub(r'Lag error', 'LagError',new_attr)
    new_attr = re.sub(r' .* ','-',new_attr)
    new_attr = re.sub(r'Svol','',new_attr)
    
    replace['attributes'][attr] = new_attr

stacked_data = stacked_data.replace(replace)



def viewHist(df,attribute = "attribute"):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set()

    # extract by attributtes
    extract = {}
    for name in df["data"].unique():
        extract[name] = df.value[(df['attributes'] == attribute) & (df['data'] == name)]
        
    fig = plt.figure(figsize=(15,5))   
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    for name,data in extract.items():
        if "W" in name:
            ax1.hist(data, bins=50, alpha=0.4,label=name)
        else:
            ax2.hist(data, bins=50, alpha=0.4,label=name)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    xlim = [df.value[(df['attributes'] == attribute)].min(), df.value[(df['attributes'] == attribute)].max()]
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax1.set_title(attribute)
    
    plt.show()
# check histgram
attr_list = stacked_data["attributes"].unique()
attr_list.sort()
print(attr_list)
viewHist(stacked_data,attribute = "pCut-LagError")
viewHist(stacked_data,attribute = "pCut-Torque")
viewHist(stacked_data,attribute = "pCut-position")
viewHist(stacked_data,attribute = "pCut-speed")
viewHist(stacked_data, attribute = "pFilm-LagError")
viewHist(stacked_data, attribute = "pFilm-position")
viewHist(stacked_data, attribute = "pFilm-speed")
attr_list = stacked_data["attributes"].unique()
attr_list.sort()
attr_list
attr_order = {"1":'pCut-position',
              "2":'pCut-speed',
              "3":'pCut-LagError',
              "4":'pCut-Torque',
              "5":'pFilm-position',
              "6":'pFilm-speed',
              "7":'pFilm-LagError'
             }

attrs =list(attr_order.values())
def compAtributesAllrange(df,data = ""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set()
    
    # extarct data
    name = data
    tmp_df = df[(df['data'] == name)]
    
    ## fix graph order 
    attr_order = {"1":'pCut-position',
                  "2":'pCut-speed',
                  "3":'pCut-LagError',
                  "4":'pCut-Torque',
                  "5":'pFilm-position',
                  "6":'pFilm-speed',
                  "7":'pFilm-LagError'
                 }
    
    fig = plt.figure(figsize=(15,2*len(attr_order)))

    
    ax_list = list(map(lambda i: fig.add_subplot(len(attr_order),1,i),range(1,len(attr_order))))
    attrs = list(attr_order.values())
                  
    for ax, attr in zip(ax_list, attr_order.values()):
        attr_df = tmp_df[(tmp_df["attributes"] == attr)]
        ax.plot(attr_df.timestamp, attr_df.value)
        ax.set_ylabel(attr)
    
    ax_list[-1].set_xlabel("timestamp")
    ax_list[0].set_title(f'Data Name: {name}')

    plt.show()
        
compAtributesAllrange(stacked_data,data = "N1")
compAtributesAllrange(stacked_data,data = "N2")
compAtributesAllrange(stacked_data,data = "N3")
compAtributesAllrange(stacked_data,data = "W1")
compAtributesAllrange(stacked_data,data = "W2")
compAtributesAllrange(stacked_data,data = "W3")

def viewTimeintervalHist(df):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set()

    # extract by attributtes
    extract = {}
    names = df["data"].unique()
    for name in names:
        extract[name] = df["timestamp"][df['data'] == name].unique()
        extract[name] = np.sort(extract[name])
        extract[name] = np.diff(extract[name],n=1)

    fig = plt.figure(figsize=(15,2*len(extract)))

    ax_list = list(map(lambda i: fig.add_subplot(len(extract),1,i),range(1,len(extract))))

    xlim = [0,0]
    for data in extract.values():
        tmp_min = data.min()
        tmp_max = data.max()
        if xlim[0] > tmp_min:
            xlim[0] = tmp_min
        if xlim[1] < tmp_max:
            xlim[1] = tmp_max
    print(xlim)
    
    for ax, data in zip(ax_list, extract.items()):
        ax.hist(data[1], bins=100, alpha=0.6)
        print(np.sort(data[1]))
        ax.set_ylabel(f"{data[0]}")
#         ax.set_xlim(xlim)
    
    ax_list[-1].set_xlabel("timestamp intervals")
    ax_list[0].set_title('timestamp intervals')
    
    plt.show()
viewTimeintervalHist(stacked_data)
def compAttributesDiff(df,data = ""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set()
    
    # extarct data
    name = data
    tmp_df = df[(df['data'] == name)]
    
    ## fix graph order 
    attr_order = {
                "1":'pCut-position',
                "2":'pCut-speed',
                "3":'pCut-LagError',
                "4":'pCut-Torque',
                "5":'pFilm-position',
                "6":'pFilm-speed',
                "7":'pFilm-LagError'
                 }
    
    fig = plt.figure(figsize=(15,2*len(attr_order)+10))

    
    ax_list = list(map(lambda i: fig.add_subplot(len(attr_order)+1,1,i),range(1,len(attr_order)+1)))
    ax2_list = list(map(lambda ax: ax.twinx(),ax_list))
    attrs = list(attr_order.values())
    
    for ax, ax2, attr in zip(ax_list,ax2_list,attr_order.values()):
  
        attr_df = tmp_df[(tmp_df["attributes"] == attr)]

        diff_df = pd.DataFrame({"diff": np.diff(attr_df.value) , "timestamp": attr_df.timestamp[1:]})

        # fix styles 
        ax2.grid(False)
        ax.set_axisbelow(True)
  
        # plotting
        ax.plot(attr_df.timestamp, attr_df.value,color="blue",label=attr,alpha=0.5)
        ax2.plot(diff_df.timestamp, diff_df["diff"],color="red", label=f"diff_{attr}",alpha=0.5)
        
        # set labels
        ax.set_ylabel(attr)
        ax2.set_ylabel(f"diff_{attr}")
        
        # set legends
        handler, label = ax.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        ax.legend(handler + handler2, label + label2)
    
    ax_list[-1].set_xlabel("timestamp")
    ax_list[0].set_title(f'Data Name: {name}')

    plt.show()

        
    
        
compAttributesDiff(stacked_data, data = "N1")
compAttributesDiff(stacked_data, data = "N2")
compAttributesDiff(stacked_data, data = "N3")
compAttributesDiff(stacked_data, data = "W1")
compAttributesDiff(stacked_data, data = "W2")
compAttributesDiff(stacked_data, data = "W3")
stacked_data[stacked_data["attributes"] == "pFilm-LagError"]
Ro
def compAttributesRolling(df,data = "",window = []):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set()
    
    # extarct data
    name = data
    tmp_df = df[(df['data'] == name)]
    
    ## fix graph order 
    attr_order = {
                "1":'pCut-position',
                "2":'pCut-speed',
                "3":'pCut-LagError',
                "4":'pCut-Torque',
                "5":'pFilm-position',
                "6":'pFilm-speed',
                "7":'pFilm-LagError'
                 }
    
    fig = plt.figure(figsize=(15,2*len(attr_order)+10))

    
    ax_list = list(map(lambda i: fig.add_subplot(len(attr_order)+1,1,i),range(1,len(attr_order)+1)))
    ax2_list = list(map(lambda ax: ax.twinx(),ax_list))
    attrs = list(attr_order.values())
    
    for ax, ax2, attr in zip(ax_list,ax2_list,attr_order.values()):
  
        attr_df = tmp_df[(tmp_df["attributes"] == attr)]

        diff_df = pd.DataFrame({"diff": np.diff(attr_df.value) , "timestamp": attr_df.timestamp[1:]})

        # fix styles 
        ax2.grid(False)
        ax.set_axisbelow(True)
  
        # plotting
        ax.plot(attr_df.timestamp, attr_df.value,color="blue",label=attr,alpha=0.5)
        ax2.plot(diff_df.timestamp, diff_df["diff"],color="red", label=f"diff_{attr}",alpha=0.5)
        
        # set labels
        ax.set_ylabel(attr)
        ax2.set_ylabel(f"diff_{attr}")
        
        # set legends
        handler, label = ax.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        ax.legend(handler + handler2, label + label2)
    
    ax_list[-1].set_xlabel("timestamp")
    ax_list[0].set_title(f'Data Name: {name}')

    plt.show()

        
    
        