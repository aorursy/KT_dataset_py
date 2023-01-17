# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
state = "CA"

CAdf = pd.read_csv("../input/ca-wx-df/CAdf.csv")

df_dict = {"CA":CAdf}

state_df = df_dict[state]



state_df.head()
# dept_id毎に販売量平均値の上位10商品を把握



dept_id_list = ["HOBBIES_1", "HOBBIES_2", "FOODS_1", "FOODS_2", "FOODS_3", "HOUSEHOLD_1", "HOUSEHOLD_2"]

fig = plt.figure(figsize=(17,8))



def RANKlist_make(DF, fig, i):

    df = DF.loc[:, DF.columns.str.contains(dept_id_list[i])].mean().sort_values(ascending=False).head(10)

    return df



RANKlist = []

for i in range(len(dept_id_list)):

    RANKlist.append(list(RANKlist_make(state_df, fig, i).index))

    

RANKlist
# 日積算降水量10mmを閾値にしてみる

p = 10



for j in range(7):

    TITLE = [i.replace("_validation", "") for i in RANKlist[j]]

    PRCP0 = state_df[state_df["PRCP"]<p][RANKlist[j]].mean()

    PRCP1 = state_df[state_df["PRCP"]>=p][RANKlist[j]].mean()



    fig = plt.figure(figsize=(18,2))

    ax = fig.add_subplot(111)



    ax.bar(np.arange(len(TITLE)), PRCP0, width=0.3, color="dodgerblue", alpha=0.4)

    ax.bar(np.arange(len(TITLE))+0.3, PRCP1, width=0.3, color="dodgerblue")



    ax.set_xticks(np.arange(len(TITLE)))

    ax.set_xticklabels(TITLE)



    plt.show()   
# 最高気温の平均値

print(state_df["MAX"].mean())



# 最高気温のヒストグラム

state_df["MAX"].hist()
# 最高気温平均値=22℃を閾値にしてみる

t = 22



for j in range(7):

    TITLE = [i.replace("_validation", "") for i in RANKlist[j]]

    Tmmn = state_df[state_df["MAX"]<t][RANKlist[j]].mean()

    Tijo = state_df[state_df["MAX"]>=t][RANKlist[j]].mean()



    fig = plt.figure(figsize=(18,2))

    ax = fig.add_subplot(111)



    ax.bar(np.arange(len(TITLE)), Tmmn, width=0.3, color="salmon", alpha=0.4)

    ax.bar(np.arange(len(TITLE))+0.3, Tijo, width=0.3, color="salmon")



    ax.set_xticks(np.arange(len(TITLE)))

    ax.set_xticklabels(TITLE)



    plt.show()   