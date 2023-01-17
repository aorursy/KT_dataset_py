# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

filePaths = []

pathCount = 0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        print(str(pathCount), ".", path)

        pathCount += 1

        filePaths.append(path)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings('ignore')

def viewInfo(table, name="Table"):

    col = table.index.tolist()

    row = table.columns.tolist()

    colLenght = len(col)

    rowLenght = len(row)

    print('[------{0}------]\n[*] Row : {1} | Col {2}'.format(name, rowLenght, colLenght))

    print(row, '\n')
# Read phanphoi data
import pandas as pd



factories = pd.read_csv(filePaths[8], index_col=0)

viewInfo(factories, "Factories")

factories
warehouses = pd.read_csv(filePaths[7])

viewInfo(warehouses, "Ware Houses")

warehouses
cost = pd.read_csv(filePaths[3], index_col=0)

viewInfo(cost, "Cost")

cost
trans = pd.read_csv(filePaths[9], index_col=0)

viewInfo(trans, "Trans")            

trans.head()
join_data_1 = pd.merge(trans, cost, left_on=["ToFC", "FromWH"], right_on=["FCID", "WHID"], how="left")

viewInfo(trans, "Trans")

viewInfo(cost, "Cost")

viewInfo(join_data, "Join Data")

join_data_1.head()
join_data_2 = pd.merge(join_data_1, factories, left_on="ToFC", right_on="FCID", how="left")

viewInfo(join_data_1, "Join 1")

viewInfo(factories, "Factories")

viewInfo(join_data_2, "Join 2")

join_data_2.head()
join_data = pd.merge(join_data_2, warehouses, left_on="FromWH", right_on="WHID", how="left")

join_data = join_data[["TransactionDate","Quantity","Cost","ToFC","FCName","FCDemand","FromWH","WHName","WHSupply","WHRegion"]]

viewInfo(join_data)

join_data.head()
kanto = join_data.loc[join_data["WHRegion"]=="関東"]

viewInfo(kanto, "Kanto Region")

kanto.head()
tohoku = join_data.loc[join_data["WHRegion"]=="東北"]

tohoku.head()
print("---総コスト---")

print("関東支社の総コスト: " + str(kanto["Cost"].sum()) + "万円")

print("東北支社の総コスト: " + str(tohoku["Cost"].sum()) + "万円")

print("---総部品輸送個数---")

print("関東支社の総部品輸送個数: " + str(kanto["Quantity"].sum()) + "個")

print("東北支社の総部品輸送個数: " + str(tohoku["Quantity"].sum()) + "個")

print("---部品一つ当たりの輸送コスト---")

tmp = (kanto["Cost"].sum() / kanto["Quantity"].sum()) * 10000

print("関東支社の部品１つ当たりの輸送コスト: " + str(int(tmp)) + "円")

tmp = (tohoku["Cost"].sum() / tohoku["Quantity"].sum()) * 10000

print("東北支社の部品１つ当たりの輸送コスト: " + str(int(tmp)) + "円")
print(cost.shape)

print(cost.loc[cost["FCID"]=="FC00001"])

print(factories)

cost_chk = pd.merge(cost, factories, on="FCID", how="left")

print("東京支社の平均輸送コスト：" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"]=="関東"].mean()) + "万円")

print("東北支社の平均輸送コスト：" + str(cost_chk["Cost"].loc[cost_chk["FCRegion"]=="東北"].mean()) + "万円")
import networkx as nx

import matplotlib.pyplot as plt



# グラフオブジェクトの作成

G=nx.Graph()



# 頂点の設定

G.add_node("nodeA")

G.add_node("nodeB")

G.add_node("nodeC")



# 辺の設定

G.add_edge("nodeA","nodeB")

G.add_edge("nodeA","nodeC")

G.add_edge("nodeB","nodeC")



# 座標の設定

pos={}

pos["nodeA"]=(0,0)

pos["nodeB"]=(1,1)

pos["nodeC"]=(0,1)



# 描画

nx.draw(G,pos)



# 表示

plt.show()
import networkx as nx

import matplotlib.pyplot as plt



# グラフオブジェクトの作成．

G=nx.Graph()



# 頂点の設定

G.add_node("nodeA")

G.add_node("nodeB")

G.add_node("nodeC")

G.add_node("nodeD")



# 辺の設定

G.add_edge("nodeA","nodeB")

G.add_edge("nodeA","nodeC")

G.add_edge("nodeB","nodeC")

G.add_edge("nodeA","nodeD")



# 座標の設定

pos={}

pos["nodeA"]=(0,0)

pos["nodeB"]=(1,1)

pos["nodeC"]=(0,1)

pos["nodeD"]=(1,0)



# 描画

nx.draw(G,pos, with_labels=True)



# 表示

plt.show()
df_w = pd.read_csv(filePaths[10])

df_p = pd.read_csv(filePaths[0])

viewInfo(df_w, "Network Weight")

viewInfo(df_p, "Network Position")

print("---Position---")

print(df_p)

print("---Weight---")

print(df_w)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import networkx as nx



# エッジの重みのリスト化

size = 10

edge_weights = []

for i in range(len(df_w)):

    for j in range(len(df_w.columns)):

        edge_weights.append(df_w.iloc[i][j]*size)

print("[+] Edge :", len(edge_weights))



# グラフオブジェクトの作成

G = nx.Graph()



# 頂点の設定

for i in range(len(df_w.columns)):

    G.add_node(df_w.columns[i])



# 辺の設定

for i in range(len(df_w.columns)):

    for j in range(len(df_w.columns)):

        G.add_edge(df_w.columns[i],df_w.columns[j])



# 座標の設定

pos = {}

for i in range(len(df_w.columns)):

    node = df_w.columns[i]

    pos[node] = (df_p[node][0],df_p[node][1])



# 描画

nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)



# 表示

plt.show()
import pandas as pd



# データ読み込み

df_tr = pd.read_csv(filePaths[11], index_col="工場")

df_pos = pd.read_csv(filePaths[6])

print(df_tr)

print(df_pos)
import pandas as pd

import matplotlib.pyplot as plt

import networkx as nx





# グラフオブジェクトの作成

G = nx.Graph()



# 頂点の設定

for i in range(len(df_pos.columns)):

    G.add_node(df_pos.columns[i])



# 辺の設定&エッジの重みのリスト化

num_pre = 0

edge_weights = []

size = 0.1

for i in range(len(df_pos.columns)):

    for j in range(len(df_pos.columns)):

        if not (i==j):

            # 辺の追加

            G.add_edge(df_pos.columns[i],df_pos.columns[j])

            # エッジの重みの追加

            if num_pre<len(G.edges):

                num_pre = len(G.edges)

                weight = 0

                if (df_pos.columns[i] in df_tr.columns)and(df_pos.columns[j] in df_tr.index):

                    if df_tr[df_pos.columns[i]][df_pos.columns[j]]:

                        weight = df_tr[df_pos.columns[i]][df_pos.columns[j]]*size

                elif(df_pos.columns[j] in df_tr.columns)and(df_pos.columns[i] in df_tr.index):

                    if df_tr[df_pos.columns[j]][df_pos.columns[i]]:

                        weight = df_tr[df_pos.columns[j]][df_pos.columns[i]]*size

                edge_weights.append(weight)

                



# 座標の設定

pos = {}

for i in range(len(df_pos.columns)):

    node = df_pos.columns[i]

    pos[node] = (df_pos[node][0],df_pos[node][1])

    

# 描画

nx.draw(G, pos, with_labels=True,font_size=16, node_size = 1000, node_color='k', font_color='w', width=edge_weights)



# 表示

plt.show()
df_tr = pd.read_csv(filePaths[11], index_col="工場")

df_tc = pd.read_csv(filePaths[5], index_col="工場")

print("---Route---")

print(df_tr)

print("\n---Cost---")

print(df_tc)
# 輸送コスト関数

def trans_cost(df_tr,df_tc):

    cost = 0

    for i in range(len(df_tc.index)):

        for j in range(len(df_tr.columns)):

            cost += df_tr.iloc[i][j]*df_tc.iloc[i][j]

    return cost



print("総輸送コスト:"+str(trans_cost(df_tr,df_tc)))
df_demand = pd.read_csv(filePaths[2])

df_supply = pd.read_csv(filePaths[1])

print("---Route---")

print(df_tr)

print("---Demand---")

print(df_demand)

print("\n---Supply---")

print(df_supply)
import pandas as pd



# データ読み込み

print("---- Fからの輸送量と需要量の比較 ---")

# 需要側の制約条件

for i in range(len(df_demand.columns)):

    temp_sum = sum(df_tr[df_demand.columns[i]])

    print(str(df_demand.columns[i])+"への輸送量:"+str(temp_sum)+" (需要量:"+str(df_demand.iloc[0][i])+")")

    if temp_sum>=df_demand.iloc[0][i]:

        print("需要量を満たしています。")

    else:

        print("需要量を満たしていません。輸送ルートを再計算して下さい。")



print("\n---- Ｗからの輸送量と供給限界の比較 ---")

# 供給側の制約条件

for i in range(len(df_supply.columns)):

    temp_sum = sum(df_tr.loc[df_supply.columns[i]])

    print(str(df_supply.columns[i])+"からの輸送量:"+str(temp_sum)+" (供給限界:"+str(df_supply.iloc[0][i])+")")

    if temp_sum<=df_supply.iloc[0][i]:

        print("供給限界の範囲内です。")

    else:

        print("供給限界を超過しています。輸送ルートを再計算して下さい。")
# データ読み込み

df_tr_new = pd.read_csv(filePaths[4], index_col="工場")

print("---Trans Route New---")

print(df_tr_new)

print("\n---Demand---")

print(df_demand)

print("\n---Supply---")

print(df_supply)
import pandas as pd

import numpy as np



# 総輸送コスト再計算 

print("総輸送コスト(変更後):"+str(trans_cost(df_tr_new,df_tc)))



# 制約条件計算関数

# 需要側

def condition_demand(df_tr,df_demand):

    flag = np.zeros(len(df_demand.columns))

    for i in range(len(df_demand.columns)):

        temp_sum = sum(df_tr[df_demand.columns[i]])

        if (temp_sum>=df_demand.iloc[0][i]):

            flag[i] = 1

    return flag

            

# 供給側

def condition_supply(df_tr,df_supply):

    flag = np.zeros(len(df_supply.columns))

    for i in range(len(df_supply.columns)):

        temp_sum = sum(df_tr.loc[df_supply.columns[i]])

        if temp_sum<=df_supply.iloc[0][i]:

            flag[i] = 1

    return flag



print("需要条件計算結果:"+str(condition_demand(df_tr_new,df_demand)))

print("供給条件計算結果:"+str(condition_supply(df_tr_new,df_supply)))