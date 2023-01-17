import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
import matplotlib.pyplot as plt
import os
print(os.listdir("../input/"))
# Any results you write to the current directory are saved as output.
# load and deal with data 
def read_Data():
    data = pd.read_csv("../input/dataset_group.csv",header=None)
    return data

def aprior_data(data):
    basket = []
    for id in data[1].unique():
        a = [data[2][i] for i, j in enumerate(data[1]) if j == id]
        basket.append(a)
    return basket

data = read_Data()
basket = aprior_data(data)
# Create Aprior Association Model 
def create_C1(basket):
    C1 = set()
    for t in basket:
        for item in t:
            item_set = frozenset([item])
            C1.add(item_set)
    return C1


def is_apriori(Ck_item, Lksub1):
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True


def create_Ck(Lksub1, k):
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                # pruning
                if is_apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck


def generate_Lk_by_Ck(data_set, Ck, min_support, support_data):
    Lk = set()
    item_count = {}
    for t in data_set:
        for item in Ck:
            if item.issubset(t):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data_set))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk


def generate_L(basket, k, min_support):
    #k=3, min_support=0.2
    support_data = {}
    C1 = create_C1(basket)
    L1 = generate_Lk_by_Ck(basket, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = []
    L.append(Lksub1)
    for i in range(2, k+1):
        Ci = create_Ck(Lksub1, i)
        Li = generate_Lk_by_Ck(basket, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data

L, support_data = generate_L(basket, k=3, min_support=0.2)
print("supports of relative commodities")
for i in support_data:
    if len(list(i))==2:
        print(list(i),"=====>",support_data[i])
def generate_big_rules(L, support_data, min_conf):
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list
big_rules_list = generate_big_rules(L, support_data, min_conf=0.7)
relation = []
for item in big_rules_list:
    a = [list(item[0])[0],list(item[1])[0],item[2]]
    relation.append(a)
relations = pd.DataFrame(relation)
relations = relations.sort_values(by=2 , ascending=False)
print(relations)
def graph(C1,big_rule_list):
    # 结点
    node = []
    for item in C1:
        node.extend(list(item))
    # 关系
    relation = []
    size = []
    for item in big_rule_list:
        a = (list(item[0])[0], list(item[1])[0], item[2])
        size.append((item[2]*100-70)**3)
        relation.append(a)
    DG = nx.Graph()
    DG.add_nodes_from(node)
    DG.add_weighted_edges_from(relation)
    nx.draw(DG,pos = nx.spring_layout(DG),with_labels=True,node_size=size)
    plt.show()
C1 = create_C1(basket)
graph(C1, big_rules_list)