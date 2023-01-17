import itertools

import copy

import matplotlib

from Bio import AlignIO

from Bio.Phylo.TreeConstruction import DistanceCalculator

from Bio.Phylo import BaseTree

from Bio import Phylo

import matplotlib.pyplot as plt
import os

print(os.listdir("../input/bio-final"))
alignment = AlignIO.read('../input/bio-final/all_230.fasta', 'fasta')

calculator = DistanceCalculator('identity')

dm = calculator.get_distance(alignment)

print(dm)
def plot_tree(title, treedata):

    matplotlib.rc('font', size=24)

    fig = plt.figure(figsize=(40, 12), dpi=100)

    axes = fig.add_subplot(1, 1, 1)

    axes.set(title=title)

    Phylo.draw(treedata, axes=axes, branch_labels=lambda c: round(c.branch_length, 4) if c.branch_length > 0.01 else "")

    return
def height_of(node):

    height=0

    # 如果是leaf ，高度就是edge 的長度

    if node.is_terminal():

        height = node.branch_length

    else:

    # 如果不是Terminal node ，高度為所有子Nodes 最大高度

        height = max(height_of(c) for c in node.clades)

    return height



def get_minimum_edge(distance_matrix):

    min_i = 1

    min_j = 0

    min_dist = distance_matrix[1, 0]

    for i in range(1, len(distance_matrix)):

        for j in range(0, i):

            if min_dist >= distance_matrix[i, j]:

                min_dist = distance_matrix[i, j]

                min_i = i

                min_j = j

    return min_dist, min_i, min_j



def build_UPGMA(distance_matrix):

    

        # deepcopy 避免影響原本的 matrix

        dm = copy.deepcopy(distance_matrix)

        

        # 根據distance_matrix 內的名字，取得所有node的名字

        nodes = [BaseTree.Clade(None, name) for name in dm.names]

        inner_count = 0

        

        while len(dm) > 1:

            # 找到最小距離的 node pairs

            min_dist, min_i, min_j = get_minimum_edge(dm)



            # 新增一個Node

            inner_count += 1

            new_node = BaseTree.Clade(None, "")

            

            # Connect node pair to new

            min_i_node = nodes[min_i]

            min_j_node = nodes[min_j]

            new_node.clades.append(min_i_node)

            new_node.clades.append(min_j_node)

            

            

            # update node i and j 的長度

            min_j_node.branch_length = min_dist/ 2

            min_i_node.branch_length = min_dist/ 2

            

            if not min_i_node.is_terminal():

                min_i_node.branch_length -= height_of(min_i_node)

            if not min_j_node.is_terminal():

                min_j_node.branch_length -= height_of(min_j_node)



            # remove node_i and node_j from nodes and add new_node to nodes

            nodes[min_j] = new_node

            del nodes[min_i]

            

            # foreach ClusterC∈Clusters

            #    do Adddistance(Cnew, C) toDM;

            # end for

            for k in range(0, len(dm)):

                if k != min_i and k != min_j:

                    dm[min_j, k] = (dm[min_i, k] + dm[min_j, k])/ 2

            

            # remove node_i and node_j from distance_matrix

            dm.names[min_j] = "Inner" + str(inner_count)

            del dm[min_i]



        new_node.branch_length = 0

        return BaseTree.Tree(new_node)
upgmatree = build_UPGMA(dm)

plot_tree("UPGMA", upgmatree)
def calculate_node_distance(dm):

    # init node distance

    node_dist = [0] * len(dm)

    for i in range(0, len(dm)):

        for j in range(0, len(dm)):

            node_dist[i] += dm[i, j]

        node_dist[i] /= len(dm) - 2

    return node_dist



def find_min_pair(node_dist, dm):

    min_dist = dm[1, 0] - node_dist[1] - node_dist[0]

    min_i = 0

    min_j = 1

    for i in range(1, len(dm)):

        for j in range(0, i):

            temp = dm[i, j] - node_dist[i] - node_dist[j]

            if min_dist > temp:

                min_dist = temp

                min_i = i

                min_j = j

    return min_dist, min_i, min_j

    



def build_NJ(distance_matrix):

        # deepcopy 避免影響原本的 matrix

        dm = copy.deepcopy(distance_matrix)

        # 根據distance_matrix 內的名字，取得所有node的名字

        nodes = [BaseTree.Clade(None, name) for name in dm.names]

        

        inner_count = 0

        while len(dm) > 2:

            node_dist = calculate_node_distance(dm)

            # find minimum distance pair

            min_dist, min_i, min_j = find_min_pair(node_dist, dm)

                        

            # create clade

            min_i_node = nodes[min_i]

            min_j_node = nodes[min_j]

            

            inner_count += 1

            new_node = BaseTree.Clade(None, "")

            new_node.clades.append(min_i_node)

            new_node.clades.append(min_j_node)

            

            

            # update node i and j 的長度

            min_i_node.branch_length = (dm[min_i, min_j] + node_dist[min_i] - node_dist[min_j]) / 2.0

            min_j_node.branch_length = dm[min_i, min_j] - min_i_node.branch_length

            

            # remove node_i and node_j from nodes and add new_node to nodes

            nodes[min_j] = new_node

            del nodes[min_i]

            

            # ∀k∈D Dk,m=Dm,k= 1/2∗(D_{k,i}+D_{k,j}−D_{i,j})

            for k in range(0, len(dm)):

                if k != min_i and k != min_j:

                    dm[min_j, k] = (dm[min_i, k] + dm[min_j, k] - dm[min_i, min_j]) / 2.0

                    

            # remove node_i and node_j from distance_matrix

            dm.names[min_j] = "Inner" + str(inner_count)

            del dm[min_i]



        # 剩下兩個點，找出未設置過的node A ，接到另一個node B 之下，

        # 再將 Node B 設置為 Root

        root = None

        if nodes[0] == new_node:

            nodes[0].branch_length = 0

            nodes[1].branch_length = dm[1, 0]

            nodes[0].clades.append(nodes[1])

            root = nodes[0]

        else:

            nodes[0].branch_length = dm[1, 0]

            nodes[1].branch_length = 0

            nodes[1].clades.append(nodes[0])

            root = nodes[1]

        return BaseTree.Tree(root, rooted=False)
njtree = build_NJ(dm)

plot_tree("NJ", njtree)
scorer = Phylo.TreeConstruction.ParsimonyScorer()

searcher = Phylo.TreeConstruction.NNITreeSearcher(scorer)

pars_constructor = Phylo.TreeConstruction.ParsimonyTreeConstructor(searcher, njtree)

pars_tree = pars_constructor.build_tree(alignment)
plot_tree("Parsimony", pars_tree)