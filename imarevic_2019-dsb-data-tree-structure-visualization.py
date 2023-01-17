import numpy as np

import pandas as pd

import pydot

from IPython.display import Image, display



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_df = pd.read_csv("../input/data-science-bowl-2019/train.csv")

test_df = pd.read_csv("../input/data-science-bowl-2019/test.csv")

train_labels_df = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")

specs_df = pd.read_csv("../input/data-science-bowl-2019/specs.csv")
train_ass = train_df.loc[train_df['event_code'].isin([4100, 4110])]

train_ass.head(15)
train_filtered = train_df.loc[train_df['installation_id'] == '0006a69f'] 

train_filtered.sort_values(by=['event_count'], ascending=False)

print(train_filtered.shape)

train_filtered.head()
# left join with labels

train_filtered_final = pd.merge(train_filtered, train_labels_df, 'left', on=['installation_id', 'game_session'])



# FILTERING:

n_samples = 20

world = 'TREETOPCITY'

train_filtered_final = train_filtered_final.loc[train_filtered_final['world'] == world]

# then get rows where assessment was present

train_with_assessment = train_filtered_final.loc[train_filtered_final['type']=='Assessment'][:n_samples]

train_filtered_final = train_filtered_final[:n_samples]

# union both

train_filtered_final = pd.concat([train_filtered_final, train_with_assessment])



# inspect df

print(train_filtered_final.shape)

train_filtered_final.head(50)
cols_to_select = ['installation_id', 'world', 'title_x', 'game_session','accuracy_group']

train_viz_df = train_filtered_final[cols_to_select]

train_viz_df.head(50)
# collect data in dict

def create_nested_dict(df):

    d = {}

    for row in df.values:

        here = d

        for elem in row[:-2]:

            if elem not in here:

                here[elem] = {}

            here = here[elem]

        here[row[-2]] = row[-1]

    return d



train_dict = create_nested_dict(train_viz_df)

train_dict
def draw(parent_name, child_name):

    edge = pydot.Edge(parent_name, child_name)

    graph.add_edge(edge)



def visit(node, parent=None):

    for k,v in node.items():

        if isinstance(v, dict):

            # start with the root node where parent is None

            # we don't want to graph the None node

            if parent:

                draw(str(parent), str(k))

            visit(v, k)

        else:

            draw(parent, k)

            # drawing the label using a distinct name

            draw(str(k), str(k)+'_'+str(v))



def show_graph(pdot_graph):

    plt = Image(pdot_graph.create_png())

    display(plt)

    

# instantiate pydot, recursive call, show graph

# and write graph to output directory

graph = pydot.Dot(graph_type='graph')

visit(train_dict)

show_graph(graph)

graph.write_png('sample_train_data_tree.png')