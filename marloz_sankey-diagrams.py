import pandas as pd

import plotly.graph_objects as go

import matplotlib.colors as mcolors

import string

import numpy as np
np.random.seed(42)

nodes = np.random.choice([letter for letter in string.ascii_letters], 10, replace=False)

node_sizes = [int(size) for size in np.random.choice(np.geomspace(100, 10000, 10), 10, replace=False)]

node_dict = dict(zip(nodes, node_sizes))



res_df = pd.DataFrame()

for source_node, source_node_size in node_dict.items():

    num_links = np.random.choice(len(node_dict) - 1)

    target_nodes = np.random.choice(nodes, num_links, replace=False)

    weights = np.random.rand(num_links)

    weights = weights / weights.sum()

    turnover = np.random.rand() * source_node_size

    link_vals = np.round(weights * turnover)

    target_nodes = np.append(target_nodes, source_node)

    link_vals = np.append(link_vals, source_node_size - sum(link_vals))

    temp_df = (pd.DataFrame({'target_node': target_nodes, 'link_val': link_vals})

               .assign(source_node=source_node))

    res_df = pd.concat([res_df, temp_df], axis=0)

res_df.sort_values('source_node').head(10)
color_list = np.random.choice(list(mcolors.CSS4_COLORS.keys()), len(nodes), replace=False)

color_dict = dict(zip(nodes, [f'rgba{mcolors.to_rgb(col) + (.4, )}' for col in color_list]))
segments_to_num = dict(zip(nodes, [*range(len(nodes))]))

res_df = res_df.assign(source_node_num = lambda x: x['source_node'].map(segments_to_num),

                       target_node_num = lambda x: x['target_node'].map(segments_to_num),

                       link_col = lambda x: x['source_node'].map(color_dict))

res_df.head()
fig = go.Figure(data=[go.Sankey(

    node = dict(label=nodes, color=color_list),

    link = dict(

      source = res_df.source_node_num,

      target = res_df.target_node_num,

      value = res_df.link_val,

      color = res_df.link_col

  ))])



fig.update_layout(

    title_text="Sankey diagram",

    font_size=10, autosize=True)

fig.show()