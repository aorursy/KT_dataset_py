import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from plotly.offline import init_notebook_mode, iplot
import igraph as ig

np.random.seed(1234)
init_notebook_mode(connected=True)
df=pd.read_csv("../input/PlayerAverages_utf8.csv")
df.columns
%%capture
# Drop fantasy related columns - we only want to cluster players based on their in-game performance
X = df[[col for col in df.columns if col not in ['Points', 'Bonus']]]

# Extract the list of player names and then drop from the data
names = X['Identifier'].values
X.drop('Identifier', axis=1, inplace=True)

# Get the overall averages for each stat for comparison later
means = np.mean(X.values, axis=0)
std_dev = np.std(X.values, axis=0)
# Initialise mapper and create lens using TSNE
mapper = km.KeplerMapper(verbose=0)
lens = mapper.fit_transform(X.values, projection=sklearn.manifold.TSNE(random_state=1234), scaler=None)

# Create the graph of the nerve of the corresponding pullback
graph = mapper.map(lens, X.values, clusterer=sklearn.cluster.KMeans(n_clusters=2, random_state=1234),
                   nr_cubes=20, overlap_perc=0.5)
def get_cluster_summary(player_list, average_mean, average_std, dataset, columns):
    # Compare players against the average and list the attributes that are above and below the average

    cluster_mean = np.mean(dataset.iloc[player_list].values, axis=0)
    diff = cluster_mean - average_mean
    std_m = np.sqrt((cluster_mean - average_mean) ** 2) / average_std

    stats = sorted(zip(columns, cluster_mean, average_mean, diff, std_m), key=lambda x: x[4], reverse=True)
    above_stats = [a[0] + ': ' + f'{a[1]:.2f}' for a in stats if a[3] > 0]
    below_stats = [a[0] + ': ' + f'{a[1]:.2f}' for a in stats if a[3] < 0]
    below_stats.reverse()

    # Create a string summary for the tooltips
    cluster_summary = 'Above Mean:<br>' + '<br>'.join(above_stats[:5]) + \
                      '<br><br>Below Mean:<br>' + '<br>'.join(below_stats[-5:])

    return cluster_summary

def make_igraph_plot(graph, data, X, player_names, layout, mean_list, std_dev_list, title, line_color='rgb(200,200,200)'):
    # Extract node information for the plot
    div = '<br>-------<br>'
    node_list = []
    cluster_sizes = []
    avg_points = []
    tooltip = []
    for node in graph['nodes']:
        node_list.append(node)
        players = graph['nodes'][node]
        cluster_sizes.append(2 * int(np.log(len(players) + 1) + 1))
        avg_points.append(np.average([data.iloc[i]['Points'] for i in players]))
        node_info = node + div + '<br>'.join([player_names[i] for i in players]) + div + \
                    get_cluster_summary(players, mean_list, std_dev_list, X, X.columns)
        tooltip += tuple([node_info])

    # Add the edges to a list for passing into iGraph:
    edge_list = []
    for node in graph['links']:
        for nbr in graph['links'][node]:
            # Need to base everything on indices for igraph
            edge_list.append((node_list.index(node), node_list.index(nbr)))

    # Make the igraph plot
    g = ig.Graph(len(node_list))
    g.add_edges(edge_list)

    links = g.get_edgelist()
    plot_layout = g.layout(layout)

    n = len(plot_layout)
    x_nodes = [plot_layout[k][0] for k in range(n)]  # x-coordinates of nodes
    y_nodes = [plot_layout[k][1] for k in range(n)]  # y-coordinates of nodes

    x_edges = []
    y_edges = []
    for e in links:
        x_edges.extend([plot_layout[e[0]][0], plot_layout[e[1]][0], None])
        y_edges.extend([plot_layout[e[0]][1], plot_layout[e[1]][1], None])

    edges_trace = dict(type='scatter', x=x_edges, y=y_edges, mode='lines', line=dict(color=line_color, width=0.3),
                       hoverinfo='none')

    nodes_trace = dict(type='scatter', x=x_nodes, y=y_nodes, mode='markers', opacity=0.8,
                       marker=dict(symbol='circle-dot', colorscale='Viridis', showscale=True, reversescale=False,
                                   color=avg_points, size=cluster_sizes,
                                   line=dict(color=line_color, width=0.3),
                                   colorbar=dict(thickness=20, ticklen=4)),
                       text=tooltip, hoverinfo='text')

    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')

    layout = dict(title=title, font=dict(size=12), showlegend=False, autosize=False, width=700, height=700,
                  xaxis=dict(axis), yaxis=dict(axis), hovermode='closest', plot_bgcolor='rgba(20,20,20, 0.8)')

    iplot(dict(data=[edges_trace, nodes_trace], layout=layout))
make_igraph_plot(graph, df, X, names, 'kk', means, std_dev, 'Player data - resolution=20')
graph = mapper.map(lens, X.values, clusterer=sklearn.cluster.KMeans(n_clusters=2, random_state=1234),
                   nr_cubes=30, overlap_perc=0.7)

make_igraph_plot(graph, df, X, names, 'kk', means, std_dev, 'Player data - resolution=30')