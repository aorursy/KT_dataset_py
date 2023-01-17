import osmnx as ox

import pandas as pd

import geopandas as gpd

import networkx as nx

%matplotlib inline

ox.config(log_console=True, use_cache=True)
# This function takes the road name and a 3-letter code for the language and it returns the colour

def colourcode(x, language):

    if (language=='GER'):

        if ('straße' in x) or ('strasse' in x): 

            return '#f6cf71'

        elif ('weg' in x):

            return '#019868'

        elif ('allee' in x) or ('gasse' in x):

            return '#ec0b88'

        elif ('damm' in x):

            return '#651eac'

        elif ('platz' in x):

            return '#e18a1e'

        elif ('chaussee' in x):

            return '#9dd292'

        elif ('see' in x):

            return '#2b7de5'

        elif ('ufer' in x):

            return '#2b7de5'

        elif ('steg' in x):

            return '#2b7de5'

        else:

            return '#c6c6c6'

    elif (language=='ENG'):

        if ('road' in x): 

            return '#019868'

        elif ('street' in x):

            return '#f6cf71'

        elif ('way' in x):

            return '#ec0b88'

        elif ('avenue' in x):

            return '#651eac'

        elif ('drive' in x):

            return '#e18a1e'

        elif ('lane' in x):

            return '#9dd292'

        else:

            return '#c6c6c6'

    elif (language=='FRA'):

        if ('rue' in x): 

            return '#019868'

        elif ('place' in x):

            return '#f6cf71'

        elif ('avenue' in x):

            return '#ec0b88'

        elif ('boulevard' in x):

            return '#651eac'

        elif ('passage' in x):

            return '#e18a1e'

        elif ('pont' in x):

            return '#9dd292'

        elif ('quai' in x):

            return '#2b7de5'

        else:

            return '#c6c6c6'

    else:

        return 'black'
# Set place and language; the place is basically a Nominatim query. It must return a POLYGON/POLYLINE, not a POINT, so you might have to play with it a little, or set which_result below accordingly    

place='Zurich, Switzerland'

language='GER'



# note the which_result parameter, as per comment above

G = ox.graph_from_place(place, network_type='all', which_result=1) 
# For the colouring, we take the attributes from each edge found extract the road name, and use the function above to create the colour array

edge_attributes = ox.graph_to_gdfs(G, nodes=False)

edge_attributes['ec'] = edge_attributes['name'].str.lower().map(lambda x: colourcode(str(x), language))

# We can finally draw the plot

fig, ax = ox.plot_graph(G, 

                        bgcolor='white', 

                        axis_off=True, 

                        node_size=0, 

                        node_color='w', 

                        node_edgecolor='gray', 

                        node_zorder=2,

                        edge_color=edge_attributes['ec'], 

                        edge_linewidth=0.5, 

                        edge_alpha=1, 

                        fig_height=20, 

                        dpi=300)
# Appendix 

# Are you curious about other feature of the streets? With this piece of code, you can see what other elements you could colour..



edge_attributes = ox.graph_to_gdfs(G, nodes=False)

edge_attributes.head()
edge_attributes.sample()
# calculate basic and extended network stats, merge them together, and display

area = ox.project_gdf(edge_attributes).unary_union.area

stats = ox.basic_stats(G, area=area)

extended_stats = ox.extended_stats(G, ecc=True, bc=True, cc=True)

for key, value in extended_stats.items():

    stats[key] = value

pd.Series(stats)
# unpack dicts into individiual keys:values

stats = ox.basic_stats(G, area=area)

for k, count in stats['streets_per_node_counts'].items():

    stats['int_{}_count'.format(k)] = count

for k, proportion in stats['streets_per_node_proportion'].items():

    stats['int_{}_prop'.format(k)] = proportion



# delete the no longer needed dict elements

del stats['streets_per_node_counts']

del stats['streets_per_node_proportion']



# load as a pandas dataframe

pd.DataFrame(pd.Series(stats)).T

G_projected = ox.project_graph(G)

max_node, max_bc = max(extended_stats['betweenness_centrality'].items(), key=lambda x: x[1])

max_node, max_bc
nc = ['r' if node==max_node else '#336699' for node in G_projected.nodes()]

ns = [50 if node==max_node else 8 for node in G_projected.nodes()]

fig, ax = ox.plot_graph(G_projected, node_size=ns, node_color=nc, node_zorder=2)
# get a color for each node

def get_color_list(n, color_map='plasma', start=0, end=1):

    return [cm.get_cmap(color_map)(x) for x in np.linspace(start, end, n)]



def get_node_colors_by_stat(G, data, start=0, end=1):

    df = pd.DataFrame(data=pd.Series(data).sort_values(), columns=['value'])

    df['colors'] = get_color_list(len(df), start=start, end=end)

    df = df.reindex(G.nodes())

    return df['colors'].tolist()



nc = get_node_colors_by_stat(G_projected, data=extended_stats['betweenness_centrality'])

fig, ax = ox.plot_graph(G_projected, node_color=nc, node_edgecolor='gray', node_size=20, node_zorder=2)