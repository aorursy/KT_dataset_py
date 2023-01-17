import numpy as np

import pandas as pd



import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



structures = pd.read_csv('../input/structures.csv')

train = pd.read_csv('../input/train.csv')



# initiate the plotly notebook mode

init_notebook_mode(connected=True)





def plot_interactions(molecule_name, structures_df, train_df):

    """Creates a 3D plot of the molecule"""

    

    atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)  

    cpk_colors = dict(C='black', F='green', H='white', N='blue', O='red')

    

    if molecule_name not in train_df.molecule_name.unique():

        print(f'Molecule "{molecule_name}" is not in the training set!')

        return

    

    molecule = structures[structures.molecule_name == molecule_name]

    coordinates = molecule[['x', 'y', 'z']].values

    x_coordinates = coordinates[:, 0]

    y_coordinates = coordinates[:, 1]

    z_coordinates = coordinates[:, 2]

    elements = molecule.atom.tolist()

    radii = [atomic_radii[element] for element in elements]

    

    data_train = train_df[train_df.molecule_name == molecule_name][['atom_index_0', 'atom_index_1', 'scalar_coupling_constant']]

    interactions = data_train.groupby('atom_index_0')['atom_index_1'].apply(set).to_dict()

    coupling_constants = data_train.set_index(['atom_index_0', 'atom_index_1']).round(2).to_dict()['scalar_coupling_constant']

    

    def get_bonds():

        """Generates a set of bonds from atomic cartesian coordinates"""

        ids = np.arange(coordinates.shape[0])

        bonds = dict()

        coordinates_compare, radii_compare, ids_compare = coordinates, radii, ids

        

        for _ in range(len(ids)):

            coordinates_compare = np.roll(coordinates_compare, -1, axis=0)

            radii_compare = np.roll(radii_compare, -1, axis=0)

            ids_compare = np.roll(ids_compare, -1, axis=0)

            distances = np.linalg.norm(coordinates - coordinates_compare, axis=1)

            bond_distances = (radii + radii_compare) * 1.3

            mask = np.logical_and(distances > 0.1, distances <  bond_distances)

            distances = distances.round(2)

            new_bonds = {frozenset([i, j]): dist for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask])}

            bonds.update(new_bonds)

        return bonds      

            

    def atom_trace():

        """Creates an atom trace for the plot"""

        colors = [cpk_colors[element] for element in elements]

        markers = dict(color=colors, line=dict(color='lightgray', width=2), size=5, symbol='circle', opacity=0.8)

        trace = go.Scatter3d(x=x_coordinates, y=y_coordinates, z=z_coordinates, mode='markers', marker=markers,

                             text=elements, name='')

        return trace



    def bond_trace():

        """"Creates a bond trace for the plot"""

        trace = go.Scatter3d(x=[], y=[], z=[], hoverinfo='none', mode='lines',

                             marker=dict(color='grey', size=7, opacity=1), line=dict(width=3))

        for i, j in bonds.keys():

            trace['x'] += (x_coordinates[i], x_coordinates[j], None)

            trace['y'] += (y_coordinates[i], y_coordinates[j], None)

            trace['z'] += (z_coordinates[i], z_coordinates[j], None)

        return trace

    

    def interaction_trace(atom_id):

        """"Creates an interaction trace for the plot"""

        trace = go.Scatter3d(x=[], y=[], z=[], hoverinfo='none', mode='lines',

                             marker=dict(color='pink', size=7, opacity=0.5),

                            visible=False)

        for i in interactions[atom_id]:

            trace['x'] += (x_coordinates[atom_id], x_coordinates[i], None)

            trace['y'] += (y_coordinates[atom_id], y_coordinates[i], None)

            trace['z'] += (z_coordinates[atom_id], z_coordinates[i], None)

        return trace

    

    bonds = get_bonds()

    

    zipped = zip(range(len(elements)), x_coordinates, y_coordinates, z_coordinates)

    annotations_id = [dict(text=num, x=x, y=y, z=z, showarrow=False, yshift=15, font = dict(color = "blue"))

                      for num, x, y, z in zipped]

    

    annotations_length = []

    for (i, j), dist in bonds.items():

        x_middle, y_middle, z_middle = (coordinates[i] + coordinates[j]) / 2

        annotation = dict(text=dist, x=x_middle, y=y_middle, z=z_middle, showarrow=False, yshift=10)

        annotations_length.append(annotation)

    

    annotations_interaction = []

    for k, v in interactions.items():

        annotations = []

        for i in v:

            x_middle, y_middle, z_middle = (coordinates[k] + coordinates[i]) / 2

            constant = coupling_constants[(k, i)]

            annotation = dict(text=constant, x=x_middle, y=y_middle, z=z_middle, showarrow=False, yshift=25,

                              font = dict(color = "hotpink"))

            annotations.append(annotation)

        annotations_interaction.append(annotations)

    

    buttons = []

    for num, i in enumerate(interactions.keys()):

        mask = [False] * len(interactions)

        mask[num] = True

        button = dict(label=f'Atom {i}',

                      method='update',

                      args=[{'visible': [True] * 2 + mask},

                            {'scene.annotations': annotations_id + annotations_length + annotations_interaction[num]}])

        buttons.append(button)

        

    updatemenus = list([

        dict(buttons = buttons,

             direction = 'down',

             xanchor = 'left',

             yanchor = 'top'

            )

    ])

    

    data = [atom_trace(), bond_trace()]

    

    # add interaction traces

    for num, i in enumerate(interactions.keys()):

        trace = interaction_trace(i)

        if num == 0:

            trace.visible = True 

        data.append(trace)

        

    axis_params = dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, titlefont=dict(color='white'))

    layout = dict(scene=dict(xaxis=axis_params, yaxis=axis_params, zaxis=axis_params,

                             annotations=annotations_id + annotations_length + annotations_interaction[0]),

                  margin=dict(r=0, l=0, b=0, t=0), showlegend=False, updatemenus=updatemenus)



    fig = go.Figure(data=data, layout=layout)

    iplot(fig)
plot_interactions('dsgdb9nsd_000117', structures, train)
plot_interactions('dsgdb9nsd_000131', structures, train)
plot_interactions('dsgdb9nsd_000161', structures, train)
plot_interactions('dsgdb9nsd_000137', structures, train)
plot_interactions('dsgdb9nsd_099964', structures, train)