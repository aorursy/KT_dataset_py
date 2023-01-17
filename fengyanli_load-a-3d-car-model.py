import json

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



with open('../input/pku-autonomous-driving/car_models_json/benchi-ML500.json') as json_file:

    data = json.load(json_file)

    vertices = np.array(data['vertices'])

    triangles = np.array(data['faces']) - 1

    plt.figure(figsize=(20,10))

    ax = plt.axes(projection='3d')

    ax.set_title('car_type: '+data['car_type'])

    ax.set_xlim([-3, 3])

    ax.set_ylim([-3, 3])

    ax.set_zlim([0, 3])

    ax.plot_trisurf(vertices[:,0], vertices[:,2], triangles, -vertices[:,1], shade=True, color='grey')