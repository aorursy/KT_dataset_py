!pip install lyft-dataset-sdk -q
#First import:  

from lyft_dataset_sdk.lyftdataset import LyftDataset  #Assuming you have already installed it

import pandas as pd

import json

import plotly

import plotly.graph_objs as go

from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
DATA_PATH = "/kaggle/input/samplyft/sampLyft/"

lyft_dataset = LyftDataset(data_path=DATA_PATH, json_path=DATA_PATH+'data')
def plotCloud(lidar_pointcloud):#Code taken from StackOverFlow, sadly forgot the link

    # Configure Plotly to be rendered inline in the notebook.

    plotly.offline.init_notebook_mode()

    # Configure the trace.

    trace = go.Scatter3d(

        x=-lidar_pointcloud.points[0,],

        y=lidar_pointcloud.points[1,],  # <-- Put your data instead

        z=-lidar_pointcloud.points[2,],  # <-- Put your data instead

        mode='markers',

        marker={

        'size': 1,

        'opacity': 0.8,

        }

    )

    # Configure the layout.

    layout = go.Layout(

        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},

         scene =dict(

        xaxis = dict(title="x", range = [-200,200]),

        yaxis = dict(title="y", range = [-200,200]),

        zaxis = dict(title="z", range = [-200,200])

         )

    )

    data = [trace]

    plot_figure = go.Figure(data=data, layout=layout)

    plotly.offline.iplot(plot_figure)
lidar_filepath = lyft_dataset.get_sample_data_path('5c3d79e1cf8c8182b2ceefa33af96cbebfc71f92e18bf64eb8d4e0bf162e01d4')

lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
plotCloud(lidar_pointcloud)
lyft_dataset.render_sample_data("f47a5d143bcebb24efc269b1a40ecb09440003df2c381a69e67cd2a726b27a0c", with_anns=False)
lyft_dataset.render_sample_data("f47a5d143bcebb24efc269b1a40ecb09440003df2c381a69e67cd2a726b27a0c")
lyft_dataset.render_sample_data("ec9950f7b5d4ae85ae48d07786e09cebbf4ee771d054353f1e24a95700b4c4af", with_anns=False)
lyft_dataset.render_sample_data("ec9950f7b5d4ae85ae48d07786e09cebbf4ee771d054353f1e24a95700b4c4af")