import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py

import plotly.graph_objs as go

import glob

import ipywidgets as widgets

from sklearn.cluster import KMeans



py.init_notebook_mode(connected=True)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

videoDF = pd.read_csv('../input/youtube_faces_with_keypoints_medium.csv')

videoDF.head(10)
npzFilesFullPath = glob.glob('../input/youtube_faces_*/*.npz')

videoIDs = [x.split('/')[-1].split('.')[0] for x in npzFilesFullPath]

fullPaths = {}

for videoID, fullPath in zip(videoIDs, npzFilesFullPath):

    fullPaths[videoID] = fullPath



# remove from the large csv file all videos that weren't uploaded yet

videoDF = videoDF.loc[videoDF.loc[:,'videoID'].isin(fullPaths.keys()),:].reset_index(drop=True)

print('Number of Videos uploaded so far is %d' %(videoDF.shape[0]))
kmeans_df = pd.DataFrame()

for i,video in enumerate(videoDF.iterrows()):

    videoID = videoDF.loc[i,'videoID']

    videoFile = np.load(fullPaths[videoID])

    landmarks3D = videoFile['landmarks3D']

    landmarks3D[:,0,0] -= np.mean(landmarks3D[:,0,0])

    landmarks3D[:,0,0]/= np.std(landmarks3D[:,0,0])

    landmarks3D[:,1,0] -= np.mean(landmarks3D[:,1,0])

    landmarks3D[:,1,0]/= np.std(landmarks3D[:,1,0])

    landmarks3D[:,2,0] -= np.mean(landmarks3D[:,2,0])

    landmarks3D[:,2,0]/= np.std(landmarks3D[:,2,0])

    kmeans_df[i] = sum(landmarks3D[:,:,0].tolist(), [])
import matplotlib.pyplot as plt

videoFile = np.load(fullPaths['Alison_Lohman_0'])

plt.figure(figsize = (10,10))

colorImages = videoFile['colorImages']

plt.imshow(colorImages[:,:,:,0])
kmeans = KMeans(n_clusters = 10).fit(kmeans_df.transpose())
def plot_kmeans(cluster):

    landmarks = np.reshape(kmeans.cluster_centers_[cluster],[68,3])

    x = landmarks[:,0] 

    y = landmarks[:,1]

    z = landmarks[:,2]



    trace1 = go.Scatter3d(

        x=x[0:17],

        y=y[0:17],

        z=z[0:17],

        name = 'Chin',

        mode='lines+markers',

        marker=dict(

            color = 'blue',

            opacity=0.5,

            size = 5

        )

    )

    trace2 = go.Scatter3d(

        x=x[17:22],

        y=y[17:22],

        z=z[17:22],

        name = 'Right eyebrow',

        mode='lines+markers',

        marker=dict(

            color = 'blue',

            opacity=0.5,

            size = 5

        )

    )

    trace3 = go.Scatter3d(

        x=x[22:27],

        y=y[22:27],

        z=z[22:27],

        name = 'Left eyebrow',

        mode='lines+markers',

        marker=dict(

            color = 'blue',

            opacity=0.5,

            size = 5

        )

    )



    trace4 = go.Scatter3d(

        x=x[27:31],

        y=y[27:31],

        z=z[27:31],

        name = 'Nose',

        mode='lines+markers',

        marker=dict(

            color = 'blue',

            opacity=0.5,

            size = 5

        )

    )



    trace5 = go.Scatter3d(

        x=x[31:36],

        y=y[31:36],

        z=z[31:36],

        name = 'Nose 2',

        mode='lines+markers',

        marker=dict(

            color = 'blue',

            opacity=0.5,

            size = 5

        )

    )



    trace6 = go.Scatter3d(

        x=x[36:42],

        y=y[36:42],

        z=z[36:42],

        name = 'Right eye',

        mode='lines+markers',

        marker=dict(

            color = 'blue',

            opacity=0.5,

            size = 5

        )

    )



    trace7 = go.Scatter3d(

        x=x[42:48],

        y=y[42:48],

        z=z[42:48],

        name = 'Left eye',

        mode='lines+markers',

        marker=dict(

            color = 'blue',

            opacity=0.5,

            size = 5

        )

    )



    trace8 = go.Scatter3d(

        x=x[48:60],

        y=y[48:60],

        z=z[48:60],

        name = 'Mouth',

        mode='lines+markers',

        marker=dict(

            color = 'blue',

            opacity=0.5,

            size = 5

        )

    )



    trace9 = go.Scatter3d(

        x=x[60:68],

        y=y[60:68],

        z=z[60:68],

        mode='lines+markers',

        marker=dict(

            color = 'blue',

            opacity=0.5,

            size = 5

        )

    )



    malpha=np.linspace(0,720,100)

    malpha=[float(xx)/180*np.pi for xx in malpha]



    mfr=[]

    for ii in malpha:

        mfr.append(dict(layout= dict(scene=dict(camera=dict(up=dict(x=-1, y=0, z=0),

                                                            center=dict(x=0, y=0, z=0),

                                                            eye=dict(x=2*np.sin(ii), y=-1, z=2*np.cos(ii)),

                                                            )

                                               ),



                                    )

                       )

                  )



    layout=dict(width=800, height=800, title='',

                scene=dict(camera=dict(up=dict(x=-1, y=0, z=0),

                                       center=dict(x=0, y=0, z=0),

                                       eye=dict(x=2.5, y=0, z=0)

                                      )

                          ),

                updatemenus=[dict(type='buttons', showactive=False,

                                    y=1,

                                    x=1,

                                    xanchor='right',

                                    yanchor='top',

                                    pad=dict(t=0, r=10),

                                    buttons=[dict(

                                    label='Play',

                                    method='animate',

                                    args=[None, dict(frame=dict(duration=0.1, redraw=True), 

                                                     transition=dict(duration=0),

                                                     fromcurrent=True,

                                                     mode='immediate'

                                                    )

                                         ]

                                                 )

                                            ]

                                   )

                              ]     

               )

    data = [trace1,trace2,trace3, trace4, trace5, trace6, trace7, trace8]

    fig=dict(data=data, layout=layout, frames=mfr)

    py.iplot(fig)
i = widgets.interactive(plot_kmeans, cluster=widgets.Dropdown(options=[0,1,2,3,4,5,6,7,8,9], description='Cluster'))

i
videoDF[videoDF.videoID == 'Sara_Silverman_2']

videoFile = np.load(fullPaths['Sara_Silverman_2'])

landmarks3D = videoFile['landmarks3D']


trace1 = go.Scatter3d(name='chin', x=landmarks3D[:,0,1][0:17],y=landmarks3D[:,1,1][0:17],

                                            z=landmarks3D[:,2,1][0:17],mode='lines+markers',marker=dict(color = 'blue',

                                            opacity=0.5,size = 5))

trace2 = go.Scatter3d(name='right eyebrow',x=landmarks3D[:,0,1][17:22],y=landmarks3D[:,1,1][17:22],

                                            z=landmarks3D[:,2,1][17:22],mode='lines+markers',marker=dict(color = 'blue',

                                            opacity=0.5,size = 5))

trace3 = go.Scatter3d(name='left eyebrow',x=landmarks3D[:,0,1][22:27],y=landmarks3D[:,1,1][22:27],

                                            z=landmarks3D[:,2,1][22:27],mode='lines+markers',marker=dict(color = 'blue',

                                            opacity=0.5,size = 5))

trace4 = go.Scatter3d(name='nose 1',x=landmarks3D[:,0,1][27:31],y=landmarks3D[:,1,1][27:31],

                                            z=landmarks3D[:,2,1][27:31],mode='lines+markers',marker=dict(color = 'blue',

                                            opacity=0.5,size = 5))

trace5 = go.Scatter3d(name='nose 2',x=landmarks3D[:,0,1][31:36],y=landmarks3D[:,1,1][31:36],

                                            z=landmarks3D[:,2,1][31:36],mode='lines+markers',marker=dict(color = 'blue',

                                            opacity=0.5,size = 5))

trace6 = go.Scatter3d(name='right eye', x=landmarks3D[:,0,1][36:42],y=landmarks3D[:,1,1][36:42],

                                            z=landmarks3D[:,2,1][36:42],mode='lines+markers',marker=dict(color = 'blue',

                                            opacity=0.5,size = 5))

trace7 = go.Scatter3d(name='left eye', x=landmarks3D[:,0,1][42:48],y=landmarks3D[:,1,1][42:48],

                                            z=landmarks3D[:,2,1][42:48],mode='lines+markers',marker=dict(color = 'blue',

                                            opacity=0.5,size = 5))

trace8 = go.Scatter3d(name='mouth 1', x=landmarks3D[:,0,1][48:60],y=landmarks3D[:,1,1][48:60],

                                            z=landmarks3D[:,2,1][48:60],mode='lines+markers',marker=dict(color = 'blue',

                                            opacity=0.5,size = 5))



trace9 = go.Scatter3d(name='mouth 2', x=landmarks3D[:,0,1][60:68],y=landmarks3D[:,1,1][60:68],

                                            z=landmarks3D[:,2,1][60:68],mode='lines+markers',marker=dict(color = 'blue',

                                            opacity=0.5,size = 5))

    

data = [trace1,trace2,trace3, trace4, trace5, trace6, trace7, trace8, trace9]

mfr = []

for i in range(len(landmarks3D[1,1,:])):

    mfr.append({'data' :[{'type' : "scatter3d",'x':landmarks3D[:,0,i][0:17],'y':landmarks3D[:,1,i][0:17],

                                            'z':landmarks3D[:,2,i][0:17],'mode':'lines+markers'},

                        {'type' : "scatter3d",'x':landmarks3D[:,0,i][17:22],'y':landmarks3D[:,1,i][17:22],

                                            'z':landmarks3D[:,2,i][17:22],'mode':'lines+markers'},

                        {'type' : "scatter3d",'x':landmarks3D[:,0,i][22:27],'y':landmarks3D[:,1,i][22:27],

                                            'z':landmarks3D[:,2,i][22:27],'mode':'lines+markers'},

                        {'type' : "scatter3d",'x':landmarks3D[:,0,i][27:31],'y':landmarks3D[:,1,i][27:31],

                                            'z':landmarks3D[:,2,i][27:31],'mode':'lines+markers'},

                        {'type' : "scatter3d",'x':landmarks3D[:,0,i][31:36],'y':landmarks3D[:,1,i][31:36],

                                            'z':landmarks3D[:,2,i][31:36],'mode':'lines+markers'},

                        {'type' : "scatter3d",'x':landmarks3D[:,0,i][36:42],'y':landmarks3D[:,1,i][36:42],

                                            'z':landmarks3D[:,2,i][36:42],'mode':'lines+markers'},

                        {'type' : "scatter3d",'x':landmarks3D[:,0,i][42:48],'y':landmarks3D[:,1,i][42:48],

                                            'z':landmarks3D[:,2,i][42:48],'mode':'lines+markers'},

                        {'type' : "scatter3d",'x':landmarks3D[:,0,i][48:60],'y':landmarks3D[:,1,i][48:60],

                                            'z':landmarks3D[:,2,i][48:60],'mode':'lines+markers'},

                        {'type' : "scatter3d",'x':landmarks3D[:,0,i][60:68],'y':landmarks3D[:,1,i][60:68],

                                            'z':landmarks3D[:,2,i][60:68],'mode':'lines+markers'}],

              'layout':{'xaxis':{'range':[40,90]}}})

                             



layout = dict(width=800, height=800, title='',

              xaxis = dict(

                  range = [40,90]),

              scene=dict(camera=dict(up=dict(x=-1, y=0, z=0),

                                                            center=dict(x=0, y=0, z=0),

                                                            eye=dict(x=0, y=0, z=2),

                                                            )

                                               ),

                updatemenus=[dict(type='buttons', showactive=False,

                                    y=1,

                                    x=1,

                                    xanchor='right',

                                    yanchor='top',

                                    pad=dict(t=0, r=10),

                                    buttons=[dict(

                                    label='Play',

                                    method='animate',

                                    args=[None, dict(frame=dict(duration=0.1, redraw=True), 

                                                     transition=dict(duration=0),

                                                     fromcurrent=True,

                                                     mode='immediate'

                                                    )

                                         ]

                                                 )

                                            ]

                                   )

                              ]     

               )

fig=dict(data=data, layout=layout, frames=mfr)

py.iplot(fig)