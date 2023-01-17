import os
print(os.listdir("../input"))
import numpy as np 
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
import plotly.graph_objs as go
init_notebook_mode(connected=True) #do not miss this line
from trackml.dataset import load_event
hits, cells, particles, truth = load_event('../input/train_1/event000001000')
fig = plt.figure()
np.random.seed(2500)
hits_sample=hits.sample(n=2000,random_state=100)
volumes=hits.volume_id.unique()
res_final=[]
for volume in volumes:
                        v=hits_sample[hits_sample.volume_id==volume]
                        hits_rep=go.Scatter3d(x=v.x,y=v.y,z=v.z,
                                              mode='markers',
                                              marker=dict(
                                                          size=4,
                                                          line=dict(
                                                                        color=volume,
                                                                        width=0.5
                                                                    ),
                                                           opacity=0.8         
                                                        ),
                                              name = 'volume ' + str(volume)
                                                )                
                        res_final.append(hits_rep)
                                                                                          
layout = go.Layout(
    title = 'Particles hits in 3D space (mm)',
    titlefont = dict(size=8),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=res_final, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')
fig=plt.figure()
particles_sample=particles.sample(n=len(particles.particle_id),random_state=100)
nhits=particles.nhits.unique()
particles_final=[]
for nhit in nhits:
                        v=particles_sample[particles_sample.nhits==nhit]
                        particles_rep=go.Scatter3d(x=v.vx,y=v.vy,z=v.vz,
                                              mode='markers',
                                              marker=dict(
                                                          size=4,
                                                          line=dict(
                                                                        color=nhit,
                                                                        width=0.5
                                                                    ),
                                                           opacity=0.8         
                                                        ),
                                                   name = str(nhit) + ' hits ' 
                                                          + '[' + str(round((len(v.particle_id)/len(particles_sample.particle_id))*100)) + '%]'
                                                )                
                        particles_final.append(particles_rep)
                                                                                          
layout = go.Layout(
    title = 'Particles and their nhits (mm)',
    titlefont = dict(size=8),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=particles_final, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')
    
