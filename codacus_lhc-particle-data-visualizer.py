import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/train_1")[:5])

# Any results you write to the current directory are saved as output.
files=os.listdir("../input/train_1")
df=pd.DataFrame(files).sort_values(0)
df['eventID'],df['infoType']=df[0].str.split('-',1).str
df['infoType'],_=df['infoType'].str.split('.',1).str
event_data_types=df['infoType'].unique()
event_data_ids=pd.DataFrame(data=df['eventID'].unique(),columns=['eventId'])
df_cells=pd.read_csv("../input/train_1/"+event_data_ids.loc[0]['eventId']+'-cells.csv')
df_hits=pd.read_csv("../input/train_1/"+event_data_ids.loc[0]['eventId']+'-hits.csv')
df_particles=pd.read_csv("../input/train_1/"+event_data_ids.loc[0]['eventId']+'-particles.csv')
df_truth=pd.read_csv("../input/train_1/"+event_data_ids.loc[0]['eventId']+'-truth.csv')
    
sample_csv_size=10
for i in range(1,sample_csv_size):
    df_cells.append(pd.read_csv("../input/train_1/"+event_data_ids.loc[i]['eventId']+'-cells.csv'), ignore_index=True)
    df_hits.append(pd.read_csv("../input/train_1/"+event_data_ids.loc[i]['eventId']+'-hits.csv') , ignore_index=True)
    df_particles.append(pd.read_csv("../input/train_1/"+event_data_ids.loc[i]['eventId']+'-particles.csv'), ignore_index=True)
    df_truth.append(pd.read_csv("../input/train_1/"+event_data_ids.loc[i]['eventId']+'-truth.csv'), ignore_index=True)
def scatter3D(x,y,z,color,name='Undefiend 3d Scatter Plot'):
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    import plotly.graph_objs as go
    init_notebook_mode(connected=True)
    trace1 = go.Scatter3d( x=x, y=y, z=z, mode='markers',
        marker=dict(
            size=12,
            color=color,           # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8,
            showscale=True
        ),
        name=name
    )

    data = [trace1]
    layout = go.Layout( margin=dict(l=0,r=0,b=0,t=0), showlegend=True, legend=dict(x=0,y=1))

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename=name)
    
sample_size=5000
x=df_truth['tx'].head(sample_size)
y=df_truth['ty'].head(sample_size)
z=df_truth['tz'].head(sample_size)

weight=df_truth['weight'].head(sample_size).apply(lambda x: x*100)

scatter3D(x,y,z,weight, 'Detecter hits location plot')

df_truth.head()
sample_size=200000
x=df_particles['vx'].head(sample_size)
y=df_particles['vy'].head(sample_size)
z=df_particles['vz'].head(sample_size)

charge=df_particles['q'].head(sample_size)

scatter3D(x,y,z,charge,"Particle's Initial locations")
sample_particle_df=df_truth.loc[df_truth['particle_id']==418835796137607168]
sample_particle_df
x=sample_particle_df['tx']
y=sample_particle_df['ty']
z=sample_particle_df['tz']

weight=sample_particle_df['weight'].apply(lambda x: x*100)

scatter3D(x,y,z,weight, 'Single particle hit location plot')
