from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np

init_notebook_mode(connected=True) #do not miss this line
#Filename for event
fn="train_1/event000001000"

#Row in particle file
n=105

A=np.loadtxt("../input/"+fn+"-truth.csv",skiprows=1,delimiter=',')
B=np.loadtxt("../input/"+fn+"-particles.csv",skiprows=1,delimiter=',')

#particle id to nth row in particle file.
part_id = B[n,0]

#Find hits from particle.
hits_from_part = np.argwhere(A[:,1]==part_id)[:,0]

#Print particle id and number of hits found.
print("PARTICLE ID:")
print(int(part_id))
print("Num hits: " + str(len(hits_from_part)))
print("")
#Get coordinates from hit-data.
coords = np.zeros((len(hits_from_part),3))
for i in range(0,len(hits_from_part)):
	coords[i,:] = A[hits_from_part[i],2:5]
	print("hit id: "+str(int(A[hits_from_part[i],0]))+ ". Absolute momentum: " + str(np.sqrt(np.sum(A[hits_from_part[i],5:8]**2))))
	
#Sort coordinates by z-component.	
idx = np.argsort(coords[:,2])
coords=coords[idx]

trace = go.Scatter3d(
    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
    marker=dict(
        size=4,
    ),
    line=dict(
        color='#1f77b4',
        width=1
    )
)

data = [trace]

layout = dict(
    width=500,
    height=500,
    autosize=False,
    title='Trajectory',
    titlefont = dict(size=8),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='simple-3d')
