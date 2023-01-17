import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import re
from IPython.display import display, Markdown
pd.options.mode.chained_assignment = None
#from adjustText import adjust_text # cant install this in kernel. It makes the plots prettier though.
players = pd.read_csv("../input/CompleteDataset.csv",low_memory=False)
player_att = pd.read_csv("../input/PlayerAttributeData.csv",low_memory=False) #to get att names easier
players = players.fillna(players.mean()) #fill missing values
players["Preferred Positions 1"] = players["Preferred Positions"].apply(lambda x: re.findall('^([\w\-]+)', x)[0]) # Let's get one position from the string of all posistions
positions_map1 = {'CAM': 1,'CB': 2, 'CDM': 2,'CF': 0,'CM': 1,'GK': 3,'LB': 2,'LM': 1,'LW': 0,'LWB': 2,'RB': 2,'RM': 1,'RW': 0,'RWB': 2,'ST': 0} # if fowards 0, midfield = 1, defender = 2, GK = 3
players["Preferred Positions int"] = players["Preferred Positions 1"].replace(positions_map1) #map it
players[["Preferred Positions","Preferred Positions 1","Preferred Positions int"]].head(50)
def get_attributes(players):
    attribute_names = player_att.columns[1:].values  # get all attribute names i.e acceleration, finishing, passing etc.
    attribute_names = attribute_names[attribute_names!="ID"] # Drop ID variable
    attributes = players[attribute_names] # get info from players
    attributes = attributes.apply(pd.to_numeric, errors='coerce', axis=0) #convert to numeric
    attributes = attributes.fillna(attributes.mean()) #fill missing values with mean
    return attributes
attributes = get_attributes(players)
attributes.head(10)
def dim_reduction(x):
    dim_reduce = PCA(n_components=5)
    dim_reduce_fit = dim_reduce.fit_transform(x)
    return dim_reduce, dim_reduce_fit
dim_reduce, dim_reduce_fit = dim_reduction(attributes)
print(dim_reduce_fit) 
var_explained2 = dim_reduce.explained_variance_ratio_[:2].sum()*100
display(Markdown("### Proportion of total variance explained by first 5 components is {0}. That is the first 2 components explain {1:.3f}% of the data' total variation".format(dim_reduce.explained_variance_ratio_,var_explained2)))
def biplot(dim_reduce,coeff,labels=None,color="blue",alpha=0.5):
    fig, ax = plt.subplots(figsize=(25,25))
    xs = dim_reduce[:,0] # First component
    ys = dim_reduce[:,1] # Second component
    n = coeff.shape[0] # number of attributes
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    
    cax = ax.scatter(xs*scalex,ys*scaley,c = color,alpha=0.35)
    ax.set_xlabel("First component")
    ax.set_ylabel("Second component")
    arrow_scale = 1.2
    annotations = []
    for i in range(n):
        ax.arrow(0, 0, coeff[i,0]*arrow_scale, coeff[i,1]*arrow_scale,color = 'red',linestyle="-",alpha=0.5,head_width=0.01)
        annotations.append(ax.text(coeff[i,0]*arrow_scale, coeff[i,1]*arrow_scale, labels[i], color = 'black', ha = 'left', va = 'top'))
    #adjust_text(annotations) This would be nice.
    return fig,cax
components = dim_reduce_fit[:,:2]
var_vectors = np.transpose(dim_reduce.components_[0:2, :])
vec_labels = attributes.columns.values
position_col = players["Preferred Positions int"]
fig,cax = biplot(components,var_vectors,labels=vec_labels,color=position_col)
cbar = fig.colorbar(cax, ticks=[0,1,2,3], orientation='horizontal')
cbar.ax.set_xticklabels(['Foward','Midfield','Defender',"Goalkeeper"])  # horizontal colorbar
plt.show()

no_gks = players[players["Preferred Positions 1"]!="GK"] # remove them
no_gks["Wage"] = no_gks["Wage"].replace('[\€K,]', '', regex=True).astype(float) #regex to grab how much they are getting paid
no_gks["Wage log"] = no_gks["Wage"].apply(np.log) # log wages
no_gks["Wage log"] = no_gks["Wage log"].fillna(0)
no_gks["Wage log"][np.isinf(no_gks["Wage log"])] = 0 #some people have negative wages :S. set them to zero
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(20,5))
ax1.hist(no_gks["Wage"],bins=10); ax1.set_xlabel("Wage"); ax1.set_ylabel("Frequency");
ax2.hist(no_gks["Wage log"],bins=10); ax2.set_ylabel("Frequency"); ax2.set_xlabel("Log wage")
plt.show()
attributes = get_attributes(no_gks)
dim_reduce, dim_reduce_fit = dim_reduction(attributes)
wage_col = no_gks["Wage log"]
fig, cax = biplot(dim_reduce_fit,np.transpose(dim_reduce.components_[0:2, :]),labels=attributes.columns.values,color=wage_col,alpha=0.9)
ticks = np.linspace(0,wage_col.max(),10)
cbar = fig.colorbar(cax, ticks=ticks)
cbar.ax.set_yticklabels(np.exp(ticks).round())  # vertically oriented colorbar
plt.show()