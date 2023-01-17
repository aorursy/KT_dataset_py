import pandas as pd



#Data importing

edges=pd.read_csv('../input/edges.csv', header=0)

nodes=pd.read_csv('../input/nodes.csv', header=0)

hero_network=pd.read_csv('../input/hero-network.csv', header=0)

#edges.csv data sample:

edges.sample(5,random_state=12228)
#nodes.csv data sample:

nodes.sample(5,random_state=12228)
#hero_network.csv data sample:

hero_network.sample(5,random_state=12228)