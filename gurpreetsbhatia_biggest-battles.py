import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Plotting graph
# Read battles.csv
file = '../input/battles.csv'
data = pd.read_csv(file)

# Compute battle size and add as new column
data['battle_size']=data['attacker_size'] + data['defender_size']

# Clean some data
data['defender_king']=data['defender_king'].fillna('NA')
data['attacker_king']=data['attacker_king'].fillna('NA')

# Subset Data
data_subset=data[['attacker_king','defender_king','battle_size']]
# Plot Graph
Xunique, X = np.unique(data_subset['attacker_king'], return_inverse=True)
Yunique, Y = np.unique(data_subset['defender_king'], return_inverse=True)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

#Size of plots
S=data_subset['battle_size']/20
ax.scatter(X, Y,s=S, c='Red',marker='o', alpha=.2)

# Set plot ticks
ax.set(xticks=range(len(Xunique)), xticklabels=Xunique,
       yticks=range(len(Yunique)), yticklabels=Yunique)

# Set graph size
fig.set_size_inches(10.5, 5.5)

# Set Colors
ax.set_axis_bgcolor('grey')
ax.tick_params(axis='x', colors='red')
ax.tick_params(axis='y', colors='green')
fig.suptitle('Battles', fontsize=30, color='blue')
plt.xlabel('Attacker',fontsize=20,color='red')
plt.ylabel('Defender',fontsize=20,color='green')

# Show Plot
plt.show() 