#Import neccessary libraries

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
#Importing the data into pandas dataframe
ipl=pd.read_excel('../input/IPL11_FINAL.xlsx')
#Analyze the dataframe
ipl.head()
#ipl.shape
#ipl.columns
#Select only required columns
ipl1=ipl.loc[:,['Match#', 'Date', 'Time (IST)', 'Match', 'WINNER', 'Venue', 'Anand',
       'Paris', 'Megha', 'Veni', 'Priya', 'Ashok', 'G3R', 'Madhan', 'Raj',
       'Thimma', 'Surajit', 'Naveen', 'Raghu', 'Manasa', 'Anil', 'Sridhar',
       'Sanjeeth', 'Anand P', 'Diwakar', 'Manish', 'Murali']]

#Same can be achieved using iloc as shown below
#samp=ipl.iloc[:,:27]
#samp.tail(5)
#Select required rows only

ipl1=ipl1.iloc[:60,:] # Final match was 60th which was played on 27th may 2018

ipl1.columns
#Create a dictionary tag a number to corresponding team

dict= {'RCB':1,'CSK':2,'MI':3,'SRH':4,'KKR':5,'DD':6,'RR':7,'KXI':8}
ipl2= ipl1.iloc[:,6:].replace(dict)

ipl2= ipl2.transpose()
ipl2.tail(5)
#Extract the names, to be used as labels 
names=list(ipl2.index.values)

names
# Calculate the linkage: mergings
mergings = linkage(ipl2,method='complete')


# Plot the dendrogram, using names as labels
dendrogram(mergings,
           labels=names,
           leaf_rotation=60,
           leaf_font_size=10,
)
plt.title('Hierarchical Clustering')
plt.figure(figsize=(200, 60))
plt.show()
