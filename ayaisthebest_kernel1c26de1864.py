import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go


import os
print(os.listdir("../input"))
data = pd.read_csv("../input/datafile.csv")

wheat = (data[data.Crop == 'Wheat']).values
#print(wheat)
labels = ['1951-1968', '1968-1981', '1981-1991'
          , '1991-1997', '1997-2002', '2002-2007', '2007-2012']
#plt.plot(wheat,labels)
#plt.show()
#d = [
#    go.Scatter(
#        x=data['Wheat'], # assign x as the dataframe column 'x'
#        y=labels
#    )
#]
wheat=np.delete(wheat, 0)
print(wheat.size, type(wheat))
plt.plot(labels, wheat)
plt.show()