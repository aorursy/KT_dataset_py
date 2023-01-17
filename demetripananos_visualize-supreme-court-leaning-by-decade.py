# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(color_codes = True)

sns.set_style('white')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/scores.csv')

df.head()
r = df.groupby('court_term', as_index = False).agg({

    

    'posterior_mean':'mean',

    'standard_deviation': lambda x: np.sqrt(np.mean(x**2)) #Average the variances

})



std = r.standard_deviation.values

m = r.posterior_mean.values

t = r.court_term.values





T = np.linspace(t.min(), t.max(), 1001)

M = np.interp(T,t,m)

STD = np.interp(T,t,std)



from matplotlib.colors import ListedColormap, BoundaryNorm

from matplotlib.collections import LineCollection









# Create a colormap for red, green and blue and a norm to color

# f' < -0.5 red, f' > 0.5 blue, and the rest green

cmap = ListedColormap(['b',  'r'])

norm = BoundaryNorm([-1, 0, 1], cmap.N)



# Create a set of line segments so that we can color them individually

# This creates the points as a N x 1 x 2 array so that we can stack points

# together easily to get the segments. The segments array for line collection

# needs to be numlines x points per line x 2 (x and y)

points = np.array([T, M]).T.reshape(-1, 1, 2)

segments = np.concatenate([points[:-1], points[1:]], axis=1)



# Create the line collection object, setting the colormapping parameters.

# Have to set the actual values used for colormapping separately.

lc = LineCollection(segments, cmap=cmap, norm=norm)

lc.set_array(M)

lc.set_linewidth(2)



fig1 = plt.figure(figsize = (13,8))

plt.gca().add_collection(lc)

plt.xlim(T.min(), T.max())

plt.ylim(M.min(), M.max())



plt.fill_between(T,M+STD,M-STD, color = 'grey', alpha = 0.2,label = r'Standard Deviation of Estimate')

plt.legend(loc = 4)

plt.title('Mean Leaning')

plt.xlabel('Year')

plt.ylabel('Leaning')