# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

from scipy.stats import probplot # for a qqplot

import matplotlib.pyplot as plt # for a qqplot

import pylab



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cereal_df = pd.read_csv('../input/cereal.csv')
cereal_df.head()
# plot a qqplot to check normality. If the varaible is normally distributed, most of the points 

# should be along the center diagonal.

probplot(cereal_df["sugars"], dist="norm", plot=pylab)
cold = cereal_df['sugars'][cereal_df['type']=='C']

hot = cereal_df['sugars'][cereal_df['type']=='H']
hot[57] = 0

hot
cold
ttest_ind(hot,cold,axis = 0,equal_var = False)
#start with cold

plt.hist(cold,color = 'g',edgecolor = 'black',alpha = .5,label = 'cold')

#then warm

plt.hist(hot,color = 'b',edgecolor = 'black',label = 'hot')

#add a legend

plt.legend(loc='upper-right')

#and a title, and labels

plt.title('Sugar(g) content for Hot & Cold cereals')




