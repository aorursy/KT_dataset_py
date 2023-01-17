import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib as mpl
mpl.style.use('ggplot')

df = pd.read_csv('../input/NationalNames.csv')
def plotname(name,sex):
    df_named=df[(df.Name==name) & (df.Gender==sex)]
    plt.figure(figsize=(12,8))
    plt.plot(df_named.Year,df_named.Count,'g-')
    plt.title('%s name variation over time'%name)
    plt.ylabel('counts')
    plt.xticks(df_named.Year,rotation='vertical')
    plt.xlim([1950,2014])    
    plt.show()    
characters = [('Young','F'),('Young','M')]
plotname('Young','M')
plotname('Megan','F')

