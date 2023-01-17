import pandas as pd

# Plotting

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

sns.set(font_scale=1,rc={'xtick.color':'white','ytick.color':'white','axes.edgecolor':'white','axes.labelcolor':'white','axes.facecolor':'grey', 'figure.facecolor':'grey','text.color':'white'})
# Load the data:

exo_df = pd.read_csv('../input/oec.csv')

exo_df.head()
# Calculate the luminosity of each host star

def calcLuminosity(radius,Teff):

    TeffSlr = 5777

    return (radius**2)*((Teff/TeffSlr)**4)
exo_df['HostStartSlrLum'] = calcLuminosity(exo_df['HostStarRadiusSlrRad'],exo_df['HostStarTempK'])

exo_df.head()



y_min = 10**-5

y_max = 10**6



fig, ax = plt.subplots()

points = plt.scatter(exo_df["HostStarTempK"], exo_df["HostStartSlrLum"],c=exo_df["HostStarTempK"], s=100, cmap="coolwarm_r")

plt.colorbar(points)

fig.set_size_inches(10, 8)

ax = sns.regplot(x='HostStarTempK',y='HostStartSlrLum',data=exo_df,fit_reg=False,scatter_kws={'alpha':0.0})

ax.invert_xaxis()

ax.set_yscale('log')

ax.set_ylim(y_min,y_max)

plt.show()
exo_df[exo_df['HostStarTempK'] > 25000].iloc[:,[0,2,4,19,22,23,24]]
