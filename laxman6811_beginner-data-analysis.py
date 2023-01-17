import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb   #seaborn to enhance the visualizations of matplotlib
sb.set()
cereals = pd.read_csv("../input/cereal.csv")
cereals.describe()
cereals.head()
mfr_counts = cereals['mfr'].value_counts()  #product counts for each unique company
mfr_counts.plot(kind='bar')
plt.xlabel("Companies")
plt.ylabel("Products")
plt.title("PRODUCTS DIVERSITY")
#plt.xticks([])  #remove x_labels and x_ticks (not available in seaborn.set),
K_mean=cereals[cereals['mfr'] == 'K']['rating'].mean()
AHFP_mean=cereals[cereals['mfr'] == 'A']['rating'].mean()
GM_mean=cereals[cereals['mfr'] == 'G']['rating'].mean()
Nab_mean=cereals[cereals['mfr'] == 'N']['rating'].mean()
P_mean=cereals[cereals['mfr'] == 'P']['rating'].mean()
Quaker_mean=cereals[cereals['mfr'] == 'Q']['rating'].mean()
RP_mean=cereals[cereals['mfr'] == 'R']['rating'].mean()
plotx=[AHFP_mean, GM_mean, K_mean, Nab_mean, P_mean, Quaker_mean, RP_mean]
plt.bar(cereals.mfr.unique(), plotx)
plt.title('Average rating of products')
plt.xlabel("Company")
plt.ylabel("Mean rating")