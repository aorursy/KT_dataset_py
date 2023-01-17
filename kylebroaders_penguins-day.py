import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
penguins = sns.load_dataset("penguins")
penguins.head()
fig = sns.scatterplot(data=penguins, x='bill_length_mm', y='bill_depth_mm', hue='species')
# Save figure
out=fig.get_figure()       # Get the figure container around our plot
plt.rc('pdf', fonttype=42) # Set the font so that we can edit it in Illustrator
out.savefig('scatter.pdf')
sns.violinplot(data=penguins,x='species', y='bill_length_mm')
plt.figure()
sns.swarmplot(data=penguins,x='species', y='bill_length_mm')
# Looking to compare above plot with island of origin
fig = sns.scatterplot(data=penguins, x='bill_length_mm', y='bill_depth_mm', hue='island')

# Set axis labels and title
plt.xlabel("Bill Depth (mm)")
plt.ylabel("Bill Length (mm)")
plt.title("Penguin bill dimensions")
# Checking if there's any difference between islands for Adelie penguins
sns.pairplot(penguins.query("species == 'Adelie'"), hue='island')
g=sns.FacetGrid(penguins,col='island',hue='species')
g.map(sns.scatterplot, "bill_length_mm", "bill_depth_mm")
sns.scatterplot(data=penguins, x='species', y='island', marker="o")
