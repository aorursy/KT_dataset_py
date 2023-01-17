import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

cites = pd.read_csv("../input/comptab_2018-01-29 16_00_comma_separated.csv")
fig_kwargs = {'figsize': (12, 6), 'fontsize': 16}

cites.head(3)
cites['Year'].value_counts()
cites['App.'].value_counts().plot.bar(**fig_kwargs, 
                                      title="Wildlife Trade by CITES Appendix")
cites['Class'].value_counts().plot.bar(**fig_kwargs, 
                                      title="Wildlife Trade by Species Class")
cites['Genus'].value_counts().head(20).plot.bar(
    **fig_kwargs, 
    title="Top Wildlife Trade by Species Genus"
)
(pd.DataFrame()
     .assign(Exports=cites.Exporter.value_counts(), 
             Imports=cites.Importer.value_counts())
     .pipe(lambda df: df.loc[df.fillna(0).sum(axis='columns').sort_values(ascending=False).index])
     .head(10)
     .plot.bar(**fig_kwargs, title="Top 10 Import/Export Countries")
)
cites['Origin'].value_counts().head(10).plot.bar(**fig_kwargs, title="Top 10 Species Origins")
cites['Term'].value_counts().tail(10)
cites['Term'].value_counts().head(20).plot.bar(**fig_kwargs, title="Top 20 Products Traded")
cites['Purpose'].value_counts().plot.bar(**fig_kwargs, title="Import/Export Purposes")
cites['Source'].value_counts().plot.bar(**fig_kwargs, title="Import/Export Sources")