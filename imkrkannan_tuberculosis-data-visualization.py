from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
%matplotlib inline
df = pd.read_csv(r"../input/TB_burden_age_sex_2020-06-14.csv")
df.head()
df.info()
# This fuctions is used a few times to display bar values in barplots
def show_values_on_bars(axs, *args, **kwargs):
    """
    Function based on Sharon Soussan answer for:
    https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values
    This function will help display information over some plots.
    """

    def _show_on_single_plot(ax, *args):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            if 'height' in kwargs:
                _y = kwargs['height']
            else:
                _y = 10
            value = f'{p.get_height()}'
            ax.text(_x, _y, value, fontsize=14, ha="center") 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs, *args):
            _show_on_single_plot(ax, *args)
    else:
        _show_on_single_plot(axs, *args)
# Setting the style for my plots
sns.set_style(style='ticks')
df['sex'].value_counts()
sex_palette = sns.color_palette(["#4287f5", "#bd3c3c"])
pclass_palette = sns.color_palette(["#FFDF00", "#c0c0c0", "#cd7f32"])
survived_palette = sns.color_palette(["#4e5245", "#57916c"])

fig = plt.figure(figsize=(16,8), constrained_layout=True)
gs = gridspec.GridSpec(nrows=3, ncols=4, figure=fig)


# Male/Female Totals
ax1 = fig.add_subplot(gs[0, 0:2])
sns.countplot(x='sex', data=df, palette=sex_palette, edgecolor=sns.color_palette(["#000"]), alpha=0.5)
show_values_on_bars(ax1)
plt.ylabel('risk_factor')





df['age_group'].value_counts()

df['age_group'].value_counts()
df['sexB'] = df['sex'].map({'male': 1,'female': 0})
df['EmbarkedNum'] = df['country'].map({'S': 0,'C': 1, 'Q': 2})
sns.heatmap(df.corr(), cmap='Pastel1')
plt.title('Correlation', fontsize=24)
df['country'].value_counts()