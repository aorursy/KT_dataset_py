import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
stats = pd.read_csv('../input/league-of-legends-soloq-ranked-games/lol_ranked_games.csv')
# Drop duplicated if any (gameId+frame)
stats.drop_duplicates(subset=['gameId', 'frame'], keep='first', inplace=True)
stats.head(20)
def show_values_on_bars(axs):
    def _show_on_single_plot(ax):        
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = f'{p.get_height():d}'
            ax.text(_x, _y, value, ha="center", fontdict={'size': 12, 'weight': 'bold'}) 

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
inspect_df = stats[stats['frame'] == 10]
tmp_palette = ['#ef233c', '#36827f']
ax = sns.countplot(inspect_df['hasWon'], palette=tmp_palette)
ax.set(title='LOSS/WIN Games',
       xlabel='Outcome',
       ylabel='Total',
       xticklabels=['LOSS', 'WIN'])

show_values_on_bars(ax)