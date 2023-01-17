import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

mdstats = pd.read_excel('/kaggle/input/mock-draft-post-draft-stats/MockDraftPostDraftStats.xlsx')

cm = sns.diverging_palette(125, 20, n=7, as_cmap=True)

mdstats_colored = mdstats.style.background_gradient(cmap=cm)
mdstats_colored