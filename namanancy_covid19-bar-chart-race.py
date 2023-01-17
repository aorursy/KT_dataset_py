# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import emoji
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')
!pip install colour
print('setup complete')
HTML('''<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/1837728" data-url="https://flo.uri.sh/visualisation/1837728/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
HTML('''<div class="flourish-embed flourish-map" data-src="visualisation/1839508" data-url="https://flo.uri.sh/visualisation/1839508/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')