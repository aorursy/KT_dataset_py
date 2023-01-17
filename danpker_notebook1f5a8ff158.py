import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
players = pd.read_csv('../input/player.csv')

players['birth_year'].plot()