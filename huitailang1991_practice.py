# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
kobe_df = pd.read_csv('../input/data.csv')
kobe_df[:5]
# the percentage of different combined_shot_type
kb_comb = kobe_df[['combined_shot_type','shot_made_flag']]
kb_comb = kb_comb.set_index('combined_shot_type')
plt.plot(kb_comb)
