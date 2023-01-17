# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
import matplotlib
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/kaggle-competions-rankings-and-kernels/competitions.csv")
data.head()
data.evaluationMetric.value_counts().to_frame().reset_index().rename(columns={"index":"evaluation_metric","evaluationMetric":"counts"})
data.hostSegment.value_counts().to_frame().reset_index().rename(columns={"index":"hostSegment","hostSegment":"counts"})
data[data["rewardTypeName"] == "USD"].sort_values(by="rewardQuantity",ascending=False).head(10)[["competitionName","enabledDate","rewardQuantity","totalTeams"]]
## More to go...
