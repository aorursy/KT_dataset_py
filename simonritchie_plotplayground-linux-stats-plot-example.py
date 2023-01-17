!pip install plot_playground
!pip install gpustat==0.5.0
from plot_playground.stats import linux_stats_plot
linux_stats_plot.display_plot()
import time

import os



import pandas as pd

import numpy as np

import torch



df = pd.DataFrame(columns=['a', 'b', 'c'], index=np.arange(0, 50000000))

time.sleep(5)

tensor_1 = torch.zeros((1000000, 5), device='cuda')

time.sleep(5)

df.to_pickle('test_1.pkl')

time.sleep(10)

tensor_2 = torch.zeros((1000000, 10), device='cuda')

time.sleep(5)

df.to_pickle('test_2.pkl')

time.sleep(10)

os.remove('test_1.pkl')

time.sleep(5)

os.remove('test_2.pkl')



df_2 = pd.DataFrame(columns=['a', 'b', 'c'], index=np.arange(0, 30000000))

time.sleep(5)

tensor_3 = torch.zeros((2000000, 10), device='cuda')

time.sleep(5)

df_2.to_pickle('test_3.pkl')

time.sleep(5)