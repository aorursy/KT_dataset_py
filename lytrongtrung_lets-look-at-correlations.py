#Ignore the seaborn warnings.
import warnings
warnings.filterwarnings("ignore");

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/primary_results.csv')
NH = df[df.state == 'New Hampshire']
for i in range(1,5):
    print(i)
    
    


