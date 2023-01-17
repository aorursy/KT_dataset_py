#import libraries



from IPython.display import FileLink

import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')
df = pd.DataFrame(np.random.randn(50,3),columns=list('ABC'))

df.head()
def create_download_link(df,filename):

    csv = df.to_csv(filename)

    return FileLink(filename)



create_download_link(df,'df.csv')