class Datos:
    def __init__(self, *d):
        self.datos = d
    def __del__(self):
        print(self.datos)
import numpy as np
np.eye(3)
np.random.randn(25)
mat = np.arange(1,26).reshape(5,5)
mat
mat.std()
import pandas as pd
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                         'C': ['C0', 'C1', 'C2', 'C3'],
                         'D': ['D0', 'D1', 'D2', 'D3']})
right
pd.merge(left, right, on=['key1', 'key2'])
df = pd.DataFrame({'A':[1,2,np.nan],
                  'B':[5,np.nan,np.nan],
                  'C':[1,2,3]})
df
df.fillna(value='FILL VALUE')
