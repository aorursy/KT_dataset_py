from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as g; import numpy as n; import os; import pandas as p
feliz = p.read_csv('/kaggle/input/world-happiness-report-2019.csv', delimiter=',')

feliz.dataframeName = 'reporte-felicidad-mundial-2019'

obs, vars = feliz.shape

print(f'Existen {obs} observaciones y {vars} variables')
feliz.head(15)