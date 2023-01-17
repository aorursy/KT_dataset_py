from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
nRowsRead = 1000 # specify 'None' if want to read whole file
# indiana_projections.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/indiana_projections.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'indiana_projections.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
nRowsRead = 1000 # specify 'None' if want to read whole file
# indiana_reports.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/indiana_reports.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'indiana_reports.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
import matplotlib.pyplot as plt
from PIL import Image
import os

n_pics_to_show = 20
ROWS = 4
COLUMNS = 5
directory = r'/kaggle/input/images/images_normalized/'
fig, ax = plt.subplots(ROWS, COLUMNS, figsize=(20, 10))

for i,filename in enumerate(os.listdir(directory)):
    if filename.endswith(".png") and i<n_pics_to_show:
        pic = Image.open(os.path.join(directory, filename))
        filename = filename.replace('.dcm.png','')
        pic_np = np.array(pic)
        ax[i%ROWS][i%COLUMNS].imshow(pic_np)
        ax[i%ROWS][i%COLUMNS].set_title(f'{filename}')
        ax[i%ROWS][i%COLUMNS].axis('off')
    else:
        break