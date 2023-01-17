%matplotlib inline
import matplotlib.pylab
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
%pylab inline

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm_notebook
import ast

sns.set_style("white")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
train_path = '../input/train_simplified/'
files = os.listdir(train_path)
categories = [category.split('.')[0] for category in files]
print('Total number of categories: ',len(categories))
print('Few Example Categories',categories[0:5])
train_data = pd.DataFrame()
for file in tqdm_notebook(files):
    train_data = train_data.append(pd.read_csv(train_path + file, index_col='word', nrows=10))    
train_data.sample(10)
train_data = train_data.reset_index()
train_data['word_count'] = train_data.groupby('word')['word'].transform('count')
sns.distplot(train_data['word_count'],kde=False)
plt.title('Word Count Distribution in Train Set')
if train_data.index.name is not 'word':
    train_data = train_data.set_index('word')
    
img_ar = None
for cat in tqdm_notebook(categories):
    df = train_data[train_data.index==cat]
    drawings = [ast.literal_eval(pts) for pts in df[:9]['drawing'].values]

    fig = Figure()
    ax = fig.subplots(1,9)
    canvas = FigureCanvas(fig)
    for i, drawing in enumerate(drawings):
        for x,y in drawing:
            ax[i].plot(x, y, marker='.')
            ax[i].axis('off')
    fig.suptitle(cat,fontsize=30)
#     plt.show()
    canvas.draw()       # draw the canvas, cache the renderer
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi() 
    img = image.reshape(int(height), int(width), 3)
    img = np.expand_dims(img,axis=0)
    if img_ar is None:
        img_ar = img
    else:
        img_ar = np.concatenate([img_ar,img],axis=0)
DataRange = (np.absolute(img_ar)).max() 
EXTENT = [0, width, 0 ,height]
NORM = matplotlib.colors.Normalize(vmin =-DataRange, vmax= DataRange, clip =True)

grid_width = 20
grid_height = len(categories)//grid_width
fig,axs = plt.subplots(grid_height,grid_width,figsize=(img_ar.shape[1], img_ar.shape[2]))
for i in range(len(categories)):
    ax = axs[int(i / grid_width), i % grid_width]
    ax.imshow(img_ar[i], norm = NORM, extent = EXTENT, aspect = 1, interpolation='none')
    ax.axis('off')

plt.show()
plt.imshow(img_ar[0])