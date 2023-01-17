import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import ceil
%matplotlib inline
N_COLS = 5
PLOT_SIZE = 3

def show_digit(row_id, row, ax):
    
    ax.imshow(row['pixel0':].reshape(28, 28), cmap='gray_r')
    ax.set_title('row_id: %d label: %d' % (row_id, row['label']))
    
def show_digits(x):
    
    num_digits = x.shape[0]
    num_rows = int(ceil((num_digits / N_COLS)))
    
    fig, axes = plt.subplots(num_rows, N_COLS, figsize=(PLOT_SIZE*N_COLS, PLOT_SIZE*num_rows))
    
    for ax, (row_id, row) in zip(axes.ravel(), x.iterrows()):
        
        show_digit(row_id, row, ax)
train = pd.read_csv('../input/train.csv')
show_digits(train.loc[1030:1054, :])
