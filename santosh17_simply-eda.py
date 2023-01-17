import os
import pandas as pd
 
path = '../input/train_1/'
train_events = os.listdir(path)
print('# File sizes')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
print('# File sizes')
count = 0
for f in os.listdir('../input/train_1/'):
    if 'zip' not in f:
        if (count < 20):
           print(f.ljust(30) + str(round(os.path.getsize('../input/train_1/' + f) / 1000000, 2)) + 'MB')
           count = count + 1
len(train_events)
hits_df = pd.read_csv(path+'event000002560-hits.csv')
cells_df = pd.read_csv(path+'event000002560-cells.csv')
particles_df = pd.read_csv(path+'event000002560-particles.csv')
truth_df = pd.read_csv(path+'event000002560-truth.csv')
hits_df.head(5)
cells_df.head(5)
truth_df.head(5)
particles_df.head(5)

