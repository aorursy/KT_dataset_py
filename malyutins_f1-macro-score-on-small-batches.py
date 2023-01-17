import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
num_classes = 28

df = pd.read_csv('../input/train.csv')

y_true = np.zeros((len(df), num_classes))

for i, row in df.iterrows():
    for lblIndex in row['Target'].split():
        y_true[i][int(lblIndex)] = 1
        
print(y_true.shape)
print(f1_score(y_true, y_true, average='macro'))
for batch_size in [64, 32, 16]:
    print("Batch size:", batch_size, "F1 macro:", f1_score(y_true[:batch_size], y_true[:batch_size], average='macro'))
