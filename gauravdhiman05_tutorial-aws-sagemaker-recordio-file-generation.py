!pip install sagemaker
import pandas as pd
import numpy as np
import sagemaker.amazon.common as smac
n = 10

x1 = np.random.random_sample(n)       # n floating point numbers between 0 and 1
x2 = np.random.randint(100,200,n)     # n integers
x3 = np.random.random_sample(n) * 10  # n floating point numbers between 0 and 10
y = np.random.randint(0,2,n)          # Response variable 0 or 1
df = pd.DataFrame({'x1':x1,
              'x2':x2, 
              'x3':x3,
              'y':y})
# X must be an array
X = df[['x1','x2','x3']].to_numpy()
X
type(X)
# Response/Target variable needs to a vector
# y must be a vector 
y = df[['y']].to_numpy()
y.shape
y
# Flatten to a single dimension array of 10 elements
y = y.ravel()
y
def write_recordio_file (filename, x, y=None):
    with open(filename, 'wb') as f:
        smac.write_numpy_to_dense_tensor(f, x, y)
        
def read_recordio_file (filename, recordsToPrint = 10):
    with open(filename, 'rb') as f:
        record = smac.read_records(f)
        for i, r in enumerate(record):
            if i >= recordsToPrint:
                break
            print ("record: {}".format(i))
            print(r)
write_recordio_file('demo_file.recordio',X,y)
read_recordio_file('demo_file.recordio')
