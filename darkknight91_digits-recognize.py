import os
import os.path
import random
import time
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from numpy.random import randint
import pickle

pd.set_option('max_columns',None)
pd.options.display.width = 2000
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

mpl.rcParams['agg.path.chunksize'] = 10000
%matplotlib inline
stime = time.time()
print(os.listdir("../input"))

update = True

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_sub = pd.read_csv("../input/sample_submission.csv")

mat = df_train.values
x_test = df_test.values
del df_train
del df_test
print("Training set",mat.shape)
print("Test set",x_test.shape)

print("Using MLPClassifier...")
# x_train,x_test = train_test_split(mat,train_size=0.7)
# y_train = x_train[:,0]
# x_train = x_train[:,1:]
# y_test = x_test[:,0]
# x_test = x_test[:,1:]
y_train = mat[:,0]
x_train = mat[:,1:]

print(x_train.shape,x_test.shape)
clf = None
try:
    clf = pickle.load(open("clf", 'rb'))
except:
    print("No pre-trained model found")
    
if(update or clf is None):
    print("Training...")
    clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(1000), random_state=1,max_iter=300,verbose=True)
    clf.fit(x_train,y_train)
    pickle.dump(clf, open("clf", 'wb'))

y1 = clf.predict(x_test)
y0 = range(1,len(y1)+1)
df_sub["ImageId"] = y0
df_sub["Label"] = y1
df_sub.to_csv("submission.csv",index=False)
#pickle.dump("submission.csv", open("submission.csv", 'wb'))
#print("success rate",(np.count_nonzero(y==y_test)/len(y))*100)
# fig,ax = plt.subplots(10,3)
# plt.title("Using MLPClassifier")
# fig.set_figheight(20)
# fig.set_figwidth(20)
# fig.tight_layout()
# count = 0
# for row in ax:
#         for subplot in row:
#             subplot.imshow(np.reshape(x_test[count],(28,28)))
#             subplot.set_title("Real:"+str(y_test[count])+",Guess:"+str(y[count]))
#             count = count + 1

# plt.show()

#display(df_sub)
print("Total Runtime = ",(time.time()-stime))

# Any results you write to the current directory are saved as output.
