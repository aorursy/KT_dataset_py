from keras.losses import binary_crossentropy, categorical_crossentropy
import keras.backend as K
import numpy as np
from prettytable import PrettyTable
from prettytable import ALL
from sklearn.metrics import f1_score, accuracy_score
from matplotlib import pyplot as plt
# ground truth
Y0 = np.zeros((12,3))
# first label is assigned to 20 % of observations
Y0[0:4,0] = 1
# # second label is assigned to 80 % of observations
Y0[4:8,1] = 1
Y0[8:,2] = 1
# Y[3,3] = 1
# Y[4,4] = 1
# ground truth with shape (BATCH_SIZE, NO_OF_LABELS)
print(Y0)

res = []
def dfs(tmp,d):
    if d == 3:
        res.append(tmp)
        return
    for i in np.arange(0,1.1,0.1):
        dfs(tmp+' '+str(i),d+1)
dfs("",0)
print(res)
import torch
import torch.nn.functional as F
import random
pred = []
for p in res:
    p = p.split()
    for i in range(len(p)):
        p[i] = round(float(p[i]),2)
    pred.append(p)
print(pred)
print(np.array(random.sample(pred, 12)))
print(Y0)
loss = F.cross_entropy(torch.from_numpy(np.array([p])), torch.tensor(np.array([0])))
from plotly import graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
def acc_loss(y_true, y_pred):
    # y_pred = y_pred.round()
    tp = (y_pred*y_true).sum(0)
#     print(y_pred*y_true)
    fp = ((1-y_true)*y_pred).sum(0)
    fn = (y_true*(1-y_pred)).sum(0)
    tn = ((1-y_true)*(1-y_pred)).sum(0)
    acc = (tp+tn)/(tp+fp+fn+tn)
    return 1-acc.mean()
losses1 = []
losses2 = []
acces_loss = []
acces = []

for i in range(100):
    Y = torch.from_numpy(Y0)
#     target = torch.randint(2, (10,), dtype=torch.int64)
    p = torch.from_numpy(np.array(random.sample(pred, 12)))
#     print(F.softmax(p, 1))
#     print(Y)
    loss = F.cross_entropy(p, Y.argmax(1).long())
    loss_of_acc = acc_loss(Y,F.softmax(p, 1))
    acc = accuracy_score(p.argmax(1).long(),Y.argmax(1).long())
    losses2.append(loss.numpy().tolist())
    acces_loss.append(loss_of_acc.tolist())
    acces.append(acc.tolist())
# print(acces)
# print(losses2)
plt.scatter(acces_loss, losses2)
plt.xlabel('Acc loss')
plt.ylabel('Cross-Entropy Loss')
plt.show()
plt.scatter(acces, losses2)
plt.xlabel('Acc')
plt.ylabel('Cross-Entropy Loss')
plt.show()
