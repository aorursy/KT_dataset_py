
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
q = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

q

y = train['label']
print(np.array(y))
x = []
for i in range(len(train)):
    x.append(np.array(train.loc[i][1:]) / 255)
print(x[0])

x_test = []
for i in range(len(test)):
    x_test.append(np.array(test.loc[i][:]) / 255)

print(len(x_test[0]))
yy = [np.zeros(10) for _ in range(len(train))]
for i in range(len(train)):
    yy[i][y[i]] = 1
y = yy
from neurnet import NeurNet
from nettraining import BackProp
net = NeurNet(28*28)
net.addLayer(200, 'sigm')
net.addLayer(10, 'sigm')
train_set = list(zip(x, y))
train = BackProp(net, train_set, 1)
def pc():
    l = len(test)
    ok = 0
    for i in range(l):
        a = net.getOutput(x[i])
        am = a.max()
        if sum(abs(a // am - y[i])) == 0:
            ok += 1
    return 100 * ok / l
train.current_iteration_number = 0
train.train(2, stop_condition=lambda train: train.current_iteration_number > 0 or train.errors[-1] < 100 or 
              len(train.errors) > 1 and abs(train.errors[-1] - train.errors[-2]) < 0.1, print_errors=True)
ok = 0
for i in range(len(x)):
    a = net.getOutput(x[i])
    am = a.max()
    if sum(abs(a // am - y[i])) == 0:
        ok += 1
print(f'{100 * ok / len(x):2f}')
ans = []
for i in range(len(test)):
    ans.append(net.getOutput(x_test[i]).argmax())
print(ans)
q = pd.DataFrame(q)
q.to_csv('/kaggle/input/ans.csv')
