# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import csv
import numpy as np
import pybrain as pb
from pybrain.structure import *
from pybrain.tools import *
from pybrain.supervised import *
from pybrain.datasets import *
from pybrain.tools.shortcuts import *
train_size = 2000

train_file = open('../input/train.csv', 'r')
train = list(csv.reader(train_file))
header_train = train[0]
train = np.array(train[1:train_size+1])
#train = np.array(train[1:])
output = train[:,0]
input = train[:,1:]
_output = np.array([[0]*10])[[0]*output.shape[0]]
for i in enumerate(_output):
    i[1][output[i[0]].astype(int)] = 1
output = _output
ds = SupervisedDataSet(input, output)

hidden_size = 5
net = buildNetwork(ds.getDimension('input'), hidden_size, ds.getDimension('target'), bias=True)

trainer = BackpropTrainer(net, ds)

trainer.trainUntilConvergence(verbose=False, maxEpochs = 100, continueEpochs = 10 )

test_file = open('../input/test.csv', 'r')
test = list(csv.reader(test_file))
header_test = test[0]
test = np.array(test[1:])
prediction_file = open("model1.csv", "w")
# pred_f = csv.writer(prediction_file, escapechar='', quotechar='', quoting=csv.QUOTE_NONE)

# pred_f.writerow(['"ImageId"', '"Label"'])

ok = 0
for i in range(len(test)):
    pred = np.argmax(net.activate(input[i]))
    act = np.argmax(output[i])
    if pred == act:
        ok += 1

#    pred = np.argmax(net.activate(test[i]))
#    pred_f.writerow([i+1, pred])

print(ok/train_size)

