# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install kaggle
from google.colab import files
files.upload()
!ls -lha kaggle.json

!pip uninstall -y kaggle
!pip install --upgrade pip
!pip install kaggle==1.5.6
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/.kaggle.json
!kaggle -v

!kaggle competitions download -c star-classifier
!unzip star-classifier.zip
import torch.optim as optim
import torch

torch.cuda.is_available()
torch.manual_seed(1)

import numpy as np 
import torch
import pandas as pd

pd_train=pd.read_csv('star_train.csv')
x_train=np.array(pd_train.iloc[:,1:7])
y_train=np.array(pd_train.iloc[:,7])

x_train=torch.FloatTensor(x_train)
y_train=torch.LongTensor(y_train)

print(x_train.shape)
print(y_train.shape)

print(x_train)
print(y_train)
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

nb_class = 6
nb_data=len(y_train)

W = torch.zeros((6, 6), requires_grad=True)
b = torch.zeros(6, requires_grad=True)

optimizer = optim.SGD([W, b], lr=1e-1,momentum=0.8)
nb_epochs = 10000

for epoch in range(nb_epochs + 1):

    cost = F.cross_entropy(x_train.matmul(W)+b,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
test = pd.read_csv('star_test.csv')
x_test = np.array(test.iloc[:,1:7])

x_test=torch.FloatTensor(x_test)

print(x_test.shape)
print(x_test)
hypo = F.softmax(x_test.matmul(W)+b,dim=1)
predict=torch.argmax(hypo, dim=1)

pd_correct=pd.read_csv('star_solution.csv')
answer = torch.LongTensor(np.array(pd_correct.iloc[:,1]))

correct_predict=predict==answer

accuracy=correct_predict.sum().item() / len(correct_predict)
print('accuracy : {:.6f}'.format(accuracy))
id = np.array([i for i in range(len(x_test))]).reshape(-1,1)
label=predict.detach().numpy().reshape(-1,1)
result = np.hstack((id,label))

df=pd.DataFrame(result,columns=('Id','Label'))
df.to_csv('baseline.csv',index=False,header=True)