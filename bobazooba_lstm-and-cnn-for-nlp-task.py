# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch
x = torch.rand(128, 64, 1024)
x.shape
# первый способ

lstm = torch.nn.LSTM(1024, 512, batch_first=True)



pred, mem = lstm(x)
pred.shape
# эмбеддинг последнего слова

# обратите внимание, что в случае с bidirectional моделью размер эмбеддинга после LSTM будет не 512, а 1024

# то есть конкатенация прохода LSTM слева направо и наоборот

# это будут две разные LSTM

pred[:, -1, :].shape
# второй способ использования LSTM

lstm = torch.nn.LSTM(1024, 512)



# меняем размерность batch и seq_len местами

x_transposed = x.transpose(0, 1)

pred_transposed, mem = lstm(x_transposed)
# у нас все еще осталась размерность (seq_len, batch, input_size)

pred_transposed.shape
# просто транспонируем еще раз

pred = pred_transposed.transpose(0, 1)

pred.shape
x.shape
# in_channels - размер входных эмбеддингов

# out_channels - количество/какой размер эмбеддингов мы хотим получить

# kernel_size - размер окна/н-граммы

cnn = torch.nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3)
# выпадет ошибка, посмотрите какая

# pred = cnn(x)
x_transposed = x.transpose(1, 2)

x_transposed.shape

# перевели в (batch, input_size, seq_len)
pred_transposed = cnn(x_transposed)

pred_transposed.shape

# осталась размерность (batch, output_size, seq_len)
# переведем обратно в (batch, seq_len, input_size)

pred = pred_transposed.transpose(1, 2)

pred.shape