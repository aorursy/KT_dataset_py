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
import pandas as pd

data_csv = pd.read_csv("/kaggle/input/temperature-and-flower-status/hirosaki_temp_cherry_bloom.csv")

df = pd.DataFrame(data_csv)



# Split date into year,month,day

dateList = df['date'].str.split('/', expand=True)

df['year'], df['month'], df['day'] = dateList[0], dateList[1], dateList[2]

df.info()
# o:Before blooming

# 1:Bloom

# 2:Full bloom

# 3:Scatter



new_df = []

for i in range(len(df)):

    year, month, day = df['year'][i], df['month'][i], df['day'][i]

    temperature = df['temperature'][i]

    flower_status = df['flower_status'][i]

    if month == '1' and day == '1':

        status = 0

    else:

        if flower_status == 'bloom':

            status = 1

        elif flower_status == 'full':

            status = 2

        elif flower_status == 'scatter':

            status = 3

    innerList = {'year':year, 'month':month, 'day':day, 'temperature':temperature, 'flower_status':status}

    new_df.append(innerList)

new_df = pd.DataFrame(new_df)

new_df
# Extract data from March 1 to May 31

new_df_2 = []

for i in range(len(new_df)):

    month = new_df['month'][i]

    if month == '3' or month == '4' or month == '5':

        innerList = {'month':month, 'day':new_df['day'][i], 'temperature':new_df['temperature'][i], 'flower_status':new_df['flower_status'][i]}

        new_df_2.append(innerList)

new_df_2 = pd.DataFrame(new_df_2)

new_df_2
# Add the cumulative temperature to the column

new_df_3 = []

for i in range(len(new_df_2)):

    month, day = new_df_2['month'][i], new_df_2['day'][i]

    if month == '3' and day == '1':

        temp_accum = 0

    temp = new_df_2['temperature'][i]

    temp_accum += temp

    status = new_df_2['flower_status'][i]

    innerList = {'month':month, 'day':day, 'temperature':temp, 'temp_accum':temp_accum, 'flower_status':status}

    new_df_3.append(innerList)

new_df_3 = pd.DataFrame(new_df_3)

new_df_3
x = pd.DataFrame(new_df_3.drop('flower_status', axis = 1))

t = pd.DataFrame(new_df_3['flower_status'])

x = np.array(x)

t = np.array(t)



t = t.ravel()

x = x.astype('float32')

t = t.astype('int32')



# 中を確認

print('x shape:', x.shape) # (n, m)

print(x[:10])

print('t shape:', t.shape) # (n,)

print(t[:10])
# TupleDataset

from chainer.datasets import TupleDataset

dataset = TupleDataset(x, t)



from chainer.datasets import split_dataset_random

train_val, test = split_dataset_random(dataset, int(len(dataset) * 0.8), seed=0)

train, valid = split_dataset_random(train_val, int(len(train_val) * 0.7), seed=0)



from chainer.iterators import SerialIterator

train_iter = SerialIterator(train, batch_size=32, repeat=True, shuffle=True)



print(dataset[0])
import chainer

import chainer.links as L

import chainer.functions as F



class Net(chainer.Chain):



    def __init__(self, n_in=4, n_hidden=100, n_out=4):

        super().__init__()

        with self.init_scope():

            self.l1 = L.Linear(n_in, n_hidden)

            self.l2 = L.Linear(n_hidden, n_hidden)

            self.l3 = L.Linear(n_hidden, n_out)



    def forward(self, x):

        h = F.relu(self.l1(x))

        h = F.relu(self.l2(h))

        h = self.l3(h)

        return h



net = Net()



from chainer import optimizers

from chainer.optimizer_hooks import WeightDecay

optimizer = optimizers.Adam(alpha=0.0001,

                            beta1=0.9,

                            beta2=0.999,

                            eps=1e-08,

                            eta=1.0,

                            weight_decay_rate=0.00001,

                            amsgrad=False,

                            adabound=False,

                            final_lr=0.1,

                            gamma=0.001)

optimizer.setup(net)



gpu_id = 0

n_epoch = 2500



net.to_gpu(gpu_id)



results_train, results_valid = {}, {}

results_train['loss'], results_train['accuracy'] = [], []

results_valid['loss'], results_valid['accuracy'] = [], []



train_iter.reset()



count = 1



for epoch in range(n_epoch):

    while True:

        train_batch = train_iter.next()

        x_train, t_train = chainer.dataset.concat_examples(train_batch, gpu_id)

        y_train = net(x_train)

        loss_train = F.softmax_cross_entropy(y_train, t_train)

        acc_train = F.accuracy(y_train, t_train)

        net.cleargrads()

        loss_train.backward()

        optimizer.update()

        count += 1



        if train_iter.is_new_epoch:

            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

                x_valid, t_valid = chainer.dataset.concat_examples(valid, gpu_id)

                y_valid = net(x_valid)

                loss_valid = F.softmax_cross_entropy(y_valid, t_valid)

                acc_valid = F.accuracy(y_valid, t_valid)

            loss_train.to_cpu()

            loss_valid.to_cpu()

            acc_train.to_cpu()

            acc_valid.to_cpu()

            if epoch % 10 == 0:

                print('epoch: {}, iteration: {}, loss(train): {:.4f}, loss(valid): {:.4f}, '

                'acc(train): {:.4f}, acc(valid): {:.4f}'.format(

                    epoch, count, loss_train.array.mean(),loss_valid.array.mean(),

                    acc_train.array.mean(), acc_valid.array.mean()))

    

            results_train['loss'].append(loss_train.array)

            results_train['accuracy'].append(acc_train.array)

            results_valid['loss'].append(loss_valid.array)

            results_valid['accuracy'].append(acc_valid.array)



            break

# Graph

import matplotlib.pyplot as plt



# lose

plt.plot(results_train['loss'], label='train')

plt.plot(results_valid['loss'], label='valid')

plt.title('Graph(loss)')

plt.xlabel('epoch')

plt.ylabel('loss')

plt.legend()

plt.show()



# accuracy

plt.plot(results_train['accuracy'], label='train')

plt.plot(results_valid['accuracy'], label='valid')

plt.title('Graph(accuracy)')

plt.xlabel('epoch')

plt.ylabel('accuracy')

plt.legend()

plt.show()
# Calculate loss and accuracy

x_test, t_test = chainer.dataset.concat_examples(test, device=gpu_id)

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

    y_test = net(x_test)

    loss_test = F.softmax_cross_entropy(y_test, t_test)

    acc_test = F.accuracy(y_test, t_test)

print('test loss: {:.4f}'.format(loss_test.array.get()))

print('test accuracy: {:.4f}'.format(acc_test.array.get()))
# Save network

net.to_cpu()

chainer.serializers.save_npz('net.npz', net)



# Check

!ls net.npz
# data from Japan Meteorological Agency (Actual and 2-week forecast)

import pandas as pd

pre_csv = pd.read_csv('/kaggle/input/hirosaki-this-year/hirosaki_this_year.csv')

pre_df = pd.DataFrame(pre_csv)

# Split date into year,month,day

dateList = pre_df['date'].str.split('/', expand=True)

pre_df['year'], pre_df['month'], pre_df['day'] = dateList[0], dateList[1], dateList[2]

pre_df.info()
# Fill in missing values by predicting future temperatures from differences between past average and this year

new_df_4 = []

for i in range(len(new_df)):

    year, month, day, temperature = new_df['year'][i], new_df['month'][i], new_df['day'][i], new_df['temperature'][i]

    # cut Feb 29

    if month == '2' and day == '29':

        dummy = ''

    else:

        innerList = {'year':year, 'month':month, 'day':day, 'temperature':temperature}

        new_df_4.append(innerList)

new_df_4 = pd.DataFrame(new_df_4)

ary_diff = []

for i in range(len(pre_df)):

    m = pre_df['month'][i]

    d = pre_df['day'][i]

    if m == '2' and d == '29':

        dummy = ''

    else:

        if pd.isnull(pre_df['temperature'][i]):

            break

        else:

            # Temperature of this year

            pre_temp = pre_df['temperature'][i]

            # Average temperature of past year(Same month and same day)

            df_m = new_df_4[new_df_4['month'] == m]

            df_m_d = df_m[df_m['day'] == d]

            temp_mean = df_m_d['temperature'].mean()

            # Difference between this year and average

            diff = pre_temp - temp_mean

            ary_diff.append(diff)

ary_diff = pd.DataFrame(ary_diff)

# Overall average of difference

diff_mean = ary_diff.mean()

pre_df_2 = []

for i in range(len(pre_df)):

    y = pre_df['year'][i]

    m = pre_df['month'][i]

    d = pre_df['day'][i]

    if pd.isnull(pre_df['temperature'][i]):

        # Predicted temperature

        df_m = new_df_4[new_df_4['month'] == m]

        df_m_d = df_m[df_m['day'] == d]

        temp_mean = df_m_d['temperature'].mean()





#--------------------------------------------------------

        weight = 1

        # Change as needed

        # Usually 1

#--------------------------------------------------------





        add = int(diff_mean * weight * 1000) / 1000

        temperature = temp_mean + add

    else:

        # Actual temperature

        temperature = pre_df['temperature'][i]

    inner_dic = {'year':y, 'month':m, 'day':d, 'temperature':temperature}

    pre_df_2.append(inner_dic)

pre_df_2 = pd.DataFrame(pre_df_2)

pre_df_2
print('Average temperature rise:', add)
# Add the cumulative temperature to the column

new_pre_df = []

temp_accum = 0

for i in range(len(pre_df)):

    year, month, day = pre_df_2['year'][i], pre_df_2['month'][i], pre_df_2['day'][i]

    if int(month) >= 3:

        temperature = pre_df_2['temperature'][i]

        temp_accum += temperature

        innerList = [month, day, temperature, temp_accum]

        new_pre_df.append(innerList)

new_pre_df = pd.DataFrame(new_pre_df)

new_pre_df
import chainer

import chainer.links as L

import chainer.functions as F



class newNet(chainer.Chain):

    def __init__(self,n_in=4, n_hidden=100, n_out=4):

        super().__init__()

        with self.init_scope():

            self.l1 = L.Linear(n_in, n_hidden)

            self.l2 = L.Linear(n_hidden, n_hidden)

            self.l3 = L.Linear(n_hidden, n_out)



    def forward(self, x):

        h = F.relu(self.l1(x))

        h = F.relu(self.l2(h))

        h = self.l3(h)

        return h



loaded_net = newNet()



chainer.serializers.load_npz('net.npz', loaded_net)



import numpy as np



new_pre_df = np.array(new_pre_df)

new_pre_df = new_pre_df.astype('float32')

with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):

    result = loaded_net(new_pre_df)



# If the status returns or jumps, it is judged as an error

# If the learning rate is low, no results are displayed

count_bud, count_bloom, count_full, count_scatter = 0, 0, 0, 0

date_bloom, date_full, date_scatter = 'none', 'none', 'none'

year = pre_df['year'][0]

for i in range(len(new_pre_df)):

    month = new_pre_df[i][0]

    day = new_pre_df[i][1]

    predict = np.argmax(result[i,:].array)

    if i == 0:

        count_bud += 1

    else:

        pre_predict = np.argmax(result[i - 1,:].array)

        if predict != pre_predict:

            if predict == 0:

                count_bud += 1

            elif predict == 1:

                count_bloom += 1

                date_bloom = '{}/{}/{}'.format(int(year), int(month), int(day))

            elif predict == 2:

                count_full += 1

                date_full = '{}/{}/{}'.format(int(year), int(month), int(day))

            elif predict == 3:

                count_scatter += 1

                date_scatter = '{}/{}/{}'.format(int(year), int(month), int(day))

#    print(predict)

if count_bud > 1:

    print('ERROR !! (Over count "Before blooming :', count_bud, '")')

if count_bloom > 1:

    print('ERROR !! (Over count "Bloom :', count_bloom, '")')

if count_full > 1:

    print('ERROR !! (Over count "Full :', count_full, '")')

if count_scatter > 1:

    print('Error !! (Over count "Scatter :', count_scatter, '")')

if count_bud == 1 and count_bloom == 1 and count_full == 1 and count_scatter == 1:

    ratio_loss = loss_test.array.get()

    ratio_acc = acc_test.array.get()

    

    # Set accuracy threshold

    specified_loss = 0.1

    specified_acc = 0.95

    

    if ratio_loss < specified_loss and ratio_acc > specified_acc:

        print('Congratulations, the prediction is successful !!')

        print('Bloom  :', date_bloom)

        print('Full   :', date_full)

        print('Scatter:', date_scatter)

        # Time stamp

        import time, datetime

        today = datetime.datetime.fromtimestamp(time.time())

        print(today.strftime('Time stamp: %Y/%m/%d %H:%M:%S (UTC)'))

    else:

        print('Low learning rate !!')

        print('Accuracy rate :', ratio_acc, '(Specified rate :', specified_acc, ')')

        print('Loss rate :', ratio_loss, '(Specified rate :', specified_loss, ')')

else:

    print('ERROR !! (Missing status)')