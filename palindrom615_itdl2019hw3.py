!mkdir dataset && tar -xzf /kaggle/input/cifar10-python/cifar-10-python.tar.gz -C dataset
import matplotlib.pyplot as plt

import itdl2019_hw3_cnn as cnn

import torch

import torch.nn.functional as F



# constants

LEARNING_RATE = 1e-3

BATCH_SIZE = 96

EPOCH_NUM = 50



# model & data loader

model = cnn.Model()

loader = cnn.DataLoader()

model.set_learning_rate(LEARNING_RATE)

loader.set_batch_size(BATCH_SIZE)
from tqdm.notebook import tqdm



model.set_learning_rate(LEARNING_RATE)

loader.set_batch_size(BATCH_SIZE)



# training

batch_num = loader.get_batch_num()

cost_list = []

for epoch_idx in tqdm(range(EPOCH_NUM)):

    cumcost = 0.0

    for idx in range(batch_num):

        x, y = loader.get_batch()

        cumcost += model.train_step(x, y)

    cumcost /= batch_num

    print('[%d] loss: %.3f' % (epoch_idx + 1, cumcost))

    cost_list.append(cumcost)

    loader.reset()



# cost_list plot

plt.plot(cost_list)

plt.show()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



testloader = cnn.DataLoader(is_train=False)

testloader.set_batch_size(32)

cost = torch.nn.MSELoss()



with torch.no_grad():

    correct = 0

    total = 0

    cumcost = 0.0

    for idx in range(testloader.get_batch_num()):

        x, y = testloader.get_batch()

        hx = model.inference(x)

        predicted = torch.argmax(hx, dim=1)

        total += y.size(0)

        correct += (predicted == y).sum().item()

        cumcost += cost(hx, F.one_hot(

    y, num_classes=10).to(device, dtype=torch.float32))

print('Accuracy of the network on the 10000 test images: %d %%' % (

100 * correct / total))

print('Loss of the network on the 10000 test images: %.3f' % (cumcost))



model.save()