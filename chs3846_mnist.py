import pandas as pd
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import random

train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
x_train = train.drop('label', axis=1)
x_train = torch.FloatTensor(x_train.values)

y_train = train['label']
y_train = torch.LongTensor(y_train.values)


dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
training_epochs = 50
model = nn.Linear(784, 10, bias=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(dataloader)

    for X, Y in dataloader:
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')
x_test = torch.FloatTensor(test.values)
answer = torch.argmax(model(x_test), 1)

submission = pd.DataFrame(answer.numpy(), columns=['Label'])
submission.index = range(1,len(submission)+1)
submission.to_csv("./submission.csv", index_label = 'ImageId')
