import os



import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_validation_split = int(train_df.shape[0] * 0.8)

train_data = torch.tensor(train_df.iloc[:train_validation_split, 1:].values, dtype=torch.float, device=device)

train_labels = torch.tensor(train_df['label'][:train_validation_split].values, dtype=torch.float, device=device)

validation_data = torch.tensor(train_df.iloc[train_validation_split:, 1:].values, dtype=torch.float, device=device)

validation_labels = torch.tensor(train_df['label'][train_validation_split:].values, dtype=torch.float, device=device)

test_data = torch.tensor(test_df.iloc[:, :].values, dtype=torch.float, device=device)
class Net(nn.Module):



    def __init__(self):

        super(Net, self).__init__()



        self.dropout = nn.Dropout2d(p=0.5)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3))

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3))

        self.fc1 = nn.Linear(400, 600)

        self.fc2 = nn.Linear(600, 250)

        self.fc3 = nn.Linear(250, 10)



    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        x = self.dropout(x)

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = self.dropout(x)

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.softmax(x, dim=1)

        return x



    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1

        for s in size:

            num_features *= s

        return num_features
net = Net()

net.to(device)



data_len = train_data.shape[0]

batch_size = 64

epochs = 300

accuracy_milestones = [100, 180, 280]



optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0001)

lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, accuracy_milestones, gamma=0.1, last_epoch=-1)

criterion = nn.MSELoss()
for epoch in range(1, epochs):

    random_order = torch.randperm(data_len)

    shuffled_data = train_data[random_order]

    shuffled_labels = train_labels[random_order]



    for j in range(0, data_len, batch_size):

        optimizer.zero_grad()



        start_index = j

        end_index = j + batch_size



        current_pics = shuffled_data[start_index:end_index]

        current_pics = current_pics.view(batch_size, 1, 28, 28)



        output = net(current_pics)



        true_digits = shuffled_labels[start_index:end_index].view(batch_size, 1).long()

        label_vec = F.one_hot(true_digits, num_classes=10).view(batch_size, 10).float()



        loss = criterion(output, label_vec)



        loss.backward()

        optimizer.step()



    net.eval()

    validation_data_len = validation_data.shape[0]

    test_output = net(validation_data.view(validation_data_len, 1, 28, 28))

    test_output_labels = torch.argmax(test_output, dim=1)

    comparision = test_output_labels == validation_labels.long()

    test_accuracy = torch.sum(comparision).cpu().numpy() / float(validation_data_len)

    net.train()

    lr_scheduler.step()

    print("Epoch: {:03d}/{:03d}, Accuracy {:4f}".format(epoch, epochs, test_accuracy))
# Generate csv result file

net.eval()



ids = []

predictions = []

for i in range(test_data.shape[0]):

    pic = test_data[i].view(1, 1, 28, 28)

    test_output = net(pic)

    test_result_digit = torch.argmax(test_output)

    ids.append(i + 1)

    predictions.append(test_result_digit.cpu().numpy())
submission = pd.DataFrame({'ImageId': ids, 'Label': predictions})

submission.to_csv('submission.csv', index=False)