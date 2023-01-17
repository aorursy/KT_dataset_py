"""

mnistをCNNで解く

Google Colaboratoryで実験してください.

"""

from pathlib import Path

from collections import namedtuple



import torch

from torchvision import datasets, transforms

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



from torch.utils.tensorboard import SummaryWriter







def load_dataloader(hparams):

    train_loader = torch.utils.data.DataLoader(

        datasets.MNIST('./', train=True, download=True,

                        transform=transforms.Compose([

                           transforms.ToTensor(),

                           transforms.Normalize((0.1307,), (0.3081,))

                         ])),

                         batch_size=hparams.batch_size, shuffle=True)

    

    

    test_loader = torch.utils.data.DataLoader(

            datasets.MNIST('./', train=False, transform=transforms.Compose([

                               transforms.ToTensor(),

                               transforms.Normalize((0.1307,), (0.3081,))

                           ])),

            batch_size=1, shuffle=True)

    return train_loader, test_loader



class Hparams():

    batch_size = 16

    learning_rate = 0.01

    epoch_num = 5



def load_hparams():

    hparams = Hparams()



    return hparams



class CNNNet(nn.Module):

    def __init__(self):

        super(CNNNet, self).__init__()

        self.cv1 = nn.Conv2d(1, 32, 5) # 28x28x32 -> 24x24x32

        self.cv2 = nn.Conv2d(32, 64, 5) # 24x24x64 -> 20x20x64 

        self.pool = nn.MaxPool2d(2, 2) # 20x20x64 -> 10x10x64

        self.fc1 = nn.Linear(10 * 10 * 64, 128)

        self.fc2 = nn.Linear(128, 10)



    def forward(self, x):

        x = F.relu(self.cv1(x))

        x = self.pool(F.relu(self.cv2(x)))

        x = x.view(-1, 10 * 10 * 64)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x



def main():

    log_dir = "logs"

    writer = SummaryWriter(log_dir=log_dir)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    hparams = load_hparams()

    train_loader, test_loader = load_dataloader(hparams)



    model = CNNNet()

    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate)



    for epoch in range(hparams.epoch_num):

        model.train()

        for idx, (data, target) in enumerate(train_loader):

             

             target = torch.tensor(target, device=device)

             data = torch.tensor(data, device=device)

             optimizer.zero_grad()

             output = model(data)

             loss = criterion(output, target)

             loss.backward()

             optimizer.step()



        writer.add_scalar("Loss/train", loss, epoch)

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

              epoch, epoch , hparams.epoch_num,

              100. * epoch / hparams.epoch_num, loss.item()))



        model.eval()

        correct = 0

        with torch.no_grad():

            for data, target in test_loader:

                target = torch.tensor(target, device=device)

                data = torch.tensor(data, device=device)

                output = model(data)

                loss += criterion(output, target).item() 

                pred = output.argmax(dim=1, keepdim=True) # 予測の最大値を取得

                correct += pred.eq(target.view_as(pred)).sum().item()

            loss /= len(test_loader.dataset)

            writer.add_scalar("Loss/valid", loss, epoch)

            writer.add_scalar("Accuracy/valid", correct / len(test_loader.dataset), epoch)

    

        writer.add_histogram("weight", model.cv1.weight, epoch)

        writer.add_image('test_image', data[0], epoch)



        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

            loss, correct, len(test_loader.dataset),

            100. * correct / len(test_loader.dataset)))



    



if __name__ == "__main__":

    main()