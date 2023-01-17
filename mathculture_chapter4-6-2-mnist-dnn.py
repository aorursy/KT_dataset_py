"""

mnistをDNNで解く

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

    input_size = 784

    output_size = 10

    learning_rate = 0.01

    epoch_num = 5



def load_hparams():

    hparams = Hparams()



    return hparams





class Net(nn.Module):

    def __init__(self, input_size, output_size):

        super(Net, self).__init__()

        self.input_size = input_size

        self.output_size = output_size

        self.linear = nn.Linear(input_size, output_size)

    

    def forward(self, x):

        x = x.view(-1, self.input_size)

        return self.linear(x)



def main():

    log_dir = "logs"

    writer = SummaryWriter(log_dir=log_dir)



    hparams = load_hparams()

    train_loader, test_loader = load_dataloader(hparams)



    model = Net(hparams.input_size, hparams.output_size)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.learning_rate)



    for epoch in range(hparams.epoch_num):

        model.train()

        for idx, (data, target) in enumerate(train_loader):

             optimizer.zero_grad()

             output = model(data)

             loss = criterion(output, target)

             loss.backward()

             optimizer.step()

        # trainとtestでロスの比較が違う

        writer.add_scalar("Loss/train", loss, epoch)

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

              epoch, epoch , hparams.epoch_num,

              100. * epoch / hparams.epoch_num, loss.item()))



        model.eval()

        correct = 0

        with torch.no_grad():

            for data, target in test_loader:

                output = model(data)

                #loss -= F.nll_loss(output, target, reduction='sum').item() # negative log likelihood sum batch 

                loss += criterion(output, target).item() 

                pred = output.argmax(dim=1, keepdim=True) # 予測の最大値を取得

                correct += pred.eq(target.view_as(pred)).sum().item()

            loss /= len(test_loader.dataset)

            writer.add_scalar("Loss/valid", loss, epoch)

            writer.add_scalar("Accuracy/valid", correct / len(test_loader.dataset), epoch)

    

        writer.add_histogram("weight", model.linear.weight, epoch)

        writer.add_image('test_image', data[0], epoch)



        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

            loss, correct, len(test_loader.dataset),

            100. * correct / len(test_loader.dataset)))



    

    



if __name__ == "__main__":

    main()