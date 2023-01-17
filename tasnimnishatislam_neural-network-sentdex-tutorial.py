import torch
import torchvision
from torchvision import transforms, datasets
train = datasets.MNIST("", train = True, download = True,
                      transform = transforms.Compose([transforms.ToTensor()]))
#I can do this with any dataset

test = datasets.MNIST("", train = False, download = True,
                     transform = transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)
for data in trainset:
    print(data)
    break;
x, y = data[0][0], data[1][0]

print(y)
import matplotlib.pyplot as plt

plt.imshow(data[0][0].view(28, 28))
#data balance - putting weight does not work
#count the datasets
total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, ys = data
    for y in ys:
        counter_dict[int(y)] += 1
        total +=1

print(counter_dict)

print(total)
for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100}")