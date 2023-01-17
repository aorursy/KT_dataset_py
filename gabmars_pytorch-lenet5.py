import cv2

from scipy.signal import correlate2d, convolve2d



import numpy as np



from IPython.display import clear_output

import PIL

import torch

import torchvision

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import matplotlib as mpl

%matplotlib inline



import torch.nn as nn

import torch.nn.functional as F



import torch.optim as optim



from IPython.display import clear_output



from collections import defaultdict



import seaborn as sn

import pandas as pd

#from torchsummary import summary
def view_classify(img, ps):

    ps = ps.cpu().data.numpy().squeeze()



    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='gray')

    ax1.axis('off')

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    ax2.set_yticklabels(np.arange(10))

    ax2.set_title('class probability')

    ax2.set_xlabel('probability')

    ax2.set_ylabel('class')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()
def show_features(features):

    if len(features.shape) < 4:

        batch, num_feature = features.shape[:2]

        for i, feature in enumerate(features):

            plt.subplot(1, num_feature, i+1)

            plt.imshow(feature.numpy().transpose(1,2,0))

    else:

        batch, num_feature = features.shape[:2]

        for i, element in enumerate(features):

            for j, feature in enumerate(element):

                plt.subplot(batch, num_feature, i * num_feature + j + 1)

                plt.imshow(feature.numpy())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
train=pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test=pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
# tt_data=pd.concat([train, test], ignore_index=True)

# tt_data=tt_data.sample(frac=1)

# test=tt_data.sample(frac=0.1)

# train=tt_data.drop(test.index)

# test=test.reset_index(drop=True)

# train=train.reset_index(drop=True)
print('train',train.shape)

print('test',test.shape)
train_data=train.drop('label',axis=1)

train_targets=train['label']



test_data=test.drop('label',axis=1)

test_targets=test['label']
train_data=torch.from_numpy(train_data.values).float().view(train_data.shape[0],28,28)

train_targets=torch.from_numpy(train_targets.values).long().view(train_data.shape[0])



test_data=torch.from_numpy(test_data.values).float().view(test_data.shape[0],28,28)

test_targets=torch.from_numpy(test_targets.values).long().view(test_data.shape[0])
num, height, width = np.array(train_data).shape

img_min = np.array(train_data).min()

img_max = np.array(train_data).max()

img_norm_mean = np.array(train_data, dtype=float).mean() / img_max

img_std = np.sqrt(np.sum((np.array(train_data) / img_max  - img_norm_mean) ** 2) / (num * height * width))

print(img_min, img_max, img_norm_mean, img_std)
print('train set shape: ', train_data.shape)

print('train labels shape: ', train_targets.shape)
print('test set shape: ', test_data.shape)

print('test labels shape: ', test_targets.shape)
plt.title('train dataset distribution')

plt.hist(train_targets);
plt.title('test dataset distribution')

plt.hist(test_targets);
class KannadaDataSet(torch.utils.data.Dataset):

    def __init__(self, images, labels, transforms = None):

        self.images = images

        self.labels = labels

        self.transforms = transforms

         

    def __len__(self):

        return self.images.shape[0]

    

    def __getitem__(self, i):

        data = np.array(self.images.iloc[i,:]).astype(np.uint8).reshape(28,28,1)

        

        if self.transforms:

            data = self.transforms(data)

            

        return (data, self.labels[i])
train_orig_T = transforms.Compose(([

    transforms.ToTensor(),

#     transforms.Normalize((img_norm_mean,), (img_std,))

]))

train_aug_T = transforms.Compose(([

    transforms.ToPILImage(),

    transforms.RandomCrop(28),

    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),

    transforms.ToTensor(),

#     transforms.Normalize((img_norm_mean,), (img_std,))

]))
test_T = transforms.Compose(([

    transforms.ToTensor(),

#     transforms.Normalize((img_norm_mean,), (img_std,))

]))
orig_train_set=KannadaDataSet(train.drop('label',axis=1), train['label'],train_orig_T)

aug_train_set=KannadaDataSet(train.drop('label',axis=1), train['label'],train_aug_T)

train_set = torch.utils.data.ConcatDataset([orig_train_set,aug_train_set])

test_set=KannadaDataSet(test.drop('label',axis=1), test['label'], test_T) 
train_loader = torch.utils.data.DataLoader(train_set, 

                                           batch_size=128, 

                                           shuffle=True)



test_loader = torch.utils.data.DataLoader(test_set,

                                          batch_size=128, 

                                          shuffle=False)



classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
images, labels = next(iter(train_loader))

images, labels = images[:8], labels[:8]

plt.imshow(torchvision.utils.make_grid(images)[0,:,:], cmap='gray')

print(labels)
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, 

                               out_channels=12, 

                               kernel_size=5, 

                               padding=0)

        self.bn1 = nn.BatchNorm2d(12)

        self.conv2 = nn.Conv2d(in_channels=12, 

                               out_channels=32, 

                               kernel_size=3, 

                               padding=0)

        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 5 * 5, 240)

        self.fc2 = nn.Linear(240, 168)

        self.fc3 = nn.Linear(168, 10)



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = F.relu(x)

        #x = F.max_pool2d(x,2)

        x = F.avg_pool2d(x,2)

        x = self.conv2(x)

        x = self.bn2(x)

        x = F.relu(x)

        #x = F.max_pool2d(x,2)

        x = F.avg_pool2d(x,2)

        x = x.view(-1, 32 * 5 * 5)

        x = self.fc1(x)

        x = F.relu(x)

        x = self.fc2(x)

        x = F.relu(x)

        x = self.fc3(x)

        x = F.log_softmax(x, dim=1)

        return x

    

    def features_2(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

#         x = F.max_pool2d(F.relu(self.conv1(x)),2)

#         x = F.max_pool2d(F.relu(self.conv2(x)),2)

        return x

    

    def features_1(self, x):

        x = F.relu(self.conv1(x))

#         x = F.max_pool2d(F.relu(self.conv1(x)),2)

        return x
net = Net().to(device)

# summary(net, (1, 28, 28))

net
plt.figure(figsize=(15,10))

show_features(net.features_1(images.to(device)).detach().cpu())
plt.figure(figsize=(15,10))

show_features(net.features_2(images.to(device)).detach().cpu())
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)



epoch=0



curve_x=[]

loss_curve_y=[]

v_loss_curve_y=[]

acc_curve_y=[]
num_epochs=5



e=0

while e < num_epochs:

    epoch+=1

    e+=1

    

#     if e == 3:

#         optimizer.param_groups[0]['lr'] = 1e-5



#     if e == 5:

#         optimizer.param_groups[0]['lr'] = 1e-8

        

    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):



        inputs, labels = data[0].to(device), data[1].to(device)



        optimizer.zero_grad()



        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item()

        

        if i != 0 and i % 100 == 0:    

            validation_loss = 0.

            correct = 0

            total = 0



            with torch.no_grad():

                net.eval()



                for v_data in test_loader:

                    v_inputs, v_labels = v_data[0].to(device), v_data[1].to(device)



                    v_outputs = net(v_inputs)

                    v_loss = criterion(v_outputs, v_labels)



                    validation_loss+=v_loss.item()



                    _, predicted = torch.max(v_outputs.data, 1)

                    total += v_labels.size(0)

                    correct += (predicted == v_labels).sum().item()

            net.train()

        #     clear_output()              

            print('epoch %d, step %5d training loss: %.3f validation loss: %.3f test acc: %.3f' %

              (epoch, i, running_loss / 1000, validation_loss/1000, 100 * correct / total))

            

            curve_x.append(len(curve_x))

            loss_curve_y.append(running_loss/1000)

            v_loss_curve_y.append(validation_loss/1000)

            acc_curve_y.append(100 * correct / total)

            

            running_loss = 0.0



print('finish')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].plot(curve_x,v_loss_curve_y, label='validation')

axes[0].plot(curve_x,loss_curve_y, '--',label='train')

axes[0].set_title('validation loss')

axes[0].set_xlabel("iterations")

axes[0].set_ylabel("loss")

axes[0].set_xticks([])

axes[0].legend()

axes[1].plot(curve_x,acc_curve_y)

axes[1].set_title('test acc')

axes[1].set_xlabel("iterations")

axes[1].set_ylabel("acc")

axes[1].set_xticks([])

fig.tight_layout()

plt.show()
plt.figure(figsize=(15,10))

show_features(net.features_1(images.to(device)).detach().cpu())
plt.figure(figsize=(15,10))

show_features(net.features_2(images.to(device)).detach().cpu())
images, labels = next(iter(test_loader))

images, labels = images.to(device), labels.to(device)

images, labels = images[:8], labels[:8]

outputs = net(images)

_, predicted = torch.max(outputs, 1)



plt.imshow(torchvision.utils.make_grid(images.cpu())[0,:,:], cmap='gray')

print('gt:', labels)

print('predict:', predicted)
correct = 0

total = 0



errors_imgs=[]

errors_labels=[]

confusion_matrix = torch.zeros(10, 10)

err_confusion_matrix = torch.zeros(10, 10)

correct_samples={i:None for i in range(10)}

with torch.no_grad():

    for data in test_loader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        errors_imgs.extend(images[predicted != labels])

        errors_labels.extend(zip(predicted[predicted != labels], labels[predicted != labels]))

        

        for i in [i for i in correct_samples if type(correct_samples[i]) != torch.Tensor]:

            ci = images[(predicted == labels) & (i == labels)]

            if ci.shape[0] > 0:

                correct_samples[i] = ci[0]

        for t, p in zip(labels.view(-1), predicted.view(-1)):

            confusion_matrix[t.long(), p.long()] += 1

            if t.long() != p.long():

                err_confusion_matrix[t.long(), p.long()] += 1

errors_imgs=torch.stack(errors_imgs)

errors_labels=np.array([(p.item(), t.item())for p, t in errors_labels])

print('accuracy of the network on the test images: %d %%' % (100 * correct / total))

#torch.save(net, 'torch_kannada_mnist_model.pt')  
print('errors: ',len(errors_labels))
df_cm = pd.DataFrame(confusion_matrix.numpy(), range(10), range(10))



plt.figure(figsize = (20,20))

sn.set(font_scale=2)

sn.heatmap(df_cm, annot=True,annot_kws={"size": 14}, fmt='g')
df_cm = pd.DataFrame(err_confusion_matrix.numpy(), range(10), range(10))



plt.figure(figsize = (20,20))

sn.set(font_scale=2)

sn.heatmap(df_cm, annot=True,annot_kws={"size": 14}, fmt='g')
sn.set(font_scale=1)

sn.set_style("whitegrid", {'axes.grid' : False})
f=lambda i, a: ({k: len(v) for k, v in a.items()} if [a[x].append(i) for x in i] else {})
errors=f(errors_labels[:,1],defaultdict(list))
plt.figure(figsize=(6,4))

plt.barh(*zip(*errors.items()))

plt.yticks(range(10))

plt.xlabel('erorrs')

plt.ylabel('labels')

plt.show()
plt.figure(figsize=(15,10))

show_features(net.features_1(errors_imgs[:8]).detach().cpu())
plt.figure(figsize=(15,10))

show_features(net.features_2(errors_imgs[:8]).detach().cpu())
for img_indx in range(10):

    pred_digit=errors_labels[img_indx][0].item()

    true_digit=errors_labels[img_indx][1].item()

    print('PREDICTED: ', pred_digit, '!=','TRUE: ', true_digit)

    ps = net.forward(errors_imgs[img_indx].unsqueeze(0))

    view_classify(errors_imgs[img_indx].cpu(), torch.softmax(ps,dim=1))

    plt.show()

    

    print(f'FEATURES_1 OF FALSE {true_digit}')

    plt.figure(figsize=(15,10))

    show_features(net.features_1(errors_imgs[img_indx:img_indx+1].to(device)).detach().cpu())

    plt.show()



    print(f'FEATURES_1 OF TRUE {true_digit}')

    plt.figure(figsize=(15,10))

    show_features(net.features_1(correct_samples[true_digit].unsqueeze(0).to(device)).detach().cpu())

    plt.show()

    

    print(f'FEATURES_1 OF TRUE {pred_digit}')

    plt.figure(figsize=(15,10))

    show_features(net.features_1(correct_samples[pred_digit].unsqueeze(0).to(device)).detach().cpu())

    plt.show()

    

    print(f'FEATURES_2 OF FALSE {true_digit}')

    plt.figure(figsize=(15,10))

    show_features(net.features_2(errors_imgs[img_indx:img_indx+1].to(device)).detach().cpu())

    plt.show()

    

    print(f'FEATURES_2 OF TRUE {true_digit}')

    plt.figure(figsize=(15,10))

    show_features(net.features_2(correct_samples[true_digit].unsqueeze(0).to(device)).detach().cpu())

    plt.show()

    

    print(f'FEATURES_2 OF TRUE {pred_digit}')

    plt.figure(figsize=(15,10))

    show_features(net.features_2(correct_samples[pred_digit].unsqueeze(0).to(device)).detach().cpu())

    plt.show()
# net=torch.load('torch_kannada_mnist_model.pt')

# net.eval()
kaggle=pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
kaggle.shape
kaggle=kaggle.drop('id', axis=1)
kaggle=torch.from_numpy(kaggle.values)
kaggle=kaggle.view(kaggle.shape[0],28,28).to(device)
# T = transforms.Compose([

#     transforms.Normalize((img_norm_mean,), (img_std,))

# ])

# kaggle=T(kaggle)
predictions=[]

with torch.no_grad():

    for i, data in enumerate(kaggle):

        images = data.to(device)

        outputs = net(images.float().unsqueeze(0).unsqueeze(0))

        _, predicted = torch.max(outputs.data, 1)

        predictions.append([i,predicted.item()])
predictions=pd.DataFrame(predictions)
predictions.columns=['id','label']
predictions.to_csv('submission.csv', index=False)