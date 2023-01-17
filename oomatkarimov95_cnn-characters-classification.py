!pip install python-resize-image
!pip install Pillow==2.2.1
import numpy as np

import matplotlib.pyplot as plt

import matplotlib



from PIL import Image

from resizeimage import resizeimage





%matplotlib inline
data1 = np.load("../input/train-1.npy")

data2 = np.load("../input/train-2.npy")

data3 = np.load("../input/train-3.npy")

data4 = np.load("../input/train-4.npy")
data = np.concatenate((data1, data2, data3, data4))



if len(data) == len(data1) + len(data2) + len(data3) + len(data4):

    print("Sizes are OK")
del data1

del data2

del data3

del data4
def resize_little_image( img , bigside = 42 ):

    image = Image.fromarray( img )  

    image_size = image.size

    width = image_size[0]

    height = image_size[1] 

    

    background = Image.new('P', (bigside, bigside), (255))

    offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))



    background.paste(image, offset)

    background.convert('1')

    

    return background
def transform_image(array_img, size_img=42, greyscale=False, tpe='train', center=0, normalize=1):

    transf_img = []

    class_of_img = []

    dict_of_class = {}

    dict_of_class_back = {}

    index_class = 0

    

    for i in range(len(array_img)):

        if tpe == 'train':

            img = Image.fromarray(array_img[i][0])  

            if img.size[0] < size_img or img.size[1] < size_img:

                img = resize_little_image(array_img[i][0])

            else:

                img = resizeimage.resize_cover(img, [size_img, size_img])

        elif tpe == 'test':

            img = Image.fromarray(array_img[i])  

            if img.size[0] < size_img or img.size[1] < size_img:

                img = resize_little_image(array_img[i])

            else:

                img = resizeimage.resize_cover(img, [size_img, size_img])

        if greyscale:

            img = img.convert('LA')

        transf_img.append(img)

        if tpe=='train':

            class_of_img.append(array_img[i][1])

            if array_img[i][1] not in dict_of_class:

                dict_of_class[ array_img[i][1] ] = index_class

                dict_of_class_back[index_class] = array_img[i][1]

                index_class += 1

                

        if i % 25000 == 0:

            print("Transform {0} images".format(i))  

    answer = []

    

    for i in range(len(transf_img)):

        temp = np.array( transf_img[i])

        temp = temp[np.newaxis, :, :]

        answer.append( temp )

   

    # it's very important to define types because pytorch has few types

    if tpe=='train':

        answer = np.array(answer ,dtype='float32')

        class_of_img = np.array(class_of_img ,dtype='long')

        

        center = np.mean(answer, axis = 0) # zero-center

        normalize = np.std(answer, axis = 0) # normalize

        answer -= center

        answer /= normalize

        return answer, class_of_img,  dict_of_class, dict_of_class_back, center, normalize

    

    elif tpe=='test':

        answer -= center

        answer /= normalize

        answer = np.array(answer ,dtype='float32')

        return answer
transform_data, class_data,  dict_of_class, dict_of_class_back, center, normalize = transform_image( data, tpe = 'train')
if len(transform_data) == len(class_data) and len(dict_of_class) == 1000 and len(dict_of_class_back) == 1000 :

    print("Sizes are OK")
transform_data_class = class_data.copy()



for i in range(len(class_data)):

    transform_data_class[i] = dict_of_class[transform_data_class[i]]
test = np.load('../input/test.npy')
transform_test = transform_image( test , tpe='test', center=center, normalize=normalize)
import matplotlib

import matplotlib.pyplot as plt

import numpy as np



%matplotlib inline



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms



import torch.utils.data as data_utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
def chin_char( train_data, target, test_data, batch_size=50, valid=0, shuffle=True, transforms = transforms.ToTensor() ):

    train = data_utils.TensorDataset(train_data, target)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    

    if valid > 0:

        num_train = len(train_data)

        indices = list(range(num_train))

        split = num_train - valid

        np.random.shuffle(indices)



        train_idx, valid_idx = indices[:split], indices[split:]

        train_sampler = SubsetRandomSampler(train_idx)

        valid_sampler = SubsetRandomSampler(valid_idx)



        train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler)

        valid_loader = DataLoader(train, batch_size=batch_size, sampler=valid_sampler)

        

        return train_loader, valid_loader, test_loader

    else:

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, transforms=transforms)

        return train_loader, test_loader
train_loader, valid_loader, test_loader = chin_char( torch.as_tensor(transform_data), 

                                        torch.as_tensor(transform_data_class), 

                                        torch.as_tensor(transform_test),

                                        valid=20000)
class ConvLayer(nn.Module):

    def __init__(self, size, padding=1, pool_layer=nn.MaxPool2d(2, stride=2),

                 bn=False, dropout=False, activation_fn=nn.LeakyReLU()):

        

        super(ConvLayer, self).__init__()

        layers = []

        layers.append(nn.Conv2d(size[0], size[1], size[2], padding=padding))

        

        if pool_layer is not None:

            layers.append(pool_layer)

        if bn:

            layers.append(nn.BatchNorm2d(size[1]))

        if dropout:

            layers.append(nn.Dropout2d())

            

        layers.append(activation_fn)

        self.model = nn.Sequential(*layers)

    

    def forward(self, x):

        return self.model(x)
class FullyConnected(nn.Module):

    def __init__(self, sizes, dropout=False, activation_fn=nn.Tanh):

        super(FullyConnected, self).__init__()

        layers = []

        

        for i in range(len(sizes) - 2):

            layers.append(nn.Linear(sizes[i], sizes[i+1]))

            

            if dropout:

                layers.append(nn.Dropout())

            layers.append(activation_fn())

        else: 

            layers.append(nn.Linear(sizes[-2], sizes[-1]))

        

        self.model = nn.Sequential(*layers)

        

    def forward(self, x):

        return self.model(x)
class Net(nn.Module):

    def __init__(self, batchnorm=False, dropout=False, optim_type='Adam',  

                 lr=1e-4, l2=0., device=device, **optim_params):

        

        super(Net, self).__init__()

        

        self._conv1 = ConvLayer([1, 64, 3], bn=batchnorm)

        self._conv12 = ConvLayer([64, 64, 3], pool_layer=None)

        self._conv2 = ConvLayer([64, 128, 3], bn=batchnorm)

        self._conv22 = ConvLayer([128, 128, 3], pool_layer=None)

        self._conv3 = ConvLayer([128, 256, 3], bn=batchnorm)

        self._conv32 = ConvLayer([256, 256, 3], pool_layer=None)

        self._conv4 = ConvLayer([256, 512, 3], bn=batchnorm)

        self._conv42 = ConvLayer([512, 512, 3], pool_layer=None)

        self.fc = FullyConnected([512*2*2, 1000], dropout=dropout)

      

        self._loss = None

        

        if optim_type == 'SGD':

            self.optim = optim.SGD(self.parameters(), **optim_params)

        elif optim_type == 'Adadelta':

            self.optim = optim.Adadelta(self.parameters(), **optim_params)

        elif optim_type == 'RMSProp':

            self.optim = optim.RMSprop(self.parameters(), **optim_params)

        elif optim_type == 'Adam':

            self.optim = optim.Adam(self.parameters(), **optim_params)

        

        self.to(device)

    

    def conv(self, x):

        x = self._conv1(x)

        x = self._conv12(x)

        x = self._conv2(x)

        x = self._conv22(x)

        x = self._conv3(x)

        x = self._conv32(x)

        x = self._conv4(x)

        x = self._conv42(x)

        return x

        

    def forward(self, x):

        x = self.conv(x)

        x = x.view(-1, 512*2*2)

        x = self.fc(x)

        return x

    

    def loss(self, output, target, **kwargs):

        self._loss = F.cross_entropy(output, target, **kwargs)

        self._correct = output.data.max(1, keepdim=True)[1]

        self._correct = self._correct.eq(target.data.view_as(self._correct)).to(torch.float).cpu().mean()

        return self._loss
def train(epoch, models, log=None):

    train_size = len(train_loader.sampler)

    

    for batch_idx, (data, target) in enumerate(train_loader):

        batch_idx, (data, target)

        data = data.to(device)

        target = target.to(device)

        

        for model in models.values():

            model.optim.zero_grad()

            output = model(data)

            loss = model.loss(output, target)

            loss.backward()

            model.optim.step()

            

        if batch_idx % 1000 == 0:

            line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(

                epoch, 

                batch_idx * len(data), 

                train_size, 100. * batch_idx / len(train_loader))

            losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])

            print(line + losses)

    else:

        batch_idx += 1

        line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(

            epoch, 

            batch_idx * len(data),

            train_size, 100. * batch_idx / len(train_loader))

        losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])

        

        if log is not None:

            for k in models:

                log[k].append((models[k]._loss, models[k]._correct))

        print(line + losses)

        

    return models
def test(models, loader, log=None):

    test_size = len(loader.sampler)

    avg_lambda = lambda l: 'Loss: {:.4f}'.format(l)

    acc_lambda = lambda c, p: 'Accuracy: {}/{} ({:.0f}%)'.format(c, test_size, p)

    line = lambda i, l, c, p: '{}: '.format(i) + avg_lambda(l) + '\t' + acc_lambda(c, p)



    test_loss = {k: 0. for k in models}

    correct = {k: 0. for k in models}

    

    with torch.no_grad():

        

        for data, target in loader:

            data = data.to(device)

            target = target.to(device)

            output = {k: m(data) for k, m in models.items()}

            

            for k, m in models.items():

                test_loss[k] += m.loss(output[k], target, reduction='sum').item() 

                pred = output[k].data.max(1, keepdim=True)[1] 

                correct[k] += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    

    for k in models:

        test_loss[k] /= test_size

        

    correct_pct = {k: c / test_size for k, c in correct.items()}

    lines = '\n'.join([line(k, test_loss[k], correct[k], 100*correct_pct[k]) for k in models]) + '\n'

    report = 'Test set:\n' + lines

    

    if log is not None:

        for k in models:

            log[k].append((test_loss[k], correct_pct[k]))

    print(report)
models = {

          'batch_drop_Adadelta': Net(True, True, optim_type = 'Adadelta')

#           'batch_drop_Adam': Net(True, True, optim_type = 'Adam')

         }



train_log = {k: [] for k in models}

test_log = {k: [] for k in models}

save_models = []
for epoch in range(1, 10):

    for model in models.values():

        model.train()

        

    save_models.append(train(epoch, models, train_log))

    

    for model in models.values():

        model.eval()

    test(models, valid_loader, test_log)
def plot_graphs(log, tpe='loss'):

    keys = log.keys()

    logs = {k:[z for z in zip(*log[k])] for k in keys}

    epochs = {k:range(len(log[k])) for k in keys}

    

    if tpe == 'loss':

        handlers, = zip(*[plt.plot(epochs[k], logs[k][0], label=k) for k in keys])

        plt.title('errors')

        plt.xlabel('epoch')

        plt.ylabel('error')

        plt.legend(handles=handlers)

        plt.show()

    elif tpe == 'accuracy':

        handlers, = zip(*[plt.plot(epochs[k], logs[k][1], label=k) for k in log.keys()])

        plt.title('accuracy')

        plt.xlabel('epoch')

        plt.ylabel('accuracy')

        plt.legend(handles=handlers)

        plt.show()
plot_graphs(test_log, 'loss')
plot_graphs(test_log, 'accuracy')
def final_test(model, loader):

    answer = []

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            output = model(data)

            pred = output.data.max(1, keepdim=True)[1] 

            answer.append((pred.cpu().data.numpy()).ravel() )

    return np.array(answer)
def to_answer(test_pred):

    answer = np.zeros(len(transform_test), dtype=int)

    index = 0

    for i in test_pred:

        for j in i:

            answer[index] = j

            index += 1

    return answer
test_pred = final_test(save_models[8]['batch_drop_Adadelta'], test_loader)
answer = to_answer(test_pred)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "prediction.csv"):  

    csv = df.to_csv(index=False)

    

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
predictions = [ dict_of_class_back[x]  for x in answer  ]



df2 = pd.DataFrame(  columns=['Id', 'Category'])

df2['Id'] = np.arange(1, len(predictions) + 1)

df2['Category'] = predictions



create_download_link(df2, filename = "prediction.csv")