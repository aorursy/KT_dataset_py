import torch

from torch import nn





cfgs = {

    'vgg_test': [16, 'M', 32, 'M', 64, 'M', 128, 'M', 512, 'M'],

    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}



class VGG(nn.Module):

    def __init__(self, cfg, vgg_name, in_channels=3, num_classes=10, Batch_norm=True, init_weights=True):

        super(VGG, self).__init__()

        self.name = vgg_name

        if Batch_norm:

            self.name += '_bn'

        self.features = self._make_layers(cfg[vgg_name], in_channels, Batch_norm)

        self.classifier = nn.Linear(512, num_classes)

        if init_weights:

            self._initialize_weights()

    

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:

                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                nn.init.constant_(m.bias, 0)

                

    def _make_layers(self, cfg, in_channels , batch_norm):

        layers = []

        for v in cfg:

            if v == 'M':

                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else:

                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

                if batch_norm:

                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

                else:

                    layers += [conv2d, nn.ReLU(inplace=True)]

                in_channels = v

        return nn.Sequential(*layers)
from dlpipeline import DLpipeline, Executor, Progbar, Reporter, Saver, FileNameManager, output_to_score_fun

import torch.optim as optim

import torchvision

import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10

from torch.utils.data import DataLoader



try:

    from torchsummary import summary

except:

    !pip install torchsummary

    from torchsummary import summary

    

''' 

try:

    from thop import profile, clever_format

except:

    !pip install thop

    from thop import profile, clever_format

'''



'''

def state_fun(pipeline):

    state = {'param': pipeline.model.state_dict(),

             'acc': pipeline.reporter.acc,

             'epoch': pipeline.epoch}

    return state

'''



def summary_fun(pipeline):

    from torchsummary import summary

    print('Model: %s' % pipeline.model_name)

    summary(pipeline.model, input_size=(3, 32, 32), device = pipeline.device.type)

    '''

    from thop import profile, clever_format

    input = torch.randn(1, 3, 32, 32)

    macs, params = profile(pipeline.model, inputs=(input, ))

    macs, params = clever_format([macs, params], "%.3f")

    print(macs, params)

    '''

    print('Optimizer:\n', pipeline.optimizer)

    

def get_device():

    if torch.cuda.is_available():

        device = 'cuda'

    else:

        device = 'cpu'

    return torch.device(device)





device = get_device()



batch_size = 256

save_dir = './checkpoint of vgg for cifar10/'

labels = [i for i in range(10)]



print('==> Preparing data ...')

transform_train = transforms.Compose([

    #transforms.RandomCrop(32, padding=4),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)),

])

transform_test = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2470, 0.2435, 0.2616)),

])



trainset = CIFAR10('./cifar10', train=True, download=True, transform=transform_train)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = CIFAR10('./cifar10', train=False, download=True, transform=transform_test)

testloader = DataLoader(testset, batch_size=250, shuffle=False, num_workers=2)





if device.type == 'cuda':

    #model = torch.nn.DataParallel(model)

    torch.backends.cudnn.benchmark = True

    

print('==> Bulding pipeline ...')

'''

# This can be use, but I prefer to use dict.

pipeline = DLpipeline(executor=Executor(),

                      progressbar=Progbar(dynamic = True),

                      reporter=Reporter(labels = labels, report_interval = 3, summary_fun = summary_fun),

                      saver=Saver(save_dir = save_dir,

                                  save_meta_file = True,

                                  save_ckpt_model = True, 

                                  save_val_model = True, 

                                  save_final_model = True,

                                  save_interval = 2, 

                                  test_model_use = 'final', 

                                  save_history = True),

                      file_name_manager=FileNameManager(),

                      device=device,

                      criterion=nn.CrossEntropyLoss(),

                      trainloader=trainloader,

                      #valloader=testloader,

                      testloader=testloader,

                     )

'''



basic_config = {'executor': Executor(),

                'progressbar': Progbar(dynamic = True),

                'reporter': Reporter(labels = labels, 

                                     need_confusion_matrix = True,

                                     output_to_score_fun = output_to_score_fun, 

                                     report_interval = 0,

                                     show_train_report = False,

                                     summary_fun = summary_fun),

                'saver': Saver(save_dir = save_dir,

                               save_meta_file = True,

                               save_ckpt_model = True, 

                               save_val_model = True, 

                               save_final_model = True,

                               save_final_optim = True,

                               save_interval = 5, 

                               test_model_use = 'final', 

                               save_history = True,

                               save_train_report = False,

                               save_test_report = True),

                'file_name_manager': FileNameManager(),

                'device': device,

                'criterion': nn.CrossEntropyLoss(),

                'trainloader': trainloader,

                'valloader': testloader,

                'testloader': testloader,

               }



pipeline = DLpipeline(**basic_config)

learning_rate = 0.05

weight_decay = 1e-3

momentum = 0.9



#print('==> Building model ...')

model = VGG(cfgs, 'vgg11', Batch_norm=False).to(device)

#summary(model, input_size=(3, 32, 32), device=device.type)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

pipeline.setup(model = model, model_name = model.name, optimizer = optimizer)

pipeline.create_pipeline()  # create a new pipeline

pipeline.summary()
pipeline(end_epoch = 20)  # train 20 epoch, from epoch=1 to epoch=20
pipeline(40)  # continue training, from epoch=21 to epoch=40
# report the result

pipeline.report()
# change model (reset optimizer) and pipeline

model = VGG(cfgs, 'vgg11', Batch_norm=True).to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

pipeline.setup(model = model, model_name = model.name, optimizer = optimizer)



# Now we specify the folder, and create a new pipeline

pipeline.create_pipeline(save_dir + 'vgg_test/')

pipeline.summary()
pipeline(20)  # train 20 epoch, from epoch=1 to epoch=20
# change model (reset optimizer) and pipeline

model = VGG(cfgs, 'vgg11', Batch_norm=True).to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

pipeline.setup(model = model, model_name = model.name, optimizer = optimizer)



# and we load the previous pipeline

pipeline.load('pipeline', save_dir + 'vgg_test/')

pipeline.summary()
# continue training

pipeline(50)  # train 30 epoch, from epoch=21 to epoch=50
import os

import re



# load history in epoch 40 (from a model file)

files = os.listdir(save_dir + 'vgg_test/')

for file in files:

    e = re.search("epoch_\d+", file)

    if e and int(e.group()[6:]) == 40:

        break



pipeline.load('history', save_dir + 'vgg_test/' + file)



# report the result, but only report train result 

# becansue we don't have test result in this file

pipeline.report(modes = 'train')
# load history in epoch 50

files = os.listdir(save_dir + 'vgg_test/')

for file in files:

    h = file[:7] == 'history'

    e = re.search("epoch_\d+", file)

    if h and e and int(e.group()[6:]) == 50:

        break



pipeline.load('history', save_dir + 'vgg_test/' + file)



# report the result, but only report test result

pipeline.report(modes = 'test')
# load pipeline in epoch 15 (from a model file)

files = os.listdir(save_dir + 'vgg_test/')

for file in files:

    e = re.search("epoch_\d+", file)

    if e and int(e.group()[6:]) == 15:

        break

pipeline.load('pipeline', save_dir + 'vgg_test/' + file)



# continue training, from epoch=16 to epoch=25

# this is a new training branch. The previous one ends at epoch=50

pipeline(25)