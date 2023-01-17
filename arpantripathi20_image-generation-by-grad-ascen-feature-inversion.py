import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

!pip install pytorch_lightning
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
PATH = '../input/fruits/fruits-360/Test/Banana/102_100.jpg'
banana = Image.open(PATH)
plt.imshow(banana)
banana_tensor = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor()])(banana)
print(banana_tensor.size())
print(banana_tensor)
normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
banana_tensor = normalizer(banana_tensor)
banana_tensor.requires_grad_(True)
print(banana_tensor.requires_grad)
model = torchvision.models.vgg16(pretrained=False)
#model.to('cuda')
#for layer in model.parameters():
    #layer.requires_grad_(False) 
model.train()
print(model.classifier[6])
print(type(model.classifier[0].children()))
print('\n\n')
for x in model.classifier.children():
    print(x)
print('\n\n')
print(type(model.features))
print('\n')
for x in model.features:
    print(x)
print(model.state_dict().keys())
print(type(model.state_dict().values()))
print()
print(len(model.state_dict().values()))
print()

#for key, value in model.state_dict().values().items():
 #   print(type(key), type(value))
#print(type(model.state_dict().values().keys()))
layer_name = 'features.24.weight'
print(type(model.state_dict()[layer_name]))
print(model.state_dict()[layer_name].size())

print(model)
a = model.features(banana_tensor.unsqueeze(dim=0))
print(a.size())
b = model.avgpool(a)
print(b.view(b.size(0),-1).size())
print(model.classifier(b.view(b.size(0),-1)).size())
#SANDBOX
t = torch.rand(1000)
print(t.argmax(dim=0))
print(t)


model.train()
model.classifier[-1].weight
print(model.features[0].__class__)
'''
A printable method?, but __print__ function can be defined only for classes?
'''
type(model.classifier[0].weight.retain_grad)

print(model.classifier[0].weight.retain_grad)
print(model.classifier[3].weight.grad)
derivative = model.classifier[3].bias.grad
print(derivative)
sample_noise = torch.randn(10,1000)
torch.argmax(sample_noise,1)
sample_noise_2 = torch.randn(10,1000)
sample_noise_2 = torch.Tensor(sample_noise)
print(sample_noise_2.data)
print(sample_noise_2 == sample_noise) #This should give all True as we just assigned the values of sample_data to sample_data_2
for i in range(sample_noise.size(0)):
    sample_noise[i] = torch.randn(1000) #Overwriting sample_data
print(sample_noise)
print(sample_noise_2)
print(sample_noise_2 == sample_noise) #This still gives true, that means, sample_data and sample_data_2 now point to the same memory, even though I used .clone() &.detach()
'''
DAY 1:


PROBLEM: 
---> The gradients do not retain in .grad attribute even though retain_grad is set to True for all parameters of model
---> There are all zero values in .grad attributes for some reason
---> There are random values in .retain_grad attribute (same name as retain_grad() method), could this be what I'm looking for?
look into '.register_hook' method for Tensors


DAY 2:

The model.parameters() and model.state_dict().values() are supposed to point to the same Tensors, however, setting .requires_grad attribute of model.parameters() does not 
affect the .requires_grad of the model.state_dict().values() Tensors

The pretrained model's parameters have .grad_fn attribute set to 'None', no matter what I do (foward passes through the whole model, 
backward pass in the loss(which yields 0 and NaN of course, as there's no grad_fn))


DAY 3:

Suppose we have a 'nn.Parameter' instance in the model's '__init__' function, let's call it 'imp_tensor', 
In the forward pass function of the model, we should use 'imp_tensor.data' as the input to layers (like nn.Conv2d)

See 'Trying to copy the value of one tensor to another, without causing them to point to the same memory!!!' section above, this is gonna be problematic with the 2nd loop of
MAIN section (below), as 'cnn_model.inp' might get overridden or override the input 'batch' attribute every forward pass

'''
#for tensor in model.parameters():
    #tensor.requires_grad_(True)
   # tensor.requires_grad = True
for tensor in model.parameters():
    print(tensor.requires_grad)
#model(zero_image)  #a forward pass hoping to update the state_dict
for layer,tensor in zip(model.state_dict().keys(),model.state_dict().values()):
    #WTF, why does it show 'False' for the 'classifier'part, I just set them all to True 3 lines above
     print(layer, tensor.requires_grad, tensor.grad_fn)
c =0 
for x in model.parameters():
    c+=1
print(c)
print(len(model.state_dict().keys()))
hparams = {
    'device': torch.device('cuda:0'),
    'batch_size': 10,
    'image_size_x' : 96,   #size of original images which were reshaped to 3x224x224 to fit the pretrained model
    'image_size_y' : 96
}
torch.set_default_tensor_type('torch.cuda.FloatTensor')
model = model.to(hparams['device'])
class cnn_model_class(nn.Module):
    def __init__(self,model,hparams):
        #self.features = model.features
        super().__init__()
        self.hparams = hparams
        self.inp = nn.Parameter(torch.empty((self.hparams['batch_size'],3,224,224),requires_grad = True)).to(hparams['device'])   
        #this is only for passing the 'noise' of size 'batch_size'
        #for passing the single image to reconstructed, directly use 'model(banana_tensor)'
        self.cnn = model.features
        self.avgpool = model.avgpool
        self.fcc_1 = nn.Parameter(torch.empty((self.hparams['batch_size'],25088),requires_grad=True)).to(hparams['device']) 
        
    def forward(self,batch):
        self.inp.data = batch.detach()
        self.fcc_1.data = self.avgpool(self.cnn(self.inp.data)).view(-1,25088)
        #final = self.classifier(self.inp.data)
        return self.fcc_1.data
    def forward(self):   #FUNCTION OVERLOADING IN PYTHON?
        self.fcc_1.data = self.avgpool(self.cnn(self.inp.data)).view(-1,25088)
        #final = self.classifier(self.inp.data)
        return self.fcc_1.data
class classifier_model_class(nn.Module):
    def __init__(self,model,hparams):
        super().__init__()
        self.hparams = hparams
        self.classifier = model.classifier
    def forward(self,batch):
        out = self.classifier(batch)
        #final = self.classifier(self.inp.data)
        return out
class model_with_fcc_1_param(nn.Module):
    def __init__(self,cnn_model):
        super().__init__()
        self.param = cnn_model.fcc_1
    def forward(self,batch):
        return self.param.data
class model_with_inp_param(nn.Module):
    def __init__(self,cnn_model):
        super().__init__()
        self.param = cnn_model.inp
    def forward(self,batch):
        return self.param.data
#def fcc_layer(out):  #creating iterable form of the first fcc layer's output so that it can be passed to optimizer 
    #yield out
model = model.to(hparams['device'])
cnn_model = cnn_model_class(model,hparams).to(hparams['device'])
classifier_model = classifier_model_class(model,hparams).to(hparams['device'])
fcc_1_param = model_with_fcc_1_param(cnn_model).to(hparams['device'])
inp_param = model_with_inp_param(cnn_model).to(hparams['device'])
model.train()
for x in cnn_model.parameters():
    x.requires_grad_(True)
for x in classifier_model.parameters():
    x.retain_grad()
for x in cnn_model.parameters():
    x.retain_grad()
for x in fcc_1_param.parameters():
    x.retain_grad()
for x in fcc_1_param.parameters():
    print(x)
    print('-----------------------------------------')
    print(x.data)
    print('-----------------------------------------')
    print(x.grad)
    print('-----------------------------------------')
    print(x.requires_grad)
#flattened = cnn_model(zero_image)
#noise_out =  classifier_model(flattened).view(zero_image.size(0), -1)


print(cnn_model.fcc_1.grad)
#loss = loss_fn(noise_out,target)
#loss.backward()
for x,y in cnn_model.named_parameters():
    print(x)
for x in cnn_model.parameters():
    #print(x.data)
    print('---------------------')
    print(x.requires_grad)
    print(x.grad)
    print('//////////////////////\n\n')
for x in model.parameters():
    x.requires_grad_(True)
for x in fcc_1_param.parameters():
    x.requires_grad_(True)
for x in cnn_model.parameters():
    x.requires_grad_(False)
for x in classifier_model.parameters():
    print(x.requires_grad)
for x in cnn_model.parameters():
    print(x.requires_grad)
for x in model.parameters():
    print(x.requires_grad)
'''
Thankfully, the claim was true

'''
epochs = 1
iters_classifier = 100
iters_cnn = 50
zero_image = torch.rand([hparams['batch_size'],banana_tensor.size(0),banana_tensor.size(1),banana_tensor.size(2)])
for i in range(zero_image.size(0)):
    zero_image[i] = normalizer(zero_image[i])
print(zero_image.size())
print(zero_image[0])
sample_noise = torch.randn(10,1000)
torch.argmax(sample_noise,1)
model.cuda()
zero_image.cuda()
banana_tensor.cuda()
for i in cnn_model.parameters():
    print(i.device)
for i in classifier_model.parameters():
    print(i.device)
for i in fcc_1_param.parameters():
    print(i)
for i in fcc_1_param.parameters():
    print(i.device)
    print('\n')
for i in inp_param.parameters():
    print(i.device)
for i in fcc_1_param.parameters():
    i.to(torch.device('cuda:0'))
for i in inp_param.parameters():
    i.to(torch.device('cuda:0'))
#THESE DIDN'T WORK
for i in cnn_model.parameters():
    i.cuda()
banana_tensor = banana_tensor.to(hparams['device'])

#single forward pass of the noise through the whole model

fcc_1 = cnn_model(zero_image)
noise_out =  classifier_model(fcc_1).view(zero_image.size(0), -1)

print(noise_out.size())
print(noise_out.requires_grad)
#fcc_1 = fcc_1.view(fcc_1.size(0),-1)



'''
making a reference for measuring the change in first FCC layer activations after Gradient Ascent
'''
fcc_1_original = fcc_1.detach()


model_opt = torch.optim.Adam(model.parameters()) #TO BE USED ONLY TO SET GRADIENTS TO ZERO
fcc_1_optimizer = torch.optim.Adam(fcc_1_param.parameters())     
inp_optimizer = torch.optim.Adam(fcc_1_param.parameters())

print(fcc_1_optimizer)
print(inp_optimizer)


#FORWARD PASS OF ORIGINAL IMAGE

'''
Calculating final layer's scores on the original image to be regenerated
'''
scores = model(banana_tensor.unsqueeze(0))

print(scores.size())
print(scores.argmax(1).data)


'''
Having faith in the model to properly identify the fruit
'''

target = torch.zeros((1,1000))
target[0,scores.argmax(1)] = 1 
target = target.expand(hparams['batch_size'],1000)

print(target.size())

loss_fn = nn.MSELoss()


fcc_grad = []
inp_grad = []


for e in range(epochs):
    
    print(f'-----------------------------------EPOCH {e}--------------------------------')
    
    fcc_1 = cnn_model(zero_image)
    noise_out =  classifier_model(fcc_1).view(zero_image.size(0), -1)
    
   
    '''
    To save computational efficiency as the earlier layers are not required in the backpropagation
    '''
    
    for i in cnn_model.parameters():             
        i.requires_grad_(False)
    for i in fcc_1_param.parameters():             
        i.requires_grad_(True)
    
    '''
    Passing the fcc_1 output as a parameter to be optimized, should be easier as we sMorTly made a new model 'fcc_1_param' with fcc_1 as the sole parameter.
    Loop to optimize the fcc_1 activations with respect to the loss
    '''
    print(f'----------------------Classifier-----------------------')
    
    for t1 in range(iters_classifier):
        model_opt.zero_grad()
        fcc_1_optimizer.zero_grad()
        
        #noise_out =  classifier_model(cnn_model.fcc_1.data).view(zero_image.size(0), -1)              
        #THIS DOESN'T UPDATE THE GRAD OF 'cnn_model.fcc_1' PARAMETER, hence, it doesn't update 'cnn_model.fcc_1' either
        
        #THIS WORKS
        noise_out =  classifier_model(cnn_model.fcc_1).view(zero_image.size(0), -1)
        loss = loss_fn(noise_out,target.detach())        #to exclude 'target' from the computational graph
        loss.backward(retain_graph= True)
        
        if t1 % 5 == 0:     #frequency of sampling results displayed 
            for i in fcc_1_param.parameters():
                print(i.grad)
                fcc_grad.append(i.grad.cpu().mean())
        fcc_1_optimizer.step()
    
    print(((fcc_1_original - cnn_model.fcc_1.data)**2).sum())      #To check the L2 distance between the original fcc_1 tensor and updated one
    
    plt.figure()
    plt.title(f'Mean fcc_grad for Epoch {e} every 5 steps')
    plt.plot(fcc_grad)
    plt.show()
    fcc_grad.clear()
    
    '''
    Now, the whole network is required in the backpropagation as we need to calculate gradients of updated fcc_1 layer with respect to the input layer
    '''
    
    for i in cnn_model.parameters():             
        i.requires_grad_(True)
    
    
    print(f'----------------------CNN-----------------------')
    
    for t2 in range(iters_cnn):
        model_opt.zero_grad()
        inp_optimizer.zero_grad()
        #noise_out =  classifier_model(cnn_model.fcc_1).view(zero_image.size(0), -1)              #THIS DOESN'T UPDATE THE GRAD OF 'cnn_model.fcc_1' PARAMETER
        
        #THIS WORKS
        noise_out =  cnn_model(cnn_model.inp.detach).view(zero_image.size(0), -1)
        loss = loss_fn(noise_out,target.detach())        #to exclude 'target' from the computational graph
        loss.backward(retain_graph= True)
        
        if t2 % 20 == 0:    #frequency of sampling results displayed 
            for i in inp_param.parameters():
                print(i)
                inp_grad.append(i.grad.cpu().mean())
        inp_optimizer.step()
        zero_image
    
    print(((zero_image - cnn_model.inp.data)**2).sum()) 
    
    plt.figure()
    plt.title(f'Mean inp_grad for Epoch {e} every 20 steps')
    plt.plot(fcc_grad)
    plt.show()
    inp_grad.clear()
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
detransform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda x: UnNormalize(mean=(0.485, 0.456, 0.406) ,std=(0.229, 0.224, 0.225))(x[0])),
                                              torchvision.transforms.ToPILImage(),
                                              torchvision.transforms.Resize((96,96))])

images = cnn_model.inp
fig, ax = plt.subplots(2,5)
for row in ax:
    for col in row:
        col.imshow()