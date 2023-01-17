import numpy as np # numpy is base of everything we see in machine leraning using python
import pandas as pd # pandas handles the data in a formatted way
import torch # it is the pytorch package
from torch import nn # Neural Network Package
from torch import optim # Optimizer
import torch.nn.functional as F # We use Functional API for flexibility
import torchvision # to deal with images
from torchvision import transforms # performs transformations
import matplotlib.pyplot as plt # Our FAV
from tqdm import tqdm # Fancy tool that does not let you get bored while in training
from sklearn.metrics import confusion_matrix # confused? DOn't be. It is just a simple confusion matrix ;)
import seaborn as sns # how can we forget seaborn when we want fancy
from torch.utils.tensorboard import SummaryWriter # this is where the magic happens in real time
import itertools # you know, some tools to do random things for you
t = torch.tensor([[1,2],[3,4]]) # create a basic tensor
print(t.shape)
t
rank = len(t.shape) # length of shape = Rank of a tensor
rank
# rank of a tensors tells us that how deep do we have to go before we can access the values. Going deep below the
# rank will always give you another tensor of a 2-D shape
print(t[0][0]) # see the output that it's a tensor

# to access a value from it, use
item = t[0][0].item()
print(item)

# item object is only for tensors having " scalar values" try using t[0].item()
print(t.device) # get the current operating device
print(type(t))
print(t.type()) # get the type of a tensor. there are many types of tensors in PyTorch. See the docs
print(t.dtype) # what is the type of data inside that specific tensor
print(t.layout) # stride is just another name for Dense
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this will automatically set operations on GPU
print('Using device:', device)

if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
t1 = torch.tensor([[1,2],[2,3]])
t2 = torch.tensor([[1.1,2.2],[2.2,3.3]])
# t3 = t1.cuda() # No CUDA Present error

print(t1.dtype)
print(t2.dtype)
t1+t2
# print(t1+t3) # it'll produce an error. Check for yourself
data = np.array([1,2,3])

t1 = torch.Tensor(data)  # pass data inside the Constructor of the Tensor class directly
# equivalent to t = torch.tensor(data,dtype=torch.float32)

print(t1)
print(f't1 belongs to: {type(t1)}')
print(f'Type of t1 is: {t1.type()}')
print(f'Type of data inside t1 is {t1.dtype}')

t2 = torch.tensor(data) # Factory Function 
#  equivalent to t = torch.tensor(data,dtype=torch.int64)


print(t2)
print(type(t2))
print(t2.type())
print(t2.dtype)
t3 = torch.from_numpy(data) # (Factory) Function

print(t3)
print(type(t3))
print(t3.type())
print(t3.dtype)

t4 = torch.as_tensor(data) # Factory Function

print(t4)
print(type(t4))
print(t4.type())
print(t4.dtype)

data[0] = 5
data[2] = 7
print(t1)
print(t2)
print(t3)
print(t4)
# each value represents a pixel of 4*4 image
one = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(4,4)  # converts into a 4*4 matrix
print(one,'\n')

two = torch.tensor([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]).reshape(4,4)
print(two,'\n')

three = torch.tensor([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]).reshape(4,4)
print(three,'\n')

stacked = torch.stack((one,two,three)) # make a batch of 3 images so shape is (3,4,4)
print('Stacked:\n',stacked)

stacked_reshaped = stacked.reshape((3,1,4,4)) # batch, color channels, height, width

stacked_reshaped_flattened = stacked_reshaped.flatten(start_dim=1) 
# leave the batch as it is but flatten all the pixels
stacked_reshaped_flattened
another_method = stacked_reshaped.reshape((3,16)) # Flatten each image
another_method
print(t1*(3))
print(t1.mul(3))
# it is not scaler multiplication but the scaler is broadcasted
t1%2 == 0 
# It is just like making an tensor of same shape as of given tensor but with all the scaler values
broadcasted_3 = torch.from_numpy(np.broadcast_to(3,t1.shape))
print(f'Broadcasted tensor: {broadcasted_3}')
print(t1*broadcasted_3)
print(t1.mul(broadcasted_3))
train_set = torchvision.datasets.FashionMNIST(download=True,root='./Data/FashionMNIST',
                                              transform=transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set,batch_size=32)
print(f'There are {len(train_set.targets)} images in our data  where each one belongs to one of 10 classes and the classes are LabelEncoded as {dict(zip(train_set.classes,range(10)))}')

fig = plt.figure(figsize=(5,5))

plt.pie(train_set.targets.bincount().numpy(),
        labels=dict(zip(train_set.classes,range(10))),
        autopct='%1.2f%%')

plt.title('Distribution of Classes in the whole dataset',size='x-large')
plt.show()
sample = next(iter(train_set))
print(type(sample))
print(len(sample))
# first entry sample[0] are the pixels and second sample[1] is the label
pixels,label = sample # or pixels,labels = sample[0], sample[1]
print(pixels.shape)
plt.imshow(pixels.reshape(28,28),cmap='gray')
# plt.imshow(pixels.squeeze()) # there is an extra image axis so we have to remove it
plt.show()
batch = next(iter(train_loader)) # iter makes the list as an iterator. Iterator gives you an object one at a time
# With each time instance a new object is generated (yield) like factory rather than a warehouse (return)
print(f'A batch has {len(batch)} elements and type of batch is {type(batch)}')
images,labels = batch
print(f'Number of images in a single batch is {len(images)} and there are {len(labels)} labels corresponding to each image')
print(f'Type of Images is {type(images)} which is a collection of images  of (28x28) pixels')
print(f'Shape of those Images is {images.shape}')
grid = torchvision.utils.make_grid(images,nrow=10) #nrow is number of elements in a single row
plt.figure(figsize=(15,12))
plt.imshow(np.transpose(grid,(1,2,0))) # transpose to (h,w,c) instead of (c,h,w)
print(labels)
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__() # Super is indeed super
        
        self.conv_1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),bias=True)
        # for single grayscale image use 6 different kernls of size (5,5) to produce 6 diff feature maps 
        
        self.conv_2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=(3,3))
        # from the existing 6 already feat maps, use 12 different kernal filters of size (3,3) to get TOTAL of 12 new feat
        
        self.dense_1 = nn.Linear(in_features=12*4*4,out_features=128) # WHYY (12*4*4) it is explained in the end
        # Flatten the output of conv_2d in 12*4*4
        
        self.fc_2 = nn.Linear(in_features=128,out_features=64)
        # Fully Connected = fc_2 = Dense Layer = dense_2
        
        self.out = nn.Linear(in_features=64,out_features=10)
        # output layer. Output number of neurons = num of classes for classification & 1 for regression
        
    
    def forward(self,t):
        '''
        implement a forward pass on a Tensor 't' of rank 'R'
        '''
        # input  layer 1 though it is never needed
        # t = t
        
        # second layer. Layer is a mix of functions that has weights 
        t = self.conv_1(t) # works by calling __call__() method inside class
        t = F.relu(input=t,) # it is not a layer but Function (layers have weights, activations don't)
        t = F.max_pool2d(t,kernel_size=(3,3),stride=2) # max pooling
        
        # third layer
        t = self.conv_2(t) # works by calling __call__() method inside class
        t = F.relu(input=t,) # it is not a leyer but Function as layers have weights
        t = F.max_pool2d(t,kernel_size=(3,3),stride=2) # max pool
        
        # fourth layer
        t = t.reshape(-1,12*4*4) 
        # due to Conv and pooling operations, our image has been reduced from (1,28,28) to (4,4)
        # use ((input_size - filter_size + 2*padding)/stride )+1  for each  cov and max_pool 
        # it assumes input and kernel size are square
        t = self.dense_1(t)
        t = F.relu(t)
        
        # Fifth layer
        t = self.fc_2(t)
        t = F.relu(t)
        
        # output
        t = self.out(t)
        # t = F.softmax(t,dim=1)
        # commented because loss function used will be cross_entropy which has softmax behind the scenes
        return t
# torch.set_grad_enabled(False) # stop making computational graphs it is True by default
tb = SummaryWriter() # instantiate the tensorboard object

network = Network() # instantiate object of Nwtwork

batch = next(iter(train_loader))
images,labels = batch

grid = torchvision.utils.make_grid(images,nrow=15)

tb.add_image('image_grid',grid)
tb.add_graph(network,images)

tb.close()
# after running this, check your local machine's address given by the shell to see the graphs and everything
sample = next(iter(train_set))
image, label = sample
image.shape # add a new index to the the image to convert it into a batch of 1
batch_image = image.reshape((1,1,28,28))
print(batch_image.shape)

# or by using 

batch_image = image.unsqueeze(dim=0)
print(batch_image.shape)
y_pred = network(batch_image) 
y_pred
# y_pred is NOT the  probabilities for each label. These are final output Tensor because we have not used softmax
F.softmax(y_pred,dim=1) # these are the probabilities for each class
y_pred.shape # shape is (1,10) means 1 image and 10 predictions
y_pred.argmax(dim=1) # get the index where value is maximum
torch.set_grad_enabled(True) # True by default but as we had turned it off so turning it on
train_set = torchvision.datasets.FashionMNIST(download=True,root='./Data/FashionMNIST',
                                              transform=transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_set,batch_size=32) 
batch = next(iter(train_loader)) # make a new batch from DataLoader
images,labels = batch # get images and labels from batch
print(len(labels)) # 32 labels for 32 images
print(images.shape) # 32 grayscale images of size (28,28)
network = Network() # weights are random everytime you initialize
pred = network(images)
pred.shape # 10 labels for each of 32 images
pred.argmax(dim=1) # get predictions for the images after 1 instance

labels # original labels
pred.argmax(dim=1).eq(labels) # element wise operation. Return True if equal else return False
def get_correct_pred(pred,labels,percent=False):
    num = pred.argmax(dim=1).eq(labels).sum().item()
    if percent:
        return (num/len(labels))*100 # 4 out of 32 correct predictions
    else:
        return num
print(f'{get_correct_pred(pred,labels,True)}% of labels have been predicted correctly')
loss = F.cross_entropy(pred,labels)
loss.item() # this is our loss function

print(network.conv_1.weight.grad) # No Gradients  present NOW on the First Pass for any of the layer
loss.backward() # works only if set_grad_enable(True)
print(network.conv_1.weight.grad.shape) # Now you can see Updated Gradients for every layer
print(network.conv_2.weight.grad.shape)
optimizer = optim.Adam(network.parameters(),lr=0.01) 
# lr is learning rate. High learning rate is fast but produce less accuracy and vice versa

optimizer.step() # update weights
# Predict again and you will see a decrease in loss

pred = network(images)
print(get_correct_pred(pred,labels))
loss = F.cross_entropy(pred,labels)
loss.item()
BATCH = 128 # set the batch size
lr = 0.001 # set learning rate
EPOCH = 5 # set epoch. 1 epoch means the whole data will be presented to the network in batches. So in this case,
# Whole data will be presented to the network 5 times  in (60000//128) steps per epoch

network = Network() # instantiate network
optimizer = optim.Adam(network.parameters(),lr=lr) # instantiate optimizer

comment = f"Hyper Parameters: BATCH={BATCH}, lr={lr}" # this is a dynamic representation.You can put code inside {}
tb = SummaryWriter(comment=comment) # instantiate Tensorboard for live evaluation

for epoch in range(EPOCH): # train for 5 epoch
    
    total_loss = 0
    total_accuracy = 0

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH)  # load the data every time for new epoch
    
    for batch in tqdm(train_loader): # it's just a fancy thing for visualization
        images,labels = batch # fet images and corresponding labels

        pred = network(images) # feed the images 
        loss = F.cross_entropy(pred,labels,reduction='sum') # calculate loss on the whole batch
        # or you can do 
        # F.cross_entropy(pred,labels)* len(images)

        optimizer.zero_grad() # make the previous epoch's gradients as 0 as these can be accumulated
        # network.zero_grad() # we can use this too
        loss.backward() # back propogate
        optimizer.step() # start and update weights for each layer

        total_loss+= loss.item() # calculate total loss
        total_accuracy+= get_correct_pred(pred,labels) 
    
    acc = (total_accuracy/len(train_set))
    avg_loss = (total_loss/len(train_set)) # average loss and accuracy after current epoch
    
    print(f"End of Epoch: {epoch+1}, Training Accuracy: {acc}, Average Training Loss: {avg_loss}")
    
    
    tb.add_scalar('AVG Loss',avg_loss,epoch) # automatically make the epoch vs loss graph using tensorboard
    tb.add_scalar('Accuracy',acc,epoch)
    tb.add_scalar('Total Correct',total_accuracy,epoch)
    
    # tb.add_histogram('conv_1 Bias',network.conv_1.bias,epoch)
    # tb.add_histogram('conv_1 Weights',network.conv_1.weight,epoch)
    # tb.add_histogram('conv_1 Gradients',network.conv_1.weight.grad,epoch)
    
    for layer_name,weight in network.named_parameters(): # it is the automated version of manual commented lines above
        tb.add_histogram(layer_name,weight,epoch) # histogram of weights of each layer per epoch
        tb.add_histogram(f'{layer_name}.grad',weight.grad,epoch)
    
tb.close()
# @torch.no_grad() # if decorator is used here, we don't need with torch.no_grad() later
def get_all_pred_from_loader(model,loader):
    all_pred = torch.Tensor([])
    for batch in loader:
        images,labels = batch
        pred = model(images)
        
        all_pred = torch.cat((all_pred,pred),dim=0)
    return all_pred
with torch.no_grad(): # we do not want to compute gradients
    loader = torch.utils.data.DataLoader(train_set,batch_size=128)
    preds = get_all_pred_from_loader(network,loader)
    
total_correct = get_correct_pred(preds,train_set.targets)
accu = total_correct/len(train_set)
print(acc)
stacked = torch.stack((train_set.targets,preds.argmax(dim=1)),dim=1)
print(stacked.shape)
stacked[:5] # original, predicted
concat = torch.cat((train_set.targets,preds.argmax(dim=1)),dim=-1) # or use dim=0 . dim=1 gives error
print(concat.shape)
concat[:5] # original, predicted
conf_mat = torch.zeros((10,10),dtype=torch.int32)
for row_val in stacked:
    true,pred = row_val.tolist() # unpack row value
    conf_mat[true,pred] = conf_mat[true,pred]+1 # whatever value was there, +1. Because we have 0-9 label

conf_mat
conf_mat2 = confusion_matrix(train_set.targets,preds.argmax(dim=1)) # sklearn
conf_mat2
# change conf_mat tensor to numpy or directly pass conf_mat2
conf_df = pd.DataFrame(conf_mat.numpy(),columns=train_set.classes,index=train_set.classes)

f,ax = plt.subplots(1,1,figsize=(8,5))
sns.heatmap(conf_df,annot=True,lw=0.8,cmap='Pastel1',ax=ax,
            fmt='d', # fmt='d'/ fmt='g' for suppressing scientific notation
            annot_kws={"size": 10})

ax.set_xlabel('Predicted Labels',size='x-large')
ax.set_ylabel('Actual Labels',size='x-large')
torch.set_grad_enabled(True) # this is True by default

batch_size_list = [32,512]
lr_list = [0.01,0.001]


all_list = [batch_size_list,lr_list] 
permutations = list(itertools.product(*all_list))
# here are (2*2) combinations to try for model for 2 epochs each

for BATCH_SIZE,lr in tqdm(permutations):
    
    network = Network()
    optimizer = optim.Adam(network.parameters(),lr=lr)

    comment = f"Hyper Parameters: BATCH={BATCH_SIZE}, lr={lr}"
    tb = SummaryWriter(comment=comment)
    
    for epoch in range(2):

        total_loss = 0
        total_accuracy = 0
        
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE) 

        for batch in train_loader: 
            images,labels = batch

            pred = network(images)
            loss = F.cross_entropy(pred,labels,reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+= loss.item()
            total_accuracy+= get_correct_pred(pred,labels)

        acc = (total_accuracy/len(train_set))
        avg_loss = (total_loss/len(train_set))

        tb.add_scalar('AVG Loss',avg_loss,epoch)
        tb.add_scalar('Accuracy',acc,epoch)
        tb.add_scalar('Total Correct',total_accuracy,epoch)

        for layer_name,weight in network.named_parameters():
            tb.add_histogram(layer_name,weight,epoch)
            tb.add_histogram(f'{layer_name}.grad',weight.grad,epoch)

tb.close()
my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
print(my_x,'\n')
my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets or labels for corresponding X)
print(my_y,'\n')

tensor_x = torch.Tensor(my_x) # transform to torch tensor
print(tensor_x)
tensor_y = torch.Tensor(my_y)

my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset
my_dataloader = torch.utils.data.DataLoader(my_dataset) # create your dataloader
def load_dataset(data_path):
    # data_path = 'data/train/' # our your own directory
    train_dataset = torchvision.datasets.ImageFolder(root=data_path,transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,num_workers=0,shuffle=True)
    return train_loader

# use the above method like

# for batch_idx, (data, target) in enumerate(load_dataset()):
    #train network
# train_dataset=datasets.ImageFolder(root="./root/",transform=train_transforms)
x = np.zeros((3,10,20))
print('Before: ',x.shape)

# %timeit np.moveaxis(x,[0,1,2],[2,0,1]) 
x1 = np.moveaxis(x,[0,1,2],[2,0,1]) # use single digit for a single axis rotation
print('Method 1:', x1.shape)

# %timeit np.einsum('ijk->kij',x)
x2 = np.einsum('ijk->jki',x)
print('Method 2: ',x2.shape)

x3 = np.rollaxis(x, 0, 3) # roll any of the two axis or in simple terms, interchange
print('Method 3: ',x3.shape)

x4 = np.transpose(x,axes=(1,2,0))
print('Method 4: ',x4.shape)

x5 = np.reshape(x,(10,20,3)) # because original shape is 3,10,20
print('Method 5: ',x5.shape)
fig=plt.figure(figsize=(15,5))
plt.imshow(plt.imread('/kaggle/input/cnn-images/stack_concat.png'))
# courtesy of Deeplizard
x = np.zeros((3,10,20))
print('Original: ',x.shape)
print('Case 1: ',np.reshape(x,(1,3,1,1,10,20,1,1,1)).shape)
print('Case 2: ',np.reshape(x,(1,25,1,12,1,2,1)).shape)
for i in range(len(x.shape)+1):
    print('Add new axis at:',i)
    print('New Shape is: ',np.expand_dims(x,axis=i).shape,'\n')
t1,t2,t3 = np.array([1,1,1]),np.array([2,2,2]),np.array([3,3,3])
x_simple_cat_dim0 = np.concatenate((t1,t2,t3),)
x_simple_cat_dim0


# Below code will produce error as there is no axis=1
# x_simple_cat_dim1 = np.concatenate((t1,t2,t3),axis=1)
# print(x_simple_cat_dim1)
np.concatenate((np.expand_dims(t1,0),
                np.expand_dims(t2,0),
                np.expand_dims(t3,0)),
               axis=0)
np.concatenate((np.expand_dims(t1,0),
                np.expand_dims(t2,0),
                np.expand_dims(t3,0)),
               axis=1)
np.concatenate((np.expand_dims(t1,1),
                np.expand_dims(t2,1),
                np.expand_dims(t3,1)),
               axis=0)
np.concatenate((np.expand_dims(t1,1),
                np.expand_dims(t2,1),
                np.expand_dims(t3,1)),
               axis=1)
x_simple_stack_dim0 = np.stack((t1,t2,t3),axis=0)
x_simple_stack_dim0
x_simple_stack_dim1 = np.stack((t1,t2,t3),axis=1)
x_simple_stack_dim1
np.stack((np.concatenate((np.expand_dims(t1,1),np.expand_dims(t2,1),np.expand_dims(t3,1)),axis=1),
                np.stack((t1,t2,t3))),axis=0)
np.stack((np.concatenate((np.expand_dims(t1,1),np.expand_dims(t2,1),np.expand_dims(t3,1)),axis=1),
                np.stack((t1,t2,t3))),axis=1)
np.concatenate((np.concatenate((np.expand_dims(t1,1),np.expand_dims(t2,1),np.expand_dims(t3,1)),axis=1),
                np.stack((t1,t2,t3))),axis=0)
np.concatenate((np.concatenate((np.expand_dims(t1,1),np.expand_dims(t2,1),np.expand_dims(t3,1)),axis=1),
                np.stack((t1,t2,t3))),axis=1)
x = np.zeros((3,3))
x
x[0] = 3
x
x[0,1:] = 1
x
x[1:,1:] = [4.4,5.5] 
x