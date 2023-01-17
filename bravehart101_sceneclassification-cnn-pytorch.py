import os 
import torch  
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 
from PIL import Image

from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid

torch.manual_seed(5)
np.random.seed(5)

%matplotlib inline
!pip install torchsummary
train_img_parent = "../input/intel-image-classification/seg_train/seg_train" # train directory address
test_img_parent = "../input/intel-image-classification/seg_test/seg_test" # test directory address

classLabels = {'buildings':0, 'mountain':1, 'street':2, 'forest':3, 'sea':4, 'glacier':5}
def csv_maker(parent_dir, class_label, csv_name):
    """Converts unstructured data stored in subdirectories into dataframe and csv file
    
    Args-
        parent_dir- String containing address of parent directory(test data or training data)
        class_label- Class label dictionary object
        csv_name- Name you want to give to your returned csv file (string) 
        
    Returns-
        dataframe- A pandas.DataFrame object
    """
    labelled_arr = np.array([]) # creates an empty array
    
    for subdir, label in class_label.items():
        img_dir = os.path.join(parent_dir, subdir) # gets the path of each subdirectory in the parent data directory
        files = np.array(os.listdir(img_dir)).reshape(-1,1) # gets the list of names of each image
        labels = np.array([label for i in range(files.shape[0])]).reshape(-1,1) #creates a label column for the images
        data = np.concatenate((files, labels), axis = 1) # concatenates file name and label arrays into a single array
        labelled_arr = np.append(labelled_arr, data)
    
    labelled_arr = labelled_arr.reshape(-1,2)
    
    np.random.seed(5)
    np.random.shuffle(labelled_arr) # shuffles the dataset
    
    dataframe = pd.DataFrame(labelled_arr)
    dataframe.columns = ['image', 'label']
    dataframe['label'] = dataframe['label'].astype('int') 
    
    dataframe.to_csv(csv_name, index = False) # creates the csv file for the dataframe
    
    return dataframe
train_df = csv_maker(train_img_parent, classLabels, csv_name = "train.csv")

test_df = csv_maker(test_img_parent, classLabels, csv_name = "test.csv")

train_csv = "./train.csv"
test_csv = "./test.csv"
print("\nTraining DF-\n")
print(train_df.head())
print("\nTesting DF-\n")
print(test_df.head())
# creating custom pytorch dataset

class ImageDataset(Dataset):
    def __init__(self, dataframe, data_dir, label_dict, transform = None):
        self.df = dataframe
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name, label = self.df.loc[idx]
        class_labels = list(self.label_dict.keys())
        img_path = self.data_dir + '/' + class_labels[label] + '/' + img_name
        img = img = Image.open(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label        
# R_sum = 0
# G_sum = 0
# B_sum = 0

# for i in range(len(train_dataset)):
#     R_sum += train_dataset[i][0][0].sum()
#     G_sum += train_dataset[i][0][1].sum()
#     B_sum += train_dataset[i][0][2].sum()

# R_mean = R_sum / (256*256*len(train_dataset))
# G_mean = G_sum / (256*256*len(train_dataset))
# B_mean = B_sum / (256*256*len(train_dataset))
# mean = (R_mean, G_mean, B_mean)
mean = (0.4302, 0.4575, 0.4539)
# R2_sum = 0
# G2_sum = 0
# B2_sum = 0

# for i in range(len(train_dataset)):
#     R2_sum += ((train_dataset[i][0][0] - mean[0])**2).sum()
#     G2_sum += ((train_dataset[i][0][1] - mean[1])**2).sum()
#     B2_sum += ((train_dataset[i][0][2] - mean[2])**2).sum()

# R_std = (R2_sum / (256*256*len(train_dataset)))**0.5
# G_std = (G2_sum / (256*256*len(train_dataset)))**0.5
# B_std = (B2_sum / (256*256*len(train_dataset)))**0.5
# std = (R_std, G_std, B_std)
std = (0.2606, 0.2588, 0.2907)
# We will try different transforms on our training data and compare the results. For now, let us stick to the training transforms given below

extra_transforms = (transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3), transforms.RandomPerspective(distortion_scale=0.2, p=0.5, interpolation=3, fill=0))


transformTrain1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    transforms.RandomErasing(),
])

transformTrain2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply(extra_transforms, p=0.2),
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.RandomErasing(scale=(0.02, 0.2)),
])

# Transforms for test data
transformTest = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
])
    
# creating training dataset
train_dataset = ImageDataset(train_df, train_img_parent, classLabels, transform = transformTrain2)

# creating testing dataset
test_dataset = ImageDataset(test_df, test_img_parent, classLabels, transform = transformTest)
print("Total images in the dataset:", len(train_dataset))
img, label = train_dataset[0]
print(img[:,:5,:5]) # printing a small 3 x 5 x 5 slice of the 3 x 256 x 256 tensor
print("Image label:", list(classLabels.keys())[label])
plt.imshow(img.permute(1,2,0))

batch_size = 32
                                                                                
train_dl = DataLoader(
    dataset = train_dataset, 
    batch_size = batch_size, 
    shuffle = True, 
    num_workers = 2, 
    pin_memory = True)

test_dl = DataLoader(
    dataset = test_dataset, 
    batch_size = batch_size, 
    shuffle = False, 
    num_workers = 2, 
    pin_memory = True)
train_dl
def batch_viewer(dataloader):
    """Shows the images in a batch returned by the PyTorch dataloader object.
    
    Args-
        dataloader- PyTorch dataloader object
    
    Returns-
        None
    """
    for images, labels in dataloader:
        fig, ax = plt.subplots(figsize = (16,16))
        ax.imshow(make_grid(images, nrow = 8).permute(1, 2, 0))
        break    
batch_viewer(train_dl)
def accuracy(output, labels):
    """Calculates the accuracy for the predicted output and the actual label values.
    
    Agrs-
        output- Output tensor generated by the model
        labels- Actual labels for the given batch of data
    
    Returns-
        accuracy- Accuracy percentage for the predictions
    """
    softmax = nn.Softmax(dim=1)
    output = softmax(output) # converts output values to probability values for each class
    preds = torch.argmax(output, axis = 1) # sets class with max probability as prediction 
    accuracy = torch.sum(preds==labels).item() / len(labels) # accuracy = correct_prediction / total_predictions
    return torch.Tensor([accuracy])

class SceneClassificationBase(nn.Module):
    def training_step(self, batch):
        """Calculates the cross entropy loss for a given batch of data.
        
        Args-
            batch- One batch of data as generated by the data loader
        
        Returns-
            batch_loss- Total cross entropy loss for the batch
        """
        images, labels = batch 
        output = self(images)                  # generates predictions for the batch of images
        batch_loss = F.cross_entropy(output, labels) # calculates loss for the predictions and actual labels
        return batch_loss
    
    def validation_step(self, batch):
        """Calculates total validation loss and validation accuracy for a given batch data during a validation step.
        
        Args-
            batch- One batch of data as generated by the data loader
            
        Returns-
            A dictionary object containing validation loss and validation accuracy for the given batch
        """
        images, labels = batch 
        output = self(images)                    # generate predictions for given batch
        batch_loss = F.cross_entropy(output, labels)   # calculates batch loss
        batch_acc = accuracy(output, labels)           # calculate batch accuracy
        return {'val_loss': batch_loss.detach(), 'val_acc': batch_acc}
        
    def validation_epoch_end(self, outputs):
        """Calculates mean validation loss and mean validation accuracy for a one validation epoch.
        
        Args-
            outputs- A list of dictionary objects containing validation accuracy and validation loss for each batch of data in one epoch
            
        Returns-
            A dictionary object containing validation loss and validation accuracy for the given batch
        """
        batch_losses = [batch_val_dict['val_loss'] for batch_val_dict in outputs] # creates a list of batch losses for all the batches in one validation epoch
        epoch_loss = torch.stack(batch_losses).mean()   # calculates mean validation loss for the epoch
        batch_accs = [batch_val_dict['val_acc'] for batch_val_dict in outputs]   # creates a list of batch accuracies for all the batches in one validation epoch 
        epoch_acc = torch.stack(batch_accs).mean()      # calculates mean validation accuracy for the epoch
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, 
            result['lrs'][-1], 
            result['train_loss'], 
            result['val_loss'], 
            result['val_acc']))
class Model_ResNeXt(SceneClassificationBase):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(    # 3 x 256 x 256
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1),    # 64 x 256 x 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
        )    
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),    # 128 x 256 x 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),    # 128 x 128 x 128 
        )
        self.resnxt1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1), # 128 x 128 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1), # 128 x 128 x 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),    # 256 x 128 x 128
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),   # 256 x 64 x 64
            nn.ReLU(inplace = True), 
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, padding = 1),    # 512 x 64 x 64
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),    # 512 x 32 x 32
            nn.ReLU(inplace = True), 
        )
        self.resnxt2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1), # 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1), # 512 x 32 x 32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
        )
        self.classifier = nn.Sequential(nn.AvgPool2d(2, 2), # 512 x 16 x 16
                                        nn.Flatten(), 
                                        nn.Linear(512 * 16 * 16, 6),
        )
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.resnxt1(output) + self.resnxt1(output) + self.resnxt1(output) + self.resnxt1(output) + output
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.resnxt2(output) + self.resnxt2(output) + self.resnxt2(output) + self.resnxt2(output) + output
        output = self.classifier(output)
        return output
        
model = Model_ResNeXt()
model
from torchsummary import summary

summary(model, input_size=(3, 256, 256))
def get_default_device():
    """Picks the trainig device-- GPU if available, else CPU.
    """
    if torch.cuda.is_available():   # checks if a cuda device is available
        return torch.device('cuda') # sets the default device as the available CUDA device
    else:
        return torch.device('cpu')  # if no CUDA device found, sets CPU as the default device
    
def to_device(data, device):
    """Move tensor(s) to chosen device
    """
    if isinstance(data, (list,tuple)): # asserts if the data is a list/tuple 
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to the default device.
    """
    def __init__(self, dataloader, device):
        self.dl = dataloader
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device.
        """
        for batch in self.dl: 
            yield to_device(batch, self.device)

    def __len__(self):
        """Prints the total number of batches.
        """
        return len(self.dl)
device = get_default_device()
device
train_dl = DeviceDataLoader(train_dl, device)
test_dl = DeviceDataLoader(test_dl, device)
to_device(model, device)
def try_batch(dataloader):
    for images, labels in dataloader:
        print('images.shape:', images.shape)
        out = model(images)
        print('out.shape:', out.shape)
        print('out[0]:', out[0])
        break

try_batch(train_dl)
from tqdm.notebook import tqdm

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()    # sets the model to evaluation mode
    outputs = [model.validation_step(batch) for batch in val_loader] # performs validation for each batch and stores it in a list
    return model.validation_epoch_end(outputs) # returns mean validation accuracy and validation loss for one complete epoch

def get_lr(optimizer):
    """Gets the learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay = 0, grad_clip = None, opt_func = torch.optim.SGD):
    
    torch.cuda.empty_cache()    # clears cache in CUDA device
    history = []    # declares an empty list to store result for each epoch
    
    # sets up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay = weight_decay)
    
    # sets up one-cycle learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs = epochs, steps_per_epoch = len(train_loader))
    
    for epoch in range(epochs): 
        model.train()    # initiate training phase
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):    # cycles through each batch of the training data
            loss = model.training_step(batch)    
            train_losses.append(loss)
            loss.backward()
            
            # perfomrs gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step() # updates parameters based on gradients obtained via optimizer.backwards()
            optimizer.zero_grad() # resets gradient values
            
            # records & updates learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()
        
        # initaites validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
initial_acc_loss = evaluate(model, test_dl)
print(initial_acc_loss)
epochs = 10
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
%%time
history = fit_one_cycle(epochs, max_lr, model, train_dl, test_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
epochs = 5
max_lr = 0.001
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

history += fit_one_cycle(epochs, max_lr, model, train_dl, test_dl, 
                             grad_clip=grad_clip, 
                             weight_decay=weight_decay, 
                             opt_func=opt_func)
trainingComplete = True
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. No. of Epochs')
    
plot_accuracies(history)
def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of Epochs')
    
plot_losses(history)
def predict_single_image(image_data):
    image, label = image_data
    label = list(classLabels.keys())[label]
    # displaying the image
    plt.imshow(image.permute(1,2,0))
    print("Actual label: ", label)
    # using model to predict image label
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    prediction = preds[0]
    prediction = list(classLabels.keys())[torch.argmax(prediction).item()]
    print("Prediction: ", prediction)
predict_single_image(test_dataset[69])
predict_single_image(test_dataset[420])
predict_single_image(test_dataset[100])
torch.save(model.state_dict(), 'sceneClassification_ResNeXt.pth')