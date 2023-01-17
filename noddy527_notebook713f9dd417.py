import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import seaborn as sns

%matplotlib inline

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import shutil
training_folder_name = './train'
img_size = (300,300)
classes = sorted(os.listdir(training_folder_name))
print(classes)

from PIL import Image


def resize_image(src_image, size=(300,300), bg_color="white"): 
    from PIL import Image, ImageOps 
    src_image.thumbnail(size, Image.ANTIALIAS)
    new_image = Image.new("RGB", size, bg_color)
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
    return new_image
training_folder_name = './train'
train_folder = './train_resized'

size = (300,300)

if os.path.exists(train_folder):
    shutil.rmtree(train_folder)


print('Transforming images...')
for root, folders, files in os.walk(training_folder_name):
    for sub_folder in folders:
        print('processing folder ' + sub_folder)
        
        saveFolder = os.path.join(train_folder,sub_folder)
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
     
        file_names = os.listdir(os.path.join(root,sub_folder))
        for file_name in file_names:
          
            file_path = os.path.join(root,sub_folder, file_name)
           
            image = Image.open(file_path)
            
            resized_image = resize_image(image, size)
            saveAs = os.path.join(saveFolder, file_name)
            
            resized_image.save(saveAs)

print('Done.')
def load_dataset(data_path):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    # Load all the images
    transformation = transforms.Compose([
        # Randomly augment the image data
            # Random horizontal flip
        transforms.RandomHorizontalFlip(0.5),
            # Random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )
    
    
    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=40,
        num_workers=0,
        shuffle=False
    )
    
    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=40,
        num_workers=0,
        shuffle=False
    )
        
    return train_loader, test_loader




#####################################################################################################




# Recall that we have resized the images and saved them into
train_folder = './train_resized'

# Get the iterative dataloaders for test and training data
train_loader, test_loader = load_dataset(train_folder)
batch_size = train_loader.batch_size
print("Data loaders ready to read", train_folder)
class Net(nn.Module):
    
    
    # Constructor
    def __init__(self, num_classes=3):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
     
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)
        
        self.fc = nn.Linear(in_features=75 * 75 * 24, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x))) 
    
        x = F.relu(self.pool(self.conv2(x)))  
        
        x = F.dropout(self.drop(x), training=self.training)
        
        x = x.view(-1, 75 * 75 * 24)
        
        x = self.fc(x)
        
        return torch.log_softmax(x, dim=1)
    
device = "cpu"
if (torch.cuda.is_available()):
    
    device = "cuda"
model = Net(num_classes=len(classes)).to(device)

print(model)
def train(model, device, train_loader, optimizer, epoch):
    
    model.train()
    train_loss = 0
    print("Epoch:", epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
       
        data, target = data.to(device), target.to(device)
        
        
        optimizer.zero_grad()
        
        
        output = model(data)
        
        loss = loss_criteria(output, target)

        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss
# Use an "Adam" optimizer to adjust weights
optimizer = optim.Adam(model.parameters(), lr=0.004)

# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 10 epochs (We restrict to 10 for time issues)
epochs = 50
print('Training on', device)
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
plt.figure(figsize=(15,15))
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()



from tqdm.notebook import tqdm

transform_test = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize((.2, .2, .2), (.5, .5, .5))

    ])

cals = {0: 'basketball_court', 1: 'beach', 2: 'forest', 3: 'others', 4: 'railway', 5: 'swimming_pool', 6: 'tennis_court'}
classes = {1: 'basketball_court', 2: 'beach', 3: 'forest', 4: 'railway', 5: 'tennis_court', 6: 'swimming_pool', 7: 'others'}
data = pd.DataFrame(columns=["ImageID", "Label"])
for filename in tqdm(os.listdir("./test_resized")):
  img = Image.open("./test_resized/"+filename)
  print(img)
  img = transform_test(img)
  img = torch.reshape(img, (1, 3, 300, 300))
  input_img = img.to(device)
  outputs = model(input_img)
  outputs = outputs.cpu()
  outputs = list(outputs[0].detach().numpy())
  pred_class = cals[outputs.index(max(outputs))]
  if pred_class =='basketball_court':
    pred_label = 1
  elif pred_class == 'beach':
    pred_label = 2
  elif pred_class == 'forest':
    pred_label = 3
  elif pred_class == 'railway':
    pred_label = 4
  elif pred_class == 'tennis_court':
    pred_label = 5
  elif pred_class == 'swimming_pool':
    pred_label = 6
  elif pred_class == 'others':
    pred_label = 7
  
  data = data.append({"ImageID": int(filename[:4]),
                      "Label": pred_label}, ignore_index=True)
data.to_csv(r'./submission.csv', index = False, header=True)







