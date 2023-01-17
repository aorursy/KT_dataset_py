# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train/train.csv')
train_df.head()
pd.value_counts(train_df.AdoptionSpeed).plot.bar()
cat_df = train_df[train_df.Type == 2]
dog_df = train_df[train_df.Type == 1]
cat_df.AdoptionSpeed.mean()
dog_df.AdoptionSpeed.mean()
train_df.groupby(['Age'])['AdoptionSpeed'].mean().plot.bar()
normalize_train_df = train_df.drop(["RescuerID", "Description"], axis=1)
test_df = pd.read_csv("../input/test/test.csv")
normalize_train_df.head()
normalize_train_df["Name"] = normalize_train_df["Name"].fillna('No name')
test_df["Name"] = test_df["Name"].fillna('No name')
names = normalize_train_df["Name"].as_matrix().tolist() + test_df["Name"].as_matrix().tolist() 
import matplotlib.pyplot as plt
plt.hist([len(name) for name in names], 20, normed=1, facecolor='green', alpha=0.75)
characters = set("".join(names))
char2idx = {char:idx+1 for idx, char in enumerate(characters)}
def name_to_ids(name):
    ids = [char2idx[c] for c in name]
    ids = ids + [0] * 10
    return ids[:10]
continuous_columns = ["Age", "MaturitySize", "Quantity", "Fee", "VideoAmt", "PhotoAmt"]
scalers = {}
for column in continuous_columns:
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((normalize_train_df[column].as_matrix(),test_df[column].as_matrix())).reshape(-1, 1))
    normalize_train_df[column] = scaler.transform(normalize_train_df[column].as_matrix().reshape(-1,1)).reshape(-1)
    scalers[column] = scaler
normalize_train_df["Type"] = normalize_train_df["Type"] - 1
normalize_train_df["Gender"] = normalize_train_df["Gender"] - 1
normalize_train_df["Vaccinated"] = normalize_train_df["Vaccinated"] - 1
normalize_train_df["Dewormed"] = normalize_train_df["Dewormed"] - 1
normalize_train_df["Sterilized"] = normalize_train_df["Sterilized"] - 1
states = [41336, 41325, 41367, 41401, 41415, 41324, 41332, 41335, 41330, 41380, 41327, 41345, 41342, 41326, 41361]
state2index = {'State': {state: idx for idx, state in enumerate(states)}}
normalize_train_df.replace(state2index, inplace=True)
normalize_train_df.head()
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data as utils
from torchvision import transforms, models
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(normalize_train_df, test_size=0.2)
import math
import random

imgtransCrop = 224
transform = transforms.Compose([transforms.RandomResizedCrop(imgtransCrop),
                                transforms.RandomHorizontalFlip(),                           
                                transforms.ToTensor(),                                
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

val_transform = transforms.Compose([transforms.Resize((imgtransCrop,imgtransCrop)),                          
                                transforms.ToTensor(),                                
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

def create_categorical_dfs(data):
    categorical_dfs = []
    for i in range(5):
        categorical_dfs.append(data[data.AdoptionSpeed == i].drop(["AdoptionSpeed"], axis=1).sample(frac=1).reset_index(drop=True))
    return categorical_dfs

def data_generator(categorical_dfs, image_transform, folder_path="../input/train_images", batch_size=15):
    result = pd.DataFrame(columns=categorical_dfs[0].columns)
    images = []
    names = []
    labels = []
    
    for i in range(batch_size//5):
        
        for idx in range(5):
            picked_data_idx = random.randint(0, len(categorical_dfs[idx])-1)
            result = result.append(categorical_dfs[idx].drop(["PetID"], axis=1).loc[picked_data_idx], ignore_index=True)
            
            image_path = os.path.join(folder_path,str(categorical_dfs[idx].loc[picked_data_idx]["PetID"])+"-1.jpg")
            try:
                image = Image.open(image_path).convert('RGB')
            except:
                image = image = Image.new('RGB', (300, 300))
            if image_transform:
                image = image_transform(image)
            images.append(image)
            names.append(name_to_ids(categorical_dfs[idx].loc[picked_data_idx]["Name"]))
            labels.append(idx)
    return result, torch.stack(images), names, labels
categorical_dfs = create_categorical_dfs(train_data)
val_categorical_dfs = create_categorical_dfs(validation_data)
class NameAnalysis(nn.Module):
    def __init__(self, device, character_size, embedding_dim, hidden_dim):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(character_size, embedding_dim).to(self.device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(1, batch_size, self.hidden_dim).to(self.device))
    def forward(self, names):
        embeds = self.embedding(torch.LongTensor(names).to(self.device))
        hidden = self.init_hidden(len(names))
        lstm_out, hidden = self.lstm(
            embeds.view(len(names[0]), len(names), -1), hidden)
        return lstm_out[-1]
class AdoptionSpeedModel(nn.Module):
    def __init__(self, device, embedding_dim, category2size, num_continues_features, character_size=len(char2idx)+1, char_embedding_dim=80, hidden_dim=5):
        super().__init__()
        self.device = device
        self.feature2embedding = {}
        for category in category2size.keys():
            self.feature2embedding[category] = nn.Embedding(category2size[category], embedding_dim).to(self.device)
        
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(character_size, char_embedding_dim).to(self.device)
        self.lstm = nn.LSTM(char_embedding_dim, hidden_dim)
        
        self.resnet = models.resnet18(pretrained=False)
        kernel_count = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(kernel_count, 300)
        self.fc = nn.Sequential(nn.Linear(embedding_dim*len(category2size.keys()) + num_continues_features + 300 + 5, 300), 
                                nn.Dropout(0.2), 
                                nn.Linear(300, 200), 
                                nn.Dropout(0.2), 
                                nn.Linear(200, 100), 
                                nn.Dropout(0.2), 
                                nn.Linear(100, 50), 
                                nn.Dropout(0.2), 
                                nn.Linear(50, 5), 
                                nn.LogSoftmax(dim=1))
        
    def init_hidden(self, batch_size):
        return (torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad=False).double().to(self.device),
                torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim), requires_grad=False).double().to(self.device))
        
    def forward(self, x):
        data, images, names = x
        embeds = None
        data = data.drop(["PetID", "Name"], axis=1)
        for feature in self.feature2embedding.keys():
            if embeds is None:
                embeds = self.feature2embedding[feature](torch.LongTensor(data[feature].values.astype(np.int64)).to(self.device))
            else:
                embeds = torch.cat((embeds, self.feature2embedding[feature](torch.LongTensor(data[feature].values.astype(np.int64)).to(self.device))), 1)
        
        char_embeds = self.embedding(torch.LongTensor(names).to(self.device))
        self.hidden = self.init_hidden(len(names))
        lstm_out, self.hidden = self.lstm(
            char_embeds.view(len(names[0]), len(names), -1), self.hidden)
        lstm_out = lstm_out[-1]
        resnet_out = self.resnet(torch.FloatTensor(images).to(self.device))
        #embeds = embeds.view(len(names),-1)
        #print(lstm_out.size(), embeds.size())
        embeds = torch.cat((embeds, 
                            torch.FloatTensor(data.drop(self.feature2embedding.keys(), axis=1).values.astype(np.float32)).to(self.device), 
                            resnet_out, 
                            lstm_out),1)
        return self.fc(embeds)
category2size = {
    "Type": 2,
    "Breed1": 308,
    "Breed2": 308,
    "Gender": 3,
    "Color1": 8,
    "Color2": 8,
    "Color3": 8,
    "FurLength": 4,
    "Vaccinated": 3,
    "Dewormed": 3,
    "Sterilized": 3,
    "Health": 4,
    "State": 15
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = AdoptionSpeedModel(device, 5, category2size, 6)
model = model.to(device)
#state_dict = torch.load('new_checkpoint.pth',map_location=device)
#model.load_state_dict(state_dict)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min', verbose=True)
import sys
epochs = 15
batch_size = 16
steps = int(12000/batch_size)
val_steps = int(steps/4)
train_losses, test_losses = [], []
loss_min = 100000
for e in range(epochs):
    running_loss = 0
    model.train()
    for step in range(steps):
        data, images, names, labels = data_generator(categorical_dfs,image_transform=transform, folder_path="../input/train_images", batch_size=batch_size)
        optimizer.zero_grad()
        log_ps = model((data, images, names))
        loss = loss_function(log_ps, torch.LongTensor(labels).to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        sys.stdout.write(f"\rEpoch {e+1}/{epochs}... Step {step+1}/{steps}... Training loss {running_loss/(step+1)}")
    else:
        test_loss = 0
        accuracy = 0
        print()
        model.eval()
        with torch.no_grad():
            for step in range(val_steps):
                data, images, names, labels = data_generator(val_categorical_dfs,image_transform=val_transform, folder_path="../input/train_images", batch_size=batch_size)
                labels = torch.LongTensor(labels).to(device)
                log_ps = model((data, images, names))
                test_loss += loss_function(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                sys.stdout.write(f"\rEpoch {e+1}/{epochs}... Step {step+1}/{val_steps}... Validation loss {test_loss/(step+1)}... Accuracy {accuracy*100/(step+1)}")
        train_losses.append(running_loss/steps)            
        test_losses.append(test_loss/val_steps)
        print()
        scheduler.step(test_loss/steps)
        if test_loss/val_steps < loss_min:
            print(f"Improve loss from {loss_min} to {test_loss/val_steps}")
            loss_min = test_loss/val_steps
            torch.save(model.state_dict(), 'new_checkpoint.pth')
        else:
            state_dict = torch.load('new_checkpoint.pth',map_location=device)
            model.load_state_dict(state_dict)
        print("\nEpoch: {}/{}.. ".format(e+1, epochs),                  
              "Training Loss: {:.3f}.. ".format(running_loss/steps),                  
              "Val Loss: {:.3f}.. ".format(test_loss/val_steps),                  
              "Val Accuracy: {:.3f}\n\n".format(accuracy/val_steps))
state_dict = torch.load('new_checkpoint.pth',map_location=device)
model.load_state_dict(state_dict)
test_df = pd.read_csv("../input/test/test.csv")
test_df["Name"] = test_df["Name"].fillna('No name')
processed_test_df = test_df.drop(["RescuerID", "Description"], axis=1)
for column in continuous_columns:
    processed_test_df[column] = scalers[column].transform(processed_test_df[column].values.reshape(-1,1)).reshape(-1)
    
processed_test_df["Type"] = processed_test_df["Type"] - 1
processed_test_df["Gender"] = processed_test_df["Gender"] - 1
processed_test_df["Vaccinated"] = processed_test_df["Vaccinated"] - 1
processed_test_df["Dewormed"] = processed_test_df["Dewormed"] - 1
processed_test_df["Sterilized"] = processed_test_df["Sterilized"] - 1

processed_test_df.replace(state2index, inplace=True)
processed_test_df.head()
model.eval()
folder_path = "../input/test_images"
image_transform = val_transform
final_result = {}
batch_test_df = processed_test_df.copy()
while batch_test_df.shape[0] > 0:
    batch = batch_test_df[:50]
    try:
        batch_test_df = batch_test_df[50:].reset_index(drop=True)
    except:
        pass
    
    images = []
    names = []
    for _, row in batch.iterrows():
        image_path = os.path.join(folder_path,str(row["PetID"])+"-1.jpg")
        image = None
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print("Cannot found any image of "+row["PetID"])
            image = image = Image.new('RGB', (300, 300))
        if image_transform:
            image = image_transform(image)
        name = name_to_ids(row["Name"])
        images.append(image)
        names.append(name)
    with torch.no_grad():
        output = model((batch, torch.stack(images), names))
    ps = torch.exp(output)
    top_p, top_class = ps.topk(1, dim=1)
    pred = top_class.cpu().numpy().reshape(-1,)
    for idx, row in batch.iterrows():
        final_result[row["PetID"]] = pred[idx]
    
final_df = pd.read_csv("../input/test/sample_submission.csv")
final_df.info()
for idx, row in final_df.iterrows():
    final_df.loc[idx,'AdoptionSpeed'] = final_result[row.PetID].astype(np.int)
final_df[final_df.AdoptionSpeed != 0]
final_df.to_csv("submission.csv",index=False)
pd.read_csv("submission.csv").head()