import torch
import torch.nn as nn 
from tqdm import tqdm 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch.optim as optim 
from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
EPOCHS =30 
BATCH_SIZE=512
TEST_BATCH_SIZE=64
TRAIN_FILE = '../input/digit-recognizer/train.csv'
TEST_FILE='../input/digit-recognizer/test.csv'
SUBMIT_FILE = '../input/digit-recognizer/sample_submission.csv'
MODEL_PATH = '/kaggle/working/model.pt'
SUBMITTION_FILE ='/kaggle/working/submission.csv'

class DigitDataset():
    def __init__(self,pixels, labels):
        self.pixels= pixels
        self.labels=labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,index):
        pixels = self.pixels[index]
        labels = self.labels[index]
#         channel, height and width 
        pixels= pixels.reshape((1, 28,28))
        
        return { 'pixels' : torch.tensor(pixels, dtype = torch.float),
                'labels': torch.tensor(labels,dtype = torch.long)            
        } 
        
class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel,self).__init__()
#         in channel, out channel, kernel , stride 
        self.conv1 = nn.Conv2d(1,32,5,1,padding=2) 
        self.conv2 = nn.Conv2d(32,32,5,1,padding=2)
#         self.bn1 = nn.BatchNorm2d(32)
        self.maxp1 = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.conv3 = nn.Conv2d(32,64,3,1,padding=1) 
        self.conv4 = nn.Conv2d(64,64,3,1,padding=1) 
#         self.bn2 = nn.BatchNorm2d(64)
        self.maxp2 = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.fc1 = nn.Linear(3136,1568)
        self.fc2 = nn.Linear(1568,10)
        self.dr1 = nn.Dropout(0.25)  
        self.dr2 = nn.Dropout(0.50)  
       
    def forward(self,x):
        
        # Convolution 1
        x = F.relu(self.conv1(x))
        
        # Convolution 2
        x = self.maxp1(F.relu(self.conv2(x)))
        x = self.dr1(x)        
        
        # Convolution 3
        x = F.relu(self.conv3(x))
        x = self.maxp2(F.relu(self.conv4(x)))
        x = self.dr1(x)
        
        # Flatten 
        x = torch.flatten(x,1)          
        
        # FC 1
        x = F.relu(self.fc1(x)) 
        x = self.dr2(x)
        
        # FC 2
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x
         
        
df_train = pd.read_csv(TRAIN_FILE).reset_index(drop=True)
df_testing = pd.read_csv(TEST_FILE).reset_index(drop=True)

# check columns
df_train.columns
df_train.describe()
# check the null value 
df_train.isna().sum()
df_train.head() 

X = df_train.loc[:,df_train.columns != 'label'].values/255
y = df_train.label.values
df_train, df_test,df_train_label,df_test_label = train_test_split(X,y, test_size =0.1,random_state=24)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
from torchvision.utils import make_grid

random_select = np.random.randint(100,size = 9)
grid = make_grid(torch.Tensor(df_train[random_select].reshape((-1, 28, 28))).unsqueeze(1), nrow=9)
plt.rcParams['figure.figsize'] =(16,2)
plt.imshow(grid.numpy().transpose((1,2,0)))
plt.axis('off')
train_dataset = DigitDataset(df_train,df_train_label)
test_dataset = DigitDataset(df_test,df_test_label)
train_dataloader = DataLoader(train_dataset,batch_size =BATCH_SIZE,num_workers=4)
test_dataloader = DataLoader(test_dataset,batch_size =BATCH_SIZE,num_workers=4)
device = torch.device("cuda")
model = DigitModel()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterian = nn.CrossEntropyLoss()
def train_fn(model,criterian,data_loader,device,optimizer):
    model.train()
    running_loss =0.0
    
    for bi , d in tqdm(enumerate(data_loader), total=len(data_loader)):
        pixels = d['pixels']
        labels = d['labels']
        pixels= pixels.to(device,dtype=torch.float)
        labels= labels.to(device,dtype=torch.long)
        optimizer.zero_grad()
        outputs  = model(pixels)
        loss = criterian(outputs,labels)
#         loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if bi % 100 == 99:    # print every 2000 mini-batches
            print('loss: %.3f' %
                  (running_loss / 2000))
            running_loss = 0.0
        
        
        
def eval_fn(data_loader, model, optimizer, device):
    model.eval()
    fin_labels=[]
    fin_outputs = []
    lossdetails =[]
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader),total =len(data_loader)):
                        
            pixels = d["pixels"]
            labels = d ["labels"]
            pixels= pixels.to(device,dtype=torch.float)
            labels= labels.to(device,dtype=torch.long)
            
            
            outputs = model(pixels)
            pred = outputs.argmax(dim=1, keepdim=True) 
            
            fin_labels.extend(labels.tolist())
            fin_outputs.extend(pred.tolist())
    return fin_outputs, fin_labels


best_accuracy =0 
for epoch in range(EPOCHS):
    train_fn(model,criterian,train_dataloader,device,optimizer)
    outputs, labels = eval_fn(test_dataloader,model,optimizer,device)
    
    accuracy = accuracy_score(labels, outputs)
    
    if accuracy > best_accuracy:
        print(f"Epoch = {epoch} : Accuracy Score = {accuracy}")
#         torch.save(model.state_dict(), MODEL_PATH)
        best_accuracy = accuracy
    
    
df_test = pd.read_csv(TEST_FILE).reset_index(drop=True)
X = df_test.values/255.
df_test['label'] = -1
y = df_test.label.values
test_dataset = DigitDataset(X,y)
test_dataloader = DataLoader(test_dataset,batch_size =16,num_workers=4)

predictions = []
model.eval()
for bi , d in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    pixels = d["pixels"]
    labels = d ["labels"]
    pixels = pixels.to(device,dtype =torch.float)
    labels = labels.to(device,dtype =torch.float)
    
    outputs = model(pixels)
    pred = outputs.argmax(dim=1, keepdim=True)   
    predictions.extend(pred.tolist())
    
from sklearn import metrics
submission = pd.read_csv(SUBMIT_FILE)
submission.head()
submission['Label'] = [int(pred[0]) for pred in  predictions]
# accuracy = metrics.f1_score(submission['Label'].values, predictions,average='macro')*100
# print(accuracy)
submission.to_csv('/kaggle/working/submit_new.csv', index=False)
submission.head()