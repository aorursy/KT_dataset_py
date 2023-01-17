# Install the PyDrive wrapper & import libraries.

# This only needs to be done once per notebook.

!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth

from pydrive.drive import GoogleDrive

from google.colab import auth

from oauth2client.client import GoogleCredentials



# Authenticate and create the PyDrive client.

# This only needs to be done once per notebook.

auth.authenticate_user()

gauth = GoogleAuth()

gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)
#Download link: https://drive.google.com/open?id=0By0bAi7hOBAFSGh0ekJqcWRnVmc

file_id_1 = 'https://drive.google.com/open?id=0By0bAi7hOBAFSGh0ekJqcWRnVmc'  

downloaded_1 = drive.CreateFile({'id': file_id_1})  



# download link: https://drive.google.com/open?id=0By0bAi7hOBAFRXNGOHBlQlZIRmc

file_id_2 = 'https://drive.google.com/open?id=0By0bAi7hOBAFRXNGOHBlQlZIRmc'  

downloaded_2 = drive.CreateFile({'id': file_id_2}) 
downloaded_1.GetContentFile('train.zip')



downloaded_2.GetContentFile('test.zip')



!unzip train.zip
!unzip test.zip
#Download link: https://drive.google.com/open?id=1eFJ5zMBTt6lLX2NT7aDh2CIKJA6RQ_CO

file_id_3 = '1eFJ5zMBTt6lLX2NT7aDh2CIKJA6RQ_CO'  

downloaded_3 = drive.CreateFile({'id': file_id_3})  



#Download link: https://drive.google.com/open?id=1voxWcD22HPAuNnvbIol8uq6rer2OOlLA

file_id_4 = '1voxWcD22HPAuNnvbIol8uq6rer2OOlLA'  

downloaded_4 = drive.CreateFile({'id': file_id_4}) 



#Download link: https://drive.google.com/open?id=19NjrmRTJuefrO7_5fmB3x3mMNC_bfrLZ

file_id_5 = '19NjrmRTJuefrO7_5fmB3x3mMNC_bfrLZ'  

downloaded_5 = drive.CreateFile({'id': file_id_5}) 



downloaded_3.GetContentFile('train_lmc.tar.gz')



downloaded_4.GetContentFile('test_lmc.tar.gz')



downloaded_5.GetContentFile('train_labels.csv')



import tarfile



with tarfile.open('train_lmc.tar.gz', 'r:gz') as tar:

    tar.extractall()



with tarfile.open('test_lmc.tar.gz', 'r:gz') as tar:

    tar.extractall()
#Download link: https://drive.google.com/open?id=12oKyXJo79yaJkBKMA0Ienb00e5EANfRz

file_id_6 = '12oKyXJo79yaJkBKMA0Ienb00e5EANfRz'  

downloaded_6 = drive.CreateFile({'id': file_id_6})  



#Download link: https://drive.google.com/open?id=1NxUe9dfVLbsfEKmJohIhk5kUqvQtJVBz

file_id_7 = '1NxUe9dfVLbsfEKmJohIhk5kUqvQtJVBz'  

downloaded_7 = drive.CreateFile({'id': file_id_7}) 



downloaded_6.GetContentFile('train_mc.tar.gz')



downloaded_7.GetContentFile('test_mc.tar.gz')



import tarfile



with tarfile.open('train_mc.tar.gz', 'r:gz') as tar:

    tar.extractall()



with tarfile.open('test_mc.tar.gz', 'r:gz') as tar:

    tar.extractall()
!pip install --upgrade librosa
import os

from PIL import Image

import matplotlib

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook 

import copy

import pickle



import torch

import torchvision

from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.transforms as transforms

from torchvision import datasets



import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import numpy as np

import pandas as pd

from sklearn import preprocessing, metrics

from sklearn.utils.multiclass import unique_labels

import seaborn as sn



import librosa

import librosa.display
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
!mkdir train_lmc/

!mkdir train_mc/



!mkdir test_lmc/

!mkdir test_mc/
def audio_to_image(source_folder_path, lmc_folder_path, mc_folder_path):

    file_names = [file for _,_,files in os.walk(source_folder_path) for file in files]

    plt.ioff()

        

    for file_count, file_name in enumerate(file_names):

        y, sr = librosa.load(source_folder_path + file_name)

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        S = np.abs(librosa.stft(y))

        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        

        for i in range(2):

            fig, (a0, a1, a2, a3) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [8, 2, 1, 1], 'hspace': 0})

            fig.set_figheight(8)

            if i == 0:

                librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), ax=a0)

                dest_path = lmc_folder_path + file_name.replace('.wav', '_lmc.png')

            else:

                librosa.display.specshow(mfcc, ax=a0)

                dest_path = mc_folder_path + file_name.replace('.wav', '_mc.png')

            librosa.display.specshow(chroma, ax=a1)

            librosa.display.specshow(contrast, ax=a2)

            librosa.display.specshow(tonnetz, ax=a3)

            fig.tight_layout()

            fig.savefig(dest_path) 

            plt.cla()

            plt.clf()

            plt.close('all')

            

        if (file_count+1) % 100 == 0:

            print(str(file_count+1) + " files processed")

            

        del y, sr, mel_spec, mfcc, chroma, S, contrast, tonnetz

            

    return 1
audio_to_image('Test/', 'test_lmc/', 'test_mc/')
class Params(object):

    def __init__(self, batch_size, epochs, seed, log_interval):

        self.batch_size = batch_size

        self.epochs = epochs

        self.seed = seed

        self.log_interval = log_interval



args = Params(64, 100, 0, 1)
torch.random.manual_seed(args.seed)

np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

np.random.seed(args.seed)



#For converting the dataset to torchvision dataset format

class UrbanSoundDataset(Dataset):

    def __init__(self, file_path, classes_file_path=None, transform=None):

        self.transform = transform

        self.file_path_lmc = file_path[0]

        self.file_path_mc = file_path[1]

        self.classes_file_path = classes_file_path

        self.file_names_lmc = sorted([file for _,_,files in os.walk(self.file_path_lmc) for file in files])

        self.file_names_mc = sorted([file for _,_,files in os.walk(self.file_path_mc) for file in files])

        self.len = len(self.file_names_lmc)

        if self.classes_file_path is not None:

            self.classes_mapping, self.classes_encoding = self.get_classes(classes_file_path)

            

    def __len__(self):

        return self.len

    

    def __getitem__(self, index):

        file_name_lmc = self.file_names_lmc[index]

        image_data_lmc = self.pil_loader(self.file_path_lmc+"/"+file_name_lmc)

        image_number_lmc = int(file_name_lmc.split("_")[0]) 

        file_name_mc = self.file_names_mc[index]

        image_data_mc = self.pil_loader(self.file_path_mc+"/"+file_name_mc)        

        image_number_mc = int(file_name_mc.split("_")[0]) 

        if image_number_lmc != image_number_mc:

            print("Image mismatch error")

        if self.transform:

            image_data_lmc = self.transform(image_data_lmc)

            image_data_mc = self.transform(image_data_mc)

            image_data = torch.cat((image_data_lmc, image_data_mc), 0)

        if self.classes_file_path is not None:          

            row = self.classes_mapping.loc[self.classes_mapping['ID'] == image_number_lmc,

                                          ['Class', 'Class_code']].values  

            class_, class_code = row[0, 0], row[0, 1]

            return image_data, image_number_lmc, class_, class_code

        else:

            return image_data, image_number_lmc

          

    def pil_loader(self,path):

        with open(path, 'rb') as f:

            img = Image.open(f)

            return img.convert('RGB')

      

    def get_classes(self, classes_file_path):

        classes_mapping = pd.read_csv(classes_file_path)

        le = preprocessing.LabelEncoder()

        classes_mapping['Class_code'] = le.fit_transform(classes_mapping['Class'].values)

        classes_encoding = {}

        for code, class_ in enumerate(list(le.classes_)):

            classes_encoding[code] = class_

        return classes_mapping, classes_encoding
transform = transforms.Compose([

                transforms.Resize((41, 85)), 

                transforms.ToTensor(),

                transforms.Normalize((0.5,), (0.5,))

])
full_data = UrbanSoundDataset(["train_lmc", "train_mc"], "train_labels.csv", transform)



full_data.classes_encoding
# Splitting the training data into 80 % training and 20 % validation datasets

train_size = int(0.8 * len(full_data))

test_size = len(full_data) - train_size

train_data, validation_data = random_split(full_data, [train_size, test_size])



print(len(full_data), len(train_data), len(validation_data))



# Data loaders for training and validation datasets. Dataloaders provide shuffled data in batches

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=len(validation_data))
test_data = UrbanSoundDataset(file_path=["test_lmc", "test_mc"], transform=transform)

test_loader = torch.utils.data.DataLoader(test_data)

print(len(test_data))



images,_,_,_ = next(iter(train_loader))

print(images.shape)



images,_,_,_ = next(iter(validation_loader))

print(images.shape)



images,_ = next(iter(test_loader))

print(images.shape)
class UrbanSoundCNN(nn.Module):

    def __init__(self, p=0):

        super(UrbanSoundCNN, self).__init__()

        self.p = p

        self.features = nn.Sequential(

            # layer 1

            nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1),  # 41 x 85 -> 21 x 43

            nn.BatchNorm2d(16),

            nn.ReLU(),



            # layer 2            

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # 21 x 43 -> 11 x 22

            nn.BatchNorm2d(16),

            nn.ReLU(),  

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 11 x 22 -> 6 x 12

            nn.Dropout2d(p=self.p),



            # layer 3

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 6 x 12 -> 4 x 7

            nn.BatchNorm2d(32),

            nn.ReLU(),   



            # layer 4            

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # 4 x 7 -> 3 x 4

            nn.BatchNorm2d(32),

            nn.ReLU(),     

            nn.Dropout2d(p=self.p),        



            # layer 5 : 

            nn.Conv2d(32, 10, kernel_size=1, stride=1, padding=0),  # 3 x 4 -> 3 x 4  

            nn.BatchNorm2d(10),

            nn.ReLU(),      

            nn.AdaptiveAvgPool2d((1,1)),            

        )



    def forward(self, x):

        x = self.features(x)

        #print(x.shape)

        x = x.view(x.size(0), -1)

        #print(x.shape)

        return x
def train(model, loss_fn, opt, epoch, batch_log_interval, dropout_prob):

    

    model.train()

    model.p = dropout_prob

    loss_avg = 0

    

    for batch_id, data in enumerate(train_loader):

        inputs, _, _, labels = data

        inputs = inputs.to(device)

        labels = labels.to(device)

        

        opt.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)        

        loss_avg += loss.data.item()*len(inputs) # as 'loss' is avg of total mini-batch loss

        loss.backward()

        opt.step()



        '''if batch_id % batch_log_interval == 0:            

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg batch loss: {:.3f}'.format(

                epoch, batch_id * len(inputs), len(train_loader.dataset),

                100. * batch_id / len(train_loader), loss.data.item()/len(inputs)))'''

            

        del inputs, labels, outputs

        torch.cuda.empty_cache()

        

    loss_avg /= len(train_loader.dataset)

    print('\nEpoch: {}, Train set: Average loss: {:.4f}'.format(epoch, loss_avg)) 

    return loss_avg
def validate(model, loss_fn, epoch, plot_confusion_mat=False):    

    model.eval()

    model.p = 0



    with torch.no_grad():

        inputs, _, _, labels = next(iter(validation_loader))

        inputs = inputs.to(device)

        labels = labels.to(device)    

        outputs = model(inputs)

        validation_loss = loss_fn(outputs, labels)

        outputs = outputs.argmax(1).cpu()

        labels = labels.cpu()

        f1_score = metrics.f1_score(labels, outputs, average='weighted') 

        validation_accuracy = metrics.accuracy_score(labels, outputs)*100

          

        print('\nEpoch: {}, Validation set: Loss: {:.4f}, Accuracy: {:.0f}%, F1 score: {:.4f}'.

              format(epoch, validation_loss, validation_accuracy, f1_score))  

        if plot_confusion_mat:

            plot_confusion_matrix(labels, outputs, normalize=True)

        

        return validation_loss, validation_accuracy, f1_score
def validate(model, loss_fn, epoch, plot_confusion_mat=False):    

    model.eval()

    model.p = 0



    with torch.no_grad():

        inputs, _, _, labels = next(iter(validation_loader))

        inputs = inputs.to(device)

        labels = labels.to(device)    

        outputs = model(inputs)

        validation_loss = loss_fn(outputs, labels)

        outputs = outputs.argmax(1).cpu()

        labels = labels.cpu()

        f1_score = metrics.f1_score(labels, outputs, average='weighted') 

        validation_accuracy = metrics.accuracy_score(labels, outputs)*100

          

        print('\nEpoch: {}, Validation set: Loss: {:.4f}, Accuracy: {:.0f}%, F1 score: {:.4f}'.

              format(epoch, validation_loss, validation_accuracy, f1_score))  

        if plot_confusion_mat:

            plot_confusion_matrix(labels, outputs, normalize=True)

        

        return validation_loss, validation_accuracy, f1_score
def main(lr, momentum, best_model=None, max_f1_score=0, best_validation_accuracy=0, dropout_prob=0):    

    train_loss = []

    validation_loss = []

    validation_accuracy = []

    f1_score = []



    print('\nLR = {}, Momentum = {}, Mini-batch size = {}\n'.format(lr, momentum, args.batch_size))

    torch.manual_seed(args.seed)

    model = UrbanSoundCNN(dropout_prob)

    if best_model:

        model.load_state_dict(best_model)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    opt = optim.SGD(model.parameters(), lr=lr, momentum = momentum)    



    for epoch in tqdm_notebook(range(1, args.epochs + 1), total=args.epochs, unit="epoch"):

        train_loss_epoch = train(model, loss_fn, opt, epoch, 1, dropout_prob)             

        if epoch % args.log_interval == 0:

            train_loss.append(train_loss_epoch)

            valid_loss_epoch, valid_accuracy_epoch, f1_score_epoch = validate(model, loss_fn, epoch, plot_confusion_mat=True)

            validation_loss.append(valid_loss_epoch)

            validation_accuracy.append(valid_accuracy_epoch)

            f1_score.append(f1_score_epoch)



            if f1_score_epoch > max_f1_score:

                max_f1_score = f1_score_epoch

                best_validation_accuracy = valid_accuracy_epoch

                best_model = copy.deepcopy(model.state_dict())

                best_epoch = epoch

            print("Maximum validation F1 score so far: {:.4f}. Corresponding validation accuracy: {:.0f}%".

                  format(max_f1_score, best_validation_accuracy))

        print("-----------------------------------------------------------------\n")



    return best_model, best_epoch, train_loss, validation_loss, validation_accuracy, f1_score
train_loss = {}

validation_loss = {}

validation_accuracy = {}

validation_f1_score = {}



lr = 0.01

momentum = 0.9

args.epochs = 100



best_model, best_epoch, train_loss[lr], validation_loss[lr], validation_accuracy[lr], validation_f1_score[lr] = main(lr, momentum, dropout_prob=0.5)



best_epoch
lr = 0.001

momentum = 0.9

args.epochs = 50 



best_model = torch.load('models/model_mc_lmc_aug_e2_50.pt')



max_f1_score = validation_f1_score[0.01][47]

best_validation_accuracy = validation_accuracy[0.01][47]
best_model, best_epoch, train_loss[lr], validation_loss[lr], validation_accuracy[lr], validation_f1_score[lr] = main(

    lr, momentum, best_model, max_f1_score, best_validation_accuracy, 0.5)



train_loss[lr] += tr_loss

validation_loss[lr] += val_loss

validation_accuracy[lr] += val_accuracy

validation_f1_score[lr] += val_f1_score
!mkdir models/ 



# model_e2 means lr=1e-2, similar names will be used

torch.save(best_model, 'models/model_mc_lmc_half_channels_no_fc_e2_mb64_100.pt')



!tar -czvf models.tar.gz models
!mkdir obj/



def save_obj(obj, name ):

    with open('obj/'+ name + '.pkl', 'wb') as f:

        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def load_obj(name ):

    with open('obj/' + name + '.pkl', 'rb') as f:

        return pickle.load(f)



save_obj(train_loss, "train_loss")

save_obj(validation_loss, "validation_loss")

save_obj(validation_accuracy, "validation_accuracy")

save_obj(validation_f1_score, "validation_f1_score")
!tar -czvf obj.tar.gz obj



with tarfile.open('models.tar.gz', 'r:gz') as tar:

    tar.extractall()



with tarfile.open('obj.tar.gz', 'r:gz') as tar:

    tar.extractall()
train_loss = load_obj("train_loss")

validation_loss = load_obj("validation_loss")

validation_accuracy = load_obj("validation_accuracy")

validation_f1_score = load_obj("validation_f1_score")



def plot_loss(epochs, metric, lr, batch_size, data_type, metric_type):

    plt.plot(epochs,metric)

    plt.title(" {} {}, lr={}, batch size={}".format(data_type, metric_type, lr, batch_size))

    plt.xlabel("epochs")

    if metric_type == "loss":

        ylabel = "Avergae CE loss"

    if metric_type == "accuracy":

        ylabel = "Accuracy (%)"

    if metric_type == "F1 score":

        ylabel = "F1 score"

    plt.ylabel(ylabel)  

    plt.show()
lr = 0.01

plot_loss(range(1,51), validation_accuracy[0.01], lr, args.batch_size, "validation", "accuracy")



plt.plot(range(1,101), train_loss[0.01], c='r', label="train")

plt.plot(range(1,101), validation_loss[0.01], c='g', label="validation")

plt.title("lr={}, momentum={}, mini-batch={}".format(lr,momentum, args.batch_size))

plt.xlabel("epochs")

plt.ylabel("Avergae CE loss")

plt.legend(loc='upper right')

plt.show()
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None,

                          cmap=plt.cm.Blues):

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = metrics.confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)

        fmt = ".2f"

    else:

      fmt = "d"



    df_cm = pd.DataFrame(cm, index = classes, columns = classes)

    plt.figure(figsize = (10,7))

    ax = sn.heatmap(df_cm, annot=True, fmt=fmt, cmap=sn.cm.rocket_r)

    ax.set_ylim(10,0)

    plt.xlabel("Predicted label")

    plt.ylabel("True label")

    plt.title(title)
!unzip best_models.zip



best_model = torch.load('best_models/model_e2_dropout.pt')



model = UrbanSoundCNN()

model.load_state_dict(best_model)

model.to(device)

loss_fn = nn.CrossEntropyLoss()

opt = optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)    
valid_tot_loader = torch.utils.data.DataLoader(full_data, batch_size=len(validation_data), shuffle=True)

inputs, _, _, labels = next(iter(valid_tot_loader))

print(len(inputs))



inputs = inputs.to(device)

labels = labels.to(device)

outputs = model(inputs).argmax(1).cpu()

labels = labels.cpu()



metrics.f1_score(labels, outputs, average='weighted') 
classes = ['air conditioner', 'car horn', 'children playing', 'dog bark', 

           'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren',

           'street_music']



# Plot non-normalized confusion matrix

plot_confusion_matrix(labels, outputs, classes=classes,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plot_confusion_matrix(labels, outputs, classes=classes, normalize=True,

                      title='Normalized confusion matrix')



plt.show()
classes = full_data.classes_encoding



submission_df = pd.read_csv("test.csv")

classes = full_data.classes_encoding

submission_df["Class"] = [""]*len(test_data)

with torch.no_grad():

    i = 0

    for img, ID in test_loader:        

        img = img.to(device)

        ID = int(ID[0])

        class_pred = int(model(img).argmax(1).cpu()[0])

        submission_df.loc[submission_df["ID"]==ID, "Class"] = classes[class_pred]



submission_df.head(10)



submission_df = submission_df[["Class", "ID"]]

submission_df.to_csv("submisision.csv", index=False)