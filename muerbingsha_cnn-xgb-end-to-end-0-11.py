import os 

import pandas as pd 

import numpy as np

import time 

import cv2 

from PIL import Image 



import torch 

import torch.nn as nn

import torch.optim as optim

import torchvision

from torch.utils.data import Dataset

from torchvision import transforms 



import xgboost as xgb

from sklearn.metrics import cohen_kappa_score

from sklearn.preprocessing import quantile_transform



DEVICE = torch.device('cuda:0')

DATA_SOURCE = os.path.join('..', 'input', 'aptos2019-blindness-detection')

MODEL_SOURCE = os.path.join('..', 'input', 'torchvisionmodelspartial1')
def crop_image(img,tol=7):

    w, h = img.shape[1],img.shape[0]

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray_img = cv2.blur(gray_img,(5,5))

    shape = gray_img.shape 

    gray_img = gray_img.reshape(-1,1)

    quant = quantile_transform(gray_img, n_quantiles=256, random_state=0, copy=True)

    quant = (quant*256).astype(int)

    gray_img = quant.reshape(shape)

    xp = (gray_img.mean(axis=0)>tol)

    yp = (gray_img.mean(axis=1)>tol)

    x1, x2 = np.argmax(xp), w-np.argmax(np.flip(xp))

    y1, y2 = np.argmax(yp), h-np.argmax(np.flip(yp))

    if x1 >= x2 or y1 >= y2 : # something wrong with the crop

        return img # return original image

    else:

        img1=img[y1:y2,x1:x2,0]

        img2=img[y1:y2,x1:x2,1]

        img3=img[y1:y2,x1:x2,2]

        img = np.stack([img1,img2,img3],axis=-1)

    return img



def process_image(image, size=512):

    image = cv2.resize(image, (size,int(size*image.shape[0]/image.shape[1])))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:

        image = crop_image(image, tol=15)

    except Exception as e:

        image = image

        print( str(e) )

    return image
os.makedir('goog')

os.listdir('./')
class RetinopathyDatasetTrain(Dataset):



    def __init__(self, transform, eval_set=False, eval_frac=0.5, random_state=42):

        if not os.path.exists("cache"): os.mkdir("cache")

        self.transform = transform

        self.base_transform = transforms.Resize((224, 224))        

        # read data list, split in train and eval, select the set

        csv_file = os.path.join(DATA_SOURCE, "train.csv")

        df = pd.read_csv(csv_file)

        df_train = df.sample(n=int(df.shape[0]*(1-eval_frac)), random_state=random_state)

        ix=[i for i in df.index if i not in df_train.index.values.tolist()]  

        df_eval = df.loc[ix]            

        if eval_set : df = df_eval

        else :        df = df_train

        self.data = df.reset_index(drop=True)

            

    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        # get image and process it to tensor ready for the model, extract features

        folder = os.path.join(DATA_SOURCE, "train_images")

        code = str(self.data.loc[idx, 'id_code'])

        file = code + ".png"

        cache_path = os.path.join("cache",code+".png")

        cached = os.path.exists(cache_path)

        if not cached : 

            path = os.path.join(folder, file)

            image = cv2.imread(path)

            image = process_image(image)

            imgpil = Image.fromarray(image)

            imgpil = self.base_transform(imgpil)

            imgpil.save(cache_path,"PNG")

        imgpil = Image.open(cache_path)

        img_tensor = self.transform(imgpil)

        label = self.data.loc[idx, "diagnosis"]

        return {'image': img_tensor, 'label': label}
class RetinopathyDatasetTest(Dataset):



    def __init__(self, eval_set=False, random_state=42):

        # read data list, split in train and eval, select the set

        csv_file = os.path.join(DATA_SOURCE, "test.csv")

        df = pd.read_csv(csv_file)

        print(df)

        self.data = df.reset_index(drop=True)

        self.transform = transforms.Compose(

                        [transforms.Resize((224, 224)),

                        transforms.ToTensor(),

                        transforms.Normalize([0.485, 0.456, 0.406], 

                                             [0.229, 0.224, 0.225])])

            

    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        # get image and process it to tensor ready for the model, extract features

        folder = os.path.join(DATA_SOURCE, "test_images")

        code = str(self.data.loc[idx, 'id_code'])

        file = code + ".png"

        path = os.path.join(folder, file)

        image = cv2.imread(path)

        image = process_image(image)

        imgpil = Image.fromarray(image)

        img_tensor = self.transform(imgpil)

        return {'image': img_tensor}
# Training loops

def train_model(model, optimizer, train_data_loader, eval_data_loader, 

                file_name, num_epochs = 50, patience = 7, prev_loss = 1000.00):

    criterion = nn.CrossEntropyLoss()

    countdown = patience

    best_loss = 1000.00

    since = time.time()

    for epoch in range(num_epochs):

        running_loss = 0.0

        counter = 0

        for bi, d in enumerate(train_data_loader):

            counter += 1

            inputs = d["image"].to(DEVICE, dtype=torch.float)

            labels = d["label"].to(DEVICE, dtype=torch.long)

            model.to(DEVICE)

            model.train()

            optimizer.zero_grad()

            outputs = model(inputs) 

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            loss_val=(running_loss / (counter * train_data_loader.batch_size))

            print("{:3} {:.4f} {:.4f}".format(counter, loss.item()*1, loss_val), end="\r")

        epoch_loss = running_loss / ( len(train_data_loader) * train_data_loader.batch_size)

        time_elapsed = time.time() - since

        print(" T{:3}/{:3} loss: {:.4f} ({:3.0f}m {:2.0f}s) ".format( 

            epoch, num_epochs - 1, epoch_loss,time_elapsed // 60, time_elapsed % 60))

        running_loss = 0.0

        counter = 0

        for bi, d in enumerate(eval_data_loader):

            counter += 1

            inputs = d["image"].to(DEVICE, dtype=torch.float)

            labels = d["label"].to(DEVICE, dtype=torch.long)

            model.to(DEVICE)

            model.eval()

            with torch.no_grad():

                outputs = model(inputs)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            loss_val=(running_loss / (counter * eval_data_loader.batch_size))

            print("{:3} {:.4f} {:.4f}".format(counter, loss.item()*1, loss_val), end="\r")

        epoch_loss = running_loss / ( len(eval_data_loader) * eval_data_loader.batch_size)

        if epoch_loss < best_loss : 

            best_loss = epoch_loss

            if epoch_loss < prev_loss:

                torch.save(model.state_dict(), file_name)

                prev_loss = epoch_loss

                print("*", end="")

            else:

                print(".", end="")

            countdown = patience

        else:

            print("{:1}".format(countdown), end="")

            countdown -= 1

        time_elapsed = time.time() - since

        print("E{:3}/{:3} loss: {:.4f} ({:3.0f}m {:2.0f}s)".format( 

            epoch, num_epochs - 1, epoch_loss,time_elapsed // 60, time_elapsed % 60 ))



        if countdown <= 0 : break



    return prev_loss

    print("done.")

# Model training
aug_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



base_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



batch_size = 48

data_train = RetinopathyDatasetTrain(aug_transform, eval_frac=0.25, random_state=69)

data_loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, 

                                                shuffle=True, num_workers=0, 

                                                drop_last=False)

data_eval = RetinopathyDatasetTrain(base_transform, eval_set=True, 

                                    eval_frac=0.25, random_state=69)

data_loader_eval = torch.utils.data.DataLoader(data_eval,batch_size=batch_size, 

                                               shuffle=False, num_workers=0, 

                                               drop_last=False)
def get_base_model():

    model = torchvision.models.densenet161(pretrained=False)

    model_path = os.path.join(MODEL_SOURCE, "densenet161.pth")

    model.load_state_dict(torch.load(model_path))

    model.classifier = nn.Sequential(

        nn.BatchNorm1d(2208),

        nn.Dropout(p=0.25),

        nn.Linear(in_features=2208, out_features=2048, bias=True),

        nn.ReLU(),

        nn.BatchNorm1d(2048),

        nn.Dropout(p=0.5),

        nn.Linear(in_features=2048, out_features=5, bias=True),

    )

    model = model.to(DEVICE)

    model.eval()

    

    return model
bst_loss = 10000.00

for no in range(5):

    print("-"*22,no)

    model = get_base_model()

    plist = [{'params': model.features.denseblock2.parameters()},

             {'params': model.features.denseblock3.parameters()},

             {'params': model.features.denseblock4.parameters()},

             {'params': model.classifier.parameters()}]

    optimizer = optim.Adam(plist, lr=0.001)

    bst_loss = train_model(model, optimizer, data_loader_train, data_loader_eval, "tmp.pth", prev_loss=bst_loss)
# load the pretrained CNN used as feature extractor

# no classifier defined, we will take the raw output from the CNN layers

extractor = torchvision.models.densenet161(pretrained=False)

extractor.classifier = nn.Sequential(

    nn.BatchNorm1d(2208),

    nn.Dropout(p=0.25),

    nn.Linear(in_features=2208, out_features=2048, bias=True),

    nn.ReLU(),

    nn.BatchNorm1d(2048),

    nn.Dropout(p=0.5),

    nn.Linear(in_features=2048, out_features=5, bias=True),

)

model_path = os.path.join("tmp.pth")

extractor.load_state_dict(torch.load(model_path))

extractor.classifier = nn.Identity()

extractor = extractor.to(DEVICE)

extractor.eval()
data_loader_train = torch.utils.data.DataLoader(RetinopathyDatasetTrain(base_transform), 

                            batch_size=64, shuffle=True, num_workers=0, drop_last=False)

data_loader_eval = torch.utils.data.DataLoader(RetinopathyDatasetTrain(base_transform, 

                                                                       eval_set=True), 

                            batch_size=64, shuffle=True, num_workers=0, drop_last=False)



def get_train_features(data_loader):

    for bi, d in enumerate(data_loader):

        print(".", end="")

        img_tensor = d["image"].to(DEVICE)

        target = d["label"].numpy()

        with torch.no_grad(): feature = extractor(img_tensor)

        feature = feature.cpu().detach().squeeze(0).numpy()

        if bi == 0 :

            features = feature 

            targets = target 

        else :

            features = np.concatenate([features, feature], axis=0)

            targets = np.concatenate([targets, target], axis=0)

    print("")

    return features, targets



print(".............................")

features_train, targets_train = get_train_features(data_loader_train)

features_eval, targets_eval = get_train_features(data_loader_eval)
XGBOOST_PARAM = {

    "random_state" : 42,

    'objective': 'multi:softmax',

    "num_class" : 5,

    "n_estimators" : 200,

    "eval_metric" : "mlogloss"

}



xgb_model_1 = xgb.XGBClassifier(**XGBOOST_PARAM)

xgb_model_1 = xgb_model_1.fit(features_train,targets_train.reshape(-1),

                        eval_set=[(features_eval, targets_eval.reshape(-1))],

                        early_stopping_rounds=20,

                        verbose=True)

prediction = xgb_model_1.predict(features_eval)
xgb_model_2 = xgb.XGBClassifier(**XGBOOST_PARAM)

xgb_model_2 = xgb_model_2.fit(features_eval,targets_eval.reshape(-1),

                        eval_set=[(features_train, targets_train.reshape(-1))],

                        early_stopping_rounds=20,

                        verbose=True)
print("Cohen Kappa quadratic score", 

      cohen_kappa_score(targets_eval, prediction, weights="quadratic"))

_ = xgb.plot_importance(xgb_model_1, max_num_features=12)
prediction1 = xgb_model_1.predict(features_eval)

prediction2 = xgb_model_2.predict(features_train)

targets = np.concatenate([targets_eval, targets_train], axis=0)

prediction = np.concatenate([prediction1, prediction2], axis=0)

print("Cohen Kappa quadratic score", 

      cohen_kappa_score(targets, prediction, weights="quadratic"))
data_loader = torch.utils.data.DataLoader(RetinopathyDatasetTest(), 

                            batch_size=2, shuffle=False, num_workers=0, drop_last=False)



def get_test_features(data_loader):

    for bi, d in enumerate(data_loader):

        if bi % 32 == 0 : print(".", end="")

        img_tensor = d["image"].to(DEVICE)

        with torch.no_grad(): feature = extractor(img_tensor)

        feature = feature.cpu().detach().squeeze(0).numpy()

        if bi == 0 :

            features = feature 

        else :

            features = np.concatenate([features, feature], axis=0)

    print("")

    return features



print("...............................")

features = get_test_features(data_loader)
prediction1 = xgb_model_1.predict_proba(features)

prediction2 = xgb_model_2.predict_proba(features)

prediction = (prediction1 + prediction2).argmax(axis=1)

csv_file = os.path.join(DATA_SOURCE, "sample_submission.csv")

df = pd.read_csv(csv_file)

df["diagnosis"] = prediction

df.to_csv('submission.csv',index=False)
df
# cleaning

for e in os.listdir("cache"):

    os.remove(os.path.join("cache", e))

os.rmdir("cache")