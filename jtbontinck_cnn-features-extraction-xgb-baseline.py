import os

import pandas as pd

import numpy as np 



from PIL import Image



import torch

import torch.nn as nn

import torchvision

from torch.utils.data import Dataset

from torchvision import transforms



import xgboost as xgb

from sklearn.metrics import cohen_kappa_score



import pickle



DEVICE = torch.device("cuda:0")

DATA_SOURCE = os.path.join("..","input","aptos2019-blindness-detection")

DATA_SOURCE = os.path.join("..","input")
class RetinopathyDatasetTrain(Dataset):



    def __init__(self, eval_set=False, random_state=42):

        # read data list, split in train and eval, select the set

        csv_file = os.path.join(DATA_SOURCE, "train.csv")

        df = pd.read_csv(csv_file)

        df_train = df.sample(n=int(df.shape[0]/2), random_state=random_state)

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

        path = os.path.join(folder, file)

        imgpil = Image.open(path)

        base_transforms = transforms.Compose([transforms.Resize((224, 224)),

                                              transforms.ToTensor(),

                                              transforms.Normalize([0.485, 0.456, 0.406], 

                                                                   [0.229, 0.224, 0.225])])

        img_tensor = base_transforms(imgpil)

        label = self.data.loc[idx, "diagnosis"]

        return {'image': img_tensor, 'labels': label}
# load the pretrained CNN used as feature extractor

# no classifier defined, we will take the raw output from the CNN layers

extractor = torchvision.models.resnet101(pretrained=True)

extractor.fc = nn.Identity() 

extractor.to(DEVICE)

extractor.eval()



data_loader_train = torch.utils.data.DataLoader(RetinopathyDatasetTrain(), 

                            batch_size=64, shuffle=False, num_workers=0, drop_last=False)

data_loader_eval = torch.utils.data.DataLoader(RetinopathyDatasetTrain(eval_set=True), 

                            batch_size=64, shuffle=False, num_workers=0, drop_last=False)



def get_extracted_data(data_loader):

    for bi, d in enumerate(data_loader):

        print(".", end="")

        img_tensor = d["image"].to(DEVICE)

        target = d["labels"].numpy()

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

features_train, targets_train = get_extracted_data(data_loader_train)

features_eval, targets_eval = get_extracted_data(data_loader_eval)
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

                        verbose=False)

prediction = xgb_model_1.predict(features_eval)

# pred1 = XGBGBDT.predict_proba(features_eval)
xgb_model_2 = xgb.XGBClassifier(**XGBOOST_PARAM)

xgb_model_2 = xgb_model_2.fit(features_eval,targets_eval.reshape(-1),

                        eval_set=[(features_train, targets_train.reshape(-1))],

                        early_stopping_rounds=20,

                        verbose=False)
print("Cohen Kappa quadratic score", 

      cohen_kappa_score(targets_eval, prediction, weights="quadratic"))

xgb.plot_importance(xgb_model_1, max_num_features=12)
torch.save(extractor.state_dict(), "resnet101.pth")

pickle.dump(xgb_model_1, open("xgb_model_1", "wb"))

pickle.dump(xgb_model_2, open("xgb_model_2", "wb"))