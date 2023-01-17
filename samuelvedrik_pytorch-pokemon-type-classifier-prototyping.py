import torch

from torchvision.transforms import transforms

from torch.utils.data import Dataset, DataLoader



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image



!pip install -q torchsummary

from torchsummary import summary



from tqdm.auto import tqdm

from time import sleep

import os



!pip -q install jcopdl

import jcopdl

from jcopdl.callback import Callback, set_config



WORKING_DIR = "/kaggle/input/pokemon-images-and-types/"

INFO_DIR = "pokemon.csv"

IMAGES_DIR = "images/images/"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





config = set_config(dict(output_size=18, batch_size=4, image_size=(120, 120), lr=5e-4, dropout=0.5))
# %%write dataset.py



from sklearn.preprocessing import OneHotEncoder, LabelEncoder





class PokemonDatasetBuilder():

    

    """

    A class used to generate the appropriate datasets.

    """

    def __init__(self, dataset_class, file_path, transform=None, splits=True):

        """

        dataset_class: one of {multilabel, multiclass}, determines the output of 

                       the dataset to be either one OneHotEncoded vector or two different labels.

                       

        file_path:     the file path to the csv file. 

        

        splits:        An optional condition to create a training, validation and testing set.

        """

        

        self._splits = splits

        self._dataset_class = dataset_class

        self.df = pd.read_csv(file_path)

        self.transform = transform

        self._preprocess_frame()

        

    def __call__(self, test_split=0.1, val_split=0.1):

        """

        Generates the datasets. 

        """

        

        dfs = []

        if self._splits:

            dfs.extend(self._create_splits(test_split, val_split))

        else:

            dfs.append(self.df)

        

        datasets = []

        

        # If multilabel, we don't encode type2 differently

        if self._dataset_class == "multilabel":

            OHE = OneHotEncoder(sparse=False, handle_unknown="ignore")

            OHE.fit(self.df[["Type1"]])

            

            for df in dfs:

                datasets.append(PokemonDatasetMultilabel(df, OHE, self.transform))

            

        

        # If multiclass, we have to encode type1 and type2 differently, 

        # as we now have 19 targets for type2 (including None)

        else:

            LE1 = LabelEncoder()

            LE2 = LabelEncoder()

            

            LE1.fit(self.df["Type1"])

            LE2.fit(self.df["Type2"])

        

            for df in dfs:

                datasets.append(PokemonDatasetMulticlass(df, LE1, LE2, self.transform))

        

        return datasets

            

            

    def _create_splits(self, test_split, val_split):

        """

        Helper function to create the different splits. 

        """

        df_test = self.df.sample(frac=test_split, random_state=42)

        df_train = self.df.drop(df_test.index)

        df_val = df_train.sample(frac=val_split, random_state=42)

        df_train = df_train.drop(df_val.index)



            

        return [df_train, df_val, df_test]



    

    def _preprocess_frame(self):

        """

        Helper function to preprocess the dataframe.

        """

        

        # From gen 7 and above the images are stored in jpg... why? 

        self.df["Name"].iloc[:721] = self.df["Name"].iloc[:721].apply(lambda x : x + ".png")

        self.df["Name"].iloc[721:] = self.df["Name"].iloc[721:].apply(lambda x : x + ".jpg")

        self.df["Type2"].fillna("None", inplace=True)

        

    

        

class PokemonDatasetMultilabel(Dataset):

    """

    A dataset that returns a multilabel vector. The encoder passed to it should be a One hot encoder.

    """

    

    def __init__(self, df, encoder, transform=None):

        

        self.df = df

        self.transform = transform

        self.encoder = encoder

        self.type1 = encoder.transform(self.df[["Type1"]])

        self.type2 = encoder.transform(self.df[["Type2"]])

    

    def __getitem__(self, idx):

        

        if torch.is_tensor(idx):

            idx = idx.tolist()

        

        image_file = WORKING_DIR + IMAGES_DIR + self.df.iloc[idx, 0]

        

        image = process_image(image_file, self.transform)



        return image, (self.type1[idx] + self.type2[idx])



    

    def __len__(self):

        return self.df.shape[0]

    

    

    def get_labels_from_vector(self, vector):

        """

        Returns labels of a pokemon given a one hot encoded vector. 

        The output is formatted as type1, type2.

        

        >>> train_dataset = PokemonDatasetMultilabel(...)

        >>> vector = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,

                     0.])

        >>> train_dataset.get_labels_from_vector(vector)

        ("Dark", "Ground")

        

        >>> vector = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,

                     0.])

        >>> train_dataset.get_labels_from_vector(vector)

        ("Fire", "None")

        """

        

        labels = self.encoder.categories_[0][vector==1]

        if vector.sum() == 1:

            return labels[0], "None"

        else:

            return tuple(labels)



    def get_labels_from_id(self, type1, type2=None):

        """

        Returns labels of a pokemon given both ids. 

        The output is formatted as type1, type2. 

        

        >>> train_dataset = PokemonDatasetMultilabel(...)

        >>> type1, type2  = (0, 15)

        >>> train_dataset.get_labels_from_id(type1, type2)

        ("Bug", "Rock")

        >>> train_dataset.get_labels_from_id(0)

        >>>("Bug", "None")

        """

        if type2 is not None: 

            return self.encoder.categories_[0][type1], self.encoder.categories_[0][type2]

        else:

            return self.encoder.categories_[0][type1], "None"

    

    

class PokemonDatasetMulticlass(Dataset):

    

    """

    A dataset that returns two outputs, which represent type1 and type2 respectively. 

    """

    def __init__(self, df, type_1_encoder, type_2_encoder, transform=None):

        

        self.df = df

        self.transform = transform

        

        self.type_1_encoder = type_1_encoder

        self.type_2_encoder = type_2_encoder

        

        self.type1 = type_1_encoder.transform(self.df["Type1"])

        self.type2 = type_2_encoder.transform(self.df["Type2"])

        

    def __getitem__(self, idx):

        

        if torch.is_tensor(idx):

            idx = idx.tolist()

            

        image_file = WORKING_DIR + IMAGES_DIR + self.df.iloc[idx, 0]

        image = process_image(image_file, self.transform)

        

        return image, self.type1[idx], self.type2[idx]

    

    def __len__(self):

        return self.df.shape[0]

    

    

    

    def get_labels(self, type1, type2):

        """

        Returns the labels of the pokemon given both ids.

        The output is formatted as type1, type2.

        """

        return self.type_1_encoder.classes_[type1], self.type_2_encoder.classes_[type2]

            

        

        

        

def process_image(image_file, transform=None):

    """

    Returns the image given the image file, and applies transform to it. 

    """

        

    # this converts PNG images to JPG images (RGBA to RGB), while giving both a white background instead of

    # a black background. 

    if image_file.split(".")[-1] == "png":

        pil_image = Image.open(image_file).convert("RGBA")

        image = Image.new('RGBA',pil_image.size,(255,255,255))

        image.paste(pil_image, (0,0), pil_image)

        image = image.convert("RGB")



    else:

        image = Image.open(image_file)





    if transform:

        image = transform(image)

        

    return image



        
def show_image(image, type1, type2):

    """

    Shows the image as well as the type of the pokemon. 

    """

    plt.imshow(image.permute(1, 2, 0))

    plt.axis("Off")

    title = f"Type: {type1}"

    if type2 != "None":

        title += f", {type2}"

    plt.title(title)
OUTPUT_TYPE = "multilabel" # can also be "multiclass"



# transformations = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])

transformations = transforms.Compose([transforms.ToTensor()])

dataset_generator = PokemonDatasetBuilder(OUTPUT_TYPE, WORKING_DIR + INFO_DIR, transformations)

train_dataset, val_dataset, test_dataset = dataset_generator()



train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True)

val_dataloader = DataLoader(val_dataset)

test_dataloader = DataLoader(test_dataset)
# Sample usage for multilabel 

pokemon = train_dataset[1]

show_image(pokemon[0], *train_dataset.get_labels_from_vector(pokemon[1]))



# Sample usage for multiclass

# show_image(pokemon[0], *train_dataset.get_labels(*pokemon[1:]))
import torch.nn as nn

import torch.nn.functional as F

from torchvision import models





class UnknownModeException(Exception):

    

    def __str__(self):

        return "Unknown mode given. Use one of: 'logistic', 'softmax', 'none'."



class PokemonFCBlock(nn.Module):

    """

    The final layer for a PokemonCNN. 

    """

    

    def __init__(self, in_features, out_features, mode="none", dropout=0.2):

        """

        Initializes a new final layer.

        

        mode represents the final activation applied to the logits.

        {"logistic", "softmax", "none"}

        """

        super().__init__()

        

        self.fc = nn.Sequential(

            nn.Linear(in_features, 1000),

            nn.LeakyReLU(),

            nn.Dropout(dropout),

            nn.Linear(1000, 1000),

            nn.LeakyReLU(), 

            nn.Dropout(dropout),

            nn.Linear(1000, out_features)

        )

        

        self.mode = mode.lower()

        

        

    def forward(self, x):

        

        if self.mode == "logistic":

            return torch.sigmoid(self.fc(x))

        elif self.mode == "softmax":

            return F.softmax(self.fc(x))

        elif self.mode == "none":

            return self.fc(x)

        else:

            raise UnknownModeException

    

        



class PokemonMultilabelCNN(nn.Module):

    """

    A CNN based on a base model that will output multilabel predictions. 

    The base model expects a ResNet architecture. 

    """

    

    def __init__(self, base_model, output_size, dropout):

        

        super().__init__()

    

        self.base_model = base_model 

        in_features = self.base_model.fc.in_features

        new_final = PokemonFCBlock(in_features, output_size, mode="logistic")

        self.base_model.fc = new_final



        

    def forward(self, x):

        return self.base_model(x)

    

    def freeze(self):

        # Freezes all the layers except for the final fully connected layer. 

        for name, child in self.base_model.named_children():

            if name != "fc":

                for param in child.parameters():

                    param.requires_grad = False 

            

    def unfreeze(self):

        for param in self.base_model.parameters():

            param.requires_grad = True

    

    

#TODO: Create the multiclass version
base_model = models.resnet50(pretrained=True, progress=False)

model = PokemonMultilabelCNN(base_model, config.output_size, config.dropout)

model.to(device)

model.freeze()

summary(model, (3, *config.image_size))
criterion = nn.BCELoss()

optimizer= torch.optim.AdamW(model.parameters(), lr=config.lr)

callback = Callback(model, config, early_stop_patience=10, outdir="model")
class ModelTrainer():

    """

    Abstract base class for model trainers.

    """

    

    def __init__(self, model, optimizer, criterion, device, callback):

        

        self.model = model

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

        self.callback = callback

        self.epoch = 1

        

    def train_loop(self, dataloader, epoch=None):

        """

        Trains the model for a single loop, i.e one epoch

        The optional argument epoch indicates which epoch this training loop belongs to.

        """

        raise NotImplementedError

        

    def validate_loop(self, dataloader):

        """

        Validate the model. 

        """

        raise NotImplementedError

        

    def train(self, train_dataloader, val_dataloader):

        """

        Trains the model for multiple epochs, and validates the model every epoch. 

        """

        raise NotImplementedError

        

class MultilabelModelTrainer(ModelTrainer):

    """

    A multilabel model trainer.

    """

    

    def __init__(self, model, optimizer, criterion, device, callback):

        

        super(MultilabelModelTrainer, self).__init__(model, optimizer, criterion, device, callback)

    

        

    def train_loop(self, dataloader, epoch=True):

        

        self.model.train()

        cost = 0 

        t = tqdm(dataloader)

        if epoch:

            t.set_description(f"Training mode, Epoch {self.epoch}")

            

        for feature, target in t:

            feature, target = feature.to(self.device), target.to(self.device)

            output = self.model(feature).double() # resolves expected dtype Double but got dtype Float

            loss = self.criterion(output, target)

            loss.backward()

            

            self.optimizer.step()

            self.optimizer.zero_grad()

            

            cost += loss.item() * feature.shape[0]

        

        return cost / len(dataloader.dataset)

            

        

    def validate_loop(self, dataloader):

        

        self.model.eval()

        cost = 0

        with torch.no_grad():

            for feature, target in tqdm(dataloader, desc="Validation mode"):

                feature, target = feature.to(self.device), target.to(self.device)

                output = self.model(feature).double()

                loss = self.criterion(output, target)

                

                cost += loss.item() * feature.shape[0]

        

        return cost/len(dataloader.dataset)

        

        

    def train(self, train_dataloader, val_dataloader):

        

        while True: 

            train_cost = self.train_loop(train_dataloader)

            val_cost = self.validate_loop(val_dataloader)

            

            self.epoch += 1



            # Prevents weird output

            _ = self.callback.log(train_cost, val_cost)

            _ = self.callback.save_checkpoint()

#             _ = self.callback.cost_runtime_plotting()

        

            if self.callback.early_stopping(model, monitor="test_cost"):

                self.callback.plot_cost()

                break

            

        
model_trainer = MultilabelModelTrainer(model, optimizer, criterion, device, callback)

model_trainer.train(train_dataloader, val_dataloader)
import random



random_idx = random.randint(0, len(train_dataset)-1)

pokemon = train_dataset[random_idx]

with torch.no_grad():

    feature = pokemon[0].to(device)

    prediction = model(feature.unsqueeze(0))
show_image(pokemon[0], *train_dataset.get_labels_from_vector(pokemon[1]))
predicted_labels = prediction.squeeze(0).argsort()

type2, type1 = predicted_labels[-2], predicted_labels[-1]



if prediction.squeeze(0)[type1] - prediction.squeeze(0)[type2] >= 0.2:

    print(train_dataset.get_labels_from_id(type1))    

    print(f"{type1}: {prediction.squeeze(0)[type1]}")

else:

    print(train_dataset.get_labels_from_id(type1, type2))

    print(f"{type1}: {prediction.squeeze(0)[type1]}")

    print(f"{type2}: {prediction.squeeze(0)[type2]}")