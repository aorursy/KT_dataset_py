!git clone https://github.com/cozek/memotion2020-code
!pip install --upgrade efficientnet-pytorch   
!git clone https://github.com/huggingface/transformers
!pip install transformers/
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(42)
import re
import os
import sys
sys.path.append('/kaggle/working/memotion2020-code/src/')
import pickle
import collections

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings  
warnings.filterwarnings('ignore')
from typing import Callable
from tqdm import notebook
import importlib
import nltk
import datetime
import time
from argparse import Namespace
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
torch.__version__
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
# ranger combo
from radam.radam import RAdam
from lookahead.optimizer import Lookahead
from transformers import RobertaTokenizer, RobertaModel
import memotion_utils.general as general_utils
import memotion_utils.transformer.data as transformer_data_utils
import memotion_utils.transformer.general as transformer_general_utils

general_utils.set_seed_everywhere()
args = Namespace(
        device = 'cuda' if torch.cuda.is_available() else 'cpu',
        n_workers = 2,
        date = datetime.datetime.now().strftime("%a_%d_%b_%Y/"),
        learning_rate = 0.0001,
        num_epochs = 20,
    )
IMAGES_DIR = '/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/images'
image_filenames = os.listdir(IMAGES_DIR)
file_extentions = [filename.split('.')[-1] for filename in image_filenames]

images_paths = [os.path.join(IMAGES_DIR,filename) for filename in image_filenames]

REF_FILE = '/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/reference_df_pickle'
LABELS_FILE = '/kaggle/input/memotion-dataset-7k/memotion_dataset_7k/labels_pd_pickle'

with open(REF_FILE, 'rb') as handle:
    reference_df_ = pickle.load(handle)

with open(LABELS_FILE, 'rb') as handle:
    labels_pd_ = pickle.load(handle)
    


image_formats = collections.Counter(file_extentions)
print(f'Num Images: {len(images_paths)}')

print('Image formats found: ', image_formats)
image_formats_df = pd.DataFrame.from_dict(image_formats, orient='index').reset_index()
image_formats_df
def get_train_val_split(train_frac, df, id_col):
    """
    Splits dataframe into train and val keeping percentage of
    labels same in both splits.
    Args:
        train_frac: Fraction of samples to use for train
        df: pd.DataFrame to split
        id_col: Column that uniquely identifies every row.
    Returns:
        split_df
    """
    val_frac = 1 - train_frac
    assert val_frac + train_frac == 1
    labels = set(df.label)
    split_df = None
    df = df.sample(frac=1) #shuffle df

    for lbl in notebook.tqdm(labels, total = len(labels)):
        lbl_df = df[df.label == lbl].copy()
        temp_df_train = lbl_df.sample(frac=train_frac).copy()
        temp_df_val = lbl_df[~lbl_df[id_col].isin(temp_df_train[id_col])].copy()
        temp_df_train['split'] = 'train'
        temp_df_val['split'] = 'val'
        if not isinstance(split_df,pd.DataFrame):
            split_df = temp_df_train.copy()
            split_df = pd.concat([split_df, temp_df_val])
        else:
            split_df = pd.concat([split_df, temp_df_train, temp_df_val])
    
    assert len(split_df) == len(df)
    return split_df
# Negative and Very Negative => 2
# Positive and Very Positive => 1
# Neutral => 0

task_a_labels = {
    'negative': 2 ,
    'very_negative': 2,
    'neutral' : 0,
    'positive' : 1,
    'very_positive': 1,
}

task_a_labels_df = labels_pd_[['image_name','overall_sentiment']].copy()
task_a_labels_df['label'] = task_a_labels_df['overall_sentiment'].map(task_a_labels)
task_a_labels_df.label.value_counts()
task_a_split_df = get_train_val_split(
    train_frac = 0.90,
    df = task_a_labels_df,
    id_col= 'image_name',
)
text_df = labels_pd_[['image_name','text_corrected']]
data_df = pd.merge(task_a_split_df,text_df, on='image_name')
del data_df['overall_sentiment']
data_df
class RobertaPreprocessor():
    def __init__(self,transformer_tokenizer,sentence_detector):
        self.transformer_tokenizer = transformer_tokenizer
        self.sentence_detector = sentence_detector
        self.bos_token = transformer_tokenizer.bos_token
        self.sep_token = ' ' + transformer_tokenizer.sep_token + ' '
    def add_special_tokens(self, text):
        text = str(text)
        sentences = self.sentence_detector.tokenize(text)
        eos_added_text  = self.sep_token.join(sentences) 
        return self.bos_token +' '+ eos_added_text + ' ' + self.transformer_tokenizer.sep_token
!python3 -c "import nltk; nltk.download('punkt')"
roberta_tokenizer = tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
punkt_sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
roberta_preproc = RobertaPreprocessor(roberta_tokenizer, punkt_sentence_detector)
data_df['text'] = data_df['text_corrected'].map(roberta_preproc.add_special_tokens)
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    

class RobertaClasasificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.swish = MemoryEfficientSwish()
        
    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class EffRoberta(nn.Module):
    def __init__(
        self,
        roberta_model_name:str = 'distilroberta-base',
        efficientnet_model_name:str = 'efficientnet-b4',
        num_classes:int = 2,
        effnet_advprop: bool = False,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(
            roberta_model_name,
            num_labels = num_classes,
        )
        self.effnet = EfficientNet.from_pretrained(
            'efficientnet-b4',
            num_classes = num_classes,
            advprop = effnet_advprop,
        )
        self.roberta_clf = RobertaClasasificationHead(self.roberta.config)
        self.dropout = nn.Dropout( p=0.1, inplace=True)
        self.fc1 = nn.Linear(2*num_classes,num_classes)
        self.swish = MemoryEfficientSwish()
        
    def forward(self, indices, attn_mask, images):
            roberta_out = self.roberta(
                input_ids = indices, 
                attention_mask =  attn_mask, 
            )[0]
            y_pred_roberta = self.roberta_clf(roberta_out)
            
            y_pred_effnet = self.effnet(images)
            self.dropout(y_pred_effnet)
            
            combined_y_pred = torch.cat([y_pred_roberta,y_pred_effnet],dim=1)
            combined_y_pred = self.swish(combined_y_pred)

            return combined_y_pred
model = EffRoberta( num_classes = len(set(data_df.label)))
model.to(args.device)
class SimpleVectorizer():
    def __init__(self,tokenizer: Callable, max_seq_len: int):
        """
        Args:
            tokenizer (Callable): transformer tokenizer
            max_seq_len (int): Maximum sequence lenght 
        """
        self.tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def vectorize(self,text :str):
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False, #already added by preproc
            max_length = self._max_seq_len,
            pad_to_max_length = True,
        )
        ids =  np.array(encoded['input_ids'], dtype=np.int64)
        attn = np.array(encoded['attention_mask'], dtype=np.int64)
        
        return ids, attn
class MemotionDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame,tokenizer:Callable, max_seq_length=None):
        """
        Args:
            data_df (pandas.DataFrame): df containing the labels and text
            tokenizer (tokenizer module for the transformer)
        """
        self.images_dir = IMAGES_DIR
        self.tokenizer = tokenizer
        
        if max_seq_length is None:
            self._max_seq_length = self._get_max_len(data_df,tokenizer)
        else:
            self._max_seq_length = max_seq_length
            
        self.vectorizer = SimpleVectorizer(tokenizer, self._max_seq_length)
        
        self.data_df = data_df
        
        
        self.train_df = self.data_df[self.data_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.data_df[self.data_df.split == 'val']
        self.val_size = len(self.val_df)

        
        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
        }

        self.set_split('train')

    def _get_max_len(self,data_df: pd.DataFrame, tokenizer: Callable):
        len_func = lambda x: len(self.tokenizer.encode_plus(x)['input_ids'])
        max_len = data_df.text.map(len_func).max() 
        return max_len

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
        if split == 'val':
            self.simple_vectorize = True
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]
        img_name = row.image_name

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        indices, attention_masks = self.vectorizer.vectorize(row.text)
        
        if self._target_split == 'train':
            tfms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            tfms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            
        img_path = os.path.join(self.images_dir,img_name) 
        img = Image.open(img_path)
        img = img.convert('RGB')
        preproc_img = tfms(img)

        label = row.label
        
        return {
            'x_images': preproc_img,
            'x_indices': indices,
            'x_attn_mask': attention_masks,
            'x_index': index,
            'y_target': label,
        }

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=False, device="cpu", pinned_memory = False, n_workers = 0): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last,
                            pin_memory= pinned_memory,
                            num_workers = n_workers,
                            )
    
    for data_dict in dataloader:
        out_data_dict = {}
        for key,val in data_dict.items():
            if key != 'x_index':
                out_data_dict[key] = data_dict[key].to(
                    device, non_blocking= (True if pinned_memory else False) 
                )
            else:
                out_data_dict[key] = data_dict[key]
                
        yield out_data_dict
dataset = MemotionDataset(
    data_df,
    roberta_tokenizer,
)
args.batch_size = 20
args.learning_rate = 2e-4
loss_func = nn.CrossEntropyLoss()

print(f'Using LR:{args.learning_rate}')
base_optimizer = RAdam(model.parameters(), lr = args.learning_rate)
optimizer = Lookahead(optimizer = base_optimizer, k = 5, alpha=0.5 )
early_stopping = transformer_general_utils.EarlyStopping(patience=4)
train_state = general_utils.make_train_state()
train_state.keys()
epoch_bar = notebook.tqdm(
    desc = 'training_routine',
    total = args.num_epochs,
    position=0,
    leave = True,
)
dataset.set_split('train')
train_bar = notebook.tqdm(
    desc = 'split=train ',
    total=dataset.get_num_batches(args.batch_size),
    position=0,
    leave=True,
)
dataset.set_split('val')
eval_bar = notebook.tqdm(
    desc = 'split=eval',
    total=dataset.get_num_batches(args.batch_size),
    position=0,
    leave=True,
)
for epoch_index in range(args.num_epochs):
    train_state['epoch_in'] = epoch_index

    dataset.set_split('train')

    batch_generator = generate_batches(
        dataset= dataset, batch_size= args.batch_size, shuffle=True,
        device = args.device, drop_last=False,
        pinned_memory = False, n_workers = 2, 
    )

    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    model.train()

    train_bar.reset(
        total=dataset.get_num_batches(args.batch_size),
    )

    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        y_pred = model(
            indices = batch_dict['x_indices'],
            attn_mask = batch_dict['x_attn_mask'],
            images = batch_dict['x_images'],
        )
        
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss.backward()
        optimizer.step()
                             
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
                             
        y_pred = y_pred.detach().cpu()
        batch_dict['y_target'] = batch_dict['y_target'].cpu()
        
        acc_t = transformer_general_utils \
            .compute_accuracy(y_pred, batch_dict['y_target'])
        
        f1_t = transformer_general_utils \
            .compute_macro_f1(y_pred, batch_dict['y_target'])

        train_state['batch_preds'].append(y_pred)
        train_state['batch_targets'].append(batch_dict['y_target'])
        train_state['batch_indexes'].append(batch_dict['x_index'])

        running_acc += (acc_t - running_acc) / (batch_index + 1)
        running_f1 += (f1_t - running_f1) / (batch_index + 1)

        train_bar.set_postfix(loss = running_loss, f1 = running_f1, acc=running_acc,
                             epoch=epoch_index)

        train_bar.update()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_state['train_accuracies'].append(running_acc)
    train_state['train_losses'].append(running_loss)
    
    train_state['train_preds'].append(
        torch.cat(train_state['batch_preds']).cpu()
    )
    train_state['train_targets'].append(
        torch.cat(train_state['batch_targets']).cpu()
    )
    train_state['train_indexes'].append(
        torch.cat(train_state['batch_indexes']).cpu()
    )
    train_f1 = transformer_general_utils \
                .compute_macro_f1(train_state['train_preds'][-1],
                                  train_state['train_targets'][-1],
                                 )
                                 
    train_state['train_f1s'].append(train_f1)
    
    train_state['batch_preds'] = []
    train_state['batch_targets'] = []
    train_state['batch_indexes'] = []
    
    
    dataset.set_split('val')
    batch_generator = generate_batches(
        dataset= dataset, batch_size= args.batch_size, shuffle=True,
        device = args.device, drop_last=False,
        pinned_memory = False, n_workers = 2, 
    )
    eval_bar.reset(
        total=dataset.get_num_batches(args.batch_size),
    )
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    
    model.eval()
    with torch.no_grad():
        optimizer._backup_and_load_cache()
        for batch_index, batch_dict in enumerate(batch_generator):
            
            y_pred = model(
                indices = batch_dict['x_indices'],
                attn_mask = batch_dict['x_attn_mask'],
                images = batch_dict['x_images'],
            )

#             y_pred = y_pred.view(-1, len(set(dataset.data_df.label)))

            loss = loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            
            y_pred = y_pred.detach().cpu()
            
            batch_dict['y_target'] = batch_dict['y_target'].cpu()
            
            acc_t = transformer_general_utils\
                .compute_accuracy(y_pred, batch_dict['y_target'])
            f1_t = transformer_general_utils \
                .compute_macro_f1(y_pred, batch_dict['y_target'])

            train_state['batch_preds'].append(y_pred.cpu())
            train_state['batch_targets'].append(batch_dict['y_target'].cpu())
            train_state['batch_indexes'].append(batch_dict['x_index'].cpu())

            running_acc += (acc_t - running_acc) / (batch_index + 1)
            running_f1 += (f1_t - running_f1) / (batch_index + 1)
            

            eval_bar.set_postfix(loss = running_loss, f1 = running_f1, acc=running_acc,
                                 epoch=epoch_index)
            eval_bar.update()
            
    train_state['val_accuracies'].append(running_acc)
    train_state['val_losses'].append(running_loss)
    
        
    train_state['val_preds'].append(
        torch.cat(train_state['batch_preds']).cpu()
    )

    train_state['val_targets'].append(
        torch.cat(train_state['batch_targets']).cpu()
    )
    train_state['val_indexes'].append(
        torch.cat(train_state['batch_indexes']).cpu()
    )
    val_f1 = transformer_general_utils \
                .compute_macro_f1(train_state['val_preds'][-1],
                                  train_state['val_targets'][-1],
                                 )
                                 
    train_state['val_f1s'].append(val_f1)
    
    train_state['batch_preds'] = []
    train_state['batch_targets'] = []
    train_state['batch_indexes'] = []
    
    early_stopping(val_f1, model)
    optimizer._clear_and_load_backup()
    epoch_bar.set_postfix( best_f1 = early_stopping.best_score, current = val_f1)
    epoch_bar.update()    
    
    if early_stopping.early_stop:
        print("Early stopping")
        break
print( f' Humor labels: {set(labels_pd_["humour"])}')
print( f' Sarcasm labels: {set(labels_pd_["sarcasm"])}')
print( f' Offensive labels: {set(labels_pd_["offensive"])}')
print( f' Motivational labels: {set(labels_pd_["motivational"])}')


humour_labels_dict = {'funny':1, 'hilarious':1, 'not_funny':0, 'very_funny':1}
sarcasm_labels_dict = {'general':1, 'twisted_meaning':1, 'not_sarcastic':0, 'very_twisted':1}
motivational_labels_dict = { 'motivational':1, 'not_motivational':0 }
offensive_labels_dict = { 'hateful_offensive':1, 'slight':1, 'not_offensive':0, 'very_offensive':1}

task_b_labels_df = labels_pd_.copy()

task_b_labels_df['humour'] = labels_pd_['humour'].map(humour_labels_dict)
task_b_labels_df['sarcasm'] = labels_pd_['sarcasm'].map(sarcasm_labels_dict)
task_b_labels_df['offensive'] = labels_pd_['offensive'].map(offensive_labels_dict)
task_b_labels_df['motivational'] = labels_pd_['motivational'].map(motivational_labels_dict)

print(task_b_labels_df.humour.value_counts(),'\n')
print(task_b_labels_df.sarcasm.value_counts(),'\n')
print(task_b_labels_df.offensive.value_counts(),'\n')
print(task_b_labels_df.motivational.value_counts(),'\n')

print('Total:\n',
     pd.concat(
        [
            task_b_labels_df['humour'],
            task_b_labels_df['sarcasm'],
            task_b_labels_df['offensive'],
            task_b_labels_df['motivational'],
        ],
        ignore_index= True,
        axis = 0,
    ).value_counts()      
)
class MemotionImageDataset(Dataset):
    def __init__(self, data_df: pd.DataFrame , images_dir:str = None):
        """
        Args:
            data_df (pandas.DataFrame): df containing the labels and text
            tokenizer (tokenizer module for the transformer)
        """

        self.images_dir = IMAGES_DIR
        
        self.data_df = data_df
        
        self.train_df = self.data_df[self.data_df.split == 'train']
        self.train_size = len(self.train_df)

        self.val_df = self.data_df[self.data_df.split == 'val']
        self.val_size = len(self.val_df)

        
        self._lookup_dict = {
            'train': (self.train_df, self.train_size),
            'val': (self.val_df, self.val_size),
        }

        self.set_split('train')

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
    
    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]
        img_name = row.image_name

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        if self._target_split == 'train':
            tfms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif self._target_split =='val':
            tfms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            
        img_path = os.path.join(self.images_dir,img_name) 
        img = Image.open(img_path)
        img = img.convert('RGB')
        preproc_img = tfms(img)

        label = row.label
        
        return {
            'x_data': preproc_img,
            'x_index': index,
            'y_target': label
        }

    
    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
def generate_image_batches(dataset, batch_size, shuffle=True,
                     drop_last=False, device="cpu", pinned_memory = False, n_workers = 0): 
    """
    A generator function which wraps the PyTorch DataLoader. It will 
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last,
                            pin_memory= pinned_memory,
                            num_workers = n_workers,
                            )
    
    for data_dict in dataloader:
        out_data_dict = {}
        out_data_dict['x_data'] = data_dict['x_data'].to(
            device, non_blocking= (True if pinned_memory else False) 
        )
        out_data_dict['x_index'] = data_dict['x_index']
        out_data_dict['y_target'] = data_dict['y_target'].to(
            device, non_blocking= (True if pinned_memory else False) 
        )
        yield out_data_dict
humor_df = task_b_labels_df[['image_name','humour']].copy()
humor_df.rename(columns = {'humour':'label'}, inplace=True)
humor_df = get_train_val_split(0.90, humor_df,'image_name')
humor_df[humor_df.split == 'train'].label.value_counts()
humor_df[humor_df.split == 'val'].label.value_counts()
model = EfficientNet.from_pretrained(
    'efficientnet-b4',
    num_classes = len(set(humor_df.label)),
 )
model.to(args.device)
dataset = MemotionImageDataset(humor_df)
loss_func = nn.CrossEntropyLoss()
print(f'Using LR:{args.learning_rate}')
base_optimizer = RAdam(model.parameters(), lr = args.learning_rate)
optimizer = Lookahead(optimizer = base_optimizer, k = 5, alpha=0.5 )
early_stopping = transformer_general_utils.EarlyStopping(patience=2)
args.batch_size = 32
train_state = general_utils.make_train_state()
train_state.keys()
epoch_bar = notebook.tqdm(
    desc = 'training_routine',
    total = args.num_epochs,
    position=0,
    leave = True,
)
dataset.set_split('train')
train_bar = notebook.tqdm(
    desc = 'split=train ',
    total=dataset.get_num_batches(args.batch_size),
    position=0,
    leave=True,
)
dataset.set_split('val')
eval_bar = notebook.tqdm(
    desc = 'split=eval',
    total=dataset.get_num_batches(args.batch_size),
    position=0,
    leave=True,
)
for epoch_index in range(args.num_epochs):
    train_state['epoch_in'] = epoch_index

    dataset.set_split('train')

    batch_generator = generate_image_batches(
        dataset= dataset, batch_size= args.batch_size, shuffle=True,
        device = args.device, drop_last=False,
        pinned_memory = False, n_workers = 2, 
    )

    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    model.train()

    train_bar.reset(
        total=dataset.get_num_batches(args.batch_size),
    )

    for batch_index, batch_dict in enumerate(batch_generator):
        optimizer.zero_grad()
        
        y_pred = model(batch_dict['x_data'])
        
        y_pred = y_pred.view(-1, len(set(dataset.data_df.label)))
        
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss.backward()
        optimizer.step()
                             
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)
                             
        y_pred = y_pred.detach().cpu()
        batch_dict['y_target'] = batch_dict['y_target'].cpu()
        
        acc_t = transformer_general_utils \
            .compute_accuracy(y_pred, batch_dict['y_target'])
        
        f1_t = transformer_general_utils \
            .compute_macro_f1(y_pred, batch_dict['y_target'])

        train_state['batch_preds'].append(y_pred)
        train_state['batch_targets'].append(batch_dict['y_target'])
        train_state['batch_indexes'].append(batch_dict['x_index'])

        running_acc += (acc_t - running_acc) / (batch_index + 1)
        running_f1 += (f1_t - running_f1) / (batch_index + 1)

        train_bar.set_postfix(loss = running_loss, f1 = running_f1, acc=running_acc,
                             epoch=epoch_index)

        train_bar.update()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_state['train_accuracies'].append(running_acc)
    train_state['train_losses'].append(running_loss)
    
    train_state['train_preds'].append(
        torch.cat(train_state['batch_preds']).cpu()
    )
    train_state['train_targets'].append(
        torch.cat(train_state['batch_targets']).cpu()
    )
    train_state['train_indexes'].append(
        torch.cat(train_state['batch_indexes']).cpu()
    )
    train_f1 = transformer_general_utils \
                .compute_macro_f1(train_state['train_preds'][-1],
                                  train_state['train_targets'][-1],
                                 )
                                 
    train_state['train_f1s'].append(train_f1)
    
    train_state['batch_preds'] = []
    train_state['batch_targets'] = []
    train_state['batch_indexes'] = []
    
    
    dataset.set_split('val')
    batch_generator = generate_image_batches(
        dataset= dataset, batch_size= args.batch_size, shuffle=True,
        device = args.device, drop_last=False,
        pinned_memory = False, n_workers = 2, 
    )
    eval_bar.reset(
        total=dataset.get_num_batches(args.batch_size),
    )
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = 0.0
    
    model.eval()
    with torch.no_grad():
        optimizer._backup_and_load_cache()
        for batch_index, batch_dict in enumerate(batch_generator):
            y_pred = model(batch_dict['x_data'])
            y_pred = y_pred.view(-1, len(set(dataset.data_df.label)))
            loss = loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            y_pred = y_pred.detach().cpu()
            batch_dict['y_target'] = batch_dict['y_target'].cpu()
            
            acc_t = transformer_general_utils\
                .compute_accuracy(y_pred, batch_dict['y_target'])
            f1_t = transformer_general_utils \
                .compute_macro_f1(y_pred, batch_dict['y_target'])

            train_state['batch_preds'].append(y_pred.cpu())
            train_state['batch_targets'].append(batch_dict['y_target'].cpu())
            train_state['batch_indexes'].append(batch_dict['x_index'].cpu())

            running_acc += (acc_t - running_acc) / (batch_index + 1)
            running_f1 += (f1_t - running_f1) / (batch_index + 1)
            

            eval_bar.set_postfix(loss = running_loss, f1 = running_f1, acc=running_acc,
                                 epoch=epoch_index)
            eval_bar.update()
            
    train_state['val_accuracies'].append(running_acc)
    train_state['val_losses'].append(running_loss)
    
        
    train_state['val_preds'].append(
        torch.cat(train_state['batch_preds']).cpu()
    )

    train_state['val_targets'].append(
        torch.cat(train_state['batch_targets']).cpu()
    )
    train_state['val_indexes'].append(
        torch.cat(train_state['batch_indexes']).cpu()
    )
    val_f1 = transformer_general_utils \
                .compute_macro_f1(train_state['val_preds'][-1],
                                  train_state['val_targets'][-1],
                                 )
                                 
    train_state['val_f1s'].append(val_f1)
    
    train_state['batch_preds'] = []
    train_state['batch_targets'] = []
    train_state['batch_indexes'] = []
    
    scheduler.step(val_f1)
    early_stopping(val_f1, model)
    optimizer._clear_and_load_backup()
    epoch_bar.set_postfix( best_f1 = early_stopping.best_score, current = val_f1)
    epoch_bar.update()    
    
    if early_stopping.early_stop:
        print("Early stopping")
        break
        