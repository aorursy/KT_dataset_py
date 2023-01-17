import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import eli5
train_val = pd.read_csv('../input/amazontrainreviews/train.csv', index_col=0)
train_val.reset_index(drop=True, inplace=True)
print(train_val.info())
display(train_val.head())
sns.countplot(train_val['labels']);
plt.title('Labels distribution');
train_val['len'] = train_val['sentences'].apply(lambda x: len(x.split()))
sns.distplot(train_val['len']);
neg_mean_len = train_val.groupby('labels')['len'].mean().values[0]
pos_mean_len = train_val.groupby('labels')['len'].mean().values[1]

print(f"Negative mean length: {neg_mean_len:.2f}")
print(f"Positive mean length: {pos_mean_len:.2f}")
print(f"Mean Difference: {neg_mean_len-pos_mean_len:.2f}")
ax = sns.catplot(x='labels', y='len', data=train_val, kind='box')
neg_array = train_val[train_val['labels']==0]['len'].values
pos_array = train_val[train_val['labels']==1]['len'].values
mean_diff = neg_mean_len - pos_mean_len
def permutation_sample(data1, data2):
    # Permute the concatenated array: permuted_data
    data = np.concatenate((data1,data2))
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2
def draw_perm_reps(data_1, data_2, size=1):

    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = np.mean(perm_sample_1) - np.mean(perm_sample_2)

    return perm_replicates
perm_replicates = draw_perm_reps(neg_array, pos_array,
                                 size=10000)

# Compute p-value: p
p = np.sum(perm_replicates >= mean_diff) / len(perm_replicates)

print(f'p-value = {p}')
def prediction(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, pred)
    f1 = f1_score(y_valid, pred)
    conf = confusion_matrix(y_valid, pred)
    joblib.dump(model, f"model_acc_{acc:.5f}.pkl")
    return model, acc, f1, conf
transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), 
                              lowercase=True, max_features=100000)
X = transformer.fit_transform(train_val['sentences'])
y = train_val.labels
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)
model = LogisticRegression(C=1, random_state=42, n_jobs=-1)
fit_model, acc, f1, conf = prediction(model, X_train, y_train, X_valid, y_valid)
print(f"Accuracy: {acc:.5f}")
print(f"F1_Score: {f1:.5f}")
print(f"Confusion Matrix: {conf}")
eli5.show_weights(estimator=fit_model, 
                  feature_names= list(transformer.get_feature_names()),
                    top=(20,20))
!pip install --pre torch==1.7.0.dev20200701+cu101 torchvision==0.8.0.dev20200701+cu101 -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
import torch
torch.__version__
import os
os.environ['WANDB_SILENT'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from typing import Mapping, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer, AutoModel

from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, OptimizerCallback, CheckpointCallback, WandbLogger
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.contrib.nn import RAdam, Lookahead, OneCycleLRWithWarmup
import wandb
MODEL_NAME = 'distilbert-base-uncased'
LOG_DIR = "./amazon" 
NUM_EPOCHS = 2 
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-3
ACCUMULATION_STEPS = 3
SEED = 42
FP_16 = dict(opt_level="O1")
set_global_seed(SEED)
prepare_cudnn(deterministic=True, benchmark=True)
class ReviewDataset(Dataset):

    
    def __init__(self,
                 sentences: List[str],
                 labels: List[str] = None,
                 max_seq_length: int = MAX_SEQ_LENGTH,
                 model_name: str = 'distilbert-base-uncased'):

        self.sentences = sentences
        self.labels = labels
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        
    def __len__(self):

        return len(self.sentences)

    
    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:

        sentence = self.sentences[index]
        encoded = self.tokenizer.encode_plus(sentence, add_special_tokens=True, 
                                        pad_to_max_length=True, max_length=self.max_seq_length, 
                                        return_tensors="pt",)
        
        output = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        output['targets'] = torch.tensor(self.labels[index], dtype=torch.long)
        
        return output
df_train, df_valid = train_test_split(
            train_val,
            test_size=0.2,
            random_state=42,
            stratify = train_val.labels.values
        )
print(df_train.shape, df_valid.shape)
train_dataset = ReviewDataset(
    sentences=df_train['sentences'].values.tolist(),
    labels=df_train['labels'].values,
    max_seq_length=MAX_SEQ_LENGTH,
    model_name=MODEL_NAME
)

valid_dataset = ReviewDataset(
    sentences=df_valid['sentences'].values.tolist(),
    labels=df_valid['labels'].values,
    max_seq_length=MAX_SEQ_LENGTH,
    model_name=MODEL_NAME
)
train_val_loaders = {
    "train": DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE, 
                        shuffle=True, num_workers=2, pin_memory=True),
    "valid": DataLoader(dataset=valid_dataset,
                        batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=2, pin_memory=True)    
}
print(df_valid.sentences.values[50])
valid_dataset[50]
class DistilBert(nn.Module):

    def __init__(self, pretrained_model_name: str = MODEL_NAME, num_classes: int = 2):

        super().__init__()

        config = AutoConfig.from_pretrained(
             pretrained_model_name)

        self.distilbert = AutoModel.from_pretrained(pretrained_model_name,
                                                    config=config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_classes)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, input_ids, attention_mask=None, head_mask=None):

        assert attention_mask is not None, "attention mask is none"
        distilbert_output = self.distilbert(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)
        hidden_state = distilbert_output[0]  # [BATCH_SIZE=32, MAX_SEQ_LENGTH = 512, DIM = 768]
        pooled_output = hidden_state[:, 0]  # [32, 768]
        pooled_output = self.pre_classifier(pooled_output)  # [32, 768]
        pooled_output = F.relu(pooled_output)  # [32, 768]
        pooled_output = self.dropout(pooled_output)  # [32, 768]
        logits = self.classifier(pooled_output)  # [32, 2]

        return logits
model = DistilBert()
param_optim = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
criterion = nn.CrossEntropyLoss()

base_optimizer = RAdam([
    {'params': [p for n,p in param_optim if not any(nd in n for nd in no_decay)],
     'weight_decay': WEIGHT_DECAY}, 
    {'params': [p for n,p in param_optim if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
])
optimizer = Lookahead(base_optimizer)
scheduler = OneCycleLRWithWarmup(
    optimizer, 
    num_steps=NUM_EPOCHS, 
    lr_range=(LEARNING_RATE, 1e-8),
    init_lr=LEARNING_RATE,
    warmup_steps=0,
)
runner = SupervisedRunner(
    input_key=(
        "input_ids",
        "attention_mask"
    )
)
# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=train_val_loaders,
    callbacks=[
        AccuracyCallback(num_classes=2),
        OptimizerCallback(accumulation_steps=ACCUMULATION_STEPS),
        WandbLogger(name="Name", project="sentiment-analysis"),
    ],
    fp16=FP_16,
    logdir=LOG_DIR,
    num_epochs=NUM_EPOCHS,
    verbose=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def prediction(model, sentence: str, max_len: int = 512, device = 'cpu'):
    x_encoded = tokenizer.encode_plus(sentence, add_special_tokens=True, pad_to_max_length=True, max_length=max_len, return_tensors="pt",).to(device)
    logits = model(x_encoded['input_ids'], x_encoded['attention_mask'])
    probabilities = F.softmax(logits.detach(), dim=1)
    output = probabilities.max(axis=1)
    print(sentence)
    print(f"Class: {['Negative' if output.indices[0] == 0 else 'Positive'][0]}, Probability: {output.values[0]:.4f}")
prediction(plain_model, df_valid['sentences'].values[20])