!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-development.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-test.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-validation.tsv -q
!ls
!pip install pytorch-pretrained-bert

!pip install https://github.com/ceshine/pytorch_helper_bot/archive/0.0.4.zip
import os



# This variable is used by helperbot to make the training deterministic

os.environ["SEED"] = "420"



import logging

from pathlib import Path



import torch

import torch.nn as nn

import numpy as np

import pandas as pd

from torch.utils.data import Dataset, DataLoader

from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.modeling import BertModel



from helperbot import BaseBot, TriangularLR, WeightDecayOptimizerWrapper
# 根据代词和候选指代A、B的偏移量，插入相应的标记，便于在tokenization后定位

def insert_tag(row):

    """Insert custom tags to help us find the position of A, B, and the pronoun after tokenization."""

    # 指代A, B和代词的偏移量降序排序

    to_be_inserted = sorted([

        (row["A-offset"], " [A] "),

        (row["B-offset"], " [B] "),

        (row["Pronoun-offset"], " [P] ")

    ], key=lambda x: x[0], reverse=True)

    text = row["Text"]

    # 插入标记

    for offset, tag in to_be_inserted:

        text = text[:offset] + tag + text[offset:]

    return text



# 对文本进行tokenization，根据插入的标记，取得指代A、B和代词的位置

def tokenize(text, tokenizer):

    """Returns a list of tokens and the positions of A, B, and the pronoun."""

    entries = {}       # 根据标记定位token化后的A，B，pronoun的位置 

    final_tokens = []  # token化后的词

    for token in tokenizer.tokenize(text):

        if token in ("[A]", "[B]", "[P]"):

            entries[token] = len(final_tokens)     

            continue

        final_tokens.append(token)

    return final_tokens, (entries["[A]"], entries["[B]"], entries["[P]"])



# 自定义pytorch dataset类，用于读取数据

class GAPDataset(Dataset):

    """Custom GAP Dataset class"""

    def __init__(self, df, tokenizer, labeled=True):

        self.labeled = labeled

        # 设置label。"A-coref"：代词是否指代A。B同理。 "Neither"：既不指代A，也不指代B

        if labeled:   

            tmp = df[["A-coref", "B-coref"]].copy()

            tmp["Neither"] = ~(df["A-coref"] | df["B-coref"])

            self.y = tmp.values.astype("bool")



        # 提取tokens和A，B，P的偏移量

        self.offsets, self.tokens = [], []

        for _, row in df.iterrows():

            text = insert_tag(row)       # 插入A、B、P标记，返回插入标记的文本

            tokens, offsets = tokenize(text, tokenizer)

            self.offsets.append(offsets)

            self.tokens.append(tokenizer.convert_tokens_to_ids(

                ["[CLS]"] + tokens + ["[SEP]"]))   # BERT输入格式，句首加入“[CLS]",句尾加入”[SEP]"

    

    # 获取数据集大小

    def __len__(self):

        return len(self.tokens)

    

    # 取数据函数

    def __getitem__(self, idx):

        if self.labeled:

            return self.tokens[idx], self.offsets[idx], self.y[idx]

        return self.tokens[idx], self.offsets[idx]

    

# 将一个batch的数据转换成tensor。将[(tokens,offsets,labels),...]转换成tokens tensor, offsets tensor, label tensor

def collate_examples(batch, truncate_len=500):

    """Batch preparation.

    

    1. Pad the sequences

    2. Transform the target.

    """

    # [(tokens1, offsets1), (tokens2, offsets2)] => [(tokens1, tokens2), (offsets1, offsets2)] 

    transposed = list(zip(*batch)) 

    # 输入序列的最大长度

    max_len = min(

        max((len(x) for x in transposed[0])),

        truncate_len

    )

    

    # tokens转成tensor

    tokens = np.zeros((len(batch), max_len), dtype=np.int64)

    for i, row in enumerate(transposed[0]):

        row = np.array(row[:truncate_len])    # tokens超过长度，截断

        tokens[i, :len(row)] = row            

    token_tensor = torch.from_numpy(tokens)

    

    # Offsets转换成tensor

    offsets = torch.stack([

        torch.LongTensor(x) for x in transposed[1]

    ], dim=0) + 1 # Account for the [CLS] token

    

    # 将长度为3的one-hot label，转换为数字label

    if len(transposed) == 2:

        return token_tensor, offsets, None

    one_hot_labels = torch.stack([

        torch.from_numpy(x.astype("uint8")) for x in transposed[2]

    ], dim=0)

    _, labels = one_hot_labels.max(dim=1)

    

    return token_tensor, offsets, labels
# 多层感知机网络

class Head(nn.Module):

    """The MLP submodule"""

    def __init__(self, bert_hidden_size: int):

        super().__init__()

        self.head_hidden_size = 1024  # MLP隐层大小

        self.bert_hidden_size = bert_hidden_size   # Bert的隐层大小

        self.fc = nn.Sequential(

            nn.BatchNorm1d(bert_hidden_size * 3),  # 批标准化，*3是因为将A，B，P的bert_output展开成1维了

            nn.Dropout(0.5),                       # 随机失活

            nn.Linear(bert_hidden_size * 3, self.head_hidden_size), # 线性层

            nn.ReLU(),                                              # 激活函数

            nn.BatchNorm1d(self.head_hidden_size),

            nn.Dropout(0.5),

            nn.Linear(self.head_hidden_size, self.head_hidden_size),

            nn.ReLU(),

            nn.BatchNorm1d(self.head_hidden_size),

            nn.Dropout(0.5),

            nn.Linear(self.head_hidden_size, self.head_hidden_size),

            nn.ReLU(),

            nn.BatchNorm1d(self.head_hidden_size),

            nn.Dropout(0.5),

            nn.Linear(self.head_hidden_size, 3)

        )

        

        # 参数初始化，不同网络块初始化方法不一样

        for i, module in enumerate(self.fc):

            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):

                nn.init.constant_(module.weight, 1)

                nn.init.constant_(module.bias, 0)

                print("Initing batchnorm")

            elif isinstance(module, nn.Linear):

                if getattr(module, "weight_v", None) is not None:

                    nn.init.uniform_(module.weight_g, 0, 1)

                    nn.init.kaiming_normal_(module.weight_v)

                    print("Initing linear with weight normalization")

                    assert model[i].weight_g is not None

                else:

                    nn.init.kaiming_normal_(module.weight)

                    print("Initing linear")

                nn.init.constant_(module.bias, 0)

    

    # 前向传播函数

    def forward(self, bert_outputs, offsets):

        # bert_outputs:[batch_size, seq_length, hidden_szie]

        assert bert_outputs.size(2) == self.bert_hidden_size   

        

        # 取出A，B，P的offsets处的embedding

        # unsqueeze(2):将2维offsets拓展为3维

        # 扩展某个size为1的维度。如(2,2,1)扩展为(2,2,3)

        # input.gather(dim,index), 对指定维进行索引。比如4*3的张量，对dim=1进行索引，那么index的取值范围就是0~2.

        extracted_outputs = bert_outputs.gather(

            1, offsets.unsqueeze(2).expand(-1, -1, bert_outputs.size(2)) 

        ).view(bert_outputs.size(0), -1)      

        return self.fc(extracted_outputs)



# 指代消解模型

class GAPModel(nn.Module):

    """The main model."""

    def __init__(self, bert_model: str, device: torch.device):

        super().__init__()

        self.device = device  # 设备：cpu 或 gpu

        if bert_model in ("bert-base-uncased", "bert-base-cased"):

            self.bert_hidden_size = 768

        elif bert_model in ("bert-large-uncased", "bert-large-cased"):

            self.bert_hidden_size = 1024

        else:

            raise ValueError("Unsupported BERT model.")

        self.bert = BertModel.from_pretrained(bert_model).to(device)

        self.head = Head(self.bert_hidden_size).to(device)

    

    def forward(self, token_tensor, offsets):

        token_tensor = token_tensor.to(self.device)

        bert_outputs, _ =  self.bert(

            token_tensor, attention_mask=(token_tensor > 0).long(), 

            token_type_ids=None, output_all_encoded_layers=False)

        head_outputs = self.head(bert_outputs, offsets.to(self.device))

        return head_outputs            
offsets = torch.tensor([[0,1,2],[1,2,3]])  # batch_size=2, len(A,B,P)= 3

print(offsets.shape)

offsets = offsets.unsqueeze(2)

print(offsets)

print(offsets.shape)
offsets=offsets.expand(-1,-1,5)   # 假设bert_hidden_size=5

print(offsets)
bert_outputs = torch.tensor([[[ 1,  2,  3,  4,  5],

                  [ 6,  7,  8,  9, 10],

                  [11, 12, 13, 14, 15],

                  [16, 17, 18, 19, 20]],

                 [[21, 22, 23, 24, 25],

                  [26, 27, 28, 29, 30],

                  [31, 32, 33, 34, 35],

                  [36, 37, 38, 39, 40]]])
print(offsets.shape)

print(bert_outputs.shape)
bert_outputs.gather(1,offsets) # 按offsets到bert_outputs的第1维取数，两者除了第一维外，其他维度大小一致
# 嵌套的网络结构，module包括很多children子网络模块

def children(m):

    return m if isinstance(m, (list, tuple)) else list(m.children())



def set_trainable_attr(m, b):

    m.trainable = b

    for p in m.parameters():

        p.requires_grad = b



def apply_leaf(m, f):

    c = children(m)

    if isinstance(m, nn.Module):

        f(m)

    if len(c) > 0:

        for l in c:

            apply_leaf(l, f)

        

def set_trainable(l, b):

    apply_leaf(l, lambda m: set_trainable_attr(m, b))
class GAPBot(BaseBot):

    def __init__(self, model, train_loader, val_loader, optimizer, clip_grad=0,

        avg_window=100, log_dir="./cache/logs/", log_level=logging.INFO,

        checkpoint_dir="./cache/model_cache/", batch_idx=0, echo=False,

        device="cuda:0", use_tensorboard=False):

        super().__init__(

            model, train_loader, val_loader, 

            optimizer=optimizer, clip_grad=clip_grad,

            log_dir=log_dir, checkpoint_dir=checkpoint_dir, 

            batch_idx=batch_idx, echo=echo,

            device=device, use_tensorboard=use_tensorboard

        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.loss_format = "%.6f"

        

    def extract_prediction(self, tensor):

        return tensor

    

    # 打印日志

    def snapshot(self):

        loss = self.eval(self.val_loader)

        loss_str = self.loss_format % loss

        self.logger.info("Snapshot loss %s", loss_str)

        self.logger.tb_scalars(

            "losses", {"val": loss},  self.step)

        target_path = (

            self.checkpoint_dir / "best.pth")        

        if not self.best_performers or (self.best_performers[0][0] > loss):

            torch.save(self.model.state_dict(), target_path)

            self.best_performers = [(loss, target_path, self.step)]

            self.logger.info("Saving checkpoint %s...", target_path)

        else:

            new_loss_str = self.loss_format % self.best_performers[0][0]

            self.logger.info("This performance:%s is not as a good as our previously saved:%s", loss_str,new_loss_str )

        assert Path(target_path).exists()

        return loss
df_train = pd.read_csv("gap-test.tsv", delimiter="\t")

df_val = pd.read_csv("gap-validation.tsv", delimiter="\t")

df_test = pd.read_csv("../input/test_stage_2.tsv", delimiter="\t")

sample_sub = pd.read_csv("../input/sample_submission_stage_2.csv")

assert sample_sub.shape[0] == df_test.shape[0]
print(len(df_train))

df_train.head()
print(len(df_test))

df_test.head()
BERT_MODEL = 'bert-large-uncased'

CASED = False



tokenizer = BertTokenizer.from_pretrained(

    BERT_MODEL,

    do_lower_case=CASED,

    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")

)

# These tokens are not actually used, so we can assign arbitrary values.

tokenizer.vocab["[A]"] = -1

tokenizer.vocab["[B]"] = -1

tokenizer.vocab["[P]"] = -1
train_ds = GAPDataset(df_train, tokenizer)

val_ds = GAPDataset(df_val, tokenizer)

test_ds = GAPDataset(df_test, tokenizer, labeled=False)

# dataset 转换成dataloader

train_loader = DataLoader(

    train_ds,

    collate_fn = collate_examples,     # 构成batch函数

    batch_size=20,

    num_workers=2,

    pin_memory=True,   # 使用锁页内存，这样tensor转传入cuda会快些

    shuffle=True,

    drop_last=True     # 丢弃不完整的batch

)

val_loader = DataLoader(

    val_ds,

    collate_fn = collate_examples,

    batch_size=128,

    num_workers=2,

    pin_memory=True,

    shuffle=False

)

test_loader = DataLoader(

    test_ds,

    collate_fn = collate_examples,

    batch_size=128,

    num_workers=2,

    pin_memory=True,

    shuffle=False

)
len(train_loader), len(test_loader), len(val_loader)
next(iter(test_loader))
model = GAPModel(BERT_MODEL, torch.device("cuda:0"))

# You can unfreeze the last layer of bert by calling set_trainable(model.bert.encoder.layer[23], True)

set_trainable(model.bert, False)

set_trainable(model.head, True)
lr=1e-3

weight_decay=5e-3

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)



bot = GAPBot(

    model, train_loader, val_loader,

    optimizer=optimizer, echo=True,

    avg_window=25

)
steps_per_epoch = len(train_loader) 

n_steps = steps_per_epoch * 27

bot.train(

    n_steps,

    log_interval=steps_per_epoch // 4,

    snapshot_interval=steps_per_epoch,

    scheduler=TriangularLR(

        optimizer, max_mul=20, ratio=2, steps_per_cycle=n_steps)

)     
# Load the best checkpoint

bot.load_model(bot.best_performers[0][1])
torch.save(model.state_dict(), './model.pth')
# 预测函数

def predict(loader, *, return_y=False):

    model.eval()

    outputs, y_global = [], []

    with torch.set_grad_enabled(False):

        for input_tensors in loader:

            input_tensors = [x.to(model.device) for x in input_tensors if x is not None]

            outputs.append(bot.predict_batch(input_tensors).cpu())

        outputs = torch.cat(outputs, dim=0)

    return outputs
preds = predict(test_loader)
len(preds)
# Create submission file

df_sub = pd.DataFrame(torch.softmax(preds, -1).cpu().numpy().clip(1e-3, 1-1e-3), columns=["A", "B", "NEITHER"])

df_sub["ID"] = df_test.ID

df_sub.to_csv("submission.csv", index=False)

df_sub.head()