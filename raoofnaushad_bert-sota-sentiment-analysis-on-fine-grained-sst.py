!pip install pytorch_transformers pytreebank
import os

import numpy as np

import pytreebank



import torch

from torch.utils.data import Dataset

from pytorch_transformers import BertTokenizer,BertConfig, BertForSequenceClassification

from tqdm import tqdm
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



MODEL_OUT_DIR = '/kaggle/working'



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def rpad(array, n=70):

    """Right padding."""

    current_len = len(array)

    if current_len > n:

        return array[: n - 1]

    extra = n - current_len

    return array + ([0] * extra)



def get_binary_label(label):

    """Convert fine-grained label to binary label."""

    if label < 2:

        return 0

    if label > 2:

        return 1

    raise ValueError("Invalid label")
class SSTDataset(Dataset):

    ## Configurable SST Dataset.



    def __init__(self, split="train", root=True, binary=True):

        """Initializes the dataset with given configuration.



        Args:

            split: str

                Dataset split, one of [train, val, test]

            root: bool

                If true, only use root nodes. Else, use all nodes.

            binary: bool

                If true, use binary labels. Else, use fine-grained.

        """

        self.sst = sst[split]

        

        if root and binary:

            self.data = [

                (

                    rpad(

                        tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66

                    ),

                    get_binary_label(tree.label),

                )

                for tree in self.sst

                if tree.label != 2

            ]

        elif root and not binary:

            self.data = [

                (

                    rpad(tokenizer.encode("[CLS] " + tree.to_lines()[0] + " [SEP]"), n=66),

                    tree.label,

                )

                for tree in self.sst

            ]

        elif not root and not binary:

            self.data = [

                (rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66), label)

                for tree in self.sst

                for label, line in tree.to_labeled_lines()

            ]

        else:

            self.data = [

                (

                    rpad(tokenizer.encode("[CLS] " + line + " [SEP]"), n=66),

                    get_binary_label(label),

                )

                for tree in self.sst

                for label, line in tree.to_labeled_lines()

                if label != 2

            ]



    def __len__(self):

        return len(self.data)



    def __getitem__(self, index):

        X, y = self.data[index]

        X = torch.tensor(X)

        return X, y

def train_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):

    generator = torch.utils.data.DataLoader(

        dataset, batch_size=batch_size, shuffle=True

    )

    model.train()

    train_loss, train_acc = 0.0, 0.0

    for batch, labels in tqdm(generator, position=0, leave=True):

        batch, labels = batch.to(device), labels.to(device)

        optimizer.zero_grad()

        loss, logits = model(batch, labels=labels)

        err = lossfn(logits, labels)

        loss.backward()

        optimizer.step()



        train_loss += loss.item()

        pred_labels = torch.argmax(logits, axis=1)

        train_acc += (pred_labels == labels).sum().item()

    train_loss /= len(dataset)

    train_acc /= len(dataset)

    return train_loss, train_acc
def evaluate_one_epoch(model, lossfn, optimizer, dataset, batch_size=32):

    generator = torch.utils.data.DataLoader(

        dataset, batch_size=batch_size, shuffle=True

    )

    model.eval()

    loss, acc = 0.0, 0.0

    with torch.no_grad():

        for batch, labels in tqdm(generator, position=0, leave=True):

            batch, labels = batch.to(device), labels.to(device)

            logits = model(batch)[0]

            error = lossfn(logits, labels)

            loss += error.item()

            pred_labels = torch.argmax(logits, axis=1)

            acc += (pred_labels == labels).sum().item()

    loss /= len(dataset)

    acc /= len(dataset)

    return loss, acc
def train(

    root=True,

    binary=False,

    bert="bert-large-uncased",

    epochs=30,

    batch_size=32,

):

    trainset = SSTDataset("train", root=root, binary=binary)

    devset = SSTDataset("dev", root=root, binary=binary)

    testset = SSTDataset("test", root=root, binary=binary)

    

    best_val_loss = np.Inf

    

    config = BertConfig.from_pretrained(bert)

    if not binary:

        config.num_labels = 5

    model = BertForSequenceClassification.from_pretrained(bert, config=config)



    model = model.to(device)

    lossfn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)



    for epoch in range(1, epochs):

        train_loss, train_acc = train_one_epoch(

            model, lossfn, optimizer, trainset, batch_size=batch_size

        )

        val_loss, val_acc = evaluate_one_epoch(

            model, lossfn, optimizer, devset, batch_size=batch_size

        )

        test_loss, test_acc = evaluate_one_epoch(

            model, lossfn, optimizer, testset, batch_size=batch_size

        )

        print(f"epoch={epoch}")

        print(

            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, test_loss={test_loss:.4f}"

        )

        print(

            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}"

        )

        

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            label = "binary" if binary else "fine"

            nodes = "root" if root else "all"

#             torch.save(model, f"{bert}__{nodes}__{label}.pickle")

            model.save_pretrained(save_directory=MODEL_OUT_DIR + '/')

            config.save_pretrained(save_directory=MODEL_OUT_DIR + '/')

            tokenizer.save_pretrained(save_directory=MODEL_OUT_DIR + '/')



    print("Done!")

## Loading Tokenizer

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")



## loading SST Dataset (Stanford Tree Bank Dataset)

sst = pytreebank.load_sst()



## Configuration Values

binary = False 

root = True

save = False

bert_config = "bert-large-uncased"

train(binary=binary, root=root, bert=bert_config)

import torch.nn.functional as F

bert="bert-large-uncased"

config = BertConfig.from_pretrained(bert)

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

config.num_labels = 5

model = BertForSequenceClassification.from_pretrained(bert, config=config)

def classify_sentiment(sentence):

    model.eval()

    with torch.no_grad():

        data = torch.tensor(rpad(tokenizer.encode("[CLS] " + sentence + " [SEP]"), n=66)).unsqueeze(dim=0)  # Sometimes Unsqueeze 

        logits = model(data)[0]

        prob = F.softmax(logits, dim=1)

        print(prob)

        pred_label = torch.argmax(prob, axis=1)

        print(pred_label)

        
sentence = "great great love"

classify_sentiment(sentence)