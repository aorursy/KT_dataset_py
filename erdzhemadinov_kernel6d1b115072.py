import os



print(os.listdir("../input/huawei-scripts/"))
import csv

import re

from collections import defaultdict

from collections import Counter

from itertools import chain

from typing import List



import numpy as np

import torch

from torch.nn.utils.rnn import pack_sequence

from torch.utils.data import Dataset, DataLoader





def read_data(path="../input/huawei-scripts/cleaned.csv"):

    data = defaultdict(list)

    with open(path, newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', )

        next(reader)

        for row in reader:

            _, text, label = row

            if label is not None and len(label):

                data[int(float(label))].append(text)

            else:

                data[None].append(text)

    return data





def read_test(path="../input/huawei-scripts/test.csv"):

    texts = []

    uuids = []

    with open(path, newline='') as csvfile:

        reader = csv.reader(csvfile, delimiter=',', )

        next(reader)

        for row in reader:

            uuid, text, _ = row

            uuids.append(uuid)

            texts.append(text)

    return uuids, texts





class Tokenizer:

    def __init__(self, word_pattern="[\w']+"):

        """

        Simple tokenizer that splits the sentence by given regex pattern

        :param word_pattern: pattern that determines word boundaries

        """

        self.word_pattern = re.compile(word_pattern)



    def tokenize(self, text):

        return self.word_pattern.findall(text)





class Vocab:

    def __init__(self, tokenized_texts: List[List[str]], max_vocab_size=None):

        """

        Builds a vocabulary by concatenating all tokenized texts and counting words.

        Most common words are placed in vocabulary, others are replaced with [UNK] token

        :param tokenized_texts: texts to build a vocab

        :param max_vocab_size: amount of words in vocabulary

        """

        counts = Counter(chain(*tokenized_texts))

        max_vocab_size = max_vocab_size or len(counts)

        common_pairs = counts.most_common(max_vocab_size)

        self.PAD_IDX = 0

        self.UNK_IDX = 1

        self.EOS_IDX = 2

        self.itos = ["<PAD>", "<UNK>", "<EOS>"] + [pair[0] for pair in common_pairs]

        self.stoi = {token: i for i, token in enumerate(self.itos)}



    def vectorize(self, text: List[str]):

        """

        Maps each token to it's index in the vocabulary

        :param text: sequence of tokens

        :return: vectorized sequence

        """

        return [self.stoi.get(tok, self.UNK_IDX) for tok in text]



    def __iter__(self):

        return iter(self.itos)



    def __len__(self):

        return len(self.itos)





class TextDataset(Dataset):

    def __init__(self, tokenized_texts, labels, vocab: Vocab):

        """

        A Dataset for the task

        :param tokenized_texts: texts from a train/val/test split

        :param labels: corresponding toxicity ratings

        :param vocab: vocabulary with indexed tokens

        """

        self.texts = tokenized_texts

        self.labels = labels

        self.vocab = vocab



    def __getitem__(self, item):

        return self.vocab.vectorize(self.texts[item]) + [self.vocab.EOS_IDX], self.labels[item]



    def __len__(self):

        return len(self.texts)



    def collate_fn(self, batch):

        """

        Technical method to form a batch to feed into recurrent network

        """

        return pack_sequence([torch.tensor(pair[0]) for pair in batch], enforce_sorted=False), torch.tensor(

            [pair[1] for pair in batch])





def train_test_split(data, train_frac=0.85):

    """

    Splits the data into train and test parts, stratifying by labels.

    Should it shuffle the data before split?

    :param data: dataset to split

    :param train_frac: proportion of train examples

    :return: texts and labels for each split

    """

    n_toxicity_ratings = 6

    train_labels = []

    val_labels = []

    train_texts = []

    val_texts = []

    for label in range(n_toxicity_ratings):

        texts = data[label]

        n_train = int(len(texts) * train_frac)

        n_val = len(texts) - n_train

        train_texts.extend(texts[:n_train])

        val_texts.extend(texts[n_train:])

        train_labels += [label] * n_train

        val_labels += [label] * n_val

    return train_texts, train_labels, val_texts, val_labels
from random import choice, choices



from jinja2 import Template





def print_sample(data, toxicity_level, n_samples=5):

    return "\n\n".join(choices(data[toxicity_level], k=n_samples))





def show_example(data, ratings=(0, 0, 1, 1, 2, 2, 3, 3, 4, 5), max_len=400):

    """

    Shows some random examples for each toxicity rating in `ratings`. Total numbers of shown comments

    is equal to `ratings` length.

    :param data: toxic comments data

    :param ratings: ratings to show

    :param max_len: maximum characters to show in a comment

    :return:

    """

    my_template = Template("""

    <!DOCTYPE html>

    <html>

    <table>

    <tr> <th>Toxicity</th> <th>Comment</th>

    {% for item in items %}

    <TR>

       <TD class="c1" style="text-align:center; font-size:large" title="Toxicity Rating: {{item.rating}}">{{item.emoji}}</TD>

       <TD class="c2" style="font-size:large">{{item.text}}</TD>

    </TR>

    {% endfor %}

    </table>

    </html>""")

    rating2emoji = {0: "&#128519;",

                    1: "&#128528;",

                    2: "&#128551;",

                    3: "&#128565;",

                    4: "&#128557;",

                    5: "&#128561;"}

    items = []

    for rating in ratings:

        text = choice(data[rating])

        if len(text) > max_len:

            text = text[:max_len] + "..."

        items.append({"text": text,

                      "rating": rating,

                      "emoji": rating2emoji[rating]})

    return my_template.render(items=items)

from typing import Dict



import torch

from torch.nn.functional import sigmoid, relu, elu, tanh

from torch.nn import Module, Embedding, LSTM, RNN, GRU, Linear, Sequential, Dropout

from torch.nn.utils.rnn import PackedSequence







def prepare_emb_matrix(gensim_model, vocab: Vocab):

    """

    Extract embedding matrix from Gensim model for words in Vocab.

    Initialize embeddings not presented in `gensim_model` randomly

    :param gensim_model: W2V Gensim model

    :param vocab: vocabulary

    :return: embedding matrix

    """

    mean = gensim_model.vectors.mean(1).mean()

    std = gensim_model.vectors.std(1).mean()

    vec_size = gensim_model.vector_size

    emb_matrix = torch.zeros((len(vocab), vec_size))

    for i, word in enumerate(vocab.itos[1:], 1):

        try:

            emb_matrix[i] = torch.tensor(gensim_model.get_vector(word))

        except KeyError:

            emb_matrix[i] = torch.randn(vec_size) * std + mean

    return emb_matrix





class RecurrentClassifier(Module):

    def __init__(self, config: Dict, vocab: Vocab, emb_matrix):

        """

        Baseline classifier, hyperparameters are passed in `config`.

        Consists of recurrent part and a classifier (Multilayer Perceptron) part

        Keys are:

            - freeze: whether word embeddings should be frozen

            - cell_type: one of: RNN, GRU, LSTM, which recurrent cell model should use

            - hidden_size: size of hidden state for recurrent cell

            - num_layers: amount of recurrent cells in the model

            - cell_dropout: dropout rate between recurrent cells (not applied if model has only one cell!)

            - bidirectional: boolean, whether to use unidirectional of bidirectional model

            - out_activation: one of: "sigmoid", "tanh", "relu", "elu". Activation in classifier part

            - out_dropout: dropout rate in classifier part

            - out_sizes: List[int], hidden size of each layer in classifier part. Empty list means that final

                layer is attached directly to recurrent part output

        :param config: configuration of model

        :param vocab: vocabulary

        :param emb_matrix: embeddings matrix from `prepare_emb_matrix`

        """

        super().__init__()

        self.config = config

        self.vocab = vocab

        self.emb_matrix = emb_matrix

        self.embeddings = Embedding.from_pretrained(emb_matrix, freeze=config["freeze"],

                                                    padding_idx=vocab.PAD_IDX)

        cell_types = {

           

            "GRU": GRU

            }

        cell_class = cell_types[config["cell_type"]]

        self.cell = cell_class(input_size=emb_matrix.size(1),

                               batch_first=True,

                               hidden_size=config["hidden_size"],

                               num_layers=config["num_layers"],

                               dropout=config["cell_dropout"],

                               bidirectional=config["bidirectional"],

                               )

        activation_types = {

            "sigmoid": sigmoid,

            "tanh": tanh,

            "relu": relu,

            "elu": elu,

        }

        self.out_activation = activation_types[config["out_activation"]]

        self.out_dropout = Dropout(config["out_dropout"])

        cur_out_size = config["hidden_size"] * config["num_layers"]

        if config["bidirectional"]:

            cur_out_size *= 2

        out_layers = []

        for cur_hidden_size in config["out_sizes"]:

            out_layers.append(Linear(cur_out_size, cur_hidden_size))

            cur_out_size = cur_hidden_size

        out_layers.append(Linear(cur_out_size, 6))

        self.out_proj = Sequential(*out_layers)



    def forward(self, input):

        embedded = self.embeddings(input.data)

        _, last_state = self.cell(PackedSequence(embedded,

                                                 input.batch_sizes,

                                                 sorted_indices=input.sorted_indices,

                                                 unsorted_indices=input.unsorted_indices))

        if isinstance(last_state, tuple):

            last_state = last_state[0]

        last_state = last_state.transpose(0, 1)

        last_state = last_state.reshape(last_state.size(0), -1)

        return self.out_proj(last_state)



from typing import Dict



import torch

from numpy import asarray

from torch.nn import CrossEntropyLoss

from torch.optim import Adam

from tqdm.notebook import tqdm

#from model import RecurrentClassifier





class Trainer:

    def __init__(self, config: Dict):

        """

        Fits end evaluates given model with Adam optimizer.

         Hyperparameters are specified in `config`

        Possible keys are:

            - n_epochs: number of epochs to train

            - lr: optimizer learning rate

            - weight_decay: l2 regularization weight

            - device: on which device to perform training ("cpu" or "cuda")

            - verbose: whether to print anything during training

        :param config: configuration for `Trainer`

        """

        self.config = config

        self.n_epochs = config["n_epochs"]

        self.setup_opt_fn = lambda model: Adam(model.parameters(),

                                               config["lr"])

        self.model = None

        self.opt = None

        self.history = None

        self.loss_fn = CrossEntropyLoss()

        self.device = config["device"]

        self.verbose = config.get("verbose", True)



    def fit(self, model, train_loader, val_loader):

        """

        Fits model on training data, each epoch evaluates on validation data

        :param model: PyTorch model for toxic comments classification (for example, `RecurrentClassifier`)

        :param train_loader: DataLoader for training data

        :param val_loader: DataLoader for validation data

        :return:

        """

        self.model = model.to(self.device)

        self.opt = self.setup_opt_fn(self.model)

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.n_epochs):

            train_info = self._train_epoch(train_loader)

            val_info = self._val_epoch(val_loader)

            self.history["train_loss"].extend(train_info["train_loss"])

            self.history["val_loss"].append(val_info["loss"])

            self.history["val_acc"].append(val_info["acc"])

            print(np.mean(train_info["train_loss"]),   val_info["loss"],  val_info["acc"])

        return self.model.eval()



    def _train_epoch(self, train_loader):

        self.model.train()

        losses = []

        if self.verbose:

            train_loader = tqdm(train_loader)

        for batch in train_loader:

            self.model.zero_grad()

            texts, labels = batch

            logits = self.model.forward(texts.to(self.device))

            loss = self.loss_fn(logits, labels.to(self.device))

            loss.backward()

            self.opt.step()

            loss_val = loss.item()

            if self.verbose:

                train_loader.set_description(f"Loss={loss_val:.3}")

            losses.append(loss_val)

        return {"train_loss": losses}



    def _val_epoch(self, val_loader):

        self.model.eval()

        all_logits = []

        all_labels = []

        if self.verbose:

            val_loader = tqdm(val_loader)

        with torch.no_grad():

            for batch in val_loader:

                texts, labels = batch

                logits = self.model.forward(texts.to(self.device))

                all_logits.append(logits)

                all_labels.append(labels)

        all_labels = torch.cat(all_labels).to(self.device)

        all_logits = torch.cat(all_logits)

        loss = CrossEntropyLoss()(all_logits, all_labels).item()

        acc = (all_logits.argmax(1) == all_labels).float().mean().item()

        if self.verbose:

            val_loader.set_description(f"Loss={loss:.3}; Acc:{acc:.3}")

        return {"acc": acc, "loss": loss}



    def predict(self, test_loader):

        if self.model is None:

            raise RuntimeError("You should train the model first")

        self.model.eval()

        predictions = []

        with torch.no_grad():

            for batch in test_loader:

                texts, labels = batch

                logits = self.model.forward(texts.to(self.device))

                predictions.extend(logits.argmax(1).tolist())

        return asarray(predictions)



    def save(self, path: str):

        if self.model is None:

            raise RuntimeError("You should train the model first")

        checkpoint = {"config": self.model.config,

                      "trainer_config": self.config,

                      "vocab": self.model.vocab,

                      "emb_matrix": self.model.emb_matrix,

                      "state_dict": self.model.state_dict()}

        torch.save(checkpoint, path)



    @classmethod

    def load(cls, path: str):

        ckpt = torch.load(path)

        keys = ["config", "trainer_config", "vocab", "emb_matrix", "state_dict"]

        for key in keys:

            if key not in ckpt:

                raise RuntimeError(f"Missing key {key} in checkpoint")

        new_model = RecurrentClassifier(ckpt["config"], ckpt["vocab"], ckpt["emb_matrix"])

        new_model.load_state_dict(ckpt["state_dict"])

        new_trainer = cls(ckpt["trainer_config"])

        new_trainer.model = new_model

        new_trainer.model.to(new_trainer.device)

        return new_trainer

%load_ext autoreload

%autoreload 2

import os

import csv

from random import seed

from pathlib import Path

from itertools import chain

import torch

from tqdm import tqdm

from IPython.display import HTML, display

import gensim.downloader as api

from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pack_sequence

import pickle

#from model import prepare_emb_matrix, RecurrentClassifier

#from trainer import Trainer
data = read_data()

seed(4)



# change this at your own risk 

ratings_to_show = (0, 1, 2)

display(HTML(show_example(data, ratings=ratings_to_show)))       
# Press shift-tab to check docstrings

tok = Tokenizer()

tok_texts = [tok.tokenize(t) for t in chain(*data.values())]

vocab = Vocab(tok_texts, max_vocab_size=40000)
train_texts, train_labels, val_texts, val_labels = train_test_split(data, train_frac=0.7)

train_dataset = TextDataset([tok.tokenize(t) for t in train_texts], train_labels, vocab)

val_dataset = TextDataset([tok.tokenize(t) for t in val_texts], val_labels, vocab)
# store embeddings in current directory

os.environ["GENSIM_DATA_DIR"] = str(Path.cwd())

# will download embeddings or load them from disk

gensim_model = api.load("glove-wiki-gigaword-300")

emb_matrix = prepare_emb_matrix(gensim_model, vocab)
config = {

    "freeze":False,

    "cell_type": "GRU",

    "cell_dropout":  0.036450680527852976,

    "num_layers": 2,

    "hidden_size":  794,

    "out_activation": "elu",

    "bidirectional": True,

    "out_dropout": 0.03363077941266784,

    "out_sizes": [200],

}







trainer_config = {

    "lr": 0.00011761138147379482,

    "n_epochs": 15,

    "weight_decay": 5.074698104035439e-06,

    "batch_size": 1024,

    "device": "cuda" if torch.cuda.is_available() else "cpu"

}

clf_model = RecurrentClassifier(config, vocab, emb_matrix)
train_dataloader = DataLoader(train_dataset, 

                              batch_size=trainer_config["batch_size"],

                              shuffle=True,

                              num_workers=0,

                              collate_fn=train_dataset.collate_fn)

val_dataloader = DataLoader(val_dataset, 

                            batch_size=trainer_config["batch_size"],

                            shuffle=False,

                            num_workers=0,

                            collate_fn=val_dataset.collate_fn)

t = Trainer(trainer_config)

t.fit(clf_model, train_dataloader, val_dataloader)
t.save("baseline_model.ckpt")
t = Trainer.load("baseline_model.ckpt")
def predict_toxicity(model, comment):

    tok_text = tok.tokenize(comment)

    indexed_text = torch.tensor(vocab.vectorize(tok_text)).to(t.device)

    rating = model(pack_sequence([indexed_text])).argmax().item()

    print(f"Toxicity rating for \"{comment}\" is: {rating}") 



predict_toxicity(t.model, "Please sir do not delete my edits")

predict_toxicity(t.model, "They are nazi pal, forget it")

predict_toxicity(t.model, "You suck")
from optuna import create_study

from pprint import pprint





BEST_ACC = 0.0



def objective(trial):

    global BEST_ACC

    

    n_hidden_layers = trial.suggest_int("n_hidden_layers", 0, 3)

    hidden_layer_size = trial.suggest_int("hidden_layer_size", 10, 1000)

    

    config = {

        "freeze": True,

        "cell_type": trial.suggest_categorical("cell_type", ["GRU"]),

        "cell_dropout": trial.suggest_loguniform("cell_dropout", 1e-9, 0.9),

        "num_layers": trial.suggest_int("num_layers", 1, 3),

        "hidden_size": trial.suggest_int("hidden_size", 10, 800),

        "out_activation": trial.suggest_categorical("out_activation", 

                                                    ["sigmoid", "tanh", "relu", "elu"]),

        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),

        "out_dropout": trial.suggest_loguniform("out_dropout", 1e-9, 0.9),

        "out_sizes": [hidden_layer_size] * n_hidden_layers,

    }



    trainer_config = {

        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),

        "n_epochs": 10,

        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-9, 1e-1),

        "batch_size": 128,

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        "verbose": False,

    }

    

    pprint({**config, **trainer_config})

        

    clf_model = RecurrentClassifier(config, vocab, emb_matrix)

    t = Trainer(trainer_config)

    t.fit(clf_model, train_dataloader, val_dataloader)

    val_acc =  t.history["val_acc"][-1]

    if val_acc > BEST_ACC:

        BEST_ACC = val_acc

        t.save("optuna_model.ckpt")

    return val_acc
# study = create_study(direction="maximize")

# # you can set more trials

# study.optimize(objective, n_trials=20)