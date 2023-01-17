# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os



import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.optim import Adam

from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import (

    Dataset,

    DataLoader,

    random_split

)



from gensim.models import Word2Vec

from sklearn.model_selection import (

    train_test_split,

    KFold

)



import pandas as pd

import seaborn as sns

    
df = pd.read_csv("/kaggle/input/a607c02890a25f1/train.csv")
def build_vocab(corpus: list, special_tokens: list=None):

    stoi = dict()

    

    if special_tokens is not None:

        for s_tok in special_tokens:

            stoi[s_tok] = len(stoi)

        

    if type(corpus[0]) == list:

        for tokens in corpus:

            for tok in tokens:

                if tok not in stoi:

                    stoi[tok] = len(stoi)

                    

    elif type(corpus[0]) == str:

        for sentence in corpus:

            for tok in sentence.split():

                if tok not in stoi:

                    stoi[tok] = len(stoi)

                    

    else:

        raise ValueError("unknown data type in corpus: %s" % type(corpus[0]))

        

    itos = {v:k for k, v in stoi.items()}

    

    return itos, stoi
train_df, val_df = train_test_split(df, test_size=0.10, shuffle=True)



special_tokens = ["<pad>", "<unk>"]

itos, stoi = build_vocab(

    corpus=df["sentence1"].tolist() + df["sentence2"].tolist(),

    special_tokens=special_tokens

)



print("Vocab Size:", len(itos))
class STSDataset(Dataset):

    def __init__(

        self,

        sent_a: list,

        sent_b: list,

        labels: list=None,

        is_train: bool=True

    ):

        assert len(sent_a) == len(sent_b)

        if is_train: assert len(sent_a) == len(labels)

        

        self.is_train = is_train

        

        self.sent_a = sent_a

        self.sent_b = sent_b

        self.labels = labels

        

        self.x_a = None

        self.x_b = None

        self.y = None

        

        

    def vectorize(self, corpus: list, stoi: dict):

        vectors = list()

        for sent in corpus:

            vec = list()

            for tok in sent.split():

                vec.append(stoi.get(tok, stoi["<unk>"]))

                    

            vectors.append(torch.tensor(vec, dtype=torch.long))

            

        return vectors

        

        

    def build(self, stoi: dict):

        self.x_a = pad_sequence(

            self.vectorize(self.sent_a, stoi),

            batch_first=True,

            padding_value=stoi["<pad>"]

        )

        self.x_b = pad_sequence(

            self.vectorize(self.sent_b, stoi),

            batch_first=True,

            padding_value=stoi["<pad>"]

        )

        

        if self.is_train:

            self.y = torch.tensor(self.labels, dtype=torch.long)

        

        

    def __len__(self):

        return len(self.sent_a)

        

        

    def __getitem__(self, idx):

        if self.x_a == None:

            raise Exception("should call <STSDataset>.build() first")

            

        if self.is_train:

            return self.x_a[idx], self.x_b[idx], self.y[idx]

        

        else:

            return self.x_a[idx], self.x_b[idx]
ds_train = STSDataset(

    train_df["sentence1"].tolist(),

    train_df["sentence2"].tolist(),

    train_df["label"].tolist()

)

ds_train.build(stoi)



ds_val = STSDataset(

    val_df["sentence1"].tolist(),

    val_df["sentence2"].tolist(),

    val_df["label"].tolist()

)

ds_val.build(stoi)
BATCH_SIZE = 1024



SAVE_PATH = "/kaggle/working/models"



os.makedirs(SAVE_PATH, exist_ok=True)



train_loader = DataLoader(dataset=ds_train, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(dataset=ds_val, batch_size=BATCH_SIZE, shuffle=False)
class SentenceEncoder(nn.Module):

    def __init__(

        self,

        model_type: str,

        config: dict,

        w2v: bool=False,

        sentences: list=None,

        emb_trainable: bool=True

    ):

        super(SentenceEncoder, self).__init__()

        

        self.model_type = model_type

        

        self.num_embeddings = config["num_embeddings"]

        self.embedding_dim  = config["embedding_dim"]

        self.hidden_size    = config["hidden_size"]

        self.use_attention  = config["use_attention"]

        

        self.embedding = nn.Embedding(

            num_embeddings=self.num_embeddings,

            embedding_dim=self.embedding_dim,

            padding_idx=0

        )

        nn.init.xavier_uniform_(self.embedding.weight)

        

        if w2v:

            w2v = Word2Vec(min_count=1, size=self.embedding_dim)

            w2v.build_vocab(sentences)

            w2v.train(sentences, total_examples=len(sentences), epochs=10)



            emb_weight = build_pretrained_embedding(w2v.wv, itos, len(special_tokens))

            self.embedding.load_state_dict({'weight': torch.tensor(emb_weight)})

            

            if not emb_trainable:

                self.embedding.weight.requires_grad = False

            

        if self.use_attention:

            self.n_heads  = config["n_heads"]

            self.attention = nn.MultiheadAttention(self.embedding_dim, self.n_heads)

        

        if self.model_type == "conv":

            self.kernel_list    = config["kernel_list"]

            self.stride         = config["stride"]

            

            self.layers = nn.ModuleList()

            for kernel_size in self.kernel_list:

                conv_layer = nn.Conv2d(

                    in_channels=1,

                    out_channels=self.hidden_size,

                    kernel_size=(kernel_size, self.embedding_dim),

                    stride=self.stride

                )

                nn.init.xavier_uniform_(conv_layer.weight)

                self.layers.append(conv_layer)

                

        elif self.model_type == "rnn":

            self.rnn_type       = config["rnn_type"]

            self.input_size     = config["input_size"]

            self.num_layers     = config["num_layers"]

            self.bidirectional  = config["bidirectional"]

            

            if self.rnn_type.lower() == "gru":

                self.rnn = nn.GRU(

                    input_size=self.input_size,

                    hidden_size=self.hidden_size,

                    num_layers=self.num_layers,

                    bidirectional=self.bidirectional

                )

                

            elif self.rnn_type.lower() == "lstm":

                self.rnn = nn.LSTM(

                    input_size=self.input_size,

                    hidden_size=self.hidden_size,

                    num_layers=self.num_layers,

                    bidirectional=self.bidirectional

                )

                

        elif self.model_type == "dan":

            self.fc1 = nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_size)

            self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)

            self.fc3 = nn.Linear(in_features=self.hidden_size, out_features=self.embedding_dim)

            self.relu = nn.LeakyReLU(0.2)

            

            nn.init.xavier_uniform_(self.fc1.weight)

            nn.init.xavier_uniform_(self.fc2.weight)

            nn.init.xavier_uniform_(self.fc3.weight)

            

        elif self.model_type == "transformer":

            self.num_layers     = config["num_layers"]

            self.n_heads        = config["n_heads"]

            

            self.encoder_layer = nn.TransformerEncoderLayer(

                d_model=self.hidden_size,

                nhead=self.n_heads,

                dim_feedforward=self.hidden_size*4

            )

            

            self.encoder = nn.TransformerEncoder(self.encoder_layer, self.num_layers)

                

                

    def generate_attention_mask(self, batch):

        mask = batch.clone()

        

        mask[mask>=1] = 1

        mask[mask==0] = 0

        

        return ~mask.type(torch.bool)

            

                

    def forward(self, x):

        attn = None

        mask = self.generate_attention_mask(x)

        x = self.embedding(x)

        

        if self.use_attention:

            x = x.transpose(0, 1)

            x, attn = self.attention(x, x, x, mask)

            x = x.transpose(0, 1)

        

        if self.model_type == "conv":

            x = torch.unsqueeze(x, dim=1)

            

            features = list()

            for conv in self.layers:

                out = conv(x).squeeze(dim=-1)

                out = nn.MaxPool1d(kernel_size=out.shape[-1])(out)



                features.append(out.squeeze(dim=-1))



            return torch.cat(features, dim=-1), None

            

        elif self.model_type == "rnn":

            out, _ = self.rnn(x)

            out = nn.MaxPool1d(out.shape[1])(out.transpose(-1, -2))



            return out.squeeze(), attn

        

        elif self.model_type == "dan":

            length = torch.sum(~mask, dim=-1)

            out = nn.AvgPool1d(x.shape[1])(x.transpose(-1, -2)).squeeze()

            out = torch.mul(out.transpose(0, 1), length).transpose(0, 1)

            

            out = self.fc1(out)

            out = self.relu(out)

            out = self.fc2(out)

            out = self.relu(out)

            out = self.fc3(out)



            return out, attn

        

        elif self.model_type == "transformer":

            out = x.transpose(0, 1)

            out = self.encoder(out, src_key_padding_mask=mask)

            out = out.transpose(0, 1)

            

            out = nn.AvgPool1d(out.shape[1])(out.transpose(-1, -2))



            return out.squeeze(), attn

        

        

        

class STSModel(nn.Module):

    def __init__(

        self,

        model_type: str,

        config: dict,

        w2v: bool=False,

        sentences: list=None,

        emb_trainable: bool=False

    ):

        super(STSModel, self).__init__()

        

        self.model_type = model_type

        

        self.conv_model_names = ["conv", "cnn", "convolution"]

        self.rnn_model_names  = ["rnn", "recurrent"]

        self.dan_model_names  = ["dan"]

        

        self.dropout = config["dropout"]

        self.hidden_size = config["hidden_size"]

        

        if self.model_type.lower() in self.conv_model_names:

            self.encoder = SentenceEncoder("conv", config, w2v, sentences, emb_trainable)

            fc_size = self.hidden_size * len(config["kernel_list"])

            

        elif self.model_type.lower() in self.rnn_model_names:

            self.encoder = SentenceEncoder("rnn", config, w2v, sentences, emb_trainable)

            if config["bidirectional"] == True:

                fc_size = self.hidden_size * 2

            else:

                fc_size = self.hidden_size

            

        elif self.model_type.lower() in self.dan_model_names:

            self.encoder = SentenceEncoder("dan", config, w2v, sentences, emb_trainable)

            fc_size = config["embedding_dim"]

            

        elif self.model_type.lower() == "transformer":

            self.encoder = SentenceEncoder("transformer", config, w2v, sentences, emb_trainable)

            fc_size = self.hidden_size

                

        else:

            raise ValueError("Unknown Model Type: %s" % model_type)

            

        self.v_linear = nn.Linear(

            in_features=fc_size,

            out_features=fc_size

        )

        self.relu = nn.LeakyReLU(0.2)

        

        fc_size *= 4

        

        self.fc = nn.Linear(

            in_features=fc_size,

            out_features=2

        )

        

        self.do = nn.Dropout(self.dropout)

        

        nn.init.xavier_uniform_(self.v_linear.weight)

        nn.init.xavier_uniform_(self.fc.weight)

        

        

    def forward(self, a, b):

        u, attn_a = self.encoder(a)

        u = self.do(u)

        

        v, attn_b = self.encoder(b)

        v = self.do(v)

        v = self.v_linear(v)

        v = self.relu(v)

        

        out = torch.cat([u, v, torch.abs(u - v), torch.mul(u, v)], dim=-1)

        

        return self.fc(out), attn_a, attn_b
# W2V_EPOCHS = 20

# word2vec = False

# emb_weight = None

# emb_trainable = True



# d_emb = 256



# if word2vec:

    

#     def build_pretrained_embedding(wv, itos, s_tok_size):

#         V, E = wv.vectors.shape

        

#         weight = np.zeros((V + s_tok_size, E))

#         for idx in range(s_tok_size, V + s_tok_size):

#             weight[idx] = wv.vectors[wv.vocab[itos[idx]].index]



#         return weight

    

    

#     sen_1 = df["sentence1"].tolist()

#     sen_2 = df["sentence2"].tolist()

#     labels = df["label"].tolist()

    

#     sentences = list()

#     for idx in range(len(df)):

#         if labels[idx] == 1:

#             sentences.append(sen_1[idx].split() + sen_2[idx].split())

            

#         else:

#             sentences.append(sen_1[idx].split())

#             sentences.append(sen_2[idx].split())

    

# #     all_tokens = list()



# #     for sentence in (sentences):

# #         tokens = list()



# #         for token in sentence.split():

# #             tokens.append(token)



# #         all_tokens.append(tokens)



#     w2v = Word2Vec(min_count=1, size=d_emb)

#     w2v.build_vocab(sentences)

#     w2v.train(sentences, total_examples=len(sentences), epochs=W2V_EPOCHS)

    

#     emb_weight = build_pretrained_embedding(w2v.wv, itos, len(special_tokens))
def build_pretrained_embedding(wv, itos, s_tok_size):

    V, E = wv.vectors.shape



    weight = np.zeros((V + s_tok_size, E))

    for idx in range(s_tok_size, V + s_tok_size):

        weight[idx] = wv.vectors[wv.vocab[itos[idx]].index]



    return weight

    

sen_1 = df["sentence1"].tolist()

sen_2 = df["sentence2"].tolist()

labels = df["label"].tolist()



sentences = list()

for idx in range(len(df)):

    if labels[idx] == 1:

        sentences.append(sen_1[idx].split() + sen_2[idx].split())



    else:

        sentences.append(sen_1[idx].split())

        sentences.append(sen_2[idx].split())
def calculate_accuracy(probs, golds):

    bsz = probs.shape[0]

    

    score = torch.sum(torch.argmax(probs, dim=-1).squeeze() == golds).item()

    

    return score / bsz



EPOCHS = 20

vocab_size = len(itos)

best_scores = [ 0.0, 0.0, 0.0, 0.0 ]



for _ in range(5):

    models = [

        STSModel(

            "conv",

            {

                "num_embeddings": vocab_size,

                "embedding_dim": 128,

                "hidden_size": 368,

                "kernel_list": [2, 2, 2, 2],

                "stride": 1,

                "use_attention": False,

                "n_heads": 0,

                "dropout": 0.3

            },

            w2v=True,

            sentences=sentences,

            emb_trainable=True

        ),

        

        STSModel(

            "rnn",

            {

                "rnn_type": "lstm",

                "num_embeddings": vocab_size,

                "embedding_dim": 128,

                "input_size": 128,

                "hidden_size": 256,

                "num_layers": 1,

                "bidirectional": True,

                "use_attention": False,

                "n_heads": 0,

                "dropout": 0.3

            },

            w2v=True,

            sentences=sentences,

            emb_trainable=False

        ),

        

        STSModel(

            "dan",

            {

                "num_embeddings": vocab_size,

                "embedding_dim": 64,

                "hidden_size": 512,

                "use_attention": False,

                "n_heads": 0,

                "dropout": 0.3

            },

            w2v=True,

            sentences=sentences,

            emb_trainable=False

        ),

        

        STSModel(

            "transformer",

            {

                "num_embeddings": vocab_size,

                "embedding_dim": 128,

                "hidden_size": 128,

                "num_layers": 1,

                "use_attention": False,

                "n_heads": 2,

                "dropout": 0.3

            },

            w2v=True,

            sentences=sentences,

            emb_trainable=True

        )

    ]

    

    optimizers = list()

    for model in models:

        optimizers.append(Adam(params=model.parameters(), lr=0.001))

        

    losses = [

        nn.CrossEntropyLoss(),

        nn.CrossEntropyLoss(),

        nn.CrossEntropyLoss(),

        nn.CrossEntropyLoss()

    ]



    if torch.cuda.is_available():

        for model in models:

            model = model.cuda()

        

    print("Define New Models")



    ensemble_size = len(models)



    for idx, (model, loss_fn, optim) in enumerate(zip(models, losses, optimizers)):

        print("%d Model Training, Total %d" % (idx+1, ensemble_size))



        for epoch in range(EPOCHS):

            total_loss = 0.0

            total_accu = 0.0

            model.train()



            for batch_idx, batch in enumerate(train_loader):

                sen_a, sen_b, labels = batch



                optim.zero_grad()



                if torch.cuda.is_available():

                    sen_a = sen_a.cuda()

                    sen_b = sen_b.cuda()

                    labels = labels.cuda()



                logits, _, _ = model(sen_a, sen_b)

                loss = loss_fn(logits, labels)

                loss.backward()

                optim.step()



                total_loss += loss.item()



                accu = calculate_accuracy(logits, labels)

                total_accu += accu



#                 print("\r| Epoch: %2d | Process: %5d / %5d |-|  Loss: %.4lf | Accu: %.4lf |" % \

#                      (epoch+1, batch_idx+1, train_loader.__len__(), total_loss / (batch_idx+1), total_accu / (batch_idx+1)), end="")



#             print("")



            test_accu = 0.0

            score = 0.0

            model.eval()



            for batch_idx, batch in enumerate(val_loader):

                sen_a, sen_b, labels = batch



                if torch.cuda.is_available():

                    sen_a = sen_a.cuda()

                    sen_b = sen_b.cuda()

                    labels = labels.cuda()



                with torch.no_grad():

                    logits, _, _ = model(sen_a, sen_b)



                accu = calculate_accuracy(logits, labels)

                test_accu += accu



                score = test_accu / (batch_idx+1)



                print("\rValidation Accu: %.4lf" % score, end="")



            if best_scores[idx] < score:

                best_scores[idx] = score

                torch.save(model.state_dict(), (SAVE_PATH + ("/sts_%s_model.pt" % model.model_type)))



                print("\nBest Model! Save at %s" % SAVE_PATH)



            print("\n")
def soft_voting(logits):

    probs = list()

    for logit in logits:

        probs.append(F.softmax(logit, dim=-1))

        

    prob = probs[0]

    for p in probs[1:]:

        prob = torch.mul(prob, p)

        

    return torch.argmax(prob, dim=-1).squeeze()

    



def soft_voting_accuracy(logits, golds=None):

    bsz = logits[0].shape[0]

    

    score = torch.sum(soft_voting(logits) == golds).item()

    

    return score / bsz





test_accu = 0.0

score = 0.0



skip_list = [

#     "conv",

#     "rnn",

#     "dan",

#     "transformer"

]



for batch_idx, batch in enumerate(val_loader):

    sen_a, sen_b, labels = batch



    if torch.cuda.is_available():

        sen_a = sen_a.cuda()

        sen_b = sen_b.cuda()

        labels = labels.cuda()



    with torch.no_grad():

        logits = list()



        for model in models:

            model.load_state_dict(torch.load(SAVE_PATH + ("/sts_%s_model.pt" % model.model_type)))

            model.eval()

            if model.model_type in skip_list: continue

                

            logit, _, _ = model(sen_a, sen_b)

            logits.append(logit)



    accu = soft_voting_accuracy(logits, labels)

    test_accu += accu



    score = test_accu / (batch_idx+1)



    print("\rValidation Accu: %.4lf" % score, end="")
def ensemble_evaluation(models, stoi):

    for model in models:

        if model.model_type in skip_list: continue

        model.load_state_dict(torch.load(SAVE_PATH + ("/sts_%s_model.pt" % model.model_type)))

        if torch.cuda.is_available():

            model = model.cuda()

            model.eval()

        

    test_df = pd.read_csv("/kaggle/input/a607c02890a25f1/test.csv")

    ids = test_df["id"].tolist()



    sts_test = STSDataset(test_df["sentence1"], test_df["sentence2"], None, is_train=False)

    sts_test.build(stoi)

    

    test_loader = DataLoader(dataset=sts_test, batch_size=BATCH_SIZE, shuffle=False)

    

    res_labels = list()

    for batch_idx, batch in enumerate(test_loader):

        sen_a, sen_b = batch

        

        if torch.cuda.is_available():

            sen_a = sen_a.cuda()

            sen_b = sen_b.cuda()



        with torch.no_grad():

            logits = list()

            

            for model in models:

                logit, _, _ = model(sen_a, sen_b)

                logits.append(logit)



        labels = soft_voting(logits).cpu().detach().numpy()

        

        res_labels.extend(labels)



    assert len(ids) == len(res_labels)

    

    results = {

        "id": ids,

        "label": res_labels

    }

    res_csv = pd.DataFrame.from_dict(results)

    res_csv.to_csv("/kaggle/working/submission.csv", index=False)

    

ensemble_evaluation(models, stoi)
# # Base Config

# EPOCHS = 20



# d_hidden = 256

# vocab_size = len(itos)

# dropout = 0.3

# use_attention = False

# n_heads = 1

# model_type = "ensemble"





# if model_type is "conv":   # Conv Config

    

#     kernel_list = [2, 3, 4, 5]

#     stride = 1



#     config = {

#         "num_embeddings": vocab_size,

#         "embedding_dim": d_emb,

#         "hidden_size": d_hidden,

#         "kernel_list": kernel_list,

#         "stride": stride,

#         "use_attention": use_attention,

#         "n_heads": n_heads,

#         "dropout": dropout

#     }

#     model = STSModel(

#         model_type,

#         config,

#         w2v=True,

#         sentences=sentences,

#         emb_trainable=True

#     )

#     criterion = nn.CrossEntropyLoss()

#     optimizer = Adam(params=model.parameters(), lr=0.001)



#     if torch.cuda.is_available():

#         model = model.cuda()

        

#     print(model)

    

# elif model_type is "rnn":   # RNN Config

    

#     num_layers = 1

#     bidirectional = True



#     config = {

#         "rnn_type": "lstm",

#         "num_embeddings": vocab_size,

#         "embedding_dim": d_emb,

#         "input_size": d_emb,

#         "hidden_size": d_hidden,

#         "num_layers": num_layers,

#         "bidirectional": bidirectional,

#         "use_attention": use_attention,

#         "n_heads": n_heads,

#         "dropout": dropout

#     }

    

#     model = STSModel(

#         model_type,

#         config,

#         w2v=True,

#         sentences=sentences,

#         emb_trainable=True

#     )

#     criterion = nn.CrossEntropyLoss()

#     optimizer = Adam(params=model.parameters(), lr=0.001)



#     if torch.cuda.is_available():

#         model = model.cuda()

        

#     print(model)

    

# elif model_type is "dan":   # DAN Config

    

#     config = {

#         "num_embeddings": vocab_size,

#         "embedding_dim": d_emb,

#         "hidden_size": d_hidden,

#         "use_attention": use_attention,

#         "n_heads": n_heads,

#         "dropout": dropout

#     }

    

#     model = STSModel(

#         model_type,

#         config,

#         w2v=True,

#         sentences=sentences,

#         emb_trainable=True

#     )

#     criterion = nn.CrossEntropyLoss()

#     optimizer = Adam(params=model.parameters(), lr=0.001)



#     if torch.cuda.is_available():

#         model = model.cuda()

        

#     print(model)

    

# elif model_type is "transformer":   # DAN Config

    

#     num_layers = 1

    

#     config = {

#         "num_embeddings": vocab_size,

#         "embedding_dim": d_emb,

#         "hidden_size": d_emb,

#         "num_layers": num_layers,

#         "use_attention": use_attention,

#         "n_heads": n_heads,

#         "dropout": dropout

#     }

    

#     model = STSModel(

#         model_type,

#         config,

#         w2v=True,

#         sentences=sentences,

#         emb_trainable=True

#     )

#     criterion = nn.CrossEntropyLoss()

#     optimizer = Adam(params=model.parameters(), lr=0.001)



#     if torch.cuda.is_available():

#         model = model.cuda()

        

#     print(model)

    

# elif model_type is "ensemble":

    

#     models = [

#         STSModel(

#             "conv",

#             {

#                 "num_embeddings": vocab_size,

#                 "embedding_dim": 128,

#                 "hidden_size": 368,

#                 "kernel_list": [2, 2, 2, 2],

#                 "stride": 1,

#                 "use_attention": False,

#                 "n_heads": 0,

#                 "dropout": 0.3

#             },

#             w2v=True,

#             sentences=sentences,

#             emb_trainable=True

#         ),

        

#         STSModel(

#             "rnn",

#             {

#                 "rnn_type": "lstm",

#                 "num_embeddings": vocab_size,

#                 "embedding_dim": 128,

#                 "input_size": 128,

#                 "hidden_size": 256,

#                 "num_layers": 1,

#                 "bidirectional": True,

#                 "use_attention": False,

#                 "n_heads": 0,

#                 "dropout": 0.3

#             },

#             w2v=True,

#             sentences=sentences,

#             emb_trainable=False

#         ),

        

#         STSModel(

#             "dan",

#             {

#                 "num_embeddings": vocab_size,

#                 "embedding_dim": 64,  # Fix

#                 "hidden_size": 512,

#                 "use_attention": False,

#                 "n_heads": 0,

#                 "dropout": 0.3

#             },

#             w2v=True,

#             sentences=sentences,

#             emb_trainable=False  # Fix

#         ),

        

#         STSModel(

#             "transformer",

#             {

#                 "num_embeddings": vocab_size,

#                 "embedding_dim": 128,

#                 "hidden_size": 128,

#                 "num_layers": 1,

#                 "use_attention": False,

#                 "n_heads": 2,

#                 "dropout": 0.3

#             },

#             w2v=True,

#             sentences=sentences,

#             emb_trainable=True

#         )

#     ]

    

#     optimizers = list()

#     for model in models:

#         optimizers.append(Adam(params=model.parameters(), lr=0.001))

        

#     losses = [

#         nn.CrossEntropyLoss(),

#         nn.CrossEntropyLoss(),

#         nn.CrossEntropyLoss(),

#         nn.CrossEntropyLoss()

#     ]



#     if torch.cuda.is_available():

#         for model in models:

#             model = model.cuda()

        

#     print(models)
# n_splits = 5

# BATCH_SIZE = 256

# EPOCHS = 10

# ensemble_size = len(models)



# kfold = KFold(n_splits=n_splits, shuffle=True)
# def calculate_accuracy(probs, golds):

#     bsz = probs.shape[0]

    

#     score = torch.sum(torch.argmax(probs, dim=-1).squeeze() == golds).item()

    

#     return score / bsz





# def soft_voting_accuracy(logits, golds=None, get_preds=False):

#     bsz = logits[0].shape[0]

    

#     probs = list()

#     for logit in logits:

#         probs.append(F.softmax(logit, dim=-1))

        

#     prob = probs[0]

#     for p in probs[1:]:

#         prob = torch.mul(prob, p)

    

#     if get_preds:

#         return torch.argmax(prob, dim=-1).squeeze()

    

#     score = torch.sum(torch.argmax(prob, dim=-1).squeeze() == golds).item()

    

#     return score / bsz





# for idx, (model, loss_fn, optim) in enumerate(zip(models, losses, optimizers)):

#     print("%d Model Training, Total %d" % (idx+1, ensemble_size))

    

#     for epoch in range(EPOCHS):

#         best_score = 0.0

#         total_score = 0.0

        

#         for val_iter, (train_index, test_index) in enumerate(kfold.split(df)):  

#             sen_a_train, sen_a_val = \

#             df.iloc[train_index]["sentence1"], df.iloc[test_index]["sentence1"]



#             sen_b_train, sen_b_val = \

#             df.iloc[train_index]["sentence2"], df.iloc[test_index]["sentence2"]



#             labels_train, labels_val = \

#             df.iloc[train_index]["label"], df.iloc[test_index]["label"]



#             ds_train = STSDataset(

#                 sen_a_train.tolist(),

#                 sen_b_train.tolist(),

#                 labels_train.tolist()

#             )

#             ds_train.build(stoi)



#             ds_val = STSDataset(

#                 sen_a_val.tolist(),

#                 sen_b_val.tolist(),

#                 labels_val.tolist()

#             )

#             ds_val.build(stoi)



#             train_loader = DataLoader(dataset=ds_train, batch_size=BATCH_SIZE, shuffle=True)

#             val_loader = DataLoader(dataset=ds_val, batch_size=BATCH_SIZE, shuffle=False)

            

#             total_loss = 0.0

#             total_accu = 0.0



#             for batch_idx, batch in enumerate(train_loader):

#                 sen_a, sen_b, labels = batch



#                 optim.zero_grad()



#                 if torch.cuda.is_available():

#                     sen_a = sen_a.cuda()

#                     sen_b = sen_b.cuda()

#                     labels = labels.cuda()



#                 logits, _, _ = model(sen_a, sen_b)

#                 loss = loss_fn(logits, labels)

#                 loss.backward()

#                 optim.step()



#                 total_loss += loss.item()



#                 accu = calculate_accuracy(logits, labels)

#                 total_accu += accu



#                 print("\r| Epoch: %2d | Process: %5d / %5d |-|  Loss: %.4lf | Accu: %.4lf |" % \

#                      (epoch+1, batch_idx+1, train_loader.__len__(), total_loss / (batch_idx+1), total_accu / (batch_idx+1)), end="")



#             print("")



#             test_accu = 0.0

#             score = 0.0



#             for batch_idx, batch in enumerate(val_loader):

#                 sen_a, sen_b, labels = batch



#                 if torch.cuda.is_available():

#                     sen_a = sen_a.cuda()

#                     sen_b = sen_b.cuda()

#                     labels = labels.cuda()



#                 with torch.no_grad():

#                     logits, _, _ = model(sen_a, sen_b)



#                 accu = calculate_accuracy(logits, labels)

#                 test_accu += accu



#                 score = test_accu / (batch_idx+1)



#             print("Validation %d Accu: %.4lf" % (val_iter+1, score))

                

#             total_score += score / n_splits



#         if best_score < total_score:

#             best_score = total_score

#             print("Total Validation Accu: %.4lf" % best_score)

#             torch.save(model.state_dict(), (SAVE_PATH + ("/sts_%s_model.pt" % model.model_type)))



#             print("\nBest Model! Save at %s" % SAVE_PATH)



#         print("\n")
# def calculate_accuracy(probs, golds):

#     bsz = probs.shape[0]

    

#     score = torch.sum(torch.argmax(probs, dim=-1).squeeze() == golds).item()

    

#     return score / bsz





# if model_type == "ensemble":

#     ensemble_size = len(models)

    

#     for idx, (model, loss_fn, optim) in enumerate(zip(models, losses, optimizers)):

#         best_score = 0.0

#         print("%d Model Training, Total %d" % (idx+1, ensemble_size))



#         for epoch in range(EPOCHS):

#             total_loss = 0.0

#             total_accu = 0.0

#             model.train()



#             for batch_idx, batch in enumerate(train_loader):

#                 sen_a, sen_b, labels = batch



#                 optim.zero_grad()



#                 if torch.cuda.is_available():

#                     sen_a = sen_a.cuda()

#                     sen_b = sen_b.cuda()

#                     labels = labels.cuda()



#                 logits, _, _ = model(sen_a, sen_b)

#                 loss = loss_fn(logits, labels)

#                 loss.backward()

#                 optim.step()



#                 total_loss += loss.item()



#                 accu = calculate_accuracy(logits, labels)

#                 total_accu += accu



#                 print("\r| Epoch: %2d | Process: %5d / %5d |-|  Loss: %.4lf | Accu: %.4lf |" % \

#                      (epoch+1, batch_idx+1, train_loader.__len__(), total_loss / (batch_idx+1), total_accu / (batch_idx+1)), end="")



#             print("")



#             test_accu = 0.0

#             score = 0.0

#             model.eval()



#             for batch_idx, batch in enumerate(val_loader):

#                 sen_a, sen_b, labels = batch



#                 if torch.cuda.is_available():

#                     sen_a = sen_a.cuda()

#                     sen_b = sen_b.cuda()

#                     labels = labels.cuda()



#                 with torch.no_grad():

#                     logits, _, _ = model(sen_a, sen_b)



#                 accu = calculate_accuracy(logits, labels)

#                 test_accu += accu



#                 score = test_accu / (batch_idx+1)



#                 print("\rValidation Accu: %.4lf" % score, end="")



#             if best_score < score:

#                 best_score = score

#                 torch.save(model.state_dict(), (SAVE_PATH + ("/sts_%s_model.pt" % model.model_type)))



#                 print("\nBest Model! Save at %s" % SAVE_PATH)



#             print("\n")

    

# else:

    

#     best_score = 0.0

#     for epoch in range(EPOCHS):

#         total_loss = 0.0

#         total_accu = 0.0

#         model.train()



#         for batch_idx, batch in enumerate(train_loader):

#             sen_a, sen_b, labels = batch

#             optimizer.zero_grad()



#             if torch.cuda.is_available():

#                 sen_a = sen_a.cuda()

#                 sen_b = sen_b.cuda()

#                 labels = labels.cuda()



#             logits, attn_a, attn_b = model(sen_a, sen_b)

#             loss = criterion(logits, labels)



#             total_loss += loss.item()



#             loss.backward()

#             optimizer.step()



#             accu = calculate_accuracy(F.softmax(logits, dim=-1).clone(), labels)

#             total_accu += accu



#             print("\r| Epoch: %2d | Process: %5d / %5d |-| Loss: %.4lf | Accu: %.4lf |" % \

#                  (epoch+1, batch_idx+1, train_loader.__len__(),

#                   total_loss / (batch_idx+1), total_accu  / (batch_idx+1)), end="")



#         print("")



#         test_accu = 0.0

#         score = 0.0

#         model.eval()



#         for batch_idx, batch in enumerate(val_loader):

#             sen_a, sen_b, labels = batch



#             if torch.cuda.is_available():

#                 sen_a = sen_a.cuda()

#                 sen_b = sen_b.cuda()

#                 labels = labels.cuda()



#             with torch.no_grad():

#                 logits, attn_a, attn_b = model(sen_a, sen_b)



#             accu = calculate_accuracy(F.softmax(logits, dim=-1), labels)

#             test_accu += accu



#             score = test_accu / (batch_idx+1)



#             print("\rValidation Accu: %.4lf" % score, end="")



#         if best_score < score:

#             best_score = score

#             torch.save(model.state_dict(), (SAVE_PATH + "/sts_model.pt"))

#             print("\nBest Model! Save at %s" % SAVE_PATH)



#         print("\n")
# def plot_attention(attn):

#     sns.heatmap(attn.cpu().detach().numpy())

    

# plot_attention(attn_a[0])
# def evaluation(model, stoi):

#     test_df = pd.read_csv("/kaggle/input/a607c02890a25f1/test.csv")

#     ids = test_df["id"].tolist()



#     sts_test = STSDataset(test_df["sentence1"], test_df["sentence2"], None, is_train=False)

#     sts_test.build(stoi)

    

#     test_loader = DataLoader(dataset=sts_test, batch_size=BATCH_SIZE, shuffle=False)

    

#     if torch.cuda.is_available():

#         model = model.cuda()

        

#     res_labels = list()

#     for batch_idx, batch in enumerate(test_loader):

#         sen_a, sen_b = batch

        

#         if torch.cuda.is_available():

#             sen_a = sen_a.cuda()

#             sen_b = sen_b.cuda()



#         with torch.no_grad():

#             logits, _, _ = model(sen_a, sen_b)



#         labels = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.uint8)



#         res_labels.extend(labels)



#     assert len(ids) == len(res_labels)

    

#     results = {

#         "id": ids,

#         "label": res_labels

#     }

#     res_csv = pd.DataFrame.from_dict(results)

#     res_csv.to_csv("/kaggle/working/submission.csv", index=False)

    

    

# model.load_state_dict(torch.load(SAVE_PATH + "/sts_model.pt"))

# evaluation(model, stoi)