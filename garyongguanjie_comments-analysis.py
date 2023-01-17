import pandas as pd

import numpy as np

import transformers

import torch

from torch import nn

from transformers import DistilBertTokenizer, DistilBertModel

from sklearn import model_selection

from tqdm import tqdm
class Config:

    MAX_LEN=512

    TRAIN_BATCH_SIZE=16

    VALID_BATCH_SIZE=16

    TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    CSV_PATH = '../input/complete-tweet-sentiment-extraction-data/tweet_dataset.csv'

    EPOCHS = 5
class CommentModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.dropouts = nn.ModuleList([nn.Dropout(0.15) for _ in range(5)])

        self.l1 = nn.Linear(768,3) # for sentiment

        self.l2 = nn.Linear(768,13) # for emotion

    def forward(self,ids,mask):

        x = self.bert(input_ids=ids,attention_mask=mask)[0]

        for i,dropout in enumerate(self.dropouts):

            if i == 0:

                out_sum = dropout(x)

            else:

                out_sum += dropout(x)

        out = out_sum/len(self.dropouts)

        out = torch.mean(out,dim=1)

        sentiment = self.l1(out)

        emotion = self.l2(out)

        return sentiment,emotion
class TweetDataset:

    """

    Dataset which stores the tweets and returns them as processed features

    """

    def __init__(self, tweet,emotion_label,sentiment_label):

        self.tweet = tweet

        self.emotion_label = emotion_label

        self.sentiment_label = sentiment_label

        self.tokenizer = Config.TOKENIZER

        self.max_len = Config.MAX_LEN

    

    def __len__(self):

        return len(self.tweet)



    def __getitem__(self, item):

        tweet = str(self.tweet[item]).strip()

        tweet = " ".join(tweet.split())



        inputs = self.tokenizer.encode_plus(

            tweet,

            None,

            add_special_tokens=True,

            max_length=self.max_len,

            pad_to_max_length=True,

        )



        ids = inputs["input_ids"]

        mask = inputs["attention_mask"]



        return {

            'ids': torch.tensor(ids),

            'mask':torch.tensor(mask),

            'emotion_label':torch.tensor(self.emotion_label[item]),

            'sentiment_label':torch.tensor(self.sentiment_label[item])

        }
class AverageMeter:

    """

    Computes and stores the average and current value

    """

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count



class EarlyStopping:

    def __init__(self, patience=7, mode="max", delta=0.001):

        self.patience = patience

        self.counter = 0

        self.mode = mode

        self.best_score = None

        self.early_stop = False

        self.delta = delta

        if self.mode == "min":

            self.val_score = np.Inf

        else:

            self.val_score = -np.Inf



    def __call__(self, epoch_score, model, model_path):



        if self.mode == "min":

            score = -1.0 * epoch_score

        else:

            score = np.copy(epoch_score)



        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(epoch_score, model, model_path)

            self.counter = 0



    def save_checkpoint(self, epoch_score, model, model_path):

        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:

            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))

            torch.save(model.state_dict(), model_path)

        self.val_score = epoch_score
def loss_fn(pred_sentiment,sentiment,pred_emotion,emotion):

    loss_fct = nn.CrossEntropyLoss()

    loss1 = loss_fct(pred_sentiment,sentiment)

    loss2 = loss_fct(pred_emotion,emotion)

    return loss1 + loss2
def train_fn(data_loader, model, optimizer, device, scheduler=None):

    model.train()

    losses = AverageMeter()

    accuracy1 = AverageMeter()

    accuracy2 = AverageMeter()



    tk0 = tqdm(data_loader, total=len(data_loader))

    

    for bi, d in enumerate(tk0):



        ids = d["ids"]

        mask = d["mask"]

        sentiment = d["sentiment_label"]

        emotion = d["emotion_label"]



        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)

        sentiment = sentiment.to(device,dtype=torch.long)

        emotion = emotion.to(device,dtype=torch.long)



        model.zero_grad()

        pred_sentiment,pred_emotion= model(

            ids=ids,

            mask=mask,

        )

        loss = loss_fn(pred_sentiment,sentiment,pred_emotion,emotion)

        loss.backward()

        optimizer.step()

        scheduler.step()

        

        pred_sentiment = torch.argmax(torch.softmax(pred_sentiment,dim=-1),dim=-1)

        pred_emotion = torch.argmax(torch.softmax(pred_emotion,dim=-1),dim=-1)

        

        s_accuracy = (pred_sentiment == sentiment).float().mean()

        e_accuracy = (pred_emotion == emotion).float().mean()

    

        losses.update(loss.item(), ids.size(0))

        accuracy1.update(s_accuracy.item(),ids.size(0))

        accuracy2.update(e_accuracy.item(),ids.size(0))

        

        tk0.set_postfix(loss=losses.avg,sentiment_accuracy=accuracy1.avg,emotion_accuracy=accuracy2.avg)
def eval_fn(data_loader, model, device):

    model.eval()

    losses = AverageMeter()

    accuracy1 = AverageMeter()

    accuracy2 = AverageMeter()



    tk0 = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():

        for bi, d in enumerate(tk0):



            ids = d["ids"]

            mask = d["mask"]

            sentiment = d["sentiment_label"]

            emotion = d["emotion_label"]



            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)

            sentiment = sentiment.to(device,dtype=torch.long)

            emotion = emotion.to(device,dtype=torch.long)



            pred_sentiment,pred_emotion= model(

                ids=ids,

                mask=mask,

            )

            loss = loss_fn(pred_sentiment,sentiment,pred_emotion,emotion)



            pred_sentiment = torch.argmax(torch.softmax(pred_sentiment,dim=-1),dim=-1)

            pred_emotion = torch.argmax(torch.softmax(pred_emotion,dim=-1),dim=-1)



            s_accuracy = (pred_sentiment == sentiment).float().mean()

            e_accuracy = (pred_emotion == emotion).float().mean()



            losses.update(loss.item(), ids.size(0))

            accuracy1.update(s_accuracy.item(),ids.size(0))

            accuracy2.update(e_accuracy.item(),ids.size(0))



            tk0.set_postfix(loss=losses.avg,sentiment_accuracy=accuracy1.avg,emotion_accuracy=accuracy2.avg)

    print(f'sentiment:{accuracy1.avg} emotion:{accuracy2.avg}')

    return (accuracy1.avg + accuracy2.avg)/2
def deEmojify(inputString):

    return inputString.encode('ascii', 'ignore').decode('ascii')



emotion_dict = {'anger':0,'boredom':1,'empty':2,'enthusiasm':3,'fun':4,

                'happiness':5,'hate':6,'love':7,'neutral':8,'relief':9,

                'sadness':10,'surprise':11,'worry':12}



sentiment_dict = {'negative':0,'neutral':1,'positive':2}
df = pd.read_csv(Config.CSV_PATH)

df = df.dropna(subset=['text','new_sentiment','sentiment'])

df['text'] = df['text'].apply(deEmojify)

df = df[df['text']!='']

df = df.sample(frac=1).reset_index(drop=True)

df['emotion_label'] = df['sentiment'].apply(lambda x : emotion_dict[x])

df['sentiment_label'] = df['new_sentiment'].apply(lambda x :sentiment_dict[x])

kf = model_selection.StratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df.sentiment.values)):

    print(len(trn_), len(val_))

    df.loc[val_, 'kfold'] = fold

df.head()
fold = 0

df_train = df[df.kfold != fold].reset_index(drop=True)

df_valid = df[df.kfold == fold].reset_index(drop=True)



train_dataset = TweetDataset(

    tweet=df_train.text.values,

    sentiment_label=df_train.sentiment_label.values,

    emotion_label=df_train.emotion_label.values

)



valid_dataset = TweetDataset(

    tweet=df_valid.text.values,

    sentiment_label=df_valid.sentiment_label.values,

    emotion_label=df_valid.emotion_label.values

)



train_data_loader = torch.utils.data.DataLoader(

    train_dataset,

    batch_size=Config.TRAIN_BATCH_SIZE,

    num_workers=4,

    shuffle=True

)



valid_data_loader = torch.utils.data.DataLoader(

    valid_dataset,

    batch_size=Config.VALID_BATCH_SIZE,

    num_workers=2

)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CommentModel()

model.to(device)



num_train_steps = int(len(df_train) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)

param_optimizer = list(model.named_parameters())

no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

optimizer_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},

]

optimizer = transformers.AdamW(optimizer_parameters, lr=6e-5)

scheduler = transformers.get_linear_schedule_with_warmup(

    optimizer, 

    num_warmup_steps=200, 

    num_training_steps=num_train_steps

)



es = EarlyStopping(patience=2, mode="max")

print(f"Training is Starting for fold={fold}")



# I'm training only for 3 epochs even though I specified 5!!!

for epoch in range(Config.EPOCHS):

    train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)

    total_accuracy = eval_fn(valid_data_loader, model, device)

    print(f"Total accuracy = {total_accuracy}")

    es(total_accuracy, model, model_path=f"model.bin")

    if es.early_stop:

        print("Early stopping")

        break