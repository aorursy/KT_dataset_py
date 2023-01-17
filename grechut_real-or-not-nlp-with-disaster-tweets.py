import pandas as pd

import numpy as np



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader



from transformers import DistilBertTokenizer, DistilBertForSequenceClassification



from sklearn.metrics import classification_report, confusion_matrix



train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
len(train_df)
max_tren_len = max([

    train_df['text'].str.split().str.len().max(),

    test_df['text'].str.split().str.len().max(),

])

max_tren_len
val_df = train_df.sample(frac=0.05)

train_df = train_df[~train_df.id.isin(val_df.id)]

len(train_df), len(val_df)
train_df.tail()
test_df.tail()
train_df['target'].value_counts()
val_df['target'].value_counts()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased').cuda()
def encode_text(text):

    return tokenizer.encode(

        text, add_special_tokens=True, max_length=max_tren_len, pad_to_max_length=True

    )



train_df['text_encoded'] = train_df['text'].map(encode_text)

val_df['text_encoded'] = val_df['text'].map(encode_text)
def predict(text, should_encode=False):

    if should_encode:

        text = encode_text(text)

    return model(torch.tensor(text).cuda().unsqueeze(0))[0].argmax().cpu().numpy()
train_tensor = torch.tensor(np.array(train_df['text_encoded'].map(np.array).tolist())).cuda()

target_tensor = torch.tensor(train_df['target'].values).cuda()

dataset = TensorDataset(train_tensor, target_tensor)
EPOCHS = 10

BATCH_SIZE = 128

LR = 0.001



dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)



for epoch in range(EPOCHS):

    print (epoch)

    model.train()



    for input_ids, labels in dl:

        optimizer.zero_grad()

        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        loss.backward()

        optimizer.step()

        

    model.eval()

    

    val_predicted = val_df['text_encoded'].map(predict)

    print(classification_report(val_df['target'], val_predicted))

    print(confusion_matrix(val_df['target'], val_predicted))



print('Finished Training')
test_df['target'] = test_df['text'].map(lambda x: predict(x, should_encode=True))

test_df.tail()
test_df.target.value_counts()
# test_df[['id', 'target']].to_csv('submission.csv', index=False)