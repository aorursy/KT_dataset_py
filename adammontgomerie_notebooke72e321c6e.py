!pip install transformers
import numpy as np
import pandas as pd
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

batch_size = 16
max_len = 128
model_name = 'bert-base-multilingual-cased'

class ContradictionDataset(Dataset):
    def __init__(self, data):
        self.examples = self.encode(data)
    
    def __getitem__(self, index):
        example = self.examples[index]
        return self.move_to_device(example)

    def __len__(self):
        return len(self.examples)

    def encode(self, data):  
        inputs = []
        for index, row in train_data.iterrows():
            encoding = tokenizer(
                text=row['premise'], 
                text_pair=row['hypothesis'],
                truncation=True,
                padding='max_length',
                max_length=max_len,
                return_tensors='pt'
            )
            if 'label' in data:
                encoding['labels'] = torch.tensor([row['label']])
            inputs.append(encoding)
        return inputs

    def move_to_device(self, inputs):
        return {key: torch.squeeze(inputs[key]).to(device) for key in inputs}

tokenizer = BertTokenizer.from_pretrained(model_name)

%cd '/content/drive/My Drive/ml_hw/kaggle/contradictory-my-dear-watson'
train_data = pd.read_csv('./train.csv')
train_data, valid_data = train_test_split(train_data, train_size=0.9, test_size=0.1)
test_data = pd.read_csv('./test.csv')

train_set = ContradictionDataset(train_data)
valid_set = ContradictionDataset(valid_data)
test_set = ContradictionDataset(test_data)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
learning_rate = 1e-5
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.to(device)
LOG_INTERVAL = round(len(train_loader) / 10)

def train(epoch):
    model.train()
    total_loss = 0

    for batch_index, batch in enumerate(train_loader):
        model.zero_grad()
        output = model(**batch)
        loss = output[0]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_index % LOG_INTERVAL == 0 and batch_index > 0:
            current_loss = total_loss / LOG_INTERVAL
            print('| epoch {:3d} | ' 
                  '{:5d}/{:5d} batches | '
                  'loss {:5.2f}'.format(
                    epoch, 
                    batch_index, len(train_loader), 
                    current_loss))
            total_loss = 0

def test(data_loader):
    model.eval()
    total_score = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            output = model(**batch)
            preds = np.argmax(output[1].cpu(), axis=1)
            total_score += preds.eq(batch['labels'].cpu()).sum()
    return (total_score.item() / (len(data_loader) *batch_size)) * 100
EPOCHS = 5

accuracy = test(valid_loader)
print('| Pretraining Accuracy: {:.2f}%\n'.format(accuracy))

for epoch in range(1, EPOCHS + 1):
    train(epoch)
    accuracy = test(valid_loader)
    print('| epoch   {} |  Accuracy: {:.2f}%\n'.format(epoch, accuracy))
model.eval()
preds = []
with torch.no_grad():
    for batch_index, batch in enumerate(test_loader):
        output = model(**batch)
        preds += np.argmax(output[0].cpu(), axis=1).tolist()


submission = pd.DataFrame(test_data['id'])
submission['prediction'] = pd.Series(preds)
submission.sample(10)
submission.to_csv("submission.csv", index = False)