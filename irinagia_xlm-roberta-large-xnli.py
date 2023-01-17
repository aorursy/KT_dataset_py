!nvidia-smi
!pip install git+https://github.com/huggingface/transformers
!pip install -U nlp
import pandas as pd

import numpy as np

import torch

import nlp



from transformers import AutoTokenizer, AutoModelForSequenceClassification



from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")

model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli").to(device)
model
train_dataset = nlp.load_dataset('csv', data_files={'train': '../input/contradictory-my-dear-watson/train.csv'}, split='train[:1024]')
train_dataset.num_rows
train_dataset[0]
def convert_to_features(batch):

    input_pairs = list(zip(batch['premise'], batch['hypothesis']))

    encodings = tokenizer.batch_encode_plus(input_pairs, 

                                            add_special_tokens=True,

                                            padding=False)



    return encodings





train_dataset = train_dataset.map(convert_to_features, batched=True)

train_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])
train_dataset[0]
batch_size = 16
train_dataloader = DataLoader(

    train_dataset,

    batch_size=batch_size,

    num_workers=0,

    collate_fn=tokenizer.pad

)
for i, batch in enumerate(train_dataloader):

    # print('Batch : {}'.format(batch))

    print('Input shape : {}'.format(batch['input_ids'].shape))

    print()

    if i == 4:

        break
batch['input_ids']
batch['attention_mask']
batch['label']
def eval(model, dataloader, device):

    model.eval()

    val_loss = 0

    correct = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch['input_ids'].to(device)

            attention_mask = batch['attention_mask'].to(device)

            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs[0]

            val_loss += loss.item()

            correct += (outputs[1].argmax(dim=-1) == labels).float().sum()



    return val_loss / len(dataloader),  correct / dataloader.dataset.shape[0]
%%time

val_loss, val_acc = eval(model, train_dataloader, device)
print(f'Loss : {val_loss:.5}, Accuracy : {val_acc:.2%}')
model.eval()

with torch.no_grad():

    input_ids = batch['input_ids'].to(device)

    attention_mask = batch['attention_mask'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    print(f'Output : {torch.argmax(outputs, -1).cpu().detach()}')

    print('Target : {}'.format(batch['label']))
train = pd.read_csv('../input/contradictory-my-dear-watson/train.csv', nrows=1024)
train.head()
train.label.replace([0, 2], [2, 0], inplace=True)
train.head()
train_dataset = nlp.Dataset.from_pandas(train)
train_dataset.num_rows
train_dataset[0]
%%time

train_dataset = train_dataset.map(convert_to_features, batched=True)

train_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])
train_dataloader = DataLoader(

    train_dataset,

    batch_size=batch_size,

    num_workers=0,

    collate_fn=tokenizer.pad

)
%%time

val_loss, val_acc = eval(model, train_dataloader, device)
print(f'Loss : {val_loss:.5}, Accuracy : {val_acc:.2%}')
sample_submission = pd.read_csv('../input/contradictory-my-dear-watson/sample_submission.csv')
test_dataset = nlp.load_dataset('csv', data_files={'test': '../input/contradictory-my-dear-watson/test.csv'})

test_dataset = test_dataset.map(convert_to_features, batched=True) 

test_dataset.set_format("torch", columns=['input_ids', 'attention_mask'])
test_dataloader = DataLoader(

        test_dataset['test'], 

        batch_size=batch_size,

        drop_last=False,

        num_workers=0,

        shuffle=False,

        collate_fn=tokenizer.pad

    )
def predict(model, dataloader, device):

    model.eval()

    val_loss = 0

    correct = 0

    test_preds = np.array([])

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch['input_ids'].to(device)

            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)[0]

            test_preds = np.concatenate((test_preds, torch.argmax(outputs, -1).cpu().detach().numpy()))



    sample_submission.prediction = test_preds.astype(int)

    sample_submission.prediction.replace([0, 2], [2, 0], inplace=True)

    sample_submission.to_csv('submission.csv', index=False)



    return sample_submission
%%time

predict(model, test_dataloader, device)