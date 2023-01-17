# Importing the libraries needed

import pandas as pd

import torch

import transformers

from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertModel, DistilBertTokenizer
# Setting up the device for GPU usage



from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
# Import the csv into pandas dataframe and add the headers

df = pd.read_csv('../input/news-corpora/newsCorpora.csv', sep='\t', names=['ID','TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# df.head()

# # Removing unwanted columns and only leaving title of news and the category which will be the target

df = df[['TITLE','CATEGORY']]

# df.head()



# # Converting the codes to appropriate categories using a dictionary

my_dict = {

    'e':'Entertainment',

    'b':'Business',

    't':'Science',

    'm':'Health'

}



def update_cat(x):

    return my_dict[x]



df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_cat(x))



encode_dict = {}



def encode_cat(x):

    if x not in encode_dict.keys():

        encode_dict[x]=len(encode_dict)

    return len(encode_dict)



df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))
# Defining some key variables that will be used later on in the training

MAX_LEN = 512

TRAIN_BATCH_SIZE = 4

VALID_BATCH_SIZE = 2

EPOCHS = 1

LEARNING_RATE = 1e-05

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
class Triage(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):

        self.len = len(dataframe)

        self.data = dataframe

        self.tokenizer = tokenizer

        self.max_len = max_len

        

    def __getitem__(self, index):

        title = str(self.data.TITLE[index])

        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(

            title,

            None,

            add_special_tokens=True,

            max_length=self.max_len,

            pad_to_max_length=True,

            return_token_type_ids=True

        )

        ids = inputs['input_ids']

        mask = inputs['attention_mask']



        return {

            'ids': torch.tensor(ids, dtype=torch.long),

            'mask': torch.tensor(mask, dtype=torch.long),

            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)

        } 

    

    def __len__(self):

        return self.len
# Creating the dataset and dataloader for the neural network



train_size = 0.8

train_dataset=df.sample(frac=train_size,random_state=200).reset_index(drop=True)

test_dataset=df.drop(train_dataset.index).reset_index(drop=True)





print("FULL Dataset: {}".format(df.shape))

print("TRAIN Dataset: {}".format(train_dataset.shape))

print("TEST Dataset: {}".format(test_dataset.shape))



training_set = Triage(train_dataset, tokenizer, MAX_LEN)

testing_set = Triage(test_dataset, tokenizer, MAX_LEN)
train_params = {'batch_size': TRAIN_BATCH_SIZE,

                'shuffle': True,

                'num_workers': 0

                }



test_params = {'batch_size': VALID_BATCH_SIZE,

                'shuffle': True,

                'num_workers': 0

                }



training_loader = DataLoader(training_set, **train_params)

testing_loader = DataLoader(testing_set, **test_params)
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 



class DistillBERTClass(torch.nn.Module):

    def __init__(self):

        super(DistillBERTClass, self).__init__()

        self.l1 = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.l2 = torch.nn.Dropout(0.3)

        self.l3 = torch.nn.Linear(768, 1)

    

    def forward(self, ids, mask):

        output_1= self.l1(ids, mask)

        output_2 = self.l2(output_1[0])

        output = self.l3(output_2)

        return output
model = DistillBERTClass()

model.to(device)
# Creating the loss function and optimizer

loss_function = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
# Defining the training function on the 80% of the dataset for tuning the distilbert model



def train(epoch):

    model.train()

    for _,data in enumerate(training_loader, 0):

        ids = data['ids'].to(device, dtype = torch.long)

        mask = data['mask'].to(device, dtype = torch.long)

        targets = data['targets'].to(device, dtype = torch.long)



        outputs = model(ids, mask).squeeze()



        optimizer.zero_grad()

        loss = loss_function(outputs, targets)

        if _%5000==0:

            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
for epoch in range(EPOCHS):

    train(epoch)
def valid(model, testing_loader):

    model.eval()

    n_correct = 0; n_wrong = 0; total = 0

    with torch.no_grad():

        for _, data in enumerate(testing_loader, 0):

            ids = data['ids'].to(device, dtype = torch.long)

            mask = data['mask'].to(device, dtype = torch.long)

            targets = data['targets'].to(device, dtype = torch.long)

            outputs = model(ids, mask).squeeze()

            big_val, big_idx = torch.max(outputs.data, dim=1)

            total+=targets.size(0)

            n_correct+=(big_idx==targets).sum().item()

    return (n_correct*100.0)/total
print('This is the validation section to print the accuracy and see how it performs')

print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')



acc = valid(model, testing_loader)

print("Accuracy on test data = %0.2f%%" % acc)
# Saving the files for re-use



# output_model_file = './models/pytorch_distilbert_news.bin'

# output_vocab_file = './models/vocab_distilbert_news.bin'



# model_to_save = model

# torch.save(model_to_save, output_model_file)

# tokenizer.save_vocabulary(output_vocab_file)



# print('All files saved')

# print('This tutorial is completed')