# For QA Roberta Large 
# 	learning_rate=1e-5
# 	batch size =3
	
# 	Epoch : 0
# 	Validation Loss : 0.6635116338729858
# 	Validation Accuracy : 62.255
# 	Train Loss : 0.6718674302101135
	
# 	Epoch : 1
# 	Validation Loss : 0.6639534831047058
# 	Validation Accuracy : 62.194
# 	Train Loss : 0.6717374920845032

# 	Epoch : 2
# 	Validation Loss : 0.6671617031097412
# 	Validation Accuracy : 62.194
# 	Train Loss : 0.6691761016845703

# -------------------------------------------------------------

# 	learning_rate=1e-5
# 	batch size =4
# Epoch : 0
# Validation Loss : 0.6570765376091003
# Validation Accuracy : 62.255
# Train Loss : 0.6687435507774353

# Epoch : 1
# Validation Loss : 0.4587351679801941
# Validation Accuracy : 79.105
# Train Loss : 0.5985463261604309

# Epoch : 2
# Validation Loss : 0.395685613155365
# Validation Accuracy : 82.475
# Train Loss : 0.4091240465641022



# test_accuracy': 83.557

!/opt/conda/bin/python3.7 -m pip install --upgrade pip
!pip install --upgrade pytorch_lightning
!pip install transformers==3.1.0
!pip install nlp
import pytorch_lightning as pl
import argparse
import numpy as np 
import pandas as pd 
from transformers import RobertaConfig,RobertaForSequenceClassification,RobertaTokenizer,AdamW,set_seed
import torch
from torch.utils.data import Dataset, DataLoader
import nlp
from collections import defaultdict
from matplotlib import pyplot as plt
import os
set_seed(6)
import torch
torch.__version__
import transformers
transformers.__version__
# model = MultiLabelClassificationModel('bert', 'bert-base-dutch-cased/bertje-base', num_labels=47, args={'n_gpu': 1, 'train_batch_size':8, 'gradient_accumulation_steps':16, 'learning_rate': 1e-4, 'num_train_epochs': 1, 'max_seq_length': 256, 'fp16': False, 'reprocess_input_data': True, 'use_cached_eval_features': False, 'evaluate_during_training': True, 'evaluate_during_training_verbose': True, 'output_dir': 'outputs/'})

tokenizer = RobertaTokenizer.from_pretrained('../input/roberta-transformers-pytorch/roberta-base')
model = RobertaForSequenceClassification.from_pretrained('../input/roberta-transformers-pytorch/roberta-base',  num_labels=2)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)
print("loss :",outputs[0])
logits = outputs[1]
print("\n\n\n",logits,"\n\n")
prob = torch.softmax(logits, dim=1)
pred = torch.argmax(prob, dim=1) 
print(pred,"\n\n",prob)
tokenizer.get_vocab()
dataset = nlp.load_dataset('boolq')
val_df_tr = pd.DataFrame()
test_df_tr = pd.DataFrame()

val_df_fl = pd.DataFrame()
test_df_fl = pd.DataFrame()

train_df =pd.DataFrame(dataset['train'])
df = pd.DataFrame(dataset['validation'])

val_df = pd.concat([df[df['answer']== True][:1016],df[df['answer']== False][:618]])

test_df = pd.concat([df[df['answer']== True][1016:],df[df['answer']== False][618:]])
train_df.to_csv('/kaggle/working/train.csv')
val_df.to_csv('/kaggle/working/val.csv')
test_df.to_csv('/kaggle/working/test.csv')
dat =[{
	"answer":"True",
	"passage": "YES Bank loan fraud: CBI charges Rana Kapoor with criminal conspiracy.",
	"question":"Does Yes bank involved in any fraud",
	"link":"https://www.business-standard.com/article/current-affairs/yes-bank-loan-fraud-cbi-charges-rana-kapoor-with-criminal-conspiracy-120062501509_1.html"
},
    {
	"answer":"False",
	"passage": "Reserve Bank of India imposes monetary penalty on Oriental Bank of Commerce.",
	"question":"Oriental Bank of Commerce was not penalized by Reserve Bank of India",
	"link":"https://m.rbi.org.in/commonman/English/Scripts/PressReleases.aspx?Id=2868"
},
{
	"answer":"True",
	"passage": "YES Bank loan fraud: CBI charges Rana Kapoor with criminal conspiracy.",
	"question":"Rana Kapoor is charged with criminal conspiracy",
	"link":"https://www.business-standard.com/article/current-affairs/yes-bank-loan-fraud-cbi-charges-rana-kapoor-with-criminal-conspiracy-120062501509_1.html"
},
{
	"answer":"True",
	"passage": "The Reserve Bank of India Tuesday slapped Rs 1 crore fine on Yes Bank for non-compliance of directions on Swift messaging software.The private sector lender in a regulatory filing said. The Reserve Bank of India (RBI) has levied an aggregate penalty of Rs 10 million (Rs 1 crore) on the bank for non-compliance of regulatory directions observed during assessment of implementation of SWIFT-related operational controls.",
	"question":"Yes Bank is fined by Reserve Bank of India",
	"link":"https://www.businesstoday.in/current/corporate/reserve-bank-of-india-rbi-yes-bank-swift-messaging-software-pnb-fraud-rbi-levies-rs-1-crore-penalty-on-yes-bank-for-non-compliance-of-instructions-related-to-swift/story/324546.html"
},
{
	"answer":"True",
	"passage": """The latest disclosure under SEBI Prohibition of Insider Trading regulations was made by N S KANNAN in ICICI Bank Ltd. where Acquisition of 75000 Equity Shares done at an average price of Rs. 153.0 was reported to the exchange on Oct. 1st 2020.There were no SAST disclosures made for ICICI Bank Ltd.Insider trades are disclosures under SEBI (Prohibition of Insider Trading) Regulations. 2015 ([Regulation 7 (2) with 6(2)] made by corporate insiders: promoters. officers. directors. employees and large shareholders who are buying and selling stock in their own companies.""",
	"question":"is N S KANNAN engaged in Insider trading",
	"link":"https://trendlyne.com/equity/insider-trading-sast/all/ICICIBANK/584/icici-bank-ltd/#:~:text=Insider%20Trading%20%26%20SAST%20disclosures%20for,an%20average%20price%20of%20Rs.&text=There%20were%20no%20SAST%20disclosures%20made%20for%20ICICI%20Bank%20Ltd."
}
,
{
	"answer":"True",
	"passage": """The latest disclosure under SEBI Prohibition of Insider Trading regulations was made by N S KANNAN in ICICI Bank Ltd. where Acquisition of 75000 Equity Shares done at an average price of Rs. 153.0 was reported to the exchange on Oct. 1st 2020.There were no SAST disclosures made for ICICI Bank Ltd.Insider trades are disclosures under SEBI (Prohibition of Insider Trading) Regulations. 2015 ([Regulation 7 (2) with 6(2)] made by corporate insiders: promoters. officers. directors. employees and large shareholders who are buying and selling stock in their own companies.""",
	"question":"is N S KANNAN engaged in Insider trading",
	"link":"https://trendlyne.com/equity/insider-trading-sast/all/ICICIBANK/584/icici-bank-ltd/#:~:text=Insider%20Trading%20%26%20SAST%20disclosures%20for,an%20average%20price%20of%20Rs.&text=There%20were%20no%20SAST%20disclosures%20made%20for%20ICICI%20Bank%20Ltd."
},

{
	"answer":"False",
	"passage": "Swiss banking giant UBS has been fined €3.7bn (£3.2bn; $4.2bn) in a French tax fraud case.A court in Paris found that the bank had illegally helped French clients hide billions of euros from French tax authorities between 2004 and 2012.",
	"question":"UBS not involved in tax fraud",
	"link":"https://www.bbc.com/news/business-47305227"
},
{
	"answer":"True",
	"passage": """After completing an over year-long investigation and under various sections of the
Forex law, the highest ever FEMA show-cause notice has been issued today. The ED has charged the
firm under the FEMA for resorting to unauthorised foreign exchange dealings, holding of foreign
exchange outside India, and willful siphoning off a whopping amount of Rs 7,220 crore as export
proceedings. At present, the firm’s three promoter brothers – Nilesh Parekh, Umesh Parekh, and
Kamlesh Parekh – are also being probed by CBI and ED. Earlier, the firm was allegedly defrauding a
consortium of 25 banks to the tune of Rs 2,672 crore by availing credit facilities in the form of
working capital loans and discounting of export bills""",
	"question":"Umesh Parekh involved in foreign exchange dealings",
	"link":""
},

]
da_df = pd.DataFrame(dat)
da_df.to_csv("test_new.csv")
da_df.count()
from random import randint
print("passage:\n",test_df.iloc[1]['passage'],"\n\n\n")
print("question:\n",test_df.iloc[1]['question'],"\n\n\n")
print("answer:\n",test_df.iloc[1]['answer'],"\n\n\n")

# train_df = pd.concat([train_df.head(100),train_df.tail(100)])
# train_df.to_csv('/kaggle/working/train.csv')
# val_df = pd.concat([val_df.head(25),val_df.tail(25)])
# val_df.to_csv('/kaggle/working/val.csv')

# test_df = pd.concat([test_df.head(25),test_df.tail(25)])
# test_df.to_csv('/kaggle/working/test.csv')
print("train_df :",train_df.count())
print("val :",val_df.count())
print("Test :",test_df.count())
class BoolqDataset(Dataset):
    def __init__(self,dir_path,type_,tokenizer):
        self.df = pd.read_csv(dir_path+type_+".csv")
        self.df['answer_']=self.df['answer'].apply(lambda x : 1 if x==True else 0)
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        passage = self.df.iloc[index]['passage']
        question = self.df.iloc[index]['question']
        label = self.df.iloc[index]['answer_']
        seq_len = len(self.tokenizer.tokenize(passage))+len(self.tokenizer.tokenize(question))
        return {'passage':passage,'question':question,'label':label,"seq_len":seq_len}
def collate_fn_seq_len(batch,params):
    input_ids =[]
    attention_mask = []
    target_ids = []
    
    max_seq_len = max([iter_data['seq_len'] for iter_data in batch]) 
    max_seq_len = 512
    for iter_data in batch:
        print("question :",iter_data['question'],"\npassage:",iter_data['passage'],"\nGT :",iter_data['label'],"\n\n\n")
        encoding = params['tokenizer'].encode_plus(iter_data['question'],iter_data['passage'],return_tensors="pt",max_length=max_seq_len,pad_to_max_length=True,truncation=True)
        input_ids.append(encoding['input_ids'])
        attention_mask.append(encoding['attention_mask'])
        target_ids.append(torch.tensor(iter_data['label'],dtype=torch.long))
    return {'input_ids':(torch.stack(input_ids)).squeeze(),'attention_mask':(torch.stack(attention_mask)).squeeze(),"target_ids":(torch.stack(target_ids)).squeeze()}

bq = BoolqDataset('/kaggle/working/','test_new',tokenizer)
dl = DataLoader(bq,batch_size=2,drop_last=True,collate_fn= lambda b, params={'tokenizer':tokenizer}: collate_fn_seq_len(b, params),shuffle=False)
for batch_ in dl:
    s=""
#     print(batch_['target_ids'])
    print(batch_['input_ids'].size())
    #print(tokenizer.convert_ids_to_tokens(batch_['input_ids'][0]))
        
    break
    
configuration = RobertaConfig()
args= dict(
    data_dir="/kaggle/working/", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='../input/roberta-transformers-pytorch/roberta-large',
    tokenizer_name_or_path='../input/roberta-transformers-pytorch/roberta-large',
    max_seq_length=512,
    learning_rate=1e-6,
    weight_decay=0.0,
    adam_epsilon=1e-5,
    warmup_steps=0,
    train_batch_size=4,
    eval_batch_size=4,
    test_batch_size=4,
    num_train_epochs=3,
    shuffle=True
)

params = argparse.Namespace(**args)
configuration.max_position_embeddings = params.max_seq_length+2
configuration.num_labels= 2
configuration.type_vocab_size=1
configuration.vocab_size=50265

#Uncomment below for Large Model
configuration.hidden_size=1024
configuration.num_attention_heads=16
configuration.num_hidden_layers=24
configuration.intermediate_size=4096




# old 12 and 
configuration

class ClassificationModel(pl.LightningModule):
    def __init__(self,hparams):
        super(ClassificationModel, self).__init__()
        self.hparams = hparams

        self.model = RobertaForSequenceClassification.from_pretrained(hparams.model_name_or_path,config=configuration)
        self.tokenizer = RobertaTokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        

    def forward(self,input_ids,attention_mask,mlabels=None,_typ=True):
        
        if _typ:
            loss,logits = self.model(input_ids = input_ids,attention_mask=attention_mask,labels=mlabels)

            return loss,logits
        else:
            loss,logits = self.model(input_ids = input_ids, token_type_ids=None,attention_mask=attention_mask)

            return loss,logits

    
    def _step(self,_batch):
        lm_labels = _batch["target_ids"]
        loss,logits = self(_batch["input_ids"],_batch["attention_mask"],lm_labels)        
        return loss,logits
    
    def training_step(self, train_batch, batch_idx):
        
        loss ,logits= self._step(train_batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
   
    def validation_step(self, val_batch, batch_idx):
        loss,logits = self._step(val_batch)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)    
        tensorboard_logs = {"val_loss": loss}
        
        prd_cnt = torch.sum(pred==val_batch["target_ids"])
        return {"loss": loss, "log": tensorboard_logs,"preds":prd_cnt}
    
    def test_step(self, test_batch, batch_idx):
        
        loss,logits = self._step(test_batch)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)    
        print("*"*30,"\n",pred,"\n\n",test_batch["target_ids"],"\n","*"*30)
        prd_cnt = torch.sum(pred==test_batch["target_ids"])
        return {"preds":prd_cnt}
    
    def training_epoch_end(self, outputs):
        print("Epoch :",self.current_epoch)
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        print("Train Loss :",avg_train_loss.item())
        print("*"*30,"\n")
        print("Deleting Log Folder")
        
        os.system('rm -r /kaggle/working/lightning_logs/')
        print("Deleted Folder /kaggle/working/lightning_logs/")
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def test_epoch_end(self, outputs):
        
        pred_tensors = torch.stack([x["preds"] for x in outputs])
        tru_pred_cnt = pred_tensors.sum()
        acc = tru_pred_cnt.item()/(len(pred_tensors)*self.hparams.eval_batch_size)
        print("Test Accuracy :",round(acc*100,3))
        tensorboard_logs = {"test_accuracy": round(acc*100,3)}
        return {"test_accuracy": round(acc,3), "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print("Validation Loss :",avg_val_loss.item())
        pred_tensors = torch.stack([x["preds"] for x in outputs])
        tru_pred_cnt = pred_tensors.sum()
        acc = tru_pred_cnt.item()/(len(pred_tensors)*self.hparams.eval_batch_size)
        print("Validation Accuracy :",round(acc*100,3))
        tensorboard_logs = {"avg_val_loss": avg_val_loss,"val_accuracy":round(acc*100,3)}
        return {"avg_val_loss": avg_val_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}
      
    def val_dataloader(self):
            bq = BoolqDataset(self.hparams.data_dir,'val',self.tokenizer)
            return DataLoader(bq,batch_size=self.hparams.eval_batch_size,drop_last=True,collate_fn=lambda b, params={'tokenizer':self.tokenizer}: collate_fn_seq_len(b, params),shuffle=True)
    
    def train_dataloader(self):
        bq = BoolqDataset(self.hparams.data_dir,'train',self.tokenizer)
        return DataLoader(bq,batch_size=self.hparams.train_batch_size,drop_last=True,collate_fn=lambda b, params={'tokenizer':self.tokenizer}: collate_fn_seq_len(b, params),shuffle=True) 
    
    def test_dataloader(self):
        print("Testing In Progress....")
        bq = BoolqDataset(self.hparams.data_dir,'test',self.tokenizer)
        return DataLoader(bq,batch_size=self.hparams.test_batch_size,drop_last=True,collate_fn=lambda b, params={'tokenizer':self.tokenizer}: collate_fn_seq_len(b, params),shuffle=True) 

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        #****
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }]
        
        optimizer = AdamW(optimizer_grouped_parameters, \
                          lr=self.hparams.learning_rate, \
                          eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

news_classification = ClassificationModel(params)
device="cuda"
test_model = news_classification.model.from_pretrained('../input/my-model/save_model/')#,num_labels=2)

test_model.to(device)
test_model.eval()

# from tqdm import tqdm 
# for batch_ in tqdm(dl, desc ="Status "): 
    
#     logits = test_model(batch_['input_ids'].to(device),batch_["attention_mask"].to(device))  
#     prob = torch.softmax(logits[0], dim=1)
#     pred = torch.argmax(prob, dim=1)   
 
#     print("pred :",pred)
    
# tensor([1, 1]) (tensor([[-2.0116,  1.0465],
#         [-1.9102,  1.7220]], grad_fn=<AddmmBackward>),)
# tensor([1, 1])
pred_l=[]
trg_l=[]
for batch_ in tqdm(dl, desc ="Status "):    
    logits = test_model(batch_['input_ids'].to(device),batch_["attention_mask"].to(device))  
    prob = torch.softmax(logits[0], dim=1)
    pred = torch.argmax(prob, dim=1)   
    print("pred :",pred)
    pred_l.append(pred)
    trg_l.append(batch_["target_ids"])
    
print("Prediction End.....")

pr=[]
tr=[]
for f,j in zip(pred_l,trg_l):
    for ff in f:
        pr.append(ff.item())
    for ff in j:
        tr.append(ff.item())
    
cnt=0
for i,j in zip(tr,pr):
    if i==j:
        cnt+=1
cnt/len(pr)

