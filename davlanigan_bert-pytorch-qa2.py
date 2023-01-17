# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import torch

import transformers

from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering,AdamW



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_df=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

test_df.head()
tokenizer = transformers.BertTokenizer.from_pretrained("/kaggle/input/bert-base-uncased/",model_max_len=48,

                                                       padding_side="right",cls_token="[CLS]",eos_token="[SEP]",pad_token="[PAD]")



config = BertConfig.from_pretrained('/kaggle/input/bertlargewholewordmaskingfinetunedsquad/bert-large-uncased-whole-word-masking-finetuned-squad-config.json', 

                                    output_hidden_states=True)



model = BertForQuestionAnswering.from_pretrained('/kaggle/input/bertlargewholewordmaskingfinetunedsquad/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin',

                                                 config=config)





train_df=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

train_df=train_df.dropna()

train_df.head()
def seed_everything(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = True



seed = 42

seed_everything(seed)





def create_input(q,c,a):

    

    def get_idxs(text_ids,selected_text_ids):

        len_text_ids,len_selected_text_ids=len(text_ids),len(selected_text_ids)

        for i in range( len_text_ids - len_selected_text_ids + 1):

            start_idx,end_idx=i,i+len_selected_text_ids

            sub_list=text_ids[start_idx:end_idx]

            if sub_list==selected_text_ids:

                return start_idx,end_idx

    

    

    MAX_LEN=84

    input_seqA,token_idsA,masksA,spA,epA=[],[],[],[],[]

    count,fails=0,0

    for i,_ in enumerate(q):

        count+=1

        cls,sep=[101],[102]

        q_encoded=tokenizer.encode(q[i],add_special_tokens=False,pad_to_max_length=False)

        c_encoded=tokenizer.encode(c[i],add_special_tokens=False,pad_to_max_length=False)

        a_encoded=tokenizer.encode(a[i],add_special_tokens=False,pad_to_max_length=False)

        

        len_c=len(c_encoded)

        try:

            start_idx,end_idx=get_idxs(c_encoded,a_encoded)

            

            input_seq=cls+q_encoded+sep+ c_encoded+sep

            token_ids=[0,0,0]+[1]*len_c+[1]



            padd=[0]*( MAX_LEN - len(input_seq) )



            mask_ids=[1]*len(input_seq) + padd

            input_seq=input_seq + padd

            token_ids=token_ids + padd



            #print("{} {} {}".format(len(mask_ids),len(input_seq),len(token_ids)))



            if len(token_ids)+len(input_seq)+len(mask_ids)==(84*3):

                input_seqA.append(input_seq)

                token_idsA.append(token_ids)

                masksA.append(mask_ids)

                spA.append(start_idx+3)

                epA.append(end_idx+3)



            else:

                fails+=1

            

        except:

            fails+=1



    print("fails: {}, % fails {}".format( fails, round( fails/count,2) ) )

    

    return np.array(input_seqA),np.array(token_idsA),np.array(masksA),np.array(spA),np.array(epA)



        

    

# iseq,ti,m=create_input(q,c,a)



# print("{} :: {}".format(len(iseq[0]),iseq[0]) )

# print("{} :: {}".format(len(ti[0]),ti[0]) )

# print("{} :: {}".format(len(m[0]),m[0]) )



class Bert_Classification_Data(torch.utils.data.Dataset):

    """

    What should a dataset look like to pytorch?

    """

    

    def __init__(self,q,c,a,create_input=create_input):

        

        eseq,ti,m,sp,ep=create_input(q,c,a)

        

        self.input_seq=torch.tensor(eseq, device="cuda",dtype=torch.long)

        self.token_ids=torch.tensor(ti, device="cuda",dtype=torch.long)

        self.masks=torch.tensor(m,device="cuda",dtype=torch.float32)

        self.start_positions=torch.tensor( sp ,dtype=torch.long,device="cuda"  )

        self.end_positions=torch.tensor( ep,dtype=torch.long, device="cuda"  )

        

    def __len__(self):

        return len(self.input_seq)

    

    def __getitem__(self,index):

        return self.input_seq[index],self.token_ids[index],self.masks[index],self.start_positions[index],self.end_positions[index]
q,c,a=train_df["sentiment"].values,train_df["text"].values,train_df["selected_text"].values

dataset=Bert_Classification_Data(q,c,a)


data_loader=train_loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=128,shuffle=True)



out=next(iter(data_loader))





input_ids=out[0][0]

si,ei=out[3][0].item(),out[4][0].item()



print(si,ei)



s=tokenizer.decode( input_ids[si:ei] )

print(s)

out=train_df.where(train_df["selected_text"]==s).dropna(how="all")

out

import matplotlib.pyplot as plt



torch.set_grad_enabled(True)



#push to gpu

model.to("cuda")



#initialize model for training

model.train()



train_loader=torch.utils.data.DataLoader(dataset=dataset,batch_size=36,shuffle=True)

optimizer=AdamW(model.parameters(),lr=2e-5,eps=1e-8)



EPOCHS=10

total_loss,score=[],[]

for epoch in range(EPOCHS):

    for batch in train_loader:

        input_seq,input_ids,masks,sp,ep=batch



        loss = model(input_seq, token_type_ids=input_ids, start_positions=sp, end_positions=ep )[0]



        optimizer.zero_grad() #zero out the previous gradiant

        loss.backward() #calculates gradients (backpropagation)

        optimizer.step() #updates the weights



        print(loss.item())

        total_loss.append( loss.item() )

    

plt.plot(total_loss)




def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    

    #print(a,b,c)

    

    return float(len(c)) / (len(a) + len(b) - len(c))

    

    



a,q,c=train_df["selected_text"].values,train_df["sentiment"].values,train_df["text"].values



torch.set_grad_enabled(False)

model.eval()



accuracy,sent=[],[]

for i,_ in enumerate(q):

    

    if q[i]!="neutral":

        encoding = tokenizer.encode_plus(q[i], c[i])



        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]



        #start_scores, end_scores = model(torch.tensor([input_ids],device="cuda"), token_type_ids=torch.tensor([token_type_ids],device="cuda"))

        out = model( torch.tensor([input_ids],device="cuda"), token_type_ids=torch.tensor([token_type_ids], device="cuda") )



        start_scores, end_scores= out[0],out[1]



        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)



        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)])

        answer=answer.replace(' ##', '')

        answer=answer.replace(" ` ","`")

        answer=answer.replace(" , ",", ")

        answer=answer.replace(" ,",", ")

        answer=answer.replace(" .",".")

        answer=answer.replace(" !","!")

        answer=answer.replace(" ?","?")

        answer=answer.replace("* * * *","****")

        answer=answer.replace("... ","...")

        answer=answer.replace("< 3","<3")

        answer=answer.replace(" )",")")

        answer=answer.replace("( ","(")

        

        

        accu_=jaccard(answer,a[i]) 

                

        accuracy.append( accu_ )

        

        if accu_ <0.6:

            sent.append( [answer,a[i],c[i]] )

    

    elif q[i]=="neutral":

        accu_=jaccard(a[i],c[i]) 

        accuracy.append( accu_ )

        

acc=np.sum(np.array(accuracy))/len(accuracy)



print(acc)

        
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    

    print(a,b,c)

    

    return float(len(c)) / (len(a) + len(b) - len(c))





print(len(sent))

for j,i in enumerate(sent):

    print(i)

    print(a[j])

        

        

print(sent[2][0])



#print(jaccard(sent[10][0],sent[10][1]))
test_df=pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

tid,q,c=test_df["textID"],test_df["sentiment"].values,test_df["text"].values





torch.set_grad_enabled(False)

model.eval()



answers=[]

for i,_ in enumerate(q):

    

    if q[i]!="neutral":

        encoding = tokenizer.encode_plus(q[i], c[i])



        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]



        #start_scores, end_scores = model(torch.tensor([input_ids],device="cuda"), token_type_ids=torch.tensor([token_type_ids],device="cuda"))

        out = model( torch.tensor([input_ids],device="cuda"), token_type_ids=torch.tensor([token_type_ids], device="cuda") )



        start_scores, end_scores= out[0],out[1]



        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)



        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)])

        answer=answer.replace(' ##', '')

        answer=answer.replace(" ` ","`")

        answer=answer.replace(" , ",", ")

        answer=answer.replace(" ,",", ")

        answer=answer.replace(" .",".")

        answer=answer.replace(" !","!")

        answer=answer.replace(" ?","?")

        answer=answer.replace("* * * *","****")

        answer=answer.replace("... ","...")

        answer=answer.replace("< 3","<3")

        answer=answer.replace(" )",")")

        answer=answer.replace("( ","(")



        #print(c[i])

        #print(answer)

        answers.append([tid[i],'"'+answer+'"'])

    

    elif q[i]=="neutral":

        answers.append([tid[i],'"'+c[i]+'"'])

        

    



df=pd.DataFrame(data=answers,columns=["textID","selected_text"])



print(df[0:30])



df.to_csv("submission.csv",index=False)