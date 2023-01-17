# import pandas as pd

# import numpy as np

# from scipy.stats import rankdata

# from scipy.special import softmax



# cols=['Depression','Alcohol','Suicide','Drugs']

# kv1=pd.read_csv('svm_sub_9.csv')

# kv2=pd.read_csv('bestsubs/Sub_v1.9.csv')

# kv3=pd.read_csv('bestsubs/Sub_v2.6.csv')

# kv4=pd.read_csv('ensemble_10.csv')

# test=pd.read_csv('Test_Set.csv')



# #ids=list(set(ids))

# #print(kv1[kv1['ID'].isin(ids)])

# #print(kv1.describe())

# #kv1.to_csv('ensemble_level_8_10_2.csv',index=False)





# def get_df(df,top=1):

#     x=pd.DataFrame(df[cols]).T

#     col=["top"+str(y) for y in range(top)]

#     rslt = pd.DataFrame(np.zeros((0,top)), columns=col)

#     for i in x.columns:

#         df1row = pd.DataFrame(x.nlargest(top, i).index.tolist(), index=col).T

#         rslt = pd.concat([rslt, df1row], axis=0)

#     rslt['ID']=df['ID'].values



#     rslt.reset_index(drop=True,inplace=True)

#     if top!=1:

#         rslt['label'] = rslt[col].apply(lambda x: ' '.join(x).split(), axis = 1)

#         rslt.drop(col,axis=1,inplace=True)

#         rslt = rslt.explode('label')

#         df = df.merge(rslt,on='ID')

#     else:

#         rslt['label'] = rslt[col]

#         rslt.drop(col,axis=1,inplace=True)

#         df = df.merge(rslt,on='ID')

#     return df



# ids=[]

# for c in cols:

#     ids.extend((kv4[(kv4[c]>=0.3) & (kv4[c]<=0.7)])['ID'])

# ids=list(set(ids))

# kv4 = kv4[kv4['ID'].isin(ids)].merge(test[test['ID'].isin(ids)])



# top2_df =get_df(kv4,1)[['ID','text','label']]

# print(top2_df['label'].value_counts())

# kv4=pd.read_csv('ensemble_10.csv')

# ids=[]

# for c in cols:

#     ids.extend((kv4[kv4[c]>=0.95])['ID'])

# ids=list(set(ids))

# kv4 = kv4[kv4['ID'].isin(ids)].merge(test[test['ID'].isin(ids)])



# top1_df =get_df(kv4,1)[['ID','text','label']]

# top1_df =top1_df[top1_df['label']!='Depression'].append(top1_df[top1_df['label']=='Depression'].sample(60,random_state=1994))



# top1_df = top1_df.append(top2_df,ignore_index=False)

# top1_df=top1_df.reset_index(drop=True)

# print(top1_df['label'].value_counts())

# print(top1_df)





# top1_df.to_csv('custom_pred_top1_2_test.csv',index=False)

# #top1_df =top1_df[top1_df['label']!='Depression'].append(top1_df[top1_df['label']=='Depression'].sample(60,random_state=1994))

# #print(top1_df['label'].value_counts())

# #rslt['labels']=" ".join(rslt[['top1','top2']].values)

# #print(rslt)









# !pip install -q kaggle

# from google.colab import files

# files.upload()

# !mkdir ~/.kaggle

# !cp kaggle.json ~/.kaggle/

# !chmod 600 ~/.kaggle/kaggle.json

# !kaggle datasets download -d chetanambi/basicneedsbasicneeds

# !unzip -q basicneedsbasicneeds.zip -d input
!pip install simpletransformers
# !pip install googletrans
def random_seed(seed_value):

    import random 

    random.seed(seed_value)  

    import numpy as np

    np.random.seed(seed_value)  

    import torch

    torch.manual_seed(seed_value)  

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)  

        torch.backends.cudnn.deterministic = True   

        torch.backends.cudnn.benchmark = False
import os

os.listdir('/kaggle/input/test-data-forzindi-techhealth')
import os

import gc

import re

import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.metrics import accuracy_score, log_loss
train = pd.read_csv('/kaggle/input/basicneedsbasicrightskenyatech4mentalhealth/Train.csv')

test = pd.read_csv('/kaggle/input/basicneedsbasicrightskenyatech4mentalhealth/Test.csv')

sub = pd.read_csv('/kaggle/input/basicneedsbasicrightskenyatech4mentalhealth/SampleSubmission.csv')



# newtrain = pd.read_csv('/kaggle/input/test-data-forzindi-techhealth/custom_pred_top1_2_test.csv')
train.shape, test.shape, sub.shape
# newtrain = newtrain[['text','label']]

# newtrain
train[train['label']=='Suicide']['text'].values
# set(train['text'][:50])
train[train['label']=='Depression']
train[train['ID']=='BXMHPTHE']['text'].values
train[train['text'].str.contains('situation')]
train[train['text'].str.contains('situation')]['text'].values

test[test['text'].str.contains('overcome')]['text'].values

# test[test['text'].str.contains('overcome')]['ID'].values
# train.loc[train['ID']=='VDCJVJSQ','text'] = 'I asked God to take away the feeling since it was so wrong' 

# train.loc[train['ID']=='AHEFY73G','text'] = 'How to cope with the alcohol situation' 

# train.loc[train['ID']=='QKBV071K','label'] = 'Suicide' 

# train.loc[train['ID']=='GUDI29F9','label'] = 'Depression' 

# train.loc[train['ID']=='D64IPYOU','label'] = 'Depression' 



# train.loc[train['ID']=='LM8GPR0X','label'] = 'Depression'

# train.loc[train['ID']=='MSEG1EUS','label'] = 'Depression'

# test.loc[test['ID']=='255YNCPV','text']='Why me?, why so?, why now? I am depressed'

# # test.loc[test['ID']=='X72WL59G','text']='I want to end my life'

# test.loc[test['ID']=='64MWIS95','text']='How to deal with depression and child support'

# # test.loc[test['ID']=='7GV9GMHR','text']='do I mean anything in this world?'

# test.loc[test['ID']=='7HYJWJTB','text']='Nobody cares?'



# test.loc[test['ID']=='DT2E8Z0Y','text']='Why is it I am going through much in life compared to others. I am depressed'



# test.loc[test['ID']=='YMT18ON6','text']='I feel like giving up'

# test.loc[test['ID']=='C8631YTB','text']='Why does God allow bad things like suicide to happen?'

# test.loc[test['ID']=='ZYIFAY98','text']='how can I overcome alcohol problem?'

# test.loc[test['ID']=='XXMTEUX8','text']='How do I go about solving my alcohol problem?'

# test.loc[test['ID']=='YCHHQT18','text']='If rejects you, what would you do?'

# test.loc[test['ID']=='USO2MI7J','text']='how will I face challenges'



train.loc[train['ID']=='CN5UZZA0','text']='I feel like I am nothing in the world'

rep={'issolated':'isolated','dieAm':'die. I am','frorge':'forget','lonelyNow':'lonely. Now','isolatedNow':'isolated. Now','whn':'when','…':' ',

    'moderatly':'moderately','deferrin':'defered in','after a lac':'for a lack','stresseed':'stressed','drugsNow':'drugs Now','mediataton':'meditation','hatredNow':'hatred Now',

    'how do I cope with ta difficult situation?what will I d to avoid it?':'how do I cope with a difficult situation? What will I do to avoid it?','cornerFeeling':'corner. Feeling',

    'avoiod':'avoid','issueS':'issues','sucidal':'suicidal','is there harm fo me when I take alcohol':'Is there harm for me when I take alcohol',

    'I feel indescribable sadness':'I feel low and extreme sadness','confusedNow':'confused. Now',

     ' It feels as if my head is exploding Lots of burnout Am feeling much better':'It feels as if my head is exploding. Lots of burnout. I am feeling much better',

    'hopelessFor':'hopeless. For','nott':'not','dizzines':'dizziness','Insomia,headache,social problems':'Insomnia, headache, social problems',

     'my own world I feel much bette':'my own world I feel much better','everythingI':'everything. I','hw':'how','oftenly':'often',' im feeling stressed':'I am feeling stressed'

    ,'whren':'when','whn':'when','lifeRight':'life. Right','npt':'not',"feel very lonelyI'm quite okay":'feel very lonely. I am quite okay','depressionI':'depression. I'

    ,'addidcted':'addicted','alccohol':'alcohol','ahd':'had','drinnking':'drinking','lowI':'low. I','SadI':'Sad. I','I felt sad,was stressed,lowi am now better':'I felt sad. I was stressed and low. I am now better'

    ,'schoolfee':'school fees','depresses':'depressed','diserted':'deserted','lonelyCurrently':'lonely. Currently','existd':'exist','GF':'girlfriend','ed results, -dissatisfied,':'ed results, dissatisfied'

    ,'negativecurrently':'negative currently','messNow':'mess. Now','FGM':'girlfriend','Feelings of defeat(post exams depression)Motivated to do better':'Feelings of defeat from post exams depression motivated to do better'

    ,'doto':'do to','finacial':'financial','worhtless':'worthless','frustratedi':'frustrated. I','benefitto':'benefit to','weatherNow':'weather. Now','doI':'do. I','incidencesof':'incidences of'

    ,' -How':'. How','Hopelessnesss,':'Hopeless ','includingmy':'including my','motivationsuicidal':'motivation suicidal','birthNow':'birth. Now','helplessStill':'helpless. Still',

    'lowt':'low','downrecovering':'down recovering',' -possible':' possible'}
def replc(vl):

    for k,v in rep.items():

        vl=vl.replace(k,v)

    return vl



test['text'] = test['text'].apply(replc)

train['text'] = train['text'].apply(replc)
train.head(3)
train.isna().sum()
train.drop('ID', axis=1, inplace=True)

test.drop('ID', axis=1, inplace=True)
train['label'].value_counts()
# train = train.append(newtrain,ignore_index=True)
train['label'].value_counts()
# test[test.duplicated()]
print(train['text'].apply(lambda x: len(x.split())).describe())
print(test['text'].apply(lambda x: len(x)).describe())
train['label'] = train['label'].map({'Depression':0, 'Alcohol': 1, 'Suicide': 2, 'Drugs': 3})
# train = train[~train['text'].duplicated()].reset_index(drop=True)

# train.drop_duplicates(inplace=True)
train.columns = ['text','labels']
train
#train_copy = train.copy()



gc.collect()

random_seed(2)



#train_copy['text'] = train_copy['text'].apply(lambda x: translator.translate(translator.translate( x, dest='fr').text, dest='en').text)

#train = train.append(train_copy,ignore_index=True)
from simpletransformers.classification import ClassificationModel

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.metrics import accuracy_score, log_loss

from scipy.special import softmax
model_args = {'train_batch_size': 50, 

              'reprocess_input_data': True,

              'overwrite_output_dir': True,

              'fp16': False,

              'do_lower_case': False,

              'num_train_epochs': 7,

              'max_seq_length': 160,

              'regression': False,

              'manual_seed': 1994,

              "learning_rate": 4e-5,

              'weight_decay': 0,

              "save_eval_checkpoints": False,

              "save_model_every_epoch": False,

              'no_cache':True,

              "silent": True,

              'gradient_accumulation_steps': 1,

              "use_early_stopping": True,

              "early_stopping_delta": 0.01,

              "early_stopping_metric": 'mcc',

              "early_stopping_metric_minimize": False,

              "early_stopping_patience": 5,

              "evaluate_during_training_steps": 1000

              }



#kv10 max seq length changed and dropped duplicates, epoch 7 to 8, tr batch size from 50 to 55
def get_model():

    model = ClassificationModel('roberta', 'roberta-base', use_cuda=True, num_labels=4, args=model_args)                            

    return model
from sklearn.utils import shuffle

train=shuffle(train,random_state=1994)
err=[]

y_pred_tot=[]

i=1

random_seed(1994)

fold=StratifiedKFold(n_splits=20, shuffle=True, random_state=1994)



for train_index, test_index in fold.split(train, train['labels']):

    train1_trn, train1_val = train.iloc[train_index], train.iloc[test_index]

    model = get_model()

    gc.collect()

    model.train_model(train1_trn)

    score, raw_outputs_val, wrong_preds = model.eval_model(train1_val) 

    raw_outputs_val = softmax(raw_outputs_val, axis=1) 

    print('Log_Loss:', log_loss(train1_val['labels'], raw_outputs_val))

    err.append(log_loss(train1_val['labels'], raw_outputs_val))

    predictions, raw_outputs_test = model.predict(test['text'])

    raw_outputs_test = softmax(raw_outputs_test, axis=1) 

    y_pred_tot.append(raw_outputs_test)
np.mean(err, 0)
y_pred = np.mean(y_pred_tot, 0)
sub[['Depression','Alcohol','Suicide','Drugs']] = y_pred

sub.head()
sub.to_csv('kv15_roberta_base.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,

    filename=filename)

    return HTML(html)



create_download_link(sub)