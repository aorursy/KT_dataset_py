import sys

package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.append(package_dir)

import gc

import numpy as np

import pandas as pd

from tqdm import tqdm

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam

from pytorch_pretrained_bert import BertConfig

import torch

import torch.utils.data

import os

import warnings

warnings.filterwarnings(action='once')



device = torch.device('cuda')



def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    print(longer)

    return np.array(all_tokens)





MAX_SEQUENCE_LENGTH = 200

SEED = 42 

BATCH_SIZE = 32

INFER_BATCH_SIZE = 64

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

#LARGE_BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16/'

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True



#bert_config = BertConfig('../input/bert-inference/bert/bert_config.json')

bert_config = BertConfig('../input/bertinference/bert_config.json')

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)



BERT_SMALL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

BERT_LARGE_PATH = '../input/bert-pretrained-models/uncased_l-24_h-1024_a-16/uncased_L-24_H-1024_A-16/'



str_='imports done'

os.system('echo '+str_)



train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

test_df['comment_text'] = test_df['comment_text'].astype(str) 

X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)

test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))   



str_='convert_lines done'

os.system('echo '+str_)



gc.collect()

model = BertForSequenceClassification(bert_config, num_labels=1)

model.load_state_dict(torch.load("../input/bertinference/pytorch_bert_6.bin"))

model.to(device)



str_='convert_lines done'

os.system('echo '+str_)



for param in model.parameters():

    param.requires_grad = False

model.eval()



test_preds = np.zeros((len(X_test)))

test_loader = torch.utils.data.DataLoader(test, batch_size=INFER_BATCH_SIZE, shuffle=False)

tk0 = tqdm(test_loader)

for i, (x_batch,) in enumerate(tk0):

    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

    test_preds[i * INFER_BATCH_SIZE:(i + 1) * INFER_BATCH_SIZE] = pred[:, 0].detach().cpu().squeeze().numpy()



predictions_bert = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()



str_='bert predicted'

os.system('echo '+str_)
del model

gc.collect()
from scipy.stats import rankdata

import numpy as np 

import pandas as pd

from IPython.display import clear_output

#from nltk.corpus import stopwords    

import nltk

#nltk.download('stopwords')

from tqdm import tqdm as tqdm 

tqdm.pandas()

import os

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense,Bidirectional, concatenate, Embedding, Input, CuDNNLSTM, Dropout, SpatialDropout1D,GlobalMaxPooling1D,GlobalAveragePooling1D, add

from keras.models import Sequential,Model

from keras import regularizers, optimizers

from keras.callbacks import LearningRateScheduler

from keras.regularizers import l1_l2

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints



print(os.listdir("../input"))
for keys in train_df.columns:

    if sum(train_df[keys].isna())!=0:

        train_df[keys] = train_df[keys].fillna(0)
class pre_processing():



    def __init__(self, data):

        self.data = data

        

    def clean_contraction(self, data):

        contraction_mapping = {

            "Trump's" : 'trump is',"'cause": 'because',',cause': 'because',';cause': 'because',"ain't": 'am not','ain,t': 'am not',

            'ain;t': 'am not','ain´t': 'am not','ain’t': 'am not',"aren't": 'are not',

            'aren,t': 'are not','aren;t': 'are not','aren´t': 'are not','aren’t': 'are not',"can't": 'cannot',"can't've": 'cannot have','can,t': 'cannot','can,t,ve': 'cannot have',

            'can;t': 'cannot','can;t;ve': 'cannot have',

            'can´t': 'cannot','can´t´ve': 'cannot have','can’t': 'cannot','can’t’ve': 'cannot have',

            "could've": 'could have','could,ve': 'could have','could;ve': 'could have',"couldn't": 'could not',"couldn't've": 'could not have','couldn,t': 'could not','couldn,t,ve': 'could not have','couldn;t': 'could not',

            'couldn;t;ve': 'could not have','couldn´t': 'could not',

            'couldn´t´ve': 'could not have','couldn’t': 'could not','couldn’t’ve': 'could not have','could´ve': 'could have',

            'could’ve': 'could have',"didn't": 'did not','didn,t': 'did not','didn;t': 'did not','didn´t': 'did not',

            'didn’t': 'did not',"doesn't": 'does not','doesn,t': 'does not','doesn;t': 'does not','doesn´t': 'does not',

            'doesn’t': 'does not',"don't": 'do not','don,t': 'do not','don;t': 'do not','don´t': 'do not','don’t': 'do not',

            "hadn't": 'had not',"hadn't've": 'had not have','hadn,t': 'had not','hadn,t,ve': 'had not have','hadn;t': 'had not',

            'hadn;t;ve': 'had not have','hadn´t': 'had not','hadn´t´ve': 'had not have','hadn’t': 'had not','hadn’t’ve': 'had not have',"hasn't": 'has not','hasn,t': 'has not','hasn;t': 'has not','hasn´t': 'has not','hasn’t': 'has not',

            "haven't": 'have not','haven,t': 'have not','haven;t': 'have not','haven´t': 'have not','haven’t': 'have not',"he'd": 'he would',

            "he'd've": 'he would have',"he'll": 'he will',

            "he's": 'he is','he,d': 'he would','he,d,ve': 'he would have','he,ll': 'he will','he,s': 'he is','he;d': 'he would',

            'he;d;ve': 'he would have','he;ll': 'he will','he;s': 'he is','he´d': 'he would','he´d´ve': 'he would have','he´ll': 'he will',

            'he´s': 'he is','he’d': 'he would','he’d’ve': 'he would have','he’ll': 'he will','he’s': 'he is',"how'd": 'how did',"how'll": 'how will',

            "how's": 'how is','how,d': 'how did','how,ll': 'how will','how,s': 'how is','how;d': 'how did','how;ll': 'how will',

            'how;s': 'how is','how´d': 'how did','how´ll': 'how will','how´s': 'how is','how’d': 'how did','how’ll': 'how will',

            'how’s': 'how is',"i'd": 'i would',"i'll": 'i will',"i'm": 'i am',"i've": 'i have','i,d': 'i would','i,ll': 'i will',

            'i,m': 'i am','i,ve': 'i have','i;d': 'i would','i;ll': 'i will','i;m': 'i am','i;ve': 'i have',"isn't": 'is not',

            'isn,t': 'is not','isn;t': 'is not','isn´t': 'is not','isn’t': 'is not',"it'd": 'it would',"it'll": 'it will',"It's":'it is',

            "it's": 'it is','it,d': 'it would','it,ll': 'it will','it,s': 'it is','it;d': 'it would','it;ll': 'it will','it;s': 'it is','it´d': 'it would','it´ll': 'it will','it´s': 'it is',

            'it’d': 'it would','it’ll': 'it will','it’s': 'it is',

            'i´d': 'i would','i´ll': 'i will','i´m': 'i am','i´ve': 'i have','i’d': 'i would','i’ll': 'i will','i’m': 'i am',

            'i’ve': 'i have',"let's": 'let us','let,s': 'let us','let;s': 'let us','let´s': 'let us',

            'let’s': 'let us',"ma'am": 'madam','ma,am': 'madam','ma;am': 'madam',"mayn't": 'may not','mayn,t': 'may not','mayn;t': 'may not',

            'mayn´t': 'may not','mayn’t': 'may not','ma´am': 'madam','ma’am': 'madam',"might've": 'might have','might,ve': 'might have','might;ve': 'might have',"mightn't": 'might not','mightn,t': 'might not','mightn;t': 'might not','mightn´t': 'might not',

            'mightn’t': 'might not','might´ve': 'might have','might’ve': 'might have',"must've": 'must have','must,ve': 'must have','must;ve': 'must have',

            "mustn't": 'must not','mustn,t': 'must not','mustn;t': 'must not','mustn´t': 'must not','mustn’t': 'must not','must´ve': 'must have',

            'must’ve': 'must have',"needn't": 'need not','needn,t': 'need not','needn;t': 'need not','needn´t': 'need not','needn’t': 'need not',"oughtn't": 'ought not','oughtn,t': 'ought not','oughtn;t': 'ought not',

            'oughtn´t': 'ought not','oughtn’t': 'ought not',"sha'n't": 'shall not','sha,n,t': 'shall not','sha;n;t': 'shall not',"shan't": 'shall not',

            'shan,t': 'shall not','shan;t': 'shall not','shan´t': 'shall not','shan’t': 'shall not','sha´n´t': 'shall not','sha’n’t': 'shall not',

            "she'd": 'she would',"she'll": 'she will',"she's": 'she is','she,d': 'she would','she,ll': 'she will',

            'she,s': 'she is','she;d': 'she would','she;ll': 'she will','she;s': 'she is','she´d': 'she would','she´ll': 'she will',

            'she´s': 'she is','she’d': 'she would','she’ll': 'she will','she’s': 'she is',"should've": 'should have','should,ve': 'should have','should;ve': 'should have',

            "shouldn't": 'should not','shouldn,t': 'should not','shouldn;t': 'should not','shouldn´t': 'should not','shouldn’t': 'should not','should´ve': 'should have',

            'should’ve': 'should have',"that'd": 'that would',"that's": 'that is','that,d': 'that would','that,s': 'that is','that;d': 'that would',

            'that;s': 'that is','that´d': 'that would','that´s': 'that is','that’d': 'that would','that’s': 'that is',"there'd": 'there had',

            "there's": 'there is','there,d': 'there had','there,s': 'there is','there;d': 'there had','there;s': 'there is',

            'there´d': 'there had','there´s': 'there is','there’d': 'there had','there’s': 'there is',

            "they'd": 'they would',"they'll": 'they will',"they're": 'they are',"they've": 'they have',

            'they,d': 'they would','they,ll': 'they will','they,re': 'they are','they,ve': 'they have','they;d': 'they would','they;ll': 'they will','they;re': 'they are',

            'they;ve': 'they have','they´d': 'they would','they´ll': 'they will','they´re': 'they are','they´ve': 'they have','they’d': 'they would','they’ll': 'they will',

            'they’re': 'they are','they’ve': 'they have',"wasn't": 'was not','wasn,t': 'was not','wasn;t': 'was not','wasn´t': 'was not',

            'wasn’t': 'was not',"we'd": 'we would',"we'll": 'we will',"we're": 'we are',"we've": 'we have','we,d': 'we would','we,ll': 'we will',

            'we,re': 'we are','we,ve': 'we have','we;d': 'we would','we;ll': 'we will','we;re': 'we are','we;ve': 'we have',

            "weren't": 'were not','weren,t': 'were not','weren;t': 'were not','weren´t': 'were not','weren’t': 'were not','we´d': 'we would','we´ll': 'we will',

            'we´re': 'we are','we´ve': 'we have','we’d': 'we would','we’ll': 'we will','we’re': 'we are','we’ve': 'we have',"what'll": 'what will',"what're": 'what are',"what's": 'what is',

            "what've": 'what have','what,ll': 'what will','what,re': 'what are','what,s': 'what is','what,ve': 'what have','what;ll': 'what will','what;re': 'what are',

            'what;s': 'what is','what;ve': 'what have','what´ll': 'what will',

            'what´re': 'what are','what´s': 'what is','what´ve': 'what have','what’ll': 'what will','what’re': 'what are','what’s': 'what is',

            'what’ve': 'what have',"where'd": 'where did',"where's": 'where is','where,d': 'where did','where,s': 'where is','where;d': 'where did',

            'where;s': 'where is','where´d': 'where did','where´s': 'where is','where’d': 'where did','where’s': 'where is',

            "who'll": 'who will',"who's": 'who is','who,ll': 'who will','who,s': 'who is','who;ll': 'who will','who;s': 'who is',

            'who´ll': 'who will','who´s': 'who is','who’ll': 'who will','who’s': 'who is',"won't": 'will not','won,t': 'will not','won;t': 'will not',

            'won´t': 'will not','won’t': 'will not',"wouldn't": 'would not','wouldn,t': 'would not','wouldn;t': 'would not','wouldn´t': 'would not',

            'wouldn’t': 'would not',"you'd": 'you would',"you'll": 'you will',"you're": 'you are','you,d': 'you would','you,ll': 'you will',

            'you,re': 'you are','you;d': 'you would','you;ll': 'you will',

            'you;re': 'you are','you´d': 'you would','you´ll': 'you will','you´re': 'you are','you’d': 'you would','you’ll': 'you will','you’re': 'you are',

            '´cause': 'because','’cause': 'because',"you've": "you have","could'nt": 'could not',

            "havn't": 'have not',"here’s": "here is",'i""m': 'i am',"i'am": 'i am',"i'l": "i will","i'v": 'i have',"wan't": 'want',"was'nt": "was not","who'd": "who would",

            "who're": "who are","who've": "who have","why'd": "why would","would've": "would have","y'all": "you all","y'know": "you know","you.i": "you i",

            "your'e": "you are","arn't": "are not","agains't": "against","c'mon": "common","doens't": "does not",'don""t': "do not","dosen't": "does not",

            "dosn't": "does not","shoudn't": "should not","that'll": "that will","there'll": "there will","there're": "there are",

            "this'll": "this all","u're": "you are", "ya'll": "you all","you'r": "you are","you’ve": "you have","d'int": "did not","did'nt": "did not","din't": "did not","dont't": "do not","gov't": "government",

            "i'ma": "i am","is'nt": "is not","‘I":'I',

            'ᴀɴᴅ':'and','ᴛʜᴇ':'the','ʜᴏᴍᴇ':'home','ᴜᴘ':'up','ʙʏ':'by','ᴀᴛ':'at','…and':'and','civilbeat':'civil beat',\

            'TrumpCare':'Trump care','Trumpcare':'Trump care', 'OBAMAcare':'Obama care','ᴄʜᴇᴄᴋ':'check','ғᴏʀ':'for','ᴛʜɪs':'this','ᴄᴏᴍᴘᴜᴛᴇʀ':'computer',\

            'ᴍᴏɴᴛʜ':'month','ᴡᴏʀᴋɪɴɢ':'working','ᴊᴏʙ':'job','ғʀᴏᴍ':'from','Sᴛᴀʀᴛ':'start','gubmit':'submit','CO₂':'carbon dioxide','ғɪʀsᴛ':'first',\

            'ᴇɴᴅ':'end','ᴄᴀɴ':'can','ʜᴀᴠᴇ':'have','ᴛᴏ':'to','ʟɪɴᴋ':'link','ᴏғ':'of','ʜᴏᴜʀʟʏ':'hourly','ᴡᴇᴇᴋ':'week','ᴇɴᴅ':'end','ᴇxᴛʀᴀ':'extra',\

            'Gʀᴇᴀᴛ':'great','sᴛᴜᴅᴇɴᴛs':'student','sᴛᴀʏ':'stay','ᴍᴏᴍs':'mother','ᴏʀ':'or','ᴀɴʏᴏɴᴇ':'anyone','ɴᴇᴇᴅɪɴɢ':'needing','ᴀɴ':'an','ɪɴᴄᴏᴍᴇ':'income',\

            'ʀᴇʟɪᴀʙʟᴇ':'reliable','ғɪʀsᴛ':'first','ʏᴏᴜʀ':'your','sɪɢɴɪɴɢ':'signing','ʙᴏᴛᴛᴏᴍ':'bottom','ғᴏʟʟᴏᴡɪɴɢ':'following','Mᴀᴋᴇ':'make',\

            'ᴄᴏɴɴᴇᴄᴛɪᴏɴ':'connection','ɪɴᴛᴇʀɴᴇᴛ':'internet','financialpost':'financial post', 'ʜaᴠᴇ':' have ', 'ᴄaɴ':' can ', 'Maᴋᴇ':' make ', 'ʀᴇʟɪaʙʟᴇ':' reliable ', 'ɴᴇᴇᴅ':' need ',

            'ᴏɴʟʏ':' only ', 'ᴇxᴛʀa':' extra ', 'aɴ':' an ', 'aɴʏᴏɴᴇ':' anyone ', 'sᴛaʏ':' stay ', 'Sᴛaʀᴛ':' start', 'SHOPO':'shop',

        }   

        

        def clean_contractions(text, mapping):

            specials = ["’", "‘", "´", "`"]

            for s in specials:

                text = text.replace(s, "'")

            text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

            return text

        

        data = data.progress_apply(lambda x: clean_contractions(x, contraction_mapping))

        return data

        

    def remove_punc(self, data):

        punct_mapping = {"_":" ", "`":" "}

        punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

        punct += '©^®` <→°€™› ♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√'

        def clean_special_chars(text, punct, mapping):

            for p in mapping:

                text = text.replace(p, mapping[p])    

            for p in punct:

                text = text.replace(p, f' {p} ')     

            return text



            data = data.astype(str)

            data = data.progress_apply(lambda x: clean_special_chars(x, punct, punct_mapping))

        return data

    

    def lower(self, data):

        data = data.astype(str).progress_apply(lambda x: x.lower())

        return data

    

    def remove_stop_words(self, data):

        #stop = set(stopwords.words('english'))

        stop = {'a',

         'about',

         'above',

         'after',

         'again',

         'against',

         'ain',

         'all',

         'am',

         'an',

         'and',

         'any',

         'are',

         'aren',

         "aren't",

         'as',

         'at',

         'be',

         'because',

         'been',

         'before',

         'being',

         'below',

         'between',

         'both',

         'but',

         'by',

         'can',

         'couldn',

         "couldn't",

         'd',

         'did',

         'didn',

         "didn't",

         'do',

         'does',

         'doesn',

         "doesn't",

         'doing',

         'don',

         "don't",

         'down',

         'during',

         'each',

         'few',

         'for',

         'from',

         'further',

         'had',

         'hadn',

         "hadn't",

         'has',

         'hasn',

         "hasn't",

         'have',

         'haven',

         "haven't",

         'having',

         'he',

         'her',

         'here',

         'hers',

         'herself',

         'him',

         'himself',

         'his',

         'how',

         'i',

         'if',

         'in',

         'into',

         'is',

         'isn',

         "isn't",

         'it',

         "it's",

         'its',

         'itself',

         'just',

         'll',

         'm',

         'ma',

         'me',

         'mightn',

         "mightn't",

         'more',

         'most',

         'mustn',

         "mustn't",

         'my',

         'myself',

         'needn',

         "needn't",

         'no',

         'nor',

         'not',

         'now',

         'o',

         'of',

         'off',

         'on',

         'once',

         'only',

         'or',

         'other',

         'our',

         'ours',

         'ourselves',

         'out',

         'over',

         'own',

         're',

         's',

         'same',

         'shan',

         "shan't",

         'she',

         "she's",

         'should',

         "should've",

         'shouldn',

         "shouldn't",

         'so',

         'some',

         'such',

         't',

         'than',

         'that',

         "that'll",

         'the',

         'their',

         'theirs',

         'them',

         'themselves',

         'then',

         'there',

         'these',

         'they',

         'this',

         'those',

         'through',

         'to',

         'too',

         'under',

         'until',

         'up',

         've',

         'very',

         'was',

         'wasn',

         "wasn't",

         'we',

         'were',

         'weren',

         "weren't",

         'what',

         'when',

         'where',

         'which',

         'while',

         'who',

         'whom',

         'why',

         'will',

         'with',

         'won',

         "won't",

         'wouldn',

         "wouldn't",

         'y',

         'you',

         "you'd",

         "you'll",

         "you're",

         "you've",

         'your',

         'yours',

         'yourself',

         'yourselves'}

        data = data.progress_apply(lambda x: (' ').join([i for i in x.split(' ') if i not in stop]))

        return data

    '''

    def replace_profanity(self, data):

        swear_words = [

       ]

        profanity1 = [i[:-2] for i in profanity]

        profanity = [i for i in profanity1 if i not in ['as', 'bust', 'caw','chin', 'mil', 'pant', 'shot', 'spun', 'bone','rap', 'dam', 'cip', 'cnu', 'coo', 'cu', 'fa', 'ga', 'horn', 'no', 'pro', 'se', 'ti', 'x', 'xx']]

        profanity_mapping = {}

        for i in profanity:

            profanity_mapping[i] = 'fuck'

            

        def rep_prof(text, mapping):

            text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

            return text                



        data = data.progress_apply(lambda x: rep_prof(x, profanity_mapping))

        return data

    '''

    

    def clean_spaces(self, data):

        data = data.progress_apply(lambda x: correct_spaces(x))

        return data

        

    def preprocess(self):

        print('correct contractions')

        self.data = self.clean_contraction(self.data)

        #print('remove punc')

        #self.data = self.remove_punc(self.data)

        #print('lower')

        #self.data = self.lower(self.data)

        #print('remove stop_word')

        #self.data = self.remove_stop_words(self.data)

        #print('replacing profanity')

        #self.data = self.replace_profanity(self.data)

        #print('cleaning spaces')

        #self.data = self.clean_spaces(self.data)

        print('done')

        return self.data



train_df['comment_text'] = pre_processing(train_df['comment_text']).preprocess()

test_df['comment_text'] = pre_processing(test_df['comment_text']).preprocess()



str_='lstm preprocess done'

os.system('echo '+str_)
IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TARGET_COLUMN = ['target']
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'



x_train = train_df['comment_text'].astype(str)

x_test = test_df['comment_text'].astype(str)



tokenizer = Tokenizer(filters=CHARS_TO_REMOVE, lower=False)

tokenizer.fit_on_texts(tqdm(list(x_train) + list(x_test)))



x_train = tokenizer.texts_to_sequences(tqdm(x_train))

x_test = tokenizer.texts_to_sequences(x_test)



MAX_LEN = 220



x_train = pad_sequences(tqdm(x_train), maxlen=MAX_LEN)

x_test = pad_sequences(x_test, maxlen=MAX_LEN)



y_train = train_df['target'].values

y_identity = (train_df[IDENTITY_COLUMNS].values>0.5).astype(int)

y_aux_train = train_df[AUX_COLUMNS].values



str_='lstm tokenizer done'

os.system('echo '+str_)
for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train_df[column] = np.where(train_df[column] >= 0.5, True, False)

    

sample_weights = np.ones(len(x_train), dtype=np.float32)

sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1).values

sample_weights += train_df[TARGET_COLUMN].mul((~train_df[IDENTITY_COLUMNS]).sum(axis=1), axis=0).values.ravel()

sample_weights += (~train_df[TARGET_COLUMN]).mul(train_df[IDENTITY_COLUMNS].sum(axis=1), axis=0).values.ravel()*5

sample_weights /= sample_weights.mean()
EMBEDDING_PATHS = ['../input/embeddings/crawl-300d-2M.vec',

                 '../input/embeddings/glove.840B.300d.txt']



def get_coefs(word, *arr):

    """

    Get word, word_embedding from a pretrained embedding file

    """

    return word, np.asarray(arr,dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():

        try:

            embedding_matrix[i] = embedding_index[word]

        except KeyError:

            pass

    return embedding_matrix
embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index, f) for f in EMBEDDING_PATHS], axis=-1)



str_='embedding matrix done'

os.system('echo '+str_)
del train_df, tokenizer

gc.collect()
BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

MAX_LEN = 220

EPOCHS=4
class Attention(Layer):

    def __init__(self, step_dim,

                 W_regularizer=None, b_regularizer=None,

                 W_constraint=None, b_constraint=None,

                 bias=True, **kwargs):

        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')



        self.W_regularizer = regularizers.get(W_regularizer)

        self.b_regularizer = regularizers.get(b_regularizer)



        self.W_constraint = constraints.get(W_constraint)

        self.b_constraint = constraints.get(b_constraint)



        self.bias = bias

        self.step_dim = step_dim

        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)



    def build(self, input_shape):

        assert len(input_shape) == 3



        self.W = self.add_weight((input_shape[-1],),

                                 initializer=self.init,

                                 name='{}_W'.format(self.name),

                                 regularizer=self.W_regularizer,

                                 constraint=self.W_constraint)

        self.features_dim = input_shape[-1]



        if self.bias:

            self.b = self.add_weight((input_shape[1],),

                                     initializer='zero',

                                     name='{}_b'.format(self.name),

                                     regularizer=self.b_regularizer,

                                     constraint=self.b_constraint)

        else:

            self.b = None



        self.built = True

        #super(Attention, self).build(input_shape) 



    def compute_mask(self, input, input_mask=None):

        return None



    def call(self, x, mask=None):

        features_dim = self.features_dim

        step_dim = self.step_dim



        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),

                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))



        if self.bias:

            eij += self.b



        eij = K.tanh(eij)



        a = K.exp(eij)



        if mask is not None:

            a *= K.cast(mask, K.floatx())



        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())



        a = K.expand_dims(a)

        weighted_input = x * a

        return K.sum(weighted_input, axis=1)



    def compute_output_shape(self, input_shape):

        return input_shape[0],  self.features_dim

    



def build_model_1(embedding_matrix, num_aux_targets):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    

    y = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    y = add([y, Dense(128*4, activation='relu')(y)])

    y = add([y, Dense(128*4, activation='relu')(y)])



    z = Attention(MAX_LEN)(x)

    z = add([z, Dense(128*2, activation='relu')(y)])

    z = add([z, Dense(128*2, activation='relu')(y)])    

    

    result = concatenate([y, z])

    result = Dense(1, activation='sigmoid')(y)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(y)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model







def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(MAX_LEN,))

    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    x = SpatialDropout1D(0.2)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)



    hidden = concatenate([

        GlobalMaxPooling1D()(x),

        GlobalAveragePooling1D()(x),

    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    result = Dense(1, activation='sigmoid')(hidden)

    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    

    model = Model(inputs=words, outputs=[result, aux_result])

    model.compile(loss='binary_crossentropy', optimizer='adam')



    return model
str_='training starting'

os.system('echo '+str_)



checkpoint_predictions = []

weights = []



model = build_model(embedding_matrix, y_aux_train.shape[-1])

for epoch in range(EPOCHS):

    str_='starting epoch: ' + str(epoch)

    os.system('echo '+str_)

    model.fit(

        x_train,

        [y_train, y_aux_train],

        batch_size=BATCH_SIZE,

        epochs=1,

        verbose=1,

        sample_weight=[sample_weights, np.ones_like(sample_weights)],

        callbacks = [

                LearningRateScheduler(lambda _: 1e-3*(0.55**epoch))

        ]

    )

    preds = model.predict(x_test, batch_size=2048)

    checkpoint_predictions.append(preds[0].flatten())

    weights.append(2*epoch)

    str_='finished epoch: ' + str(epoch)

    os.system('echo '+str_)
predictions_lstm = np.average(checkpoint_predictions, weights=weights, axis=0)
model.save('my_model.h5')
del model

gc.collect()
def ensemble_predictions(predictions, weights, type_="linear"):

    assert np.isclose(np.sum(weights), 1.0)

    if type_ == "linear":

        res = np.average(predictions, weights=weights, axis=0)

    elif type_ == "harmonic":

        res = np.average([1 / p for p in predictions], weights=weights, axis=0)

        return 1 / res

    elif type_ == "geometric":

        numerator = np.average(

            [np.log(p) for p in predictions], weights=weights, axis=0

        )

        res = np.exp(numerator / sum(weights))

        return res

    elif type_ == "rank":

        res = np.average([rankdata(p) for p in predictions], weights=weights, axis=0)

        return res / (len(res) + 1)

    return res
# model1 0.97301

#predictions = ensemble_predictions([predictions_bert, predictions_lstm], weights=[0.6, 0.4], type_='rank') 

# model

predictions = ensemble_predictions([predictions_bert, predictions_lstm], weights=[0.6, 0.4], type_='rank')

#predictions = ensemble_predictions([predictions_bert, predictions_lstm], weights=[0.666, 0.334], type_='rank')

#predictions = ensemble_predictions([predictions_bert, predictions_lstm], weights=[0.666, 0.334], type_='rank') 
'''submission_lstm = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': predictions_lstm

})

submission_bert = pd.DataFrame.from_dict({

    'id': test['id'],

    'prediction': predictions_bert

})

predictions = submission_lstm['prediction'].rank(pct=True)*0.4 + submission_bert['prediction'].rank(pct=True)*0.6'''
submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    'prediction': predictions

})

submission.to_csv('submission.csv', index=False)