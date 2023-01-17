# VERSION SUMMARY



# version 6: small bugfix

# version 5: added example for tokenization and prediction

# version 4: added apex install for mixed precision training 
import numpy as np 

import pandas as pd 

import os

import torch
!cd ../input/apex-master/apex-master/apex-master/ && pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
os.system('pip install --no-index --find-links="../input/pytorchpretrainedbert/" pytorch_pretrained_bert')
from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.modeling import BertModel
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import datetime

import pkg_resources

import seaborn as sns

import time

import scipy.stats as stats

import gc

import re

import operator 

import sys

from sklearn import metrics

from sklearn import model_selection

import torch

import torch.nn as nn

import torch.utils.data

import torch.nn.functional as F

from nltk.stem import PorterStemmer

from sklearn.metrics import roc_auc_score

%load_ext autoreload

%autoreload 2

%matplotlib inline

from tqdm import tqdm, tqdm_notebook

import os

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import warnings

warnings.filterwarnings(action='once')

import pickle

from apex import amp

import shutil

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
device=torch.device('cuda')

MAX_SEQUENCE_LENGTH = 220

SEED = 1234

EPOCHS = 5

Data_dir="../input/jigsaw-unintended-bias-in-toxicity-classification"

Input_dir = "../input"

WORK_DIR = "working_dir/"

TOXICITY_COLUMN = 'target'

run_type = 'val'
import os

'    '.join(os.listdir(Data_dir))




from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification,BertAdam
WORK_DIR = "../input/torch-bert-model"

os.listdir(WORK_DIR)



from pytorch_pretrained_bert import BertConfig

bert_config = BertConfig(WORK_DIR+"/bert_config.json")
def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm_notebook(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    print(longer)

    return np.array(all_tokens)
from sklearn.model_selection import KFold



if run_type != 'tst':

    train_dfall = pd.read_csv(os.path.join(Data_dir, 'train.csv'))

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    trn_ids, val_ids = next(cv.split(train_dfall))

else:

    train_df = pd.read_csv(os.path.join(Data_dir, 'train.csv'))

test_df = pd.read_csv(os.path.join(Data_dir, 'test.csv'))



print(train_df.shape)


from nltk.tokenize.treebank import TreebankWordTokenizer

symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'

symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'

tokenizer = TreebankWordTokenizer()

isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}

remove_dict = {ord(c):f'' for c in symbols_to_delete}



def words(text): return re.findall(r'\w+', text.lower())



def get_embedding_words(path):

    with open(path) as f:

        return set([line.strip().split(' ')[0] for line in f])



def handle_punctuation(x):

    x = x.translate(remove_dict)

    x = x.translate(isolate_dict)

    return x



def handle_contractions(x):

    x = tokenizer.tokenize(x)

    return x



def fix_quote(x):

    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]

    x = ' '.join(x)

    return x
def clean_contractions(text):

    mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text



def preprocess(x):

    x = handle_punctuation(x)

    x = handle_contractions(x)

    x = fix_quote(x)

    x = clean_contractions(x)

    return x
import multiprocessing as mp

pool = mp.Pool(processes= 2)

if run_type != 'tst':

    train_dfall['comment_text'] = pool.map(preprocess, train_dfall['comment_text'].to_list())

else:

    train_df['comment_text'] = pool.map(preprocess, train_df['comment_text'].to_list())

test_df['comment_text'] = pool.map(preprocess, test_df['comment_text'].to_list())

pool.close
%%time

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=False)

print('loaded %d records' % len(train_dfall))



# Make sure all comment_text values are strings

train_dfall['comment_text'] = train_dfall['comment_text'].astype(str) 



sequences = convert_lines(train_dfall["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)



train_dfall=train_dfall.fillna(0)

# List all identities

identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

y_columns=['target']



train_dfall = train_dfall.drop(['comment_text'],axis=1)

# convert target to 0,1

train_dfall['target']=(train_dfall['target']>=0.5).astype(float)

X = sequences[trn_ids]                

y = train_dfall[y_columns].values[trn_ids]

X_val = sequences[val_ids]                

y_val = train_dfall[y_columns].values[val_ids]
train_df, valid_df = train_dfall.loc[trn_ids], train_dfall.loc[val_ids]

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))

# Set up a folder for each epoch

for i in range(EPOCHS):

    chkpt_dir = WORK_DIR+'{}'.format(i)

    os.mkdir(chkpt_dir)

    shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', chkpt_dir + '/bert_config.json')
EPOCHS = 5

lr=2e-5

batch_size = 32

accumulation_steps=1

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

model = BertForSequenceClassification.from_pretrained(WORK_DIR,cache_dir=None,num_labels=len(y_columns))

model.zero_grad()

model = model.to(device)

param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}

    ]

train = train_dataset



num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

optimizer = BertAdam(optimizer_grouped_parameters,

                     lr=lr,

                     warmup=0.05,

                     t_total=num_train_optimization_steps)



model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

model=model.train()
tq = tqdm_notebook(range(EPOCHS))

for epoch in tq:

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    avg_loss = 0.

    avg_accuracy = 0.

    lossf=None

    tk0 = tqdm_notebook(enumerate(train_loader),total=len(train_loader),leave=False)

    for i,(x_batch, y_batch) in tk0:

        optimizer.zero_grad()

        y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)

        loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))

        with amp.scale_loss(loss, optimizer) as scaled_loss:

            scaled_loss.backward()

        if (i+1) % accumulation_steps == 0:             # Wait for several backward steps

            optimizer.step()                            # Now we can do an optimizer step

            optimizer.zero_grad()

        if lossf:

            lossf = 0.98*lossf+0.02*loss.item()

        else:

            lossf = loss.item()

        tk0.set_postfix(loss = lossf)

        avg_loss += loss.item() / len(train_loader)

        avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)

    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)

    print('Loss {} // Accuracy // Epoch {}'.format(avg_loss, avg_accuracy, epoch))

    output_model_file = "working_dir/{}/pytorch_model.bin".format(epoch)

    torch.save(model.state_dict(), output_model_file)

# Run validation

# The following 2 lines are not needed but show how to download the model for prediction

model = BertForSequenceClassification(bert_config,num_labels=len(y_columns))

model.load_state_dict(torch.load(output_model_file ))

model.to(device)

for param in model.parameters():

    param.requires_grad=False

model.eval()

valid_preds = np.zeros((len(X_val)))

valid = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long))

valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)



tk0 = tqdm_notebook(valid_loader)

for i,(x_batch,)  in enumerate(tk0):

    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)

    valid_preds[i*32:(i+1)*32]=pred[:,0].detach().cpu().squeeze().numpy()

from sklearn.metrics import roc_auc_score

print('AUC score : {:.5f}'.format(roc_auc_score((y_val>0.5).astype(int), valid_preds)))
print('AUC score : {:.5f}'.format(roc_auc_score((y_val>0.5).astype(int), valid_preds)))
