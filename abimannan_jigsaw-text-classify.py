import sys

package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.append(package_dir)
%reload_ext autoreload

%autoreload 2

%matplotlib inline



import fastai

from fastai.train import Learner

from fastai.train import DataBunch

from fastai.callbacks import *

from fastai.basic_data import DatasetType

import fastprogress

from fastprogress import force_console_behavior

import numpy as np

from pprint import pprint

import pandas as pd

import os

import time

import gc

import random

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

from keras.preprocessing import text, sequence

import torch

from torch import nn

from torch.utils import data

from torch.nn import functional as F



from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import torch.utils.data

from tqdm import tqdm

import warnings

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam

from pytorch_pretrained_bert import BertConfig

from nltk.tokenize.treebank import TreebankWordTokenizer



from gensim.models import KeyedVectors
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

    return np.array(all_tokens)



def is_interactive():

    return 'SHLVL' not in os.environ



def seed_everything(seed=123):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    #with open(path,'rb') as f:

    emb_arr = KeyedVectors.load(path)

    return emb_arr



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((max_features + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        if i <= max_features:

            try:

                embedding_matrix[i] = embedding_index[word]

            except KeyError:

                try:

                    embedding_matrix[i] = embedding_index[word.lower()]

                except KeyError:

                    try:

                        embedding_matrix[i] = embedding_index[word.title()]

                    except KeyError:

                        unknown_words.append(word)

    return embedding_matrix, unknown_words



def sigmoid(x):

    return 1 / (1 + np.exp(-x))



class SpatialDropout(nn.Dropout2d):

    def forward(self, x):

        x = x.unsqueeze(2)    # (N, T, 1, K)

        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)

        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked

        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)

        x = x.squeeze(2)  # (N, T, K)

        return x



def train_model(learn,test,output_dim,lr=0.0001,

                batch_size=512, n_epochs=4,

                enable_checkpoint_ensemble=True):

    

    all_test_preds = []

    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    n = len(learn.data.train_dl)

    phases = [(TrainingPhase(n).schedule_hp('lr', lr * (0.6**(i)))) for i in range(n_epochs)]

    sched = GeneralScheduler(learn, phases)

    learn.callbacks.append(sched)

    for epoch in range(n_epochs):

        learn.fit(1)

        test_preds = np.zeros((len(test), output_dim))    

        for i, x_batch in enumerate(test_loader):

            X = x_batch[0].cuda()

            y_pred = sigmoid(learn.model(X).detach().cpu().numpy())

            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred



        all_test_preds.append(test_preds)





    if enable_checkpoint_ensemble:

        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)    

    else:

        test_preds = all_test_preds[-1]

        

    return test_preds



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



def preprocess(x):

    x = handle_punctuation(x)

    x = handle_contractions(x)

    x = fix_quote(x)

    return x



class SequenceBucketCollator():

    def __init__(self, choose_length, sequence_index, length_index, label_index=None):

        self.choose_length = choose_length

        self.sequence_index = sequence_index

        self.length_index = length_index

        self.label_index = label_index

        

    def __call__(self, batch):

        batch = [torch.stack(x) for x in list(zip(*batch))]

        

        sequences = batch[self.sequence_index]

        lengths = batch[self.length_index]

        

        length = self.choose_length(lengths)

        mask = torch.arange(start=maxlen, end=0, step=-1) < length

        padded_sequences = sequences[:, mask]

        

        batch[self.sequence_index] = padded_sequences

        

        if self.label_index is not None:

            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]

    

        return batch

    

class NeuralNet(nn.Module):

    def __init__(self, embedding_matrix, num_aux_targets):

        super(NeuralNet, self).__init__()

        embed_size = embedding_matrix.shape[1]

        

        self.embedding = nn.Embedding(max_features, embed_size)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        self.embedding.weight.requires_grad = False

        self.embedding_dropout = SpatialDropout(0.3)

        

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

    

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)

        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)

        

    def forward(self, x, lengths=None):

        h_embedding = self.embedding(x.long())

        h_embedding = self.embedding_dropout(h_embedding)

        

        h_lstm1, _ = self.lstm1(h_embedding)

        h_lstm2, _ = self.lstm2(h_lstm1)

        

        # global average pooling

        avg_pool = torch.mean(h_lstm2, 1)

        # global max pooling

        max_pool, _ = torch.max(h_lstm2, 1)

        

        h_conc = torch.cat((max_pool, avg_pool), 1)

        h_conc_linear1  = F.relu(self.linear1(h_conc))

        h_conc_linear2  = F.relu(self.linear2(h_conc))

        

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        

        result = self.linear_out(hidden)

        aux_result = self.linear_aux_out(hidden)

        out = torch.cat([result, aux_result], 1)

        

        return out

    

def custom_loss(data, targets):

    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])

    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])

    return (bce_loss_1 * loss_weight) + bce_loss_2



def reduce_mem_usage(df):

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
warnings.filterwarnings(action='once')

device = torch.device('cuda')

MAX_SEQUENCE_LENGTH = 300

SEED = 1234

BATCH_SIZE = 512

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'

np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True

bert_config = BertConfig('../input/bert-inference/bert/bert_config.json')

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)



tqdm.pandas()

CRAWL_EMBEDDING_PATH = '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim'

GLOVE_EMBEDDING_PATH = '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'

NUM_MODELS = 2

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

MAX_LEN = 220

if not is_interactive():

    def nop(it, *a, **k):

        return it



    tqdm = nop



    fastprogress.fastprogress.NO_BAR = True

    master_bar, progress_bar = force_console_behavior()

    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar



seed_everything()
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

test_df['comment_text'] = test_df['comment_text'].astype(str) 

X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)
model = BertForSequenceClassification(bert_config, num_labels=1)

model.load_state_dict(torch.load("../input/bert-inference/bert/bert_pytorch.bin"))

model.to(device)

for param in model.parameters():

    param.requires_grad = False

model.eval()
test_preds = np.zeros((len(X_test)))

test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))

test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=False)

tk0 = tqdm(test_loader)

for i, (x_batch,) in enumerate(tk0):

    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

    test_preds[i * 512:(i + 1) * 512] = pred[:, 0].detach().cpu().squeeze().numpy()



test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()
submission_bert = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': test_pred

})
train_df = reduce_mem_usage(pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'))
symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'

symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'
tokenizer = TreebankWordTokenizer()



isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}

remove_dict = {ord(c):f'' for c in symbols_to_delete}
x_train = train_df['comment_text'].progress_apply(lambda x:preprocess(x))

y_aux_train = train_df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

x_test = test_df['comment_text'].progress_apply(lambda x:preprocess(x))



identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

# Overall

weights = np.ones((len(x_train),)) / 4

# Subgroup

weights += (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4

# Background Positive, Subgroup Negative

weights += (( (train_df['target'].values>=0.5).astype(bool).astype(np.int) +

   (train_df[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

# Background Negative, Subgroup Positive

weights += (( (train_df['target'].values<0.5).astype(bool).astype(np.int) +

   (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

loss_weight = 1.0 / weights.mean()



y_train = np.vstack([(train_df['target'].values>=0.5).astype(np.int),weights]).T



max_features = 410047
tokenizer = text.Tokenizer(num_words = max_features, filters='',lower=False)
tokenizer.fit_on_texts(list(x_train) + list(x_test))



crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)

print('n unknown words (crawl): ', len(unknown_words_crawl))



glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)

print('n unknown words (glove): ', len(unknown_words_glove))



max_features = max_features or len(tokenizer.word_index) + 1

max_features



embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

embedding_matrix.shape



del crawl_matrix

del glove_matrix

gc.collect()



y_train_torch = torch.tensor(np.hstack([y_train, y_aux_train]), dtype=torch.float32)

x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)
lengths = torch.from_numpy(np.array([len(x) for x in x_train]))

 

maxlen = 300

x_train_padded = torch.from_numpy(sequence.pad_sequences(x_train, maxlen=maxlen))
test_lengths = torch.from_numpy(np.array([len(x) for x in x_test]))



x_test_padded = torch.from_numpy(sequence.pad_sequences(x_test, maxlen=maxlen))

batch_size = 512

test_dataset = data.TensorDataset(x_test_padded, test_lengths)

train_dataset = data.TensorDataset(x_train_padded, lengths, y_train_torch)

valid_dataset = data.Subset(train_dataset, indices=[0, 1])



train_collator = SequenceBucketCollator(lambda lenghts: lenghts.max(), 

                                        sequence_index=0, 

                                        length_index=1, 

                                        label_index=2)

test_collator = SequenceBucketCollator(lambda lenghts: lenghts.max(), sequence_index=0, length_index=1)



train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator)

valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collator)

test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collator)



databunch = DataBunch(train_dl=train_loader, valid_dl=valid_loader, collate_fn=train_collator)

all_test_preds = []



for model_idx in range(NUM_MODELS):

    print('Model ', model_idx)

    seed_everything(1 + model_idx)

    model = NeuralNet(embedding_matrix, y_aux_train.shape[-1])

    learn = Learner(databunch, model, loss_func=custom_loss)

    test_preds = train_model(learn,test_dataset,output_dim=7)    

    all_test_preds.append(test_preds)
submission_lstm = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': np.mean(all_test_preds, axis=0)[:, 0]

})
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

submission['prediction'] = ((submission_bert.prediction + submission_lstm.prediction)/2 + submission_lstm.prediction)/2

submission.to_csv('submission.csv', index=False)