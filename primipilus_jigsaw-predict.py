debug = False
import time
start_time = time.time()
import sys
package_dir = "../input/pytorchpretrainedberthaqishen/pytorch-pretrained-bert/pytorch-pretrained-BERT/"
sys.path = [package_dir] + sys.path
import os
import pickle
import argparse
import multiprocessing
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling_gpt2 import GPT2Config, GPT2Model, GPT2PreTrainedModel
from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from pytorch_pretrained_bert.modeling_xlnet import XLNetConfig, XLNetModel, XLNetPreTrainedModel
from pytorch_pretrained_bert.tokenization_xlnet import XLNetTokenizer

from scipy.stats import rankdata
from nltk.tokenize.treebank import TreebankWordTokenizer
Ttokenizer = TreebankWordTokenizer()

symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'
symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'
CONTRACTION_MAPPING = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c):f'' for c in symbols_to_delete}

from tqdm import tqdm
import warnings
import traceback

warnings.filterwarnings(action='once')
device = torch.device('cuda')

print('Import done! Time past %.2f secs' % (time.time() - start_time))
# Pandas multiprocessing
def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))
class BertForJigsaw(BertPreTrainedModel):

    def __init__(self, config, out_dim=7):
        super(BertForJigsaw, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, out_dim)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits
    
class GPT2ClassificationHeadModel(GPT2PreTrainedModel):

    def __init__(self, config, clf_dropout=0.4, out_dim=8):
        super(GPT2ClassificationHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.linear = nn.Linear(config.n_embd * 2, out_dim)

        nn.init.normal_(self.linear.weight, std = 0.02)
        nn.init.normal_(self.linear.bias, 0)
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens):
        pass

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None, **kwargs):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        if isinstance(hidden_states, list):
            hidden_states = hidden_states[-1]
        avg_pool = torch.mean(hidden_states, 1)
        max_pool, _ = torch.max(hidden_states, 1)
        h_conc = torch.cat((avg_pool, max_pool), 1)
        return self.linear(h_conc)
    
    
class XLNetForJigSaw(XLNetPreTrainedModel):
    def __init__(self, config, out_dim):
        
        super(XLNetForJigSaw, self).__init__(config)
        self.attn_type = config.attn_type
        self.same_length = config.same_length
        self.summary_type = "last"

        self.transformer = XLNetModel(config, output_attentions=False, keep_multihead_output=False)
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()
        self.linear = nn.Linear(config.d_model, out_dim, bias=True)
        self.apply(self.init_xlnet_weights)

    def forward(self, input_ids, seg_id=None, input_mask=None,
                mems=None, perm_mask=None, target_mapping=None, inp_q=None,
                target=None, output_all_encoded_layers=True, head_mask=None, **kargs):

        output, hidden_states, new_mems = self.transformer(input_ids, seg_id, input_mask,
                                            mems, perm_mask, target_mapping, inp_q,
                                            output_all_encoded_layers, head_mask)
        first_token_tensor = output[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)

        return self.linear(pooled_output)
def convert_line(row, max_seq_length, tokenizer, model_name='bert'):
    guid = row['id']
    text_a = row['comment_text']

    if 'label' in row.keys():
        label = row['label']
    else:
        label = None

    tokens_a = tokenizer.tokenize(text_a)

    if 'bert' in model_name:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
    else:
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
        tokens = tokens_a
        input_ids = tokenizer.convert_tokens_to_ids(tokens_a)

    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, input_mask, segment_ids, label
def preprocess(x):
    for k, v in CONTRACTION_MAPPING.items():
        x = x.replace(' %s ' % k, ' %s ' % v)
    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    x = Ttokenizer.tokenize(x)
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x
def get_input_data(test_data):
    all_input_ids, all_input_mask,  all_segment_ids, all_label_ids = [], [], [], []
    for i, (input_ids, input_mask, segment_ids, label) in test_data.items():
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_label_ids.append(label)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
    try:
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.float32)
    except:
        pass
    
    return all_input_ids, all_input_mask,  all_segment_ids, all_label_ids

print('Def functions done! Time past %.2f secs' % (time.time() - start_time))
print('Loading data...')
df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
if debug:
    df = df.loc[:25]
test_ids = df['id'].tolist()
df['comment_text'] = df['comment_text'].astype(str)
print('Preprocessing...')
df['comment_text'] = apply_by_multiprocessing(df['comment_text'], preprocess, workers=4)
print('Done! Time past %.2f secs' % (time.time() - start_time))
try:
    model_dir = '../input/jigsawmodels/bert_small_v2_29bin_naus_300seq/bert_small_v2_29bin_naus_300seq/'
    max_seq_length = 300
    short_length = 100
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    print('Converting data to sequences...')
    test_data = apply_by_multiprocessing(df, convert_line, axis=1, max_seq_length=max_seq_length, tokenizer=tokenizer, model_name='bert', workers=4)  # takes 2 mins
    all_input_ids, all_input_mask,  all_segment_ids, all_label_ids = get_input_data(test_data)
    long_idx = (all_input_ids[:, short_length-max_seq_length:].sum(1) > 0).nonzero().squeeze().numpy()
    short_idx = (all_input_ids[:, short_length-max_seq_length:].sum(1) == 0).nonzero().squeeze().numpy()
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    # Load a trained model and vocabulary that you have fine-tuned
    print('Loading model from %s ...' % model_dir)
    bert_config = BertConfig(os.path.join(model_dir, 'config.json'))
    model = BertForJigsaw(bert_config, out_dim=29+6)  # with NAUX
    model.load_state_dict(torch.load('%s/pytorch_model.bin' % model_dir))
    model.to(device)
    model.eval()
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    print('Predicting model Bert Small...')
    predictions_bert_small = np.zeros(df.shape[0])
    with torch.no_grad():
        for i, idx in enumerate([short_idx, long_idx]):
            test_data = TensorDataset(all_input_ids[idx], all_input_mask[idx], all_segment_ids[idx]) if i == 1 else \
                        TensorDataset(all_input_ids[idx, :short_length], all_input_mask[idx, :short_length], all_segment_ids[idx, :short_length])
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=128)

            pred = []
            for input_ids, input_mask, segment_ids in tqdm(test_dataloader):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                logits = model(input_ids, segment_ids, input_mask, labels=None)
                logits = torch.sigmoid(logits).detach().cpu().numpy()
                pred.append(logits[:, :-6])  # remove NAUX
            predictions_bert_small[idx] = np.vstack(pred).mean(1)

    print('Done! Time past %.2f secs' % (time.time() - start_time))
except:
    print('Something wrong with Bert Small.')
    traceback.print_exc()
len(predictions_bert_small)
predictions_bert_small[:10]
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Load and show train data
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
train_df.sample()
train_df.target.hist(bins=30)
train_df.severe_toxicity.hist(bins=30)
# What problem do we address? 
# Synonyms proposal.
# What cases do we have? 
# 1. User looks for synonism for offensive words. In this case we shouldn't provide them. Basically, it already works.
# 2. Variety/Deduplication/Synonyms proposal – proposed words in the given context shoudn't be neither offensive nor sensitive.
# 2.1 Some of such words could be neutral by itself but offensive in some context. This couldn't be checked via dictionary only and we should address exact this problem.
# 3. Let's find some examples that falls under this denition. Let's focus on short examples. 
# 4. Then check if they could be converted back to normal by replacing some words certain words. 
# Show statistic on sentence length
train_df['comment_text_len'] = train_df.comment_text.apply(lambda t: len(t))
train_df['comment_text_len'].hist(bins=100, figsize=(10,5))
max_text_len = 160
short_text_train_df = train_df[train_df.comment_text_len <= max_text_len]
len(train_df)
len(short_text_train_df)
# Show sample of both severe and not 
# Show prediction on portion of this data
# For now, let's focus on certain category – black. black >= 0.5, target >= 0.5
black_df = short_text_train_df[(short_text_train_df.target >= .5) & (short_text_train_df.black >= .5)]
len(black_df)
for t in black_df[:100].comment_text: print(t); print("\n *** \n"); 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
try:
    model_dir = '../input/jigsawmodels/bert_large_v2_99bin_250seq/bert_large_v2_99bin_250seq/'
    max_seq_length = 250
    short_length = 100
    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    print('Converting data to sequences...')
    test_data = apply_by_multiprocessing(df, convert_line, axis=1, max_seq_length=max_seq_length, tokenizer=tokenizer, model_name='bert', workers=4)  # takes 2 mins
    all_input_ids, all_input_mask,  all_segment_ids, all_label_ids = get_input_data(test_data)
    long_idx = (all_input_ids[:, short_length-max_seq_length:].sum(1) > 0).nonzero().squeeze().numpy()
    short_idx = (all_input_ids[:, short_length-max_seq_length:].sum(1) == 0).nonzero().squeeze().numpy()
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    # Load a trained model and vocabulary that you have fine-tuned
    print('Loading model from %s ...' % model_dir)
    bert_config = BertConfig(os.path.join(model_dir, 'config.json'))
    model = BertForJigsaw(bert_config, out_dim=99)
    model.load_state_dict(torch.load('%s/pytorch_model.bin' % model_dir))
    model.to(device)
    model.eval()
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    print('Predicting model Bert Large')
    predictions_bert_large = np.zeros(df.shape[0])
    with torch.no_grad():
        for i, idx in enumerate([short_idx, long_idx]):
            test_data = TensorDataset(all_input_ids[idx], all_input_mask[idx], all_segment_ids[idx]) if i == 1 else \
                        TensorDataset(all_input_ids[idx, :short_length], all_input_mask[idx, :short_length], all_segment_ids[idx, :short_length])
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=64)

            pred = []
            for input_ids, input_mask, segment_ids in tqdm(test_dataloader):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)

                logits = model(input_ids, segment_ids, input_mask, labels=None)
                logits = torch.sigmoid(logits).detach().cpu().numpy()
                pred.append(logits)
            predictions_bert_large[idx] = np.vstack(pred).mean(1)

    print('Done! Time past %.2f secs' % (time.time() - start_time))
except:
    print('Something wrong with Bert Large.')
    traceback.print_exc()
try:
    model_dir = '../input/jigsawmodels/xlnet_large_9bin_220seq/xlnet_large_9bin_220seq/'
    max_seq_length = 220
    tokenizer = XLNetTokenizer.from_pretrained(model_dir)
    print('Converting to sequences...')
    test_data = []
    for i, row in tqdm(df.iterrows()):
        test_data.append(convert_line(
            row,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            model_name='xlnet'
        ))
    test_data = pd.Series(test_data)
    all_input_ids, all_input_mask,  all_segment_ids, all_label_ids = get_input_data(test_data)
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

    # Load a trained model and vocabulary that you have fine-tuned
    print('Loading model from %s ...' % model_dir)
    xlnet_config = XLNetConfig(os.path.join(model_dir, 'config.json'))
    model = XLNetForJigSaw(xlnet_config, out_dim=9)
    model.load_state_dict(torch.load('%s/pytorch_model.bin' % model_dir))
    model.to(device)
    model.eval()
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    print('Predicting model xlnet_large_v2_9bin_220seq')
    predictions_xlnet = []
    with torch.no_grad():

        for input_ids, input_mask, segment_ids in tqdm(test_dataloader):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            logits = model(input_ids)
            logits = torch.sigmoid(logits).detach().cpu().numpy()
            predictions_xlnet.append(logits)

    predictions_xlnet = np.vstack(predictions_xlnet).mean(1)
    print('Done! Time past %.2f secs' % (time.time() - start_time))
except:
    print('Something wrong with xlnet_large_v2_9bin_220seq.')
    traceback.print_exc()
try:
    model_dir = '../input/jigsawmodels/gpt2_29bin_350seq_aus/gpt2_29bin_350seq_aus/'
    max_seq_length = 350
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    print('Converting data to sequences...')
    test_data = apply_by_multiprocessing(df, convert_line, axis=1, max_seq_length=max_seq_length, tokenizer=tokenizer, model_name='gpt2', workers=4)  # takes 2 mins
    all_input_ids, all_input_mask,  all_segment_ids, all_label_ids = get_input_data(test_data)
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=64)

    # Load a trained model and vocabulary that you have fine-tuned
    print('Loading model from %s ...' % model_dir)
    gpt2_config = GPT2Config(os.path.join(model_dir, 'config.json'))
    model = GPT2ClassificationHeadModel(gpt2_config, out_dim=29+6)  # with NAUX
    model.load_state_dict(torch.load('%s/pytorch_model.bin' % model_dir))
    model.to(device)
    model.eval()
    print('Done! Time past %.2f secs' % (time.time() - start_time))

    print('Predicting model GPT2...')
    predictions_gpt2 = []
    with torch.no_grad():

        for input_ids, input_mask, segment_ids in tqdm(test_dataloader):
            
            if (time.time() - start_time) > 7140:
                print('STOP GPT2 FOR THE TIME SACK.')
                break

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            logits = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )
            logits = torch.sigmoid(logits).detach().cpu().numpy()
            predictions_gpt2.append(logits[:, :-6])  # remove NAUX

    predictions_gpt2 = np.vstack(predictions_gpt2).mean(1)
    print('Done! Time past %.2f secs' % (time.time() - start_time))
except:
    print('Something wrong with GPT2:')
    traceback.print_exc()
    print('But the program is still running')
predictions = np.zeros(df.shape[0])

try:
    predictions += predictions_bert_small * 0.9
except: pass

try:
    predictions += predictions_bert_large * 1.0 
except: pass

try:
    predictions += predictions_gpt2 * 0.7
except: pass

try:
    predictions += predictions_xlnet * 0.8
except: pass
df_prediction = pd.DataFrame({
    'id': test_ids,
    'prediction': predictions,
})
df_prediction.to_csv('./submission.csv', index=False)
print('Output done! Time past %.2f secs' % (time.time() - start_time))
print(df_prediction.head(25))