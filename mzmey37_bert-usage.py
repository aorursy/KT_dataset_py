# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import sys

package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"

sys.path.append(package_dir)
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import torch.utils.data

import numpy as np

import pandas as pd

from tqdm import tqdm

import os

import warnings

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertConfig

from torch.utils.data import Dataset, DataLoader

from torch import nn





warnings.filterwarnings(action='once')

device = torch.device('cuda')
TARGET_COL = 'target'

TEXT_COL = 'comment_text'

DF_TRAIN_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv'

DF_TEST_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv'



UNCASED_BERT_CONFIG = BertConfig('../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_config.json')



FIRST_MODEL_PATH = '../input/bert-accumulate-500-weighted/bert_accumulate_500_weighted.bin'

SECOND_MODEL_PATH = '../input/bert-accumulate-250/bert_accumulate_250.bin'



UNCASED_DIR_PATH = '../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'



uncased_tokenizer = BertTokenizer.from_pretrained(UNCASED_DIR_PATH, cache_dir=None, do_lower_case=True)
replacings = [

    ('tRump', 'Trump'),

    ("gov't", 'goverment'),

    ('Brexit', 'Britain leaving EU'),

    ('theglobeandmail', 'the globe and mail'),

    ('Drumpf', 'Donald Trump'),

    ('SB91', 'Senate Bill 91'),

    ("Gov't", 'goverment'),

    ('Trumpcare', 'AHCA'),

    ('Trumpism', 'the policies advocated by Donald Trump'),

    ('bigly', 'on a large scale'),

    ("y'all", 'you all'),

    ('Auwe', 'darn'),

    ('Trumpian', 'bombast egotism lies'),

    ('Trumpsters', 'Trump ideas'),

    ('Vinis', 'vinis'),

    ('Saullie', 'Evans'),

    ('Koncerned', 'concerned'),

    ('SJWs', 'SJW'),

    ('TFWs', 'TFW'),

    ('RangerMC', 'Ranger MC'),

    ('civilbeat', 'civil beat'),

    ('BCLibs', 'BC Libraries'),

    ('garycrum', 'Gary Crum'),

    ('Trudope', 'prime minister of Canada'),

    ('Daesh', 'Islamic State'),

    ("Qur'an", 'Quran'),

    ('wiliki', 'wiki'),

    ('OBAMAcare', 'Obamacare'),

    ('cashapp24', 'cash app'),

    ('Finicum', 'LaVoy'),

    ('Donkel', 'dark'),

    ('Trumpkins', 'Donald Trump meme'),

    ('Cheetolini', 'Trump'),

    ('brotherIn', 'brother in'),

    ('Trudeaus', 'Trudeau'),

    ('Beyak', 'Lynn'),

    ('dailycaller', 'daily caller'),

    ('Layla4', 'Layla'),

    ('Tridentinus', 'mushroom'),

    ('Ontariowe', 'Ontario WE'),

    ('washingtontimes', 'Washington Times')

]





symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'

symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'



from nltk.tokenize.treebank import TreebankWordTokenizer

tree_tokenizer = TreebankWordTokenizer()





isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}

remove_dict = {ord(c):f'' for c in symbols_to_delete}





def handle_punctuation(x):

    x = x.translate(remove_dict)

    x = x.translate(isolate_dict)

    return x



def handle_contractions(x):

    x = tree_tokenizer.tokenize(x)

    return x



def fix_quote(x):

    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]

    x = ' '.join(x)

    return x



def preprocess(x):

    x = handle_punctuation(x)

    x = handle_contractions(x)

    x = fix_quote(x)

    for was, became in replacings:

        x = x.replace(was, became)

    return x





class Collator(object):

    def __init__(self, device='cpu', maxlen=500):

        self.maxlen = maxlen

        self.device = device



    def __call__(self, batch):

        xs = batch

        maxlen = min(max(map(len, xs)), self.maxlen)

        x = np.zeros([len(xs), maxlen], dtype=np.int64)

        mask = np.zeros([len(xs), maxlen], dtype=np.int64)

        for i in range(len(xs)):

            if len(xs[i]) > 500:

                xs[i] = xs[i][:100] + xs[i][-400:]

            x[i, :len(xs[i])] = xs[i]

            mask[i, :len(xs[i])] = 1

        return {

            'input_ids': torch.LongTensor(x).to(self.device),

            'attention_mask': torch.LongTensor(mask).to(self.device),

        }





class TextsDataset(Dataset):

    def __init__(self, df, tokenizer):

        self.df = df

        self.tokenizer = tokenizer



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        return self.tokenizer.convert_tokens_to_ids(['[CLS]'] + self.tokenizer.tokenize(self.df.iloc[idx][TEXT_COL]) + ['[SEP]'])





class BertClassifier(nn.Module):

    def __init__(self, config, n_labels):

        super(BertClassifier, self).__init__()

        self.bert = BertModel(config)

        self.out = nn.Linear(config.hidden_size, n_labels)

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')



    def forward(self, input_ids, token_type_ids=None, attention_mask=None,

                output_all_encoded_layers=True, increase_weight=None, labels=None):

        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,

                                     output_all_encoded_layers=False)

        logits = self.out(pooled_output)

        if labels is None or increase_weight is None:

            return logits





def get_dataloader(df, tokenizer, collator, batch_size):

    return DataLoader(TextsDataset(df, tokenizer), batch_size, shuffle=False, collate_fn=collator)





def get_preds(model, dl):

    model.eval()

    with torch.no_grad():

        preds = np.zeros([len(test_dl.dataset)], dtype=np.float32)

        idx = 0

        for batch in tqdm(test_dl):

            pr = torch.sigmoid(model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], increase_weight=None)).cpu().numpy()[:, 0]

            preds[idx: idx + len(pr)] = pr

            idx += len(pr)

    return preds
test_df = pd.read_csv(DF_TEST_PATH)

test_df[TEXT_COL] = test_df[TEXT_COL].apply(preprocess)

device = torch.device('cuda')

collator = Collator(device)

test_dl = get_dataloader(test_df, uncased_tokenizer, collator, 32)
model_path2n_out = {

    FIRST_MODEL_PATH: 6,

    SECOND_MODEL_PATH: 6

}
preds = []

for i, p in enumerate([FIRST_MODEL_PATH, SECOND_MODEL_PATH]):

    model = BertClassifier(UNCASED_BERT_CONFIG, model_path2n_out[p]).to(device)

    model.load_state_dict(torch.load(p, map_location=device))

    preds.append(get_preds(model, test_dl))

submission = pd.DataFrame.from_dict({

    'id': test_df['id'],

    'prediction': np.average(preds, axis=0)

})

submission.to_csv('submission.csv', index=False)