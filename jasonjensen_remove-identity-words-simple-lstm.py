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
import time

import numpy as np

import pandas as pd

from keras.models import Model, Sequential

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate

from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D

from keras.preprocessing import text, sequence

from keras.callbacks import LearningRateScheduler

import gc

import random

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from tensorflow.keras.preprocessing import text

import shap

from tqdm._tqdm_notebook import tqdm_notebook as tqdm

import pickle
tqdm.pandas()
absStart = time.time();
CRAWL_EMBEDDING_PATH = '../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl'

GLOVE_EMBEDDING_PATH = '../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl'



NUM_MODELS = 2

BATCH_SIZE = 512

LSTM_UNITS = 128

DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

MAX_LEN = 220

IDENTITY_COLUMNS = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

TEXT_COLUMN = 'comment_text'

TARGET_COLUMN = 'target'

CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

MAX_FEATURES = 400000



NUM_IDENTITY_WORDS = 15



symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'

symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()





isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}

remove_dict = {ord(c):f'' for c in symbols_to_delete}





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
def get_balanced_set(df, FOCUS_COLUMN, size = 0):

    if (size == 0):

        size = df.loc[df[FOCUS_COLUMN] == True].shape[0] * 2

    true_df = df.loc[df[FOCUS_COLUMN] == True][:round(size/2)]

    false_df = df.loc[df[FOCUS_COLUMN] == False][:round(size/2)]

    joined = pd.concat([true_df, false_df])

    return joined    
def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')





def load_embeddings(path):

    with open(path,'rb') as f:

        emb_arr = pickle.load(f)

    return emb_arr



def build_matrix(word_index, path):

    embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((MAX_FEATURES + 1, 300))

    unknown_words = []

    

    for word, i in word_index.items():

        if i <= MAX_FEATURES:

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

    



    ## build the model (a couple of layers)

def build_model(embedding_matrix, num_aux_targets):

    words = Input(shape=(None,))

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

    
chunkStart = time.time()



train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

print('CHUNK', time.time() - chunkStart)

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

print('CHUNK', time.time() - chunkStart)

train_df[TEXT_COLUMN] = train_df[TEXT_COLUMN].apply(lambda x:preprocess(x))

print('CHUNK', time.time() - chunkStart)

test_df[TEXT_COLUMN] = test_df[TEXT_COLUMN].apply(lambda x:preprocess(x))

gc.collect()



print('CHUNK', time.time() - chunkStart)

print('OVERALL', time.time() - absStart)
class TextPreprocessor(object):

    def __init__(self, vocab_size):

        self._vocab_size = vocab_size

        self._tokenizer = None

        self._embedding_matrix = None

    

    def create_tokenizer(self, text_list):

        tokenizer = text.Tokenizer(num_words = MAX_FEATURES, filters='',lower=False)

        crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)

        print('n unknown words (crawl): ', len(unknown_words_crawl))



        glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)

        print('n unknown words (glove): ', len(unknown_words_glove))

        self._tokenizer = tokenizer

        self._embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

        del crawl_matrix

        del glove_matrix

        gc.collect()

    

    def transform_text(self, text_list):

        print(text_list[:10])

        text_matrix = self._tokenizer.texts_to_matrix(text_list)

        return text_matrix

# for extracting the most predictive words from the model

def get_top_words(vals, num_words, word_index):

    means = np.matrix(vals[0]).mean(0)

    means = np.absolute(means)

    words = set()

    while len(words) < num_words:

        idx = means.argmax()

        idx

        words.add(word_index[idx])

        means[0, idx] = -1000



    return words



def create_identity_model(vocab_size, num_tags):

    model = Sequential()

    model.add(Dense(50, input_shape=(vocab_size,), activation='relu'))

    model.add(Dense(25, activation='relu'))

    model.add(Dense(num_tags, activation='sigmoid'))



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

  
def get_words_for_identity(identity_df, IDENTITY_COLUMN, VOCAB_SIZE = 400):

    id_train = get_balanced_set(identity_df, IDENTITY_COLUMN)



    tokenizer = text.Tokenizer(num_words = VOCAB_SIZE, filters='',lower=False)

    tokenizer.fit_on_texts(id_train['comment_text'])



#     body_train = processor.transform_text(id_train['comment_text'])

    body_train = tokenizer.texts_to_matrix(id_train['comment_text'])

    

    ## fit model

    id_model = create_identity_model(VOCAB_SIZE, 1)

    id_model.fit(

        x=body_train, 

        y=id_train[IDENTITY_COLUMN], 

        epochs=10, 

        batch_size=128, 

        validation_split=0.1, 

        verbose=0)

    

    ## create word lookup

    words = tokenizer.word_index

    word_lookup = list()

    for i in words.keys():

      word_lookup.append(i)



    word_lookup = [''] + word_lookup

    

    ## get shap values

    num_explainers = 2000

    explainer_idx = np.random.randint(body_train.shape[0], size=num_explainers)

    attrib_data = body_train[explainer_idx,:]

    explainer = shap.DeepExplainer(id_model, attrib_data)



    #### a bit sloppy to use the training data here, but it's probably ok

    num_explanations = 1000

    explanations_idx = np.random.randint(body_train.shape[0], size=num_explanations)

    shap_vals = explainer.shap_values(body_train[explanations_idx,:])

    

    top_words = get_top_words(shap_vals, NUM_IDENTITY_WORDS, word_lookup)



    return top_words
# chunkStart = time.time()



# all_id_words = set()

# for id in IDENTITY_COLUMNS:

#     print(id.upper())

#     id_words = get_words_for_identity(train_df, id)

#     for word in id_words:

#         all_id_words.add(word)



# print('CHUNK', time.time() - chunkStart)

# print('OVERALL', time.time() - absStart)



# all_id_words
static_all_id_words = {

 'Black',

 'Blacks',

 'Catholic',

 'Catholics',

 'Christian',

 'Christianity',

 'Christians',

 'Church',

 'Gay',

 'Germany',

 'Hitler',

 'Islamic',

 'Jesus',

 'Jew',

 'Jewish',

 'Jews',

 'Men',

 'Mental',

 'Muslim',

 'Muslims',

 'Nazi',

 'Nazis',

 'She',

 'White',

 'Women',

 'believe',

 'bishops',

 'black',

 'blacks',

 'church',

 'disorder',

 'faith',

 'female',

 'gay',

 'gays',

 'guy',

 'her',

 'homosexual',

 'homosexuals',

 'illness',

 'issues',

 'jews',

 'm',

 'male',

 'males',

 'man',

 'marriage',

 'men',

 'mental',

 'mentally',

 'muslim',

 'muslims',

 'priests',

 'race',

 'racial',

 'racism',

 'rights',

 'schools',

 'sexual',

 'supremacist',

 'supremacists',

 'terrorist',

 'wedding',

 'white',

 'whites',

 'woman',

 'women',

 }
print('TIME', time.time() - absStart)
def remove_id_words(s):

    s = s.lower()

    for w in static_all_id_words:

        s = s.replace(w, '')

    return s
## For main analysis

chunkStart = time.time()



x_train = train_df[TEXT_COLUMN].astype(str).apply(remove_id_words)

y_train = train_df[TARGET_COLUMN].values

y_aux_train = train_df[AUX_COLUMNS].values

x_test = test_df[TEXT_COLUMN].astype(str).apply(remove_id_words)



print('CHUNK', time.time() - chunkStart)

print('OVERALL', time.time() - absStart)



for column in IDENTITY_COLUMNS + [TARGET_COLUMN]:

    train_df[column] = np.where(train_df[column] >= 0.5, True, False)



MAX_FEATURES = 400000

tokenizer = text.Tokenizer(num_words = MAX_FEATURES, filters='',lower=False)

tokenizer.fit_on_texts(list(x_train) + list(x_test))

crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)

print('n unknown words (crawl): ', len(unknown_words_crawl))



glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)

print('n unknown words (glove): ', len(unknown_words_glove))



MAX_FEATURES = MAX_FEATURES or len(tokenizer.word_index) + 1

MAX_FEATURES



embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

embedding_matrix.shape



del crawl_matrix

del glove_matrix

gc.collect()



# tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)

# tokenizer.fit_on_texts(list(x_train) + list(x_test))



x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)

x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)



sample_weights = np.ones(len(x_train), dtype=np.float32)

sample_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)

sample_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)

sample_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5

sample_weights /= sample_weights.mean()



# embedding_matrix = np.concatenate(

#     [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)



checkpoint_predictions = []

weights = []



print('CHUNK', time.time() - chunkStart)

print('OVERALL', time.time() - absStart)
chunkStart = time.time()

for model_idx in range(NUM_MODELS):

    model = build_model(embedding_matrix, y_aux_train.shape[-1])

    for global_epoch in range(EPOCHS):

        model.fit(

            x_train,

            [y_train, y_aux_train],

            batch_size=BATCH_SIZE,

            epochs=1,

            verbose=1,

            sample_weight=[sample_weights.values, np.ones_like(sample_weights)],

            callbacks=[

                LearningRateScheduler(lambda _: 1e-3 * (0.55 ** global_epoch))

            ]

        )

        checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())

        weights.append(2 ** global_epoch)

        print('EPOCH', time.time() - chunkStart)

    print('MODEL', time.time() - chunkStart)

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)



submission = pd.DataFrame.from_dict({

    'id': test_df.id,

    'prediction': predictions

})

submission.to_csv('submission.csv', index=False)



print('TOTAL TIME')

print(time.time() - absStart)