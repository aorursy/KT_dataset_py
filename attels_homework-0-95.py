import pickle

import tensorflow as tf

from collections import Counter, defaultdict
#path = r'../input/'

file_word_counters = pickle.load(open(r'../input/os_mk_ru_en.pickle', 'rb'))

filenames = {r'OpenSubtitles.mk-ru.mk': 'mk', 

             r'OpenSubtitles.mk-ru.ru': 'ru',

             r'OpenSubtitles.en-ru.en':'en'}
def lang_dicts_filter(lang_dicts, f_exclude=lambda w, cnt, lng: False, f_action=lambda w, cnt, lng: None, new_counters=False):

    if new_counters:

        result = {}

    else:

        result = lang_dicts

    for lng, wc in lang_dicts.items():

        if new_counters:

            wc = Counter(wc)

        for w, cnt in list(wc.items()):

            if f_exclude(w, cnt, lng):

                wc.pop(w)

            else:

                f_action(w, cnt, lng)

        if new_counters:

            result[lng] = wc        

    return result
# Фильтруем редкие и длинные слова и составляем общий словарь

MIN_WORD_COUNT = 5

MAX_WORD_LENGTH = 20



# меняем ключи словаря на коды языка:

lang_dicts2 = {filenames[fn]: cnt for fn, cnt in file_word_counters.items()}

all_lang_dict2 = defaultdict(list)

lang_dicts2 = lang_dicts_filter(lang_dicts2, 

                               f_exclude=lambda w, cnt, lng: cnt <= MIN_WORD_COUNT or len(w) > MAX_WORD_LENGTH,

                               f_action=lambda w, cnt, lng: all_lang_dict2[w].append((lng, cnt)),

                               new_counters=True)
multilang_set = set(w for w, lng in all_lang_dict2.items() if len(lng) > 1)



lang_dicts2 = lang_dicts_filter(lang_dicts2, 

                               f_exclude=lambda w, cnt, lng:  w in multilang_set)
langs_final = ['ru', 'mk']



# меняем ключи словаря на коды языка:

lang_dicts3 = {ln: wc for ln, wc in lang_dicts2.items() if ln in langs_final}



def gen_alphabet_counter():

    alph_cnt = Counter()

    def alph_counter(w, cnt, lng):

        nonlocal alph_cnt

        alph_cnt += Counter(w)    

    return alph_cnt, alph_counter

alphabet_cnt, alphabet_counter = gen_alphabet_counter()

lang_dicts3 = lang_dicts_filter(lang_dicts3, f_action=alphabet_counter)

#print(alphabet_cnt.most_common())
MIN_SYMBOL_COUNT = 100

alphabet_cnt_flt = Counter({s: cnt for s, cnt in alphabet_cnt.items() if cnt > MIN_SYMBOL_COUNT})

# print(alphabet_cnt_flt.most_common())

alphabet_set = set(alphabet_cnt_flt)



lang_dicts4 = lang_dicts_filter(lang_dicts3, 

                               f_exclude=lambda w, cnt, lng:  len(set(w) - alphabet_set) > 0,

                               new_counters=True)



alphabet_cnt_f, alphabet_counter_f = gen_alphabet_counter()

lang_dicts4 = lang_dicts_filter(lang_dicts4, f_action=alphabet_counter_f)

#print(alphabet_cnt_f.most_common())
# кодирование символов



# Добавляем перым пустой символ (кодируем его подчеркиванием):

alphabet_cnt_f['_'] = alphabet_cnt_f.most_common()[0][1] **2

code_symb_to_ind = {s: i for i, (s, cnt) in enumerate(alphabet_cnt_f.most_common())}

CODE_LEN = len(code_symb_to_ind)

code_ind_to_symb = {i: s for s, i in code_symb_to_ind.items()}

from collections import namedtuple

import numpy as np

TrnValTst = namedtuple('TrnValTst', 'trn val tst')
# train validation test

def vocabulary_tvt_split(cntr, train=0.8, validation=0.1, test=0.1):

    assert sum((train, validation, test)) == 1.0, f'Incorrect shares: train:{train}, validation:{validation}, test:{test}!'

    cl = len(cntr)

    val_l = int(cl * validation)

    tst_l = int(cl * test)

    trn_l = cl - val_l - tst_l

    split_types = np.concatenate((np.full((trn_l,), 0), np.full((val_l,), 1), np.full((tst_l,), 2)))

    np.random.shuffle(split_types)        

    words_l_tvt = TrnValTst([], [], [])

    wordscount_l_tvt = TrnValTst([], [], [])

    for ind, (w, cnt) in enumerate(cntr.most_common()):

        t = split_types[ind]

        words_l_tvt[t].append(w)

        wordscount_l_tvt[t].append(cnt)                

    wordsprob_ar_tvt = TrnValTst._make(np.array(l, dtype=np.float) for l in wordscount_l_tvt)

    for ar in wordsprob_ar_tvt:

        ar /= ar.sum()

    return words_l_tvt, wordsprob_ar_tvt
def generate_wrods_tvt(words_l_tvt, wordsprob_ar_tvt, train, validation, test):

    par_tvt = TrnValTst(train, validation, test)

    res_words_l_tvt = TrnValTst([], [], [])

    for q, words_l, wordsprob_ar, lst in zip(par_tvt, words_l_tvt, wordsprob_ar_tvt, res_words_l_tvt):

        lst.extend(words_l[i] for i in np.random.choice(len(words_l), size=q, p=wordsprob_ar))

    return res_words_l_tvt
def vectorize_word_list(w_list, word_len=MAX_WORD_LENGTH, code_s_to_i=code_symb_to_ind):

    code_len = len(code_s_to_i)

    rest_t = np.zeros((len(w_list), word_len, code_len), dtype=np.float16)

    for i, w in enumerate(w_list):

        for j, s in enumerate(w):

            rest_t[i, j, code_s_to_i[s]] = 1.0

    return rest_t
def unison_shuffled_copies(*arrs):

    lens = [len(a) for a in arrs]

    assert min(lens) == max(lens)

    p = np.random.permutation(lens[0])

    return tuple(a[p] for a in arrs)
TRAIN = 200000

VALIDATION = 20000

TEST = 20000

ru = generate_wrods_tvt(*vocabulary_tvt_split(lang_dicts4['ru']), TRAIN, VALIDATION, TEST)

mk = generate_wrods_tvt(*vocabulary_tvt_split(lang_dicts4['mk']), TRAIN, VALIDATION, TEST)

data = [ru, mk]
x_d = {}

y_d = {}

for fld in data[0]._fields:    

    x_d[fld] = [getattr(tvt, fld) for tvt in data]

    y_d[fld] = np.concatenate([np.full(len(w_l), i, dtype=np.float16) \

                               for i, w_l in enumerate(getattr(tvt, fld) for tvt in data)])



xv_d = {fld: vectorize_word_list([w for word_l in words_ll for w in word_l]) for fld, words_ll in x_d.items()}
vdata = TrnValTst(**{fld: unison_shuffled_copies(xv_d[fld], y_d[fld], np.arange(len(y_d[fld]))) for fld in TrnValTst._fields})
x_train = vdata.trn[0]

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

x_train.shape
x_validate = vdata.val[0]

x_validate = x_validate.reshape((x_validate.shape[0], x_validate.shape[1] * x_validate.shape[2]))

x_validate.shape
y_train = vdata.trn[1]

y_train = y_train.reshape((y_train.size, 1))

y_train.shape 
y_validate = vdata.val[1]

y_validate = y_validate.reshape((y_validate.size, 1))

y_validate.shape 
x_train = x_train.astype(np.float64)

y_train = y_train.astype(np.float64)

y_validate = y_validate.astype(np.float64)

x_validate =x_validate.astype(np.float64)
from keras.models import Sequential

from keras.layers import Input, Dense, Activation, Dropout

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD

import keras
model = Sequential()



#input layer

model.add(Dense(940,input_shape = (MAX_WORD_LENGTH*CODE_LEN,)))

model.add(Activation('relu'))



# hidden layear

model.add(Dense(200,))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())



# output layer

model.add(Dense(80,))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())



model.add(Dense(1,))

model.add(BatchNormalization())

model.add(Activation('sigmoid'))
model.summary()
model.compile(loss=keras.losses.binary_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])
history = model.fit(x_train, y_train,

          batch_size=256,

          epochs=5,

          verbose=1,

          validation_data=(x_validate, y_validate))
model.evaluate(x_validate,y_validate)
import matplotlib.pyplot as plt
history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']



acc = history_dict['acc']

epochs = range(1, len(acc) + 1)



plt.plot(epochs, loss_values, 'b', label='Training loss', color = 'red') 

plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']



plt.plot(epochs, acc_values, 'b', label='Training acc', color = 'red')

plt.plot(epochs, val_acc_values, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()