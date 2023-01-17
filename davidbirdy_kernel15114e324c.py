# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import sys

import sklearn

import tensorflow as tf

from tensorflow import keras

import numpy as np

import os

import pandas as pd 



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")
# loading data

input_dir = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/'

def load_csv_data(input_dir, filename):

    csv_path = input_dir + filename + ".csv"

    return pd.read_csv(csv_path)



train_data_b01 = load_csv_data(input_dir, "jigsaw-toxic-comment-train-processed-seqlen128")
# construct toxic label in trainset



train_data_b01['total_toxicity'] = 0

s = train_data_b01.shape[0]



for i in range(s):

    counter = 0

    if train_data_b01["toxic"][i] > 0:

        counter += 1

        train_data_b01['total_toxicity'][i] += train_data_b01["toxic"][i]

        

    if train_data_b01["severe_toxic"][i] > 0:

        counter += 1

        train_data_b01['total_toxicity'][i] += train_data_b01["severe_toxic"][i]

        

    if train_data_b01["obscene"][i] > 0:

        counter += 1

        train_data_b01['total_toxicity'][i] += train_data_b01["obscene"][i]

        

    if train_data_b01["threat"][i] > 0:

        counter += 1

        train_data_b01['total_toxicity'][i] += train_data_b01["threat"][i]

        

    if train_data_b01["insult"][i] > 0:

        counter += 1

        train_data_b01['total_toxicity'][i] += train_data_b01["insult"][i]

    

    if train_data_b01["identity_hate"][i] > 0:

        counter += 1

        train_data_b01['total_toxicity'][i] += train_data_b01["identity_hate"][i]

        

    if counter > 0 :

        train_data_b01['total_toxicity'][i] = train_data_b01['total_toxicity'][i]/counter
# construct trainset



data = { 'comment_text' : (train_data_b01['comment_text']),

        'toxic' : train_data_b01['total_toxicity']   

}



train_data_mod = pd.DataFrame(data, columns = ['comment_text', 'toxic'])
# separate normal and toxic comments



max_comment_size = 300



# remove emoji from comments

import emoji

def give_emoji_free_text(text):

    return emoji.get_emoji_regexp().sub(r'', text)



comment_normal = []

comment_toxic = []

size = train_data_mod.shape[0]



with tf.device('/CPU:0'):

    for i in range(size):

        comment = train_data_mod['comment_text'][i]

        comment = give_emoji_free_text(comment)

        comment = comment[0:max_comment_size].replace("<r\s*/?>", " ").replace("[^a-zA-Z']", " ")

        if train_data_mod['toxic'][i] == 0:

            comment_normal.append(comment)

        else:

            comment_toxic.append(comment)

        

normar_labels = np.zeros(len(comment_normal))

toxic_labels = np.ones(len(comment_toxic))
# find duplicates



import collections

print([count for item, count in collections.Counter(comment_normal).items() if count > 1])

print([count for item, count in collections.Counter(comment_toxic).items() if count > 1])
# delete duplicates



comment_normal = list(dict.fromkeys(comment_normal))

comment_toxic = list(dict.fromkeys(comment_toxic))
# check it

print([count for item, count in collections.Counter(comment_normal).items() if count > 1])

print([count for item, count in collections.Counter(comment_toxic).items() if count > 1])
# function for combining  text with separators from comment data



def text_packet(comment_normal, comment_toxic, inds_n, inds_t, packet_size =25, separator = "######"):

    normal_comments = []

    toxic_comments = []

    

    for i in range(packet_size):

        n_ind = inds_n.pop(i)

        t_ind = inds_t.pop(i)

        normal_comments.append(comment_normal[n_ind])

        toxic_comments.append(comment_toxic[t_ind])

        

    text_normal = ''

    text_toxic = ''

    

    for comment in normal_comments:

        text_normal += comment + separator

    

    for comment in toxic_comments:

        text_toxic += comment + separator

        

    return text_normal, text_toxic  
# back from text to list of comments



def text_to_list(text, separator = "######"):

    text_comments = []

    text_comments = text.split(separator)

    try:

        text_comments.remove('')

    except:

        text_comments = text_comments

    return text_comments
# append prepared lists of comments, labels, langs



def append_data_lists(text, toxic_list, lang_list, comment_list, toxic = 1, lang = 'tr'):

    text_list = text_to_list(text, separator = "######")

    n = len(text_list) 

    for i in range(n):

        toxic_list.append(toxic)

        lang_list.append(lang)

        comment_list.append(text_list[i])

    return toxic_list, lang_list, comment_list
# prepare random indecex for data selection 

def random_inds(size):

    inds = list(np.random.permutation(size))

    return inds
# translate english comments



np.random.seed(42)



len_n = len(comment_normal)

len_t = len(comment_toxic)



inds_n_tr = random_inds(len_n)

inds_t_tr = random_inds(len_t)



inds_n_pt = random_inds(len_n)

inds_t_pt = random_inds(len_t)



inds_n_ru = random_inds(len_n)

inds_t_ru = random_inds(len_t)



inds_n_fr = random_inds(len_n)

inds_t_fr = random_inds(len_t)



inds_n_it = random_inds(len_n)

inds_t_it = random_inds(len_t)



inds_n_es = random_inds(len_n)

inds_t_es = random_inds(len_t)
#pip install translators
try:

    import time

    import translators as ts

    toxic_list = []

    lang_list = []

    comment_list = []

    API = ts.google

    packet_size = 25

    sleep_time = 2

    num_iter = 24

    

    for iteration in range(num_iter):

        import translators as ts    

        print("Iteration: " + str(iteration)) 

       

        lang = 'tr'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_tr, inds_t_tr, 

                                     packet_size = packet_size)

        text_t_tr = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_tr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_tr = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_tr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

    

        lang = 'pt'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_pt, inds_t_pt, 

                                     packet_size = packet_size)

        text_t_pt = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_pt, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_pt = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_pt, 

                                                            toxic_list, lang_list, comment_list,

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

       

        lang = 'ru'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_ru, inds_t_ru,

                                    packet_size = packet_size)

        text_t_ru = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_ru, 

                                                            toxic_list, lang_list, comment_list,                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_ru = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_ru, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

       

    

        lang = 'fr'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_fr, inds_t_fr, 

                                     packet_size = packet_size)

        text_t_fr = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_fr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_fr = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_fr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

    

        lang = 'it'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_it, inds_t_it, 

                                     packet_size = packet_size)

        text_t_it = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_it, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_it = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_it, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

    

        lang = 'es'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_es, inds_t_es, 

                                     packet_size = packet_size)

        text_t_es = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_es, 

                                                            toxic_list, lang_list, 

                                                            comment_list, toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_es = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_es, 

                                                            toxic_list, lang_list, 

                                                            comment_list, toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

        

except:

    print("Error")

       
"""

mini_multilang_trainset_1 = pd.DataFrame(columns = ['comment', 'toxic', 'lang'])

mini_multilang_trainset_1['comment'] = comment_list

mini_multilang_trainset_1['toxic'] = toxic_list

mini_multilang_trainset_1['lang'] = lang_list



mini_multilang_trainset_1.to_csv('mini_multilang_trainset_1.csv',index = False)

"""

preprocessed_dir = '/kaggle/input/preprocessed-data/'

output_dir = '/kaggle/working/'



mini_multilang_trainset_1 = load_csv_data(preprocessed_dir, "mini_multilang_trainset_1")

mini_multilang_trainset_1.to_csv(output_dir + 'mini_multilang_trainset_1.csv', index = False)
"""

unused_normal_indeces_1 = pd.DataFrame(columns = ['ind_tr', 'ind_pt', 'ind_ru', 'ind_fr', 'ind_it', 'ind_es'])

unused_normal_indeces_1['ind_tr'] = inds_n_tr

unused_normal_indeces_1['ind_pt'] = inds_n_pt

unused_normal_indeces_1['ind_ru'] = inds_n_ru

unused_normal_indeces_1['ind_fr'] = inds_n_fr

unused_normal_indeces_1['ind_it'] = inds_n_it

unused_normal_indeces_1['ind_es'] = inds_n_es



unused_toxic_indeces_1 = pd.DataFrame(columns = ['ind_tr', 'ind_pt', 'ind_ru', 'ind_fr', 'ind_it', 'ind_es'])

unused_toxic_indeces_1['ind_tr'] = inds_t_tr

unused_toxic_indeces_1['ind_pt'] = inds_t_pt

unused_toxic_indeces_1['ind_ru'] = inds_t_ru

unused_toxic_indeces_1['ind_fr'] = inds_t_fr

unused_toxic_indeces_1['ind_it'] = inds_t_it

unused_toxic_indeces_1['ind_es'] = inds_t_es



unused_normal_indeces_1.to_csv('unused_normal_indeces_after_step_1',index = False)

unused_toxic_indeces_1.to_csv('unused_toxic_indeces_after_step_1.csv',index = False)

"""



unused_normal_indeces_after_step_1 = load_csv_data(preprocessed_dir, "unused_normal_indeces_after_step_1")

unused_normal_indeces_after_step_1.to_csv(output_dir + 'unused_normal_indeces_after_step_1.csv', index = False)



unused_toxic_indeces_after_step_1 = load_csv_data(preprocessed_dir, "unused_toxic_indeces_after_step_1")

unused_toxic_indeces_after_step_1.to_csv(output_dir + 'unused_toxic_indeces_after_step_1.csv', index = False)
# prepare random indeces from step 1



inds_n_tr = list(unused_normal_indeces_after_step_1['ind_tr'])

inds_t_tr = list(unused_toxic_indeces_after_step_1['ind_tr'])



inds_n_pt = list(unused_normal_indeces_after_step_1['ind_pt'])

inds_t_pt = list(unused_toxic_indeces_after_step_1['ind_pt'])



inds_n_ru = list(unused_normal_indeces_after_step_1['ind_ru'])

inds_t_ru = list(unused_toxic_indeces_after_step_1['ind_ru'])



inds_n_fr = list(unused_normal_indeces_after_step_1['ind_fr'])

inds_t_fr = list(unused_toxic_indeces_after_step_1['ind_fr'])



inds_n_it = list(unused_normal_indeces_after_step_1['ind_it'])

inds_t_it = list(unused_toxic_indeces_after_step_1['ind_it'])



inds_n_es = list(unused_normal_indeces_after_step_1['ind_es'])

inds_t_es = list(unused_toxic_indeces_after_step_1['ind_es'])
try:

    import time

    import translators as ts

    toxic_list = []

    lang_list = []

    comment_list = []

    API = ts.google

    packet_size = 25

    sleep_time = 2

    num_iter = 11

    

    for iteration in range(num_iter):

        import translators as ts    

        print("Iteration: " + str(iteration)) 

       

        lang = 'tr'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_tr, inds_t_tr, 

                                     packet_size = packet_size)

        text_t_tr = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_tr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_tr = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_tr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

    

        lang = 'pt'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_pt, inds_t_pt, 

                                     packet_size = packet_size)

        text_t_pt = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_pt, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_pt = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_pt, 

                                                            toxic_list, lang_list, comment_list,

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

       

        lang = 'ru'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_ru, inds_t_ru,

                                    packet_size = packet_size)

        text_t_ru = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_ru, 

                                                            toxic_list, lang_list, comment_list,                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_ru = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_ru, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

       

    

        lang = 'fr'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_fr, inds_t_fr, 

                                     packet_size = packet_size)

        text_t_fr = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_fr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_fr = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_fr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

    

        lang = 'it'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_it, inds_t_it, 

                                     packet_size = packet_size)

        text_t_it = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_it, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_it = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_it, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

    

        lang = 'es'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_es, inds_t_es, 

                                     packet_size = packet_size)

        text_t_es = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_es, 

                                                            toxic_list, lang_list, 

                                                            comment_list, toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_es = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_es, 

                                                            toxic_list, lang_list, 

                                                            comment_list, toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

        

except:

    print("Error")

       
"""

mini_multilang_trainset_2 = pd.DataFrame(columns = ['comment', 'toxic', 'lang'])

mini_multilang_trainset_2['comment'] = comment_list

mini_multilang_trainset_2['toxic'] = toxic_list

mini_multilang_trainset_2['lang'] = lang_list



mini_multilang_trainset_2.to_csv('mini_multilang_trainset_2.csv',index = False)

"""

mini_multilang_trainset_2 = load_csv_data(preprocessed_dir, "mini_multilang_trainset_2")

mini_multilang_trainset_2.to_csv(output_dir + 'mini_multilang_trainset_2.csv', index = False)
"""

unused_normal_indeces_2 = pd.DataFrame(columns = ['ind_tr', 'ind_pt', 'ind_ru', 'ind_fr', 'ind_it', 'ind_es'])

unused_normal_indeces_2['ind_tr'] = inds_n_tr

unused_normal_indeces_2['ind_pt'] = inds_n_pt

unused_normal_indeces_2['ind_ru'] = inds_n_ru

unused_normal_indeces_2['ind_fr'] = inds_n_fr

unused_normal_indeces_2['ind_it'] = inds_n_it

unused_normal_indeces_2['ind_es'] = inds_n_es



unused_toxic_indeces_2 = pd.DataFrame(columns = ['ind_tr', 'ind_pt', 'ind_ru', 'ind_fr', 'ind_it', 'ind_es'])

unused_toxic_indeces_2['ind_tr'] = inds_t_tr

unused_toxic_indeces_2['ind_pt'] = inds_t_pt

unused_toxic_indeces_2['ind_ru'] = inds_t_ru

unused_toxic_indeces_2['ind_fr'] = inds_t_fr

unused_toxic_indeces_2['ind_it'] = inds_t_it

unused_toxic_indeces_2['ind_es'] = inds_t_es



unused_normal_indeces_2.to_csv('unused_normal_indeces_2.csv',index = False)

unused_toxic_indeces_2.to_csv('unused_toxic_indeces_2.csv',index = False)

"""



unused_normal_indeces_after_step_2 = load_csv_data(preprocessed_dir, "unused_normal_indeces_after_step_2")

unused_normal_indeces_after_step_2.to_csv(output_dir + 'unused_normal_indeces_after_step_2.csv', index = False)



unused_toxic_indeces_after_step_2 = load_csv_data(preprocessed_dir, "unused_toxic_indeces_after_step_2")

unused_toxic_indeces_after_step_2.to_csv(output_dir + 'unused_toxic_indeces_after_step_2.csv', index = False)
# prepare random indeces from step 2



inds_n_tr = list(unused_normal_indeces_after_step_2['ind_tr'])

inds_t_tr = list(unused_toxic_indeces_after_step_2['ind_tr'])



inds_n_pt = list(unused_normal_indeces_after_step_2['ind_pt'])

inds_t_pt = list(unused_toxic_indeces_after_step_2['ind_pt'])



inds_n_ru = list(unused_normal_indeces_after_step_2['ind_ru'])

inds_t_ru = list(unused_toxic_indeces_after_step_2['ind_ru'])



inds_n_fr = list(unused_normal_indeces_after_step_2['ind_fr'])

inds_t_fr = list(unused_toxic_indeces_after_step_2['ind_fr'])



inds_n_it = list(unused_normal_indeces_after_step_2['ind_it'])

inds_t_it = list(unused_toxic_indeces_after_step_2['ind_it'])



inds_n_es = list(unused_normal_indeces_after_step_2['ind_es'])

inds_t_es = list(unused_toxic_indeces_after_step_2['ind_es'])
try:

    import time

    import translators as ts

    toxic_list = []

    lang_list = []

    comment_list = []

    API = ts.google

    packet_size = 25

    sleep_time = 2

    num_iter = 15

    

    for iteration in range(num_iter):

        import translators as ts    

        print("Iteration: " + str(iteration)) 

       

        lang = 'tr'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_tr, inds_t_tr, 

                                     packet_size = packet_size)

        text_t_tr = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_tr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_tr = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_tr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

    

        lang = 'pt'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_pt, inds_t_pt, 

                                     packet_size = packet_size)

        text_t_pt = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_pt, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_pt = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_pt, 

                                                            toxic_list, lang_list, comment_list,

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

       

        lang = 'ru'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_ru, inds_t_ru,

                                    packet_size = packet_size)

        text_t_ru = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_ru, 

                                                            toxic_list, lang_list, comment_list,                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_ru = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_ru, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

       

    

        lang = 'fr'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_fr, inds_t_fr, 

                                     packet_size = packet_size)

        text_t_fr = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_fr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_fr = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_fr, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

    

        lang = 'it'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_it, inds_t_it, 

                                     packet_size = packet_size)

        text_t_it = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_it, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_it = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_it, 

                                                            toxic_list, lang_list, comment_list, 

                                                            toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

    

        lang = 'es'

        text_n, text_t = text_packet(comment_normal, comment_toxic, inds_n_es, inds_t_es, 

                                     packet_size = packet_size)

        text_t_es = API(text_t, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_t_es, 

                                                            toxic_list, lang_list, 

                                                            comment_list, toxic = 1, lang = lang)

        time.sleep(sleep_time)

        text_n_es = API(text_n, to_language=lang)

        toxic_list, lang_list, comment_list = append_data_lists(text_n_es, 

                                                            toxic_list, lang_list, 

                                                            comment_list, toxic = 0, lang = lang)

        time.sleep(sleep_time*2)

        

except:

    print("Error")

       
"""

mini_multilang_trainset_3 = pd.DataFrame(columns = ['comment', 'toxic', 'lang'])

mini_multilang_trainset_3['comment'] = comment_list

mini_multilang_trainset_3['toxic'] = toxic_list

mini_multilang_trainset_3['lang'] = lang_list



mini_multilang_trainset_3.to_csv('mini_multilang_trainset_3.csv',index = False)

"""

mini_multilang_trainset_3 = load_csv_data(preprocessed_dir, "mini_multilang_trainset_3")

mini_multilang_trainset_3.to_csv(output_dir + 'mini_multilang_trainset_3.csv', index = False)
"""

unused_normal_indeces_3 = pd.DataFrame(columns = ['ind_tr', 'ind_pt', 'ind_ru', 'ind_fr', 'ind_it', 'ind_es'])

unused_normal_indeces_3['ind_tr'] = inds_n_tr

unused_normal_indeces_3['ind_pt'] = inds_n_pt

unused_normal_indeces_3['ind_ru'] = inds_n_ru

unused_normal_indeces_3['ind_fr'] = inds_n_fr

unused_normal_indeces_3['ind_it'] = inds_n_it

unused_normal_indeces_3['ind_es'] = inds_n_es



unused_toxic_indeces_3 = pd.DataFrame(columns = ['ind_tr', 'ind_pt', 'ind_ru', 'ind_fr', 'ind_it', 'ind_es'])

unused_toxic_indeces_3['ind_tr'] = inds_t_tr

unused_toxic_indeces_3['ind_pt'] = inds_t_pt

unused_toxic_indeces_3['ind_ru'] = inds_t_ru

unused_toxic_indeces_3['ind_fr'] = inds_t_fr

unused_toxic_indeces_3['ind_it'] = inds_t_it

unused_toxic_indeces_3['ind_es'] = inds_t_es



unused_normal_indeces_3.to_csv('unused_normal_indeces_3.csv',index = False)

unused_toxic_indeces_3.to_csv('unused_toxic_indeces_3.csv',index = False)

"""



unused_normal_indeces_after_step_3 = load_csv_data(preprocessed_dir, "unused_normal_indeces_after_step_3")

unused_normal_indeces_after_step_3.to_csv(output_dir + 'unused_normal_indeces_after_step_3.csv', index = False)



unused_toxic_indeces_after_step_3 = load_csv_data(preprocessed_dir, "unused_toxic_indeces_after_step_3")

unused_toxic_indeces_after_step_3.to_csv(output_dir + 'unused_toxic_indeces_after_step_3.csv', index = False)
# add english comments



len_n = len(comment_normal)

len_t = len(comment_toxic)



num_eng_comments = 2800



inds_n_en =random_inds(len_n)

inds_t_en =random_inds(len_t)



toxic_list_en = []

lang_list_en = []

comment_list_en = []



for i in range(num_eng_comments//2):

    ind = inds_n_en[i]

    comment_list_en.append(comment_normal[ind])

    lang_list_en.append('en')

    toxic_list_en.append(0)

    

for i in range(num_eng_comments//2):

    ind = inds_t_en[i]

    comment_list_en.append(comment_toxic[ind])

    lang_list_en.append('en')

    toxic_list_en.append(1)
"""

mini_multilang_trainset_en = pd.DataFrame(columns = ['comment', 'toxic', 'lang'])

mini_multilang_trainset_en['comment'] = comment_list_en

mini_multilang_trainset_en['toxic'] = toxic_list_en

mini_multilang_trainset_en['lang'] = lang_list_en



mini_multilang_trainset_en.to_csv('mini_multilang_trainset_en_s2800.csv',index = False)

"""

mini_multilang_trainset_en = load_csv_data(preprocessed_dir, "mini_multilang_trainset_en_s2800")

mini_multilang_trainset_en.to_csv(output_dir + 'mini_multilang_trainset_en_s2800.csv', index = False)
frames = [mini_multilang_trainset_1, mini_multilang_trainset_2, mini_multilang_trainset_3, mini_multilang_trainset_en]



mini_multilang_trainset_combo  = pd.concat(frames)

mini_multilang_trainset_combo.head()
mini_multilang_trainset_combo = load_csv_data(preprocessed_dir, "mini_multilang_trainset_combo_manually_corrected")

mini_multilang_trainset_combo.to_csv(output_dir + 'mini_multilang_trainset_combo_manually_corrected.csv', index = False)



mini_multilang_trainset_combo.drop_duplicates(keep=False,inplace=True)

mini_multilang_trainset_combo.head()
mini_multilang_trainset_combo['lang'].value_counts()
import re



def clean_text(text):

    # clear data and time

    text = re.sub('\d{2}.\d{2}.\d{2}, \d{2}:\d{2}:\d{2}', '', text)

    text = re.sub('\d{2}.\d{2} \d{2}:\d{2}', '', text)

    

    # remove whitespace before and after word

    text = re.sub('-\s\r\n\|-\s\r\n|\r\n|[«»]|[""]|[><]|"[\[]]|//"', '', text)

    text = re.sub('[«»]|[""]|[><]|"[\[]]"', '', text)

    text = re.sub('[~-¿:;_"?*!@#$^&%()]|[+=]|[[]|[]]|[/]', ' ', text)

    

    text = re.sub(r'\r\n\t|\n|\r\t|\\n|&gt', ' ', text)

    text = re.sub(r'[\xad]|[\s+]', ' ', text)

    text = text.strip().lower()

    

    return text
def translit(string):

    """ This function works just fine """

    capital_letters = {



    }



    lower_case_letters = {

        u'а': u'a',

        u'б': u'b',

        u'в': u'v',

        u'г': u'g',

        u'д': u'd',

        u'е': u'e',

        u'ё': u'e',

        u'ж': u'zh',

        u'з': u'z',

        u'и': u'i',

        u'й': u'y',

        u'к': u'k',

        u'л': u'l',

        u'м': u'm',

        u'н': u'n',

        u'о': u'o',

        u'п': u'p',

        u'р': u'r',

        u'с': u's',

        u'т': u't',

        u'у': u'u',

        u'ф': u'f',

        u'х': u'h',

        u'ц': u'ts',

        u'ч': u'ch',

        u'ш': u'sh',

        u'щ': u'sch',

        u'ъ': u'',

        u'ы': u'y',

        u'ь': u'',

        u'э': u'e',

        u'ю': u'yu',

        u'я': u'ya',

        

        u'ö': u'o',

        u'ü': u'u',

        u'ş': u's',

        u'ç': u'c',

        u'ğ': u'g',

        u'â': u'a',

        u'i̇': u'i',

        

        u'ó': u'o',

        u'é': u'e',

        u'ñ': u'n',

        u'á': u'a',

        u'í': u'i',

        

        u'ã': u'a',

        u'ú': u'u',

        u'ê': u'e',

        u'à': u'a',

        u'õ': u'o',

        u'ĩ': u'i',

        u'è': u'i',

    }



    translit_string = ""



    for index, char in enumerate(string):

        if char in lower_case_letters.keys():

            char = lower_case_letters[char]

        elif char in capital_letters.keys():

            char = capital_letters[char]

            if len(string) > index+1:

                if string[index+1] not in lower_case_letters.keys():

                    char = char.upper()

            else:

                char = char.upper()

        translit_string += char



    return translit_string
def separate_to_lists(dataframe):

    comment_normal = []

    comment_toxic = []

    size = dataframe.shape[0]



    for i in range(size):

        comment = dataframe['comment'][i]

        lang = dataframe['lang'][i]

        comment = clean_text(comment)

        if lang != 'en':

            comment = translit(comment)            

        if dataframe['toxic'][i] == 0:

            comment_normal.append(comment)

        else:

            comment_toxic.append(comment)

        

    normar_labels = np.zeros(len(comment_normal))

    toxic_labels = np.ones(len(comment_toxic))

    

    return comment_normal, comment_toxic, normar_labels, toxic_labels
comment_normal, comment_toxic, labels_normal, labels_toxic = separate_to_lists(mini_multilang_trainset_combo)
# again preprocess text data



def preprocess(X_batch, y_batch):

    n_words = 128

    shape = tf.shape(X_batch) * tf.constant([1, 0]) + tf.constant([0, n_words])

    X_batch = tf.strings.substr(X_batch, 0, 300)

    X_batch = tf.strings.lower(X_batch)

    X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")

    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")

    X_batch = tf.strings.split(X_batch)

    X_batch =X_batch.to_tensor(shape=shape, default_value=b"<pad>")

    return X_batch, y_batch
# make tensorflow datasets



dataset_normal =  tf.data.Dataset.from_tensor_slices((tf.constant(comment_normal, dtype=tf.string), 

                                                      tf.constant(labels_normal, dtype=tf.float32)))

dataset_toxic =  tf.data.Dataset.from_tensor_slices((tf.constant(comment_toxic, dtype=tf.string), 

                                                     tf.constant(labels_toxic, dtype=tf.float32)))
from collections import Counter



toxic_vocabulary = Counter()

for X_batch, y_batch in dataset_toxic.batch(32).map(preprocess):

    for comment in X_batch:

        toxic_vocabulary.update(list(comment.numpy()))

        

normal_vocabulary = Counter()

for X_batch, y_batch in dataset_normal.batch(32).map(preprocess):

    for comment in X_batch:

        normal_vocabulary.update(list(comment.numpy()))
toxic_vocabulary.most_common()[:10]
normal_vocabulary.most_common()[:10]
len(toxic_vocabulary), len(normal_vocabulary)
new_toxic_vocabulary = toxic_vocabulary

new_normal_vocabulary = normal_vocabulary



toxic_vocabulary_list = list(new_toxic_vocabulary)

normal_vocabulary_list = list(new_normal_vocabulary)



for word in normal_vocabulary_list:

    if new_toxic_vocabulary[word] != 0:

        del new_toxic_vocabulary[word]



len(new_toxic_vocabulary), len(new_normal_vocabulary)
new_toxic_vocabulary.most_common()[:10]
normal_vocab_size = 40000

truncated_normal_vocabulary = [

    word for word, count in new_normal_vocabulary.most_common()[:normal_vocab_size]]



toxic_vocab_size = 20000

truncated_toxic_vocabulary = [

    word for word, count in new_toxic_vocabulary.most_common()[:toxic_vocab_size]]



merged_vocabulary = truncated_normal_vocabulary + truncated_toxic_vocabulary

len(merged_vocabulary)
word_to_id = {word: index for index, word in enumerate(merged_vocabulary)}



vocab_size = normal_vocab_size + toxic_vocab_size



for word in b"fuck this shit i hate it".split():

    print(word_to_id.get(word) or vocab_size)
for word in b"pochel v pizda urod".split():

    print(word_to_id.get(word) or vocab_size)
word_to_id
num_oov_buckets = 5000



words = tf.constant(merged_vocabulary)

word_ids = tf.range(len(merged_vocabulary), dtype=tf.int64)

vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)



table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)



table.lookup(tf.constant([b"fuck this shit i hate it".split()]))
# prepare efficient tensorflow train set



all_comments = comment_normal + comment_toxic

all_labels = np.concatenate((np.zeros(len(comment_normal)), np.ones(len(comment_toxic))), axis = 0)



train_dataset=  tf.data.Dataset.from_tensor_slices((tf.constant(all_comments, dtype=tf.string), 

                                                      tf.constant(all_labels, dtype=tf.float32)))



def encode_words(X_batch, y_batch):

    return table.lookup(X_batch), y_batch



batch_size = 128



train_set = train_dataset.repeat().shuffle(50000).batch(batch_size).map(preprocess)

train_set = train_set.map(encode_words).prefetch(1)
np.random.seed(42)

tf.random.set_seed(42)



train_size = len(all_comments)

embed_size = 128

model = keras.models.Sequential([

    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,

                           mask_zero=True,

                           input_shape=[None]),

    keras.layers.GRU(128, return_sequences=True),

    keras.layers.GRU(128),

    keras.layers.Dense(1, activation="sigmoid")

])



optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001)



model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])



history = model.fit(train_set, steps_per_epoch=train_size // batch_size, epochs=8)
model.save(output_dir + "model.h5")

#model.load_weights(output_dir + "model.h5")
plt.plot(np.arange(len(history.history["loss"])) + 0.5, history.history["loss"], "b.-", label="Training loss")
plt.plot(np.arange(len(history.history["accuracy"])) + 0.5, history.history["accuracy"], "b.-", label="Training accuracy")
pred_list = []

label_list = []



num_taken = all_labels.shape[0]//batch_size



for (X_batch, y_batch) in train_set.take(num_taken):

    batch_predictions = model.predict(X_batch)

    

    for prediction in batch_predictions:

        pred_list.append(prediction)

        

    for label in y_batch:

        label_list.append(label)

        

y_pred = np.asarray(pred_list).reshape((len(pred_list),))

y_train = np.asarray(label_list).reshape((len(label_list),))

y_train_pred = np.around(y_pred)
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_train, y_train_pred)

cm
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score, recall_score



print("accuracy_score: " + str(accuracy_score(y_train, y_train_pred)))

print("precision_score: " + str(precision_score(y_train, y_train_pred)))

print("recall_score: " + str(recall_score(y_train, y_train_pred)))
# prepare validation set



valid_data_b = load_csv_data(input_dir,"validation-processed-seqlen128")



valid_data_b['lang'].value_counts()
def separate_to_lists_2(dataframe):

    comments = []

    labels = []

    size = dataframe.shape[0]



    for i in range(size):

        lang = dataframe['lang'][i]

        label = dataframe['toxic'][i]

        comment = dataframe['comment_text'][i]

        comment = clean_text(comment)

        if lang != 'en':

            comment = translit(comment)                

        comments.append(comment)

        labels.append(label)

    

    return comments, labels



valid_comments, valid_labels = separate_to_lists_2(valid_data_b)
# tensorflow dataset



dataset_valid =  tf.data.Dataset.from_tensor_slices((tf.constant(valid_comments, dtype=tf.string), 

                                                      tf.constant(valid_labels, dtype=tf.float32)))



valid_set = dataset_valid.repeat().batch(batch_size).map(preprocess)

valid_set = valid_set.map(encode_words).prefetch(1)
num_taken = len(valid_labels)//batch_size



model.evaluate(valid_set.take(num_taken))
valid_pred_list = []

valid_label_list = []



num_taken = len(valid_labels)//batch_size



for (X_batch, y_batch) in valid_set.take(num_taken):

    batch_predictions = model.predict(X_batch)

    

    for prediction in batch_predictions:

        valid_pred_list.append(prediction)

        

    for label in y_batch:

        valid_label_list.append(label)

        

y_valid_prob = np.asarray(valid_pred_list).reshape((len(valid_pred_list),))

y_valid = np.asarray(valid_label_list).reshape((len(valid_label_list),))

y_valid_pred = np.around(y_valid_prob)
cm = confusion_matrix(y_valid, y_valid_pred)

cm
print("accuracy_score: " + str(accuracy_score(y_valid, y_valid_pred)))

print("precision_score: " + str(precision_score(y_valid, y_valid_pred)))

print("recall_score: " + str(recall_score(y_valid, y_valid_pred)))
# continue training on validation data



train_size = len(valid_labels)



history = model.fit(valid_set, steps_per_epoch=train_size // batch_size, epochs=5)
valid_pred_list = []

valid_label_list = []



num_taken = len(valid_labels)//batch_size



for (X_batch, y_batch) in valid_set.take(num_taken):

    batch_predictions = model.predict(X_batch)

    

    for prediction in batch_predictions:

        valid_pred_list.append(prediction)

        

    for label in y_batch:

        valid_label_list.append(label)

        

y_valid_prob = np.asarray(valid_pred_list).reshape((len(valid_pred_list),))

y_valid = np.asarray(valid_label_list).reshape((len(valid_label_list),))

y_valid_pred = np.around(y_valid_prob)



cm = confusion_matrix(y_valid, y_valid_pred)

cm
test_data_b = load_csv_data(input_dir,"test-processed-seqlen128")

test_data_b.head()
test_data_b.shape
def separate_to_lists_test_data(dataframe):

    comments = []

    size = dataframe.shape[0]



    for i in range(size):

        comment = dataframe['comment_text'][i]

        comment = clean_text(comment)

        comment = translit(comment)

        comments.append(comment)

    

    return comments
test_comments = separate_to_lists_test_data(test_data_b)

pseudo_labels = np.zeros(len(test_comments))
len(test_comments)
dataset_test =  tf.data.Dataset.from_tensor_slices((tf.constant(test_comments, dtype=tf.string), 

                                                      tf.constant(pseudo_labels, dtype=tf.float32)))
test_set = dataset_test.batch(batch_size=1).map(preprocess)

test_set = test_set.map(encode_words).prefetch(1)
test_pred_list = []

#counter = 0

for data in test_set.as_numpy_iterator():

    X, y = data

    prediction = model.predict(X)

    test_pred_list.append(prediction)

    #counter += 0

    #if counter%5000 == 0:

        #print(str(counter/63812) +"%")

len(test_pred_list)    
y_test_prob = np.asarray(test_pred_list).reshape((len(test_pred_list),))

#y_test_pred = np.around(y_test_prob)

y_test_pred = y_test_prob
counter = 0



for prediction in y_test_pred:

    if prediction >= 0.5:

        counter+=1

        

counter, counter/len(y_test_pred), len(y_test_pred)
submission = load_csv_data(input_dir, "sample_submission")

submission['toxic'] = y_test_pred

submission.to_csv(output_dir + 'submission.csv',index = False)