import os; import joblib; import numpy as np
import pandas as pd; import random; import time
import pickle; import sys; import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk; from random import shuffle
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.text import one_hot, text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline

# INPUT PATHS
metadata_path = '../input/CORD-19-research-challenge/metadata.csv'
text_folders = ['../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json',
                '../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json',
                '../input/CORD-19-research-challenge/custom_license/custom_license/pdf_json',
                '../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json']

# OUTPUT PATHS
clean_text_folder = '/kaggle/working/CleanTexts'
tokens_path = '/kaggle/working/tokens.txt'
clean_metadata_path = '/kaggle/working/metadata_clean.csv'
cluster_model_path = '/kaggle/working/cluster_model.h5'
cluster_data_path = '/kaggle/working/clusters.csv'

# USER-DEFINED FUNCTIONS

def GetTextFromJSON(path):
    replace_phrases = ['ACKNOWLEDGMENTS',
                       'This is an Open Access article distributed under the terms of the Creative Commons Attribution Non-Commercial License',
                       'http://creativecommons.org/licenses/bync/4.0/','It is made available under a',
                       'CC-BY-NC-ND 4.0 International license author/funder.',
                       'No reuse allowed without permission.','All rights reserved.','(which was not peer-reviewed)',
                       'bioRxiv preprint','The copyright holder for this preprint','author/funder',
                       'doi:',
                       'The copyright holder for this preprint (which was not peer-reviewed) is the']
    with open(path, 'r') as f:
        content = f.read()
    text = json.loads(content)['body_text']
    txt_out = ''
    for body_text in text:
        txt_out += body_text['text']
    for p in replace_phrases:
        if p in txt_out: txt_out = txt_out.replace(p, '')
    return txt_out

def Lemmatize(instr):
    vowels = ['a','e','i','o','u']; words = list(set(instr.split()))
    ignore_words = ['yes','hivaids','politics','building','series','analysis','analyses','fed','bled','bed','subcutaneous','viscous','synthesis',
                    'species','whereas','sars','mers','mumps','bias','physics','paralysis','diagnosis',
                    'bronchitis','king','string','apply','diabetes','nothing','fly','sly','offspring',
                    'dues','wing','during','early','mumps','ring','sing','fling','spring','bing','daily',
                    'does','have','only','comply','family','supply','italy','led','lied','lying','dying',
                    'tied','died','need','untreated','according','unexpected','going','especially','rings','differing','ongoing','bleed','freed','always','creed','springs','thing','something','nothing','anything']
    for word in words:
        if word not in ignore_words:
            new_word = word
            if len(new_word) > 4:
                lemmatized = False
                if new_word[-2:] == 'ss': double_s = True
                else: double_s = False
                if new_word[-2:] in ['ed','es'] and new_word[-3:] not in ['ies','ied'] and not lemmatized:
                    new_word = new_word[:-2]
                    if new_word[-1] in vowels or new_word[-1] in ['c','g','v','u'] or new_word[-2:] in ['pl','yp','gl','cl','yt']: new_word = new_word + 'e'
                    lemmatized = True
                if new_word[-1] == 's' and not double_s and new_word[-2] not in ['u','i'] and new_word[-3:] != 'ies' and new_word[-4:] != 'ings' and not lemmatized:
                    new_word = new_word[:-1]
                    if new_word[-1] in vowels: new_word = new_word + 'e'
                    lemmatized = True
                if new_word[-3:] == 'ing' and not lemmatized:
                    new_word = new_word[:-3]
                    if new_word[-1] in vowels or new_word[-1] in ['c','g','u','i','v'] or new_word[-2:] in ['pl','yp','gl','cl','yt']: new_word = new_word + 'e'
                    lemmatized = True
                if new_word[-3:] == 'ings' and not lemmatized:
                    new_word = new_word[:-4]
                    if new_word[-1] in vowels or new_word[-1] in ['c','g','u','i','v'] or new_word[-2:] in ['pl','yp','cl','gl','yt']: new_word = new_word + 'e'
                    lemmatized = True
                if new_word[-3:] == 'ied' and not lemmatized:
                    new_word = new_word[:-3] + 'y'
                    lemmatized = True
                if new_word[-3:] == 'ies' and not lemmatized:
                    new_word = new_word[:-3] + 'y'
                    lemmatized = True
                if new_word[-2:] == 'ly' and not lemmatized:
                    if word == 'probably': new_word = 'probable'
                    else: new_word = new_word[:-2]
                    lemmatized = True
                if new_word[-3:] == 'ful' and not lemmatized:
                    new_word = new_word[:-3]
                    lemmatized = True
                if new_word == 'viruse': new_word = 'virus'
                if new_word == 'detaile': new_word = 'detail'
                if new_word == 'obtaine': new_word = 'obtain'
                if new_word == 'looke': new_word = 'look'
                if new_word == 'neede': new_word = 'need'
                if new_word == 'showe': new_word = 'show'
                if new_word in ['modell','modele']: new_word = 'model'
                if new_word == 'staine': new_word = 'stain'
                if new_word == 'spreade': new_word = 'spread'
                if new_word == 'mixe': new_word = 'mix'
                if new_word == 'maintaine': new_word = 'maintain'
                if new_word == 'transferr': new_word = 'transfer'
                if new_word == 'transmitt': new_word = 'transmit'
                if new_word == 'followe': new_word = 'follow'
                if new_word == 'administere': new_word = 'administer'
                if new_word == 'includ': new_word = 'include'
                if new_word == 'compar': new_word = 'compare'
                if new_word == 'determin': new_word = 'determine'
                if new_word == 'analyz': new_word = 'analyze'
                if new_word == 'treate': new_word = 'treat'
                if new_word == 'reveale': new_word = 'reveal'
                if new_word == 'possib': new_word = 'possible'
                if new_word == 'provid': new_word = 'provide'
                if new_word == 'especial': new_word = 'especially'
                if new_word == 'unexpect': new_word = 'unexpected'
                if new_word == 'considere': new_word = 'considering'
                if new_word == 'ensur': new_word = 'ensure'
                if new_word == 'filtere': new_word = 'filter'
                if new_word == 'adenoviruse': new_word = 'adenovirus'
                if new_word == 'blott': new_word = 'blot'
                if new_word == 'licens': new_word = 'license'
                if new_word == 'preferr': new_word = 'prefer'
                if new_word == 'creat': new_word = 'create'
                if new_word == 'wheez': new_word = 'wheeze'
                if new_word == 'playe': new_word = 'play'
                if new_word == 'compris': new_word = 'comprise'
                if new_word == 'examin': new_word = 'examine'
                if new_word == 'triggere': new_word = 'trigger'
                if new_word == 'offere': new_word = 'offer'
                if new_word == 'indicat': new_word = 'indicate'
                if new_word == 'requir': new_word = 'require'
                if new_word == 'isolat': new_word = 'isolate'
                if new_word == 'describ': new_word = 'describe'
                if new_word == 'diseas': new_word = 'disease'
                if new_word == 'associat': new_word = 'associate'
                if new_word == 'increas': new_word = 'increase'
                instr = instr.replace(word, new_word)
    return instr

def TopDictValues(in_dict, N = 5):
    if len(in_dict) < N: N = len(in_dict)
    values = list(in_dict.values())
    idx_list = []
    for i in range(N):
        idx = np.argmax(values)
        values[idx] = -1
        idx_list.append(idx)
    return idx_list

def TopWords(instr, N = 5, frequency_based = False, wc = None):
    vocab = list(set(instr.split()))
    for word in vocab:
        if word.isdigit() or word in STOPWORDS:
            instr = instr.replace(word, '')
    vocab = list(set(instr.split()))
    if len(vocab) < N: N = len(vocab)
    i = 0; counts = np.zeros([len(vocab)])
    for word in vocab:
        if frequency_based and len(wc.keys()) > 0: counts[i] = instr.count(word) / np.log(wc[word])
        else: counts[i] = instr.count(word)
        i += 1
    return list(np.array(vocab)[TopValues(list(counts), N = N)])

def TopValues(in_list, N = 5):
    if len(in_list) < N: N = len(in_list)
    idx_list = []
    for i in range(N):
        idx = np.argmax(in_list)
        in_list[idx] = -1
        idx_list.append(idx)
    return idx_list

def TokenizeText(text, tokens):
    text = np.array(text.split())
    keep_idxs = []
    for w in set(text):
        if w in tokens:
            keep_idxs += list(np.where(text == w)[0])
    keep_idxs.sort()
    return ' '.join(list(np.array(text)[keep_idxs]))

def ReplaceSymbols(instr, symbols = ['®', '∼', '°', '–', '′', '⪢',
                                     '‘', '’', '♀', '!', '?', '±', '·', '“', '”', '"', "'",
                                   '(', ')', '>', '<', '[', ']', '{', '}', '/', '.',
                                     ',', ':', ';', '-', '=',
                                   '+', '~', '@', '#', '$', '%', '^', '&', '*', '_']):
    for sym in symbols:
        if sym in instr: instr = instr.replace(sym, '')
    return instr
if not os.path.exists(clean_text_folder):
    os.mkdir(clean_text_folder)
    df = pd.read_csv(metadata_path)
    # FILTER FOREIGN JOURNALS
    foreign_journals = ['Traité de médecine vasculaire.','Zentralblatt für Bakteriologie','Zeitschrift für Immunitaetsforschung','Experimentelle und Klinische Immunologie',
                    'Zentralblatt für Bakteriologie, Mikrobiologie und Hygiene. 1. Abt. Originale. A, Medizinische Mikrobiologie, Infektionskrankheiten und Parasitologie',
                    'Zentralblatt für Bakteriologie, Mikrobiologie und Hygiene. Series A: Medical Microbiology, Infectious Diseases, Virology, Parasitology',
                    'Zentralblatt für Bakteriologie. 1. Abt. Originale A, Medizinische Mikrobiologie, Infektionskrankheiten und Parasitologie',
                    'Z Gesundh Wiss','Yi chuan = Hereditas','Zdr Varst','Zhejiang Da Xue Xue Bao Yi Xue Ban','Zhongguo Dang Dai Er Ke Za Zhi','Zhongguo Fei Ai Za Zhi','Zhonghua Bing Li Xue Za Zhi','Zhonghua Er Bi Yan Hou Tou Jing Wai Ke Za Zhi',
                    'Zhonghua Er Ke Za Zhi','Zhonghua Fu Chan Ke Za Zhi','Zhonghua Gan Zang Bing Za Zhi','Zhonghua Jie He He Hu Xi Za Zhi',
                    'Zhonghua Kou Qiang Yi Xue Za Zhi','Zhonghua Lao Dong Wei Sheng Zhi Ye Bing Za Zhi','Zhonghua Liu Xing Bing Xue Za Zhi','Zhonghua Nei Ke Za Zhi','Zhonghua Shao Shang Za Zhi','Zhonghua Wai Ke Za Zhi','Zhonghua Wei Chang Wai Ke Za Zhi',
                    'Zhonghua Xin Xue Guan Bing Za Zhi','Zhonghua Xue Ye Xue Za Zhi','Zhonghua Yan Ke Za Zhi','Zhonghua Yi Xue Za Zhi','Zhonghua Yu Fang Yi Xue Za Zhi','Zhonghua Zhong Liu Za Zhi',
                    'Wirtschaftsdienst','Virologica Sinica','Técnicas y Métodos de Laboratorio Clínico','Türk Pediatri Arşivi','Tratado de medicina de urgencias pediátricas','Trauma Berufskrankh','Inmunología','Journal Européen des Urgences et de Réanimation',
                    'Nota Técnica-SVS/SES-RJ','Actualités Pharmaceutiques Hospitalières','Réanimation',"Revue Française d'Allergologie et d'Immunologie Clinique",
                    'Revista Española de Anestesiología y Reanimación','Klinische Infektiologie','Acta Genetica Sinica','Journal de Mycologie Médicale','Sheehy. Manual de urgencia de enfermería','Rev Saude Publica','Manual de Otorrinolaringología Infantil',
                    '250 Examens de Laboratoire','Nursing (Ed. española)','La Presse Médicale Formation','European Research in Telemedicine / La Recherche Européenne en Télémédecine',
                    "Canadian Journal of Anesthesia/Journal canadien d'anesthésie",'Memórias do Instituto Oswaldo Cruz','Atención Primaria','Bulletin du Cancer',
                    'Revue des Maladies Respiratoires']
    sdf = df
    for j in foreign_journals:
        sdf = sdf[sdf['journal'] != j]
    foreign_tags = ['è','é','ü',' des ',' du ',' de ',' y ','í','ã','ç','ñ',' un ',' une ']
    foreign_j = []
    for j in set(sdf['journal'].dropna()):
        if j in foreign_journals:
            sdf = sdf[sdf['journal'] != j]    
            foreign_j.append(j)
        else:
            for tag in foreign_tags:
                if tag in j:
                    sdf = sdf[sdf['journal'] != j]    
                    foreign_j.append(j)
                    break
    # IGNORE ENTRIES WITH NO TEXT
    sdf = sdf[sdf['full_text_file'].isna() == False]
    sdf = sdf[sdf['sha'].isna() == False]
    sdf = sdf[sdf['abstract'].isna() == False]
    sdf.to_csv(clean_metadata_path, index = False)
    metadf = pd.read_csv(clean_metadata_path)
    sha_keys = list(set(metadf['sha']))
    for folder in text_folders:
        flist = os.listdir(folder)
        print('\nProcessing Text in Folder:  {} ({} files)'.format(folder, len(flist)))
        i = 0
        for f in flist:
            key = f.split('.')[0]
            if key in sha_keys:
                fpath = '{}/{}'.format(folder, f)
                text = GetTextFromJSON(fpath)
                text = text.lower()
                text = ReplaceSymbols(text)
                text = Lemmatize(text)
                outpath = '{}/{}.txt'.format(clean_text_folder, key)
                if len(text) > 100 and len(set(text)) > 3:
                    try:
                        with open(outpath,'w') as f:
                            f.write(text)
                    except:
                        if os.path.exists(outpath): os.remove(outpath)
            if i%1000==0: print(i)
            i += 1
    vocab_size = 5000
    all_texts, keep_idxs, wc = [],[],{}
    print('\nLoading Cleaned Text...'.format(vocab_size))
    for f in os.listdir(clean_text_folder):
        fpath = '{}/{}'.format(clean_text_folder, f)
        with open(fpath, 'r') as f:
            text = f.read()
        all_texts.append(text)
    print('\nCollecting {} Tokens...'.format(vocab_size))
    T = Tokenizer(num_words = vocab_size, lower = True, split = ' ')
    T.fit_on_texts(all_texts)
    word_counts = T.get_config()['word_counts']
    word_counts = word_counts.split(',')
    fragments = ['copyright','serv','involv','sequenc','sampl','framehift','differe','possib','analys','provid','differ','perpetuityis','ntly','ly','refseq','followe','allowe','vadr','authorfunder','perpetuitythe','httpsdoiorg','et','al','perpetuitywhich','permissionauthorfunder','peerreviewed','rights','medrxiv','permissionthe','biorxiv','preprint','use','cell','virus','case','study','display','figure','number','table','graph','holder','peerreviewe','fig','data','license','doi']
    ignore_words = fragments + ['ccbyncnd','grant','though','although','however','result','include','report','viral','genera']
    for entry in word_counts:
        st_idx = entry.index('"')
        end_idx = entry[st_idx+1:].index('"') + st_idx + 1
        key = entry[st_idx+1:end_idx]
        value = int(entry.split(': ')[-1].strip('}'))
        if '\\' not in key and not key.isdigit() and key.isalpha() and key not in STOPWORDS and key not in ignore_words and len(key) > 4 and value > 4:
            wc[key] = value
    # Retrieve the indices of the top N most common words
    idx_list = TopDictValues(wc, N = vocab_size)
    tokens = list(np.asarray(list(wc.keys()))[idx_list])
    print('Extracted Top {} Most Frequent Tokens.'.format(vocab_size))
    print('\nTOP 100:\n', ' | '.join(tokens[0:100]))
    print('\nWriting Tokens to File...')
    outstr=''
    for key in list(wc.keys()):
        outstr += '{}:{},'.format(key, wc[key])
    with open(tokens_path,'w') as f:
        f.write(outstr)
else: print(clean_text_folder, 'already exists. If you want to recreate or update this folder, you must first delete the folder and its contents.')
print('\n\nDONE.')
vocab_size = 2500; seq_len = 50; bag_size = 10; n_entries = 20000

# LOAD TEXT FILES
df = pd.read_csv(clean_metadata_path)

# GET TOKENS
keep_idxs = []; wc = {}; tokened_texts = []
with open(tokens_path,'r') as f:
    WC = f.read()
WC = WC.split(',')[:-1]
for pair in WC:
    key, value = pair.split(':')
    wc[key] = int(value)
idx_list = TopDictValues(wc, N = vocab_size)
tokens = list(np.asarray(list(wc.keys()))[idx_list]) 
print('Extracted {} Most Frequent Words.'.format(vocab_size))
print('\nTOP 100:\n', ' | '.join(tokens[0:100]))

# CREATE LOOKUP DICTIONARIES
word_to_int = dict((c, i) for i, c in enumerate(tokens))
int_to_word = dict((i, c) for i, c in enumerate(tokens))

# CREATE WORD VECTORS (LSTM inputs & outputs)
X,Y,sha_keys = [],[],[]
flist = os.listdir(clean_text_folder); nf = len(flist)
print('\nCreating {} {}-Word Vectors... (this part takes a while)'.format(n_entries, seq_len))
for i in range(n_entries):
    fname = random.choice(flist)
    fpath = '{}/{}'.format(clean_text_folder, fname)
    key = fname.split('.')[0]
    with open(fpath, 'r') as f:
        body = f.read()
    sequence = TokenizeText(body, tokens).split()
    if len(sequence) > seq_len:
        st_idx = random.randint(0, len(sequence) - seq_len - 1)
        X.append(to_categorical([word_to_int[word] for word in sequence[st_idx:st_idx + seq_len]], num_classes = vocab_size))
        Y.append(to_categorical([word_to_int[word] for word in TopWords(' '.join(sequence), N = bag_size, frequency_based = True, wc = wc)], num_classes = vocab_size).sum(axis = 0))
        sha_keys.append(key)
    if i%1000 == 0: print('  Progress:  {:6}%'.format(round(i/n_entries*100, 2)))
X = np.array(X)
Y = np.array(Y)
n_entries = len(X)
print('\nSuccessfully Created {} Entries.'.format(n_entries))
n_epochs = 100

# ASSEMBLE LSTM MODEL
model = Sequential()
model.add(LSTM(32, input_shape = np.shape(X)[1:], return_sequences = True))
model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(LSTM(32))
model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(Y.shape[-1], activation = 'relu'))

opt_type = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
model.compile(loss = 'mean_squared_error', optimizer = opt_type, metrics = ['accuracy'])
print(model.summary())

# TRAIN THE MODEL
history = model.fit(X, Y, batch_size = 128, epochs = n_epochs)

# PLOT TRAINING HISTORY
if n_epochs > 1:
    plt.plot(history.history['accuracy'], label = 'train acc')
    plt.title('LSTM Training History', fontsize = 16)
    plt.ylabel('Accuracy (%)', fontsize = 12)
    plt.xlabel('Epoch', fontsize = 12)
    plt.legend(loc = "upper left")
    plt.show()
n_pts = 2000; n_clusters = 6; layer_idx = -1

# GENERATE DL LAYER OUTPUT
print('\nGenerating Features...')
idxs = list(range(0,len(X)))
shuffle(idxs)
idxs = idxs[0:n_pts]
get_3rd_layer_output = keras.backend.function([model.layers[0].input], [model.layers[layer_idx].output])
layer_outputs = get_3rd_layer_output([X[idxs]])[0]

# CLUSTER FEATURES
print('\nClustering Outputs...')
cluster_model = KMeans(n_clusters = n_clusters, n_init = 30, n_jobs = 2)
labels = cluster_model.fit_predict(layer_outputs)
centroids = cluster_model.cluster_centers_

# SAVE FITTED CLUSTER MODEL
joblib.dump(cluster_model, cluster_model_path)

alpha = 0.80; pt_size = 3; cmap = 'Paired'

print('\nCalculating PCAs...')
pca = PCA(n_components = 3)
pca_df = pca.fit_transform(layer_outputs)
pca_df = pd.DataFrame(pca_df, columns = ['PC1','PC2','PC3'])
pca_df['labels'] = labels
print('\nGenerating Cluster Plots...')
fig, axs = plt.subplots(1,3)
fig.set_figheight(6)
fig.set_figwidth(18)
axs[0].scatter(pca_df['PC1'], pca_df['PC2'], c = labels, s = pt_size, alpha = alpha, cmap = plt.cm.get_cmap(cmap, n_clusters))
axs[1].scatter(pca_df['PC2'], pca_df['PC3'], c = labels, s = pt_size, alpha = alpha, cmap = plt.cm.get_cmap(cmap, n_clusters))
axs[2].scatter(pca_df['PC1'], pca_df['PC3'], c = labels, s = pt_size, alpha = alpha, cmap = plt.cm.get_cmap(cmap, n_clusters))
plt.show()
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
xs = pca_df['PC1']
ys = pca_df['PC2']
zs = pca_df['PC3']
p = ax.scatter(xs, ys, zs, s = pt_size, alpha = alpha, c = labels, cmap = cmap)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
fig.colorbar(p)
plt.show()
# ASSIGN DOCUMENTS TO CLUSTER
print('\nProcessing Cluster Data...')
n_entries = len(X)
clustered_df = []
for i in range(len(sha_keys)):
    key = sha_keys[i]
    layer_output = get_3rd_layer_output([X[i:i+1]])[0]
    label = cluster_model.predict(layer_output)[0]
    dist = np.sqrt(np.sum((layer_output[0] - centroids[label])**2))
    title = df['title'][df['sha'] == key].values[0]
    abstr = df['abstract'][df['sha'] == key].values[0]
    clustered_df.append([key, label, dist, title, abstr])
    if i%1000 == 0: print('  Progress:  {:6}%'.format(round(i/n_entries*100, 2)))

# WRITE CLUSTER DATA TO FILE
clustered_df = pd.DataFrame(clustered_df, columns = ['sha','cluster','distance','title','abstract'])
print('\nSaving Cluster Data to File...')
clustered_df.to_csv(cluster_data_path, index = False)

# VIEW RESULTS CLOSEST TO EACH CLUSTER CENTROID
for i in range(centroids.shape[0]):
    sdf = clustered_df[clustered_df['cluster'] == i]
    top_titles = sdf.sort_values(by = 'distance')['title'][0:5]
    print('\nCLUSTER {} ({} entries):'.format(i, len(sdf)))
    j = 0
    for title in top_titles:
        print(' {:2})  {}'.format(j, title[0:100]))
        j += 1
