import gensim

print("gensim version: ", gensim.__version__)



import pandas as pd

print("pandas version: ", pd.__version__)



import requests

print("requests version: ", requests.__version__)



import re

print("re version: ", re.__version__)



import argparse

print("argparse version: ", argparse.__version__)



import time



import sys

import os.path

import multiprocessing
DIR_DATA_A = "../input/ukara-test-phase/"

DIR_DATA_B = "../input/ukara-test-phase/"

data_A_train = pd.read_csv("{}/data_train_A.csv".format(DIR_DATA_A))

data_A_dev = pd.read_csv("{}/data_dev_A.csv".format(DIR_DATA_A))

data_A_test = pd.read_csv("{}/data_test_A.csv".format(DIR_DATA_A))



data_B_train = pd.read_csv("{}/data_train_B.csv".format(DIR_DATA_B))

data_B_dev = pd.read_csv("{}/data_dev_B.csv".format(DIR_DATA_B))

data_B_test = pd.read_csv("{}/data_test_B.csv".format(DIR_DATA_B))
def preprocess(text):

    text = text.strip()

    text = text.lower()

    text = re.sub('[^0-9a-zA-Z]+', ' ', text)

    text = re.sub(' +', ' ', text).strip()

    return text





start_preprocess1 = time.time()

data_A_train['RESPONSE'] = data_A_train['RESPONSE'].apply(lambda x: preprocess(x))

data_A_dev['RESPONSE'] = data_A_dev['RESPONSE'].apply(lambda x: preprocess(x))

data_A_test['RESPONSE'] = data_A_test['RESPONSE'].apply(lambda x: preprocess(str(x)))



data_B_train['RESPONSE'] = data_B_train['RESPONSE'].apply(lambda x: preprocess(x))

data_B_dev['RESPONSE'] = data_B_dev['RESPONSE'].apply(lambda x: preprocess(x))

data_B_test['RESPONSE'] = data_B_test['RESPONSE'].apply(lambda x: preprocess(str(x)))

end_preprocess1 = time.time()





stimulus_a = ["Pemanasan global terjadi karena peningkatan produksi karbon dioksida yang dihasilkan oleh pembakaran fosil dan konsumsi bahan bakar yang tinggi.",

"Salah satu akibat adalah mencairnya es abadi di kutub utara dan selatan yang menimbulkan naiknya ketinggian air laut.",

"kenaikan air laut akan terjadi terus menerus meskipun dalam hitungan centimeter akan mengakibatkan perubahan yang signifikan.",

"Film “Waterworld”, adalah film fiksi ilmiah yang menunjukkan akibat adanya pemanasan global yang sangat besar sehingga menyebabkan bumi menjadi tertutup oleh lautan.",

"Negara-negara dan daratan yang dulunya kering menjadi tengelamn karena terjadi kenaikan permukaan air laut.",

"Penduduk yang dulunya bisa berkehidupan bebas menjadi terpaksa mengungsi ke daratan yang lebih tinggi atau tinggal diatas air.",

"Apa yang akan menjadi tantangan bagi suatu penduduk ketika terjadi situasi daratan tidak dapat ditinggali kembali karena tengelam oleh naiknya air laut."]



stimulus_b = ["Sebuah toko baju berkonsep self-service menawarkan promosi dua buah baju bertema tahun baru seharga Rp50.000,00. sebelum baju bertema tahun baru dibagikan kepada pembeli, sebuah layar akan menampilkan tampilan gambar yang menampilkan kondisi kerja di dalam sebuah pabrik konveksi/pembuatan baju. ",

"Kemudian pembeli diberi program pilihan untuk menyelesaikan pembeliannya atau menyumpangkan Rp50.000,00 untuk dijadikan donasi pembagian baju musim dingin di suatu daerah yang membutuhkan.",

"Delapan dari sepuluh pembeli memilih untuk memberikan donasi.",

"Menurut anda mengapa banyak dari pembeli yang memilih berdonasi?"]



data_stimulus = []



for text in stimulus_a:

    data_stimulus.append(preprocess(text))

    

for text in stimulus_b:

    data_stimulus.append(preprocess(text))

    

data_stimulus.extend(data_A_train['RESPONSE'].values)

data_stimulus.extend(data_A_dev['RESPONSE'].values)

data_stimulus.extend(data_A_test['RESPONSE'].values)

data_stimulus.extend(data_B_train['RESPONSE'].values)

data_stimulus.extend(data_B_dev['RESPONSE'].values)

data_stimulus.extend(data_B_test['RESPONSE'].values)
print(len(data_stimulus))

data_stimulus[0:3]
!wget http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/mono/OpenSubtitles.raw.id.gz -O dataset.txt.gz

!gzip -d dataset.txt.gz

!tail dataset.txt
def download(link, file_name):

    with open(file_name, "wb") as f:

        print("Downloading %s" % file_name)

        response = requests.get(link, stream=True)

        total_length = response.headers.get('content-length')



        if total_length is None: # no content length header

            f.write(response.content)

        else:

            dl = 0

            total_length = int(total_length)

            for data in response.iter_content(chunk_size=4096):

                dl += len(data)

                f.write(data)

                done = int(50 * dl / total_length)

                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )

                sys.stdout.flush()



def get_id_wiki(dump_path):

    if not os.path.isfile(dump_path):

        url = 'https://dumps.wikimedia.org/idwiki/latest/idwiki-latest-pages-articles.xml.bz2'

        download(url, dump_path)

    return gensim.corpora.WikiCorpus(dump_path, lemmatize=False, dictionary={})
dump_path = 'idwiki-latest-pages-articles.xml.bz2'

id_wiki = get_id_wiki(dump_path)
dim = 100

model_path = 'idwiki_word2vec_{}.model'.format(dim)

extracted_path = 'idwiki.txt'
print('Extracting text...')

start_preprocess2 = time.time()

with open(extracted_path, 'w') as f:

    # ukara

    i_ukara = 0

    word_ukara = 0

    for text in data_stimulus:

        test_processed = text.strip()

        f.write(test_processed + '\n')

        i_ukara += 1

        word_ukara += len(test_processed.split())



    # opensubs

    i_opensubs = 0

    word_opensubs = 0

    with open('dataset.txt') as f_opensubs:

        opensubs = f_opensubs.readlines()

        for text in opensubs:

            test_processed = preprocess(text).strip()

            f.write(test_processed + '\n')

            i_opensubs += 1

            word_opensubs += len(test_processed.split())



    # wikipedia

    i_wiki = 0

    word_wiki = 0

    for text in id_wiki.get_texts():

        text = ' '.join(text)

        f.write(text + '\n')

        i_wiki += 1

        word_wiki += len(text.split())

        

    end_preprocess2 = time.time()

            

    print('total ukara text: ', str(i_ukara))

    print('total ukara word:', str(word_ukara))

    print('total opensubs text: ', str(i_opensubs))

    print('total opensubs word:', str(word_opensubs)) 

    print('total wikipedia text: ', str(i_wiki))

    print('total wikipedia word:', str(word_wiki)) 
def build_model(extracted_path, model_path, dim):

    sentences = gensim.models.word2vec.LineSentence(extracted_path)

    id_w2v = gensim.models.word2vec.Word2Vec(sentences, size=dim, workers=multiprocessing.cpu_count()-1)

    id_w2v.save(model_path)

    return id_w2v
print('Building the model...')

start_training1 = time.time()

model = build_model(extracted_path, model_path, dim)

end_training1 = time.time()

print('Saved model:', model_path)
print("Total word2vec vocabulary: ", len(model.wv.vocab))
print("Time elapsed preprocessing data: {} second".format((end_preprocess1-start_preprocess1)+(end_preprocess2-start_preprocess2)))
print("Time elapsed training wordembedding: {} second".format((end_training1-start_training1)))