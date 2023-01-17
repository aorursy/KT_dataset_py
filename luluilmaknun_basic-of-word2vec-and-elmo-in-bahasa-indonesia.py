#Preparation



from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

import numpy as np

from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings('ignore')
!cp -r /kaggle/input/praktikumword2vecnlp/158 /kaggle/working/158

!git clone https://github.com/HIT-SCIR/ELMoForManyLangs

!pip install -e ELMoForManyLangs/

!cp ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json /kaggle/working/158/config.json
file = open("/kaggle/working/158/config.json", "w")



teks = '{"seed": 1, "gpu": 3, "train_path": "/users4/conll18st/raw_text/Indonesian/id-20m.raw", "valid_path": null, "test_path": null, "config_path": "/kaggle/working/ELMoForManyLangs/configs/cnn_50_100_512_4096_sample.json", "word_embedding": null, "optimizer": "adam", "lr": 0.001, "lr_decay": 0.8, "model": "/users4/conll18st/elmo/src/final_models/id.model", "batch_size": 32, "max_epoch": 10, "clip_grad": 5, "max_sent_len": 20, "min_count": 3, "max_vocab_size": 150000, "save_classify_layer": false, "valid_size": 0, "eval_steps": 10000}'



file.write(teks)

file.close()
import ELMoForManyLangs.elmoformanylangs as elmo
from gensim.models import Word2Vec



modelword2vec = Word2Vec.load("/kaggle/input/praktikumword2vecnlp/idwiki_word2vec_300.model")
def plot(model, words):

    

    arr = np.empty((0,100), dtype='f')

    word_labels = []

    

    for word in words:

        wrd_vector = model[word]

        word_labels.append(word)

        arr = np.append(arr, np.array([wrd_vector]), axis=0)

        

        

    # find tsne coords for 2 dimensions

    pca = PCA(n_components=2, copy=False, whiten=True)

    Y = pca.fit_transform(arr)



    x_coords = Y[:, 0]

    y_coords = Y[:, 1]

    # display scatter plot

    plt.scatter(x_coords, y_coords)



    for label, x, y in zip(word_labels, x_coords, y_coords):

        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)

    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)

    plt.show()
print(modelword2vec["ui"])
# Jawaban kode Soal 1



print(modelword2vec)



print("Each vector size: ", modelword2vec["ui"].shape)
plot(modelword2vec, ["jakarta", "bandung" , "bekasi" , "serpong" ,# Barat Jawa

             "surabaya" , "malang" , "yogyakarta"  , # Timur Jawa

             "banjarmasin" , "balikpapan" , "samarinda" , # Kalimantan

             "medan" , "palembang" , "jambi" , # Sumatera

             "manado" , "gorontalo" , "palu"  , # Sulawesi

            "ambon" , "sofifi" , "tual" , # Maluku

             "fakfak" , "jayapura" , "mamuju" ]) # Papua
# Jawaban kode nomor 2



## Mencari tahu kata apa saja yang ada pada model

# word_vectors = modelword2vec.wv

# print(word_vectors.vocab)



plot(modelword2vec, ["menaruh", "mencukupi", "mengambil", "menyebabkan",# Kata Kerja (verb)

                     "fakta" , "ilmu" , "dunia", "unsur", # Kata benda (noun)

                     "tetapi" , "agar" , "sehingga" , # Kata sambung (conj)

                    ])
modelword2vec.most_similar(positive = ["presiden"], topn=5)
modelword2vec.most_similar(positive = ["makan"], topn=5)
modelword2vec.most_similar(positive = ["inggris", "jakarta"], negative = ["indonesia"], topn=5)
plot(modelword2vec, ["inggris" , "london",

                     "filipina" , "manila" ,

                    "rusia" , "moscow" , 

                    "jepang" , "tokyo",

                    "taiwan" , "taipei",

                    "kanada" , "ottawa"])
modelword2vec.similarity('raja' , 'presiden')
modelword2vec.doesnt_match(['jokowi' , 'sby' , 'suharto' , 'sule'])
# Jawaban kode nomor 3



# Mencari similaritas dari semua dimensi vektor

print("Similarity masing-masing pasangan kata:")

print("dompet:uang ->", modelword2vec.similarity('dompet' , 'uang'))

print("a. gunung:harimau ->", modelword2vec.similarity('gunung' , 'harimau'))

print("b. tas sekolah:buku ->", modelword2vec.similarity('tas' , 'buku'))

print("c. laut:garam ->", modelword2vec.similarity('laut' , 'garam'))

print("d. burung:sangkar ->", modelword2vec.similarity('burung' , 'sangkar'))

print("e. kandang:ayam ->", modelword2vec.similarity('kandang' , 'ayam'))



# Yang divisualisasikan hanya 2 vektor dimensi penting <-- jadi mungkin tidak sesuai yang di atas visualisasinya

plot(modelword2vec, ["dompet" , "uang",

                     "gunung" , "harimau" ,

                     "tas" , "buku" , 

                     "laut" , "garam",

                     "burung" , "sangkar",

                     "kandang" , "ayam"])
# Jawaban kode nomor 4



def cosine_sim(w1, w2):

    wv1 = modelword2vec[w1]

    wv2 = modelword2vec[w2]

    

    return np.dot(wv1.T, wv2) / (np.linalg.norm(wv1) * np.linalg.norm(wv2))



# tes pada raja dan presiden, expected: 0.3518753

print(cosine_sim('raja', 'presiden'))
# Ambil word2vec setiap kata

w2v = dict(zip(modelword2vec.wv.index2word, modelword2vec.wv.syn0))



class MeanEmbeddingVectorizer(object):

    def __init__(self, word2vec):

        self.word2vec = word2vec

        self.dim = len(word2vec['dan'])

        

    def tokenize(self, sentences):

        return [sentence.lower().split(" ") for sentence in sentences]



    

    def transform(self, X):

        # Ambil kata-katanya lalu rata-rata

        return np.array([

            np.mean([self.word2vec[w] for w in words if w in self.word2vec]

                    or [np.zeros(self.dim)], axis=0)

            for words in X

        ])

    

vectorizer = MeanEmbeddingVectorizer(w2v)
# Task : Sentimen Analisis

# 0 negatif , 1 positif



train_teks = ["Saya sedih karena warung pasta ditutup" ,

              "Sekarang adalah waktunya untuk berbahagia dan bersyukur" , 

              "Bangun segan , mati tidak mau ketika menghadapi sprint" ,

              "OH MY GOD AKU DAPAT TANDA TANGAN LISA DARI BLACKPINK" ,

              "NLP itu seru !" ,

              "Gue bahagia karena keterima magang" ,

              "' Mampus aku bisnis aku bakal bangkrut ' , pikir CEO Traveloka" , 

              "Cacing di perut mencuri semua nutrisi penting"

             ]



train_y = [0 , 1, 0 ,1 , 1,  1, 0 , 0]



train_X = vectorizer.transform(vectorizer.tokenize(train_teks))



test_teks = ["Memang tidak salah untuk berharap , tapi aku harus tahu kapan berhenti" ,

             "Mengapur berdebu , kita semua gagal , ambil s'dikit tisu , bersedihlah secukupnya" , 

             "Ini adalah waktunya kamu untuk bersinar" ,

             "Kita akan berhasil menghadapi cobaan "

             ]



test_y = [0 , 0 , 1 , 1]



test_X = vectorizer.transform(vectorizer.tokenize(test_teks))

# Jawaban kode nomor 5



from sklearn.svm import LinearSVC



# Accuracy report

def classification_report(predicted, actual):

    unique_pred = np.unique(predicted, return_counts=True)[1]

    p = unique_pred[1]

    n = unique_pred[0]

    N = p + n

    

    tp= sum((int(actual[i] == 1 and predicted[i] == 1) for i in range(len(actual))))

    tn = sum((int(actual[i] != 1 and predicted[i] != 1) for i in range(len(actual))))

    

    fp = n - tn

    fn = p - tp

    tp_rate = tp / p

    tn_rate = tn / n

    

    accuracy = (tp + tn) / N



    # Macro average precision

    precision1 = tp / (tp + fp)

    precision2 = tn / (tp + fp)

    precision = precision1 + precision2 / 2

    

    recall = tp_rate

    f_measure = 2 * ((precision * recall) / (precision + recall))

    

    return (accuracy,precision,recall,f_measure)





# Predicting

svm = LinearSVC()

svm.fit(train_X, train_y)



test_pred = svm.predict(test_X)

report = classification_report(test_pred, test_y)



print("Accuracy: ", report[0])

print("Precision: ", report[1])

print("Recall: ", report[2])

print("F1-score: ", report[3])
from gensim.test.utils import common_texts

from gensim.models import Word2Vec
common_texts
model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)

model.save("word2vec.model")

model = Word2Vec.load("word2vec.model")
e = elmo.Embedder('/kaggle/working/158')
def encode_elmo(elmo,  kalimat):

    vektor = elmo.sents2elmo([kalimat.split(" ")])

    return vektor[0]
matematika = encode_elmo(e, "Tujuh kali dua sama dengan empat belas")[1]

sungai1 = encode_elmo(e, "Saya tinggal di samping kali Ciliwung")[4]

sungai2 = encode_elmo(e, "Indonesia Lawyers Club mempertanyakan kualitas kali yang menjadi water way")[5]

sekarang = encode_elmo(e, "Untuk kali ini dia yang kena batunya")[1]

perbandingan = encode_elmo(e, "Sejak korona , harga telur menjadi dua kali lipat")[7]

frekuensi = encode_elmo(e, "Saya sudah tidur lima kali")[4]
arr = np.empty((0,1024), dtype='f')

arr = np.append(arr, np.array([matematika]), axis=0)

arr = np.append(arr, np.array([sungai1]), axis=0)

arr = np.append(arr, np.array([sungai2]), axis=0)

arr = np.append(arr, np.array([sekarang]), axis=0)

arr = np.append(arr, np.array([perbandingan]), axis=0)

arr = np.append(arr, np.array([frekuensi]), axis=0)



pca = PCA(n_components=2, copy=False, whiten=True)

Y = pca.fit_transform(arr)



x_coords = Y[:, 0]

y_coords = Y[:, 1]



plt.scatter(x_coords, y_coords)



nama_label = ['matematika' , 'sungai1' , 'sungai2' , 

              'sekarang' , 'perbandingan' , 'frekuensi']

for label, x, y in zip(nama_label, x_coords, y_coords):

    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)

plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)

plt.show()