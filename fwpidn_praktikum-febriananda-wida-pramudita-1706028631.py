c
!ls /kaggle/input/praktikumword2vecnlp/idwiki.txt
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
import os
from IPython.display import FileLink
FileLink(r'/kaggle/input/praktikumword2vecnlp/idwiki.txt')
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
print("dimensi dari vektor word2vec yang kita gunakan sekarang adalah:", modelword2vec["ui"].shape[0])
plot(modelword2vec, ["jakarta", "bandung" , "bekasi" , "serpong" ,# Barat Jawa
             "surabaya" , "malang" , "yogyakarta"  , # Timur Jawa
             "banjarmasin" , "balikpapan" , "samarinda" , # Kalimantan
             "medan" , "palembang" , "jambi" , # Sumatera
             "manado" , "gorontalo" , "palu"  , # Sulawesi
            "ambon" , "sofifi" , "tual" , # Maluku
             "fakfak" , "jayapura" , "mamuju" ]) # Papua
# Jawaban kode nomor 2
plot(modelword2vec, ["sungai", "danau", "laut", "rawa", # tempat alami di perairan
                     "sabana", "stepa", "hutan", "gurun", "pegunungan", # tempat alami di daratan
                     "waduk", "kolam", # tempat buatan manusia di perairan
                     "kebun", "sawah", "pabrik", "rumah", # tempat buatan manusia di daratan
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
modelword2vec.similarity('zebra' , 'refrigerator')
modelword2vec.similarity('zebra' , 'house')
modelword2vec.doesnt_match(['jokowi' , 'sby' , 'suharto' , 'sule'])
# Jawaban kode nomor 3
plot(modelword2vec, [
    "semut", "gula",
    "harimau", "hutan",
    "tentara", "perang",
    "kerbau", "kandang",
    "burung", "langit",
    "penjara", "narapidana"
])
print('setelah diplot yang paling mirip arah dan jaraknya adalah kerbau:kandang')
# Jawaban kode nomor 4
def cosine_similarity(word1, word2):
    vec1 = modelword2vec[word1]
    vec2 = modelword2vec[word2]
    return (vec1.dot(vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

cosine_similarity('raja', 'presiden')
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
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

SVCpipe = Pipeline([('scale', StandardScaler()),
                   ('SVC',LinearSVC())])

# Gridsearch to determine the value of C
param_grid = {'SVC__C':[10**(i/8) for i in range(-24, 25)]}
clf = GridSearchCV(SVCpipe,param_grid,cv=4,return_train_score=True)
clf.fit(train_X,train_y)
print(clf.best_params_)
#linearSVC.coef_
#linearSVC.intercept_

bestlinearSVC = clf.best_estimator_
bestlinearSVC.fit(train_X,train_y)
bestlinearSVC.coef_ = bestlinearSVC.named_steps['SVC'].coef_
bestlinearSVC.score(test_X,test_y)
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
common_texts

!ls /kaggle/input/praktikumword2vecnlp/idwiki.txt
!wc -l /kaggle/input/praktikumword2vecnlp/idwiki.txt
import nltk
common_texts = []
f = open('/kaggle/input/praktikumword2vecnlp/idwiki.txt')
counter = 0
for line in f:
    counter += 1
    if counter % 1000 == 0:
        print(counter * 100 / 392172, '%')
    common_texts.append(nltk.word_tokenize(line))
print(len(common_texts))
model = Word2Vec(common_texts, size=300, window=3, min_count=1, workers=4)
!ls
model.save("word2vec.300.model")
!ls
!cp word2vec.300.model* /kaggle/working/
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