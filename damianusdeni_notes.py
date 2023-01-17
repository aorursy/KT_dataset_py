import pandas as pd

# excel
df = pd.read_excel('/kaggle/input/allianz-news-jan-jun-2020/allianz_news_jan_jun_2020.xlsx')
df.head()

# csv
df = pd.read_csv('/kaggle/input/allianz-news-jan-jun-2020/allianz_news_jan_jun_2020.csv')
df.head()
import pandas as pd
import json

# option 1
path = '../input/stanford-covid-vaccine/'
df = pd.read_json(f'{path}/train.json', lines=True).drop(columns='index')
test = pd.read_json(f'{path}/test.json', lines=True).drop(columns='index')
submission = pd.read_csv(f'{path}/sample_submission.csv')

# option 2
with open("../input/allianz-twitter-tweets-2011-2020/tweet.js", 'rb') as handle:
    df = json.load(handle)
    
# option 3
df = pd.read_json('/kaggle/input/allianz-twitters-tweet-2011-2019/20191102-tweet.js')
target = model.predict(df_submit_text_list)
target
feature, target = next(iter(testloader))
feature, target = feature.to(device), target.to(device)


# alt 1
with torch.no_grad():
    model.eval()
    output = model(feature)
    preds = output.argmax(1)
preds

# alt 2
with torch.no_grad():
    model.eval()
    output = model(feature)
    preds = (output > 0.5).to(torch.float32)
preds
# Visualize Target Label

import seaborn as sns
import matplotlib.pyplot as plt

y_train.shape, y_test.shape


# y_train.shape
sns.set(style="darkgrid")
sns.countplot(x=y_train)
plt.title("y_train");


# y_test.shape
sns.set(style="darkgrid")
sns.countplot(x=y_test)
plt.title("y_test");
# Word Cloud

from matplotlib import pyplot as plt


fig, axes = plt.subplots(6, 6, figsize=(24, 24))
for image, label, pred, ax in zip(feature, target, preds, axes.flatten()):
    ax.imshow(image.permute(1, 2, 0).cpu())
    font = {"color": 'r'} if label != pred else {"color": 'g'}        
    label, pred = label2cat[label.item()], label2cat[pred.item()]
    ax.set_title(f"L: {label} | P: {pred}", fontdict=font);
    ax.axis('off');
from jcopml.utils import save_model

save_model(model_logreg_tfidf, "model_logreg_tfidf.pkl")
import pickle

pickle.dump(model, open("knn.pkl", 'wb'))
import os

os.makedirs("model/fasttext/", exist_ok=True)
model_fasttext.save("model/fasttext/capres_sentiment.fasttext")
from jcopml.utils import load_model

model = load_model('model/knn.pkl')
import pickle

model = pickle.load(open("knn.pkl", "rb"))

df_submit = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_submit.drop(columns=["keyword", "location"], inplace=True)
df_submit_id_list = df_submit.id.values.tolist()
df_submit_text_list = df_submit.text.values.tolist()
df_submit.head()



print("df_submit.id: ", len(df_submit.id))
print("df_submit.text: ", len(df_submit.text))
print("df_submit_id_list: ", len(df_submit_id_list))
print("df_submit_text_list: ", len(df_submit_text_list))



target = model.predict(df_submit_text_list)
target



df_submit_final = pd.DataFrame({
    "id": df_submit_id_list,
    "target": target
})



df_submit_final.set_index('id', inplace=True)
df_submit_final.head()



df_submit_final.to_csv("disaster_tweet_v10.csv")



test_set_final = datasets.ImageFolder("/kaggle/input/chest-xray-pneumonia/chest_xray/test", transform=test_transform)
testloader_final = DataLoader(test_set_final, batch_size=bs)



with torch.no_grad():
    test_cost_final, test_score_final = loop_fn("test", test_set_final, testloader_final, model, criterion, optimizer, device)
    print(f"Test Accuracy: {test_score_final}")


import nltk 
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from string import punctuation

sw = stopwords.words("indonesian") + stopwords.words("english")



# 1) Normalization to Lower Case & Removing "https: ..."
def clean_text(text):
    text = text.lower()
    text = re.sub("\n", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = text.split()
    text = " ".join(text)
    return text

df1 = df.Isi_Tweet.apply(str).apply(lambda x:clean_text(x))


# display text samples
for i in range(len(df1)):
    print(df1[i])
    if i == 10:
        break

print("The length of dataframe is", len(df1), "rows")
        


    
    
# 2) Sentence & Word Tokenization; Punctuation and Words Removal
df1_clean_text = []
for i in range(len(df1)):
    x = df1[i]
    # x_sent_token = sent_tokenize(x)
    # x_sent_token
    x_word_tokens = word_tokenize(x)
    # x_word_tokens
#     print(x)
    
#     print(df1[i])
    
    # punctuation removal
    x_word_tokens_removed_punctuations = [w for w in x_word_tokens if w not in punctuation]
#     print(x_word_tokens_removed_punctuations, "punctuation")
    
    # numeric removal
    x_word_tokens_removed_punctuations = [w for w in x_word_tokens_removed_punctuations if w.isalpha()]
#     print(x_word_tokens_removed_punctuations, "numeric")
    
    # stopwords removal
    x_word_tokens_removed_punctuation_removed_sw = [w for w in x_word_tokens_removed_punctuations if w not in sw]
#     print(x_word_tokens_removed_punctuation_removed_sw, "stopwords")

    # rejoining the words into one string/sentence as inputted before being tokenized
    x_word_tokens_removed_punctuation_removed_sw = " ".join(x_word_tokens_removed_punctuation_removed_sw)
#     print(x_word_tokens_removed_punctuation_removed_sw)
    
    df1_clean_text.append(x_word_tokens_removed_punctuation_removed_sw)
    
    
# display text vs processed text
for i,j in zip(df1[0:10], df1_clean_text[0:10]):
    print(i)
    print(j)
    print()
    
    
    


# list (df1_clean_text) to series (df1_clean_text_series)

# list
print(type(df1_clean_text))
print(len(df1_clean_text))

# converting list to pandas series
df1_clean_text_series = pd.Series(df1_clean_text)

print(type(df1_clean_text_series))
print(len(df1_clean_text_series))


# create new df
df['Isi_Tweet'] = df1_clean_text_series
df.head(10)


# count total of words of old df vs new df 

total_words_new_df = df.Isi_Tweet.apply(lambda x: len(x.split(" "))).sum()

print("old df: ", total_words_old_df, "words")
print("new df: ", total_words_new_df, "words")
print("text processing has reduced the number of words by", round((total_words_old_df-total_words_new_df)/total_words_old_df*100), "%")
# 1) Prepare Corpus
from tqdm.auto import tqdm
from gensim.models import FastText

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# sw is list of stopwords (ID + EN) and punctuation
sw = stopwords.words("indonesian") + stopwords.words("english") + list(punctuation)


sentences = [word_tokenize(text.lower()) for text in tqdm(df.Isi_Tweet)]


# 2) Train FastText Model
import os

model_fasttext = FastText(sentences, size=128, window=5, min_count=3, workers=4, iter=100, sg=0, hs=0)

# save
os.makedirs("model/fasttext/", exist_ok=True)
model_fasttext.save("model/fasttext/capres_sentiment.fasttext")


# 3) Encoding
from tqdm.auto import tqdm
from gensim.models import FastText

w2v = FastText.load("model/fasttext/capres_sentiment.fasttext").wv


def simple_encode_sentence(sentence, w2v, stopwords=None):
    if stopwords is None:
        vecs = [w2v[word] for word in word_tokenize(sentence)]
    else:
        vecs = [w2v[word] for word in word_tokenize(sentence) if word not in stopwords]
    sentence_vec = np.mean(vecs, axis=0)
    return sentence_vec

def better_encode_sentence(sentence, w2v, stopwords=None):
    if stopwords is None:
        vecs = [w2v[word] for word in word_tokenize(sentence)]
    else:
        vecs = [w2v[word] for word in word_tokenize(sentence) if word not in stopwords]
        
    vecs = [vec / np.linalg.norm(vec) for vec in vecs if np.linalg.norm(vec) > 0]
    sentence_vec = np.mean(vecs, axis=0)
    return sentence_vec


vecs = [better_encode_sentence(sentence, w2v, stopwords=sw) for sentence in df.Isi_Tweet]
vecs = np.array(vecs)
vecs


# 4) When used in Dataset Splitting
X = vecs
y = df.Sentimen

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from jcopml.tuning import random_search_params as rsp
from jcopml.tuning import random_search_params as gsp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 


pipeline = Pipeline([
    ('prep', TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
#     ('prep', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))), #choose bow or tfidf
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])

model_logreg_tfidf = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=5, n_iter=50, n_jobs=-1, verbose=1, random_state=42)
# model_logreg_tfidf = GridSearchCV(pipeline, gsp.logreg_params, cv=5, n_jobs=-1, verbose=1) # it takes longer time
model_logreg_tfidf.fit(X_train, y_train)

print(model_logreg_tfidf.best_params_)
print(model_logreg_tfidf.score(X_train, y_train), model_logreg_tfidf.best_score_, model_logreg_tfidf.score(X_test, y_test))

# LINEAR SVM
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from jcopml.tuning import random_search_params as rsp
from jcopml.tuning import random_search_params as gsp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 


pipeline = Pipeline([
    ('prep', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])


parameter = {
    'algo__loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
    'algo__penalty': ['l2', 'l1', 'elasticnet'],
    'algo__alpha': [0.0001, 0.0002, 0.0003], 
    'algo__max_iter': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'algo__tol': [0.0001, 0.0002, 0.0003]
}
# model_sgd_bow = GridSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1) # it takes longer time
model_sgd_bow = RandomizedSearchCV(pipeline, parameter, cv=50, n_jobs=-1, verbose=1)
model_sgd_bow.fit(X_train, y_train)


print(model_sgd_bow.best_params_)
print(model_sgd_bow.score(X_train, y_train), model_sgd_bow.best_score_, model_sgd_bow.score(X_test, y_test))
# Architecture & Config
from torchvision.models import densenet121
from jcopdl.layers import linear_block
from tqdm.auto import tqdm

dnet = densenet121(pretrained=True)

# freeze model
for param in dnet.parameters():
    param.requires_grad = False
    

dnet.classifier = nn.Sequential(
    nn.Linear(1024, 42),
    nn.LogSoftmax()
)
dnet


class CustomDensenet121(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.dnet = densenet121(pretrained=True)
        self.freeze()
        self.dnet.classifier = nn.Sequential(
#             linear_block(1024, 1, activation="lsoftmax")
            nn.Linear(1024, output_size),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        return self.dnet(x)

    def freeze(self):
        for param in self.dnet.parameters():
            param.requires_grad = False
            
    def unfreeze(self):        
        for param in self.dnet.parameters():
            param.requires_grad = True  

            
config = set_config({
    "output_size": len(train_set.classes),
    "batch_size": bs,
    "crop_size": crop_size
})



    
# Phase 1: Adaptation (lr standard + patience low)
model = CustomDensenet121(config.output_size).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
callback = Callback(model, config, early_stop_patience=2, outdir="model")

# Phase 1: Training
def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, device):
    if mode == "train":
        model.train()
    elif mode == "test":
        model.eval()
    cost = correct = 0
    for feature, target in tqdm(dataloader, desc=mode.title()):
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        cost += loss.item() * feature.shape[0]
        correct += (output.argmax(1) == target).sum().item()
    cost = cost / len(dataset)
    acc = correct / len(dataset)
    return cost, acc

while True:
    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    with torch.no_grad():
        test_cost, test_score = loop_fn("test", val_set, valloader, model, criterion, optimizer, device)
    
    # Logging
    callback.log(train_cost, test_cost, train_score, test_score)

    # Checkpoint
    callback.save_checkpoint()
        
    # Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()
    
    # Early Stopping
    if callback.early_stopping(model, monitor="test_score"):
        callback.plot_cost()
        callback.plot_score()
        break




# Phase 2: Fine Tuning (lr low + patience high)
model.unfreeze()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

callback.reset_early_stop()
callback.early_stop_patience = 5

# Phase 2: Training
while True:
    train_cost, train_score = loop_fn("train", train_set, trainloader, model, criterion, optimizer, device)
    with torch.no_grad():
        test_cost, test_score = loop_fn("test", val_set, valloader, model, criterion, optimizer, device)
    
    # Logging
    callback.log(train_cost, test_cost, train_score, test_score)

    # Checkpoint
    callback.save_checkpoint()
        
    # Runtime Plotting
    callback.cost_runtime_plotting()
    callback.score_runtime_plotting()
    
    # Early Stopping
    if callback.early_stopping(model, monitor="test_score"):
        callback.plot_cost()
        callback.plot_score()
        break
# Colab library to upload files to notebook
from google.colab import files

# Upload kaggle API key file
# download kaggle.json from Kaggle - My Account - Create New API Token
uploaded = files.upload()


# Install Kaggle library & set on kaggle.json on the root
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json


# Download data for productdetection2
# copy the API Command Link by right click menu at the right side of new notebook and copy the link
!kaggle datasets download -d maharajaarizona/productdetection2


# unzip data
!unzip /content/productdetection2.zip
def stratified_split_folder(input_dir, output_dir, valid_size=0.2):
    labels = [folder for folder in os.listdir(input_dir) if not folder.startswith(".")]
    for label in labels:
        # Dapatkan semua nama file untuk label tertentu
        files = glob(f"{input_dir}/{label}/*.jpg")
        
        # Shuffle split
        shuffle(files)
        n_test = int(valid_size * len(files))
        train, valid = files[-n_test:], files[-n_test:]
        
        # Untuk semua yang merupakan validation data, pindahkan ke folder baru
        os.makedirs(f"{output_dir}/{label}", exist_ok=True)
        for file in valid:
            fname = os.path.basename(file)
            os.rename(file, f"{output_dir}/{label}/{fname}")
            

            
stratified_split_folder(input_dir="resized/train", output_dir="resized/valid", valid_size=0.2)
with open("../input/ndsc-beginner/categories.json", 'rb') as handle:
    category_details = json.load(handle)