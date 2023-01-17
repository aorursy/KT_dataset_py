import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df.head()
df.info()
# количество уникальных ключевых слов

df.loc[:,'keyword'].nunique()
# частота использования keywords

df.loc[:,'keyword'].value_counts()
# количество уникальных локаций

df.loc[:,'location'].nunique()
# частота использования различных location

df.loc[:,'location'].value_counts()
# описание данных

print(df.groupby('target')['target'].count())



# процент target=1

target_count = df[df['target'] == 1]['target'].count()

total = df['target'].count()

target_share = target_count/total

print("Доля данных, показывающих реалные происшествия \"target=1\" {0:.2f}".format(target_share))



# гистограмма

df[df['target'] == 0]['target'].astype(int).hist(label='Фейк', grid = False, bins=1, rwidth=0.8)

df[df['target'] == 1]['target'].astype(int).hist(label='Реальные', grid = False, bins=1, rwidth=0.8)

plt.xticks((0,1),('Фейк', 'Реальные'))

plt.show()
# т.к. в признаке location отсутствуют координаты, необходимые для построения карты, то добавим новую базу world-cities-database

# в этой базе каждой стране или городу проставлены координаты latitude / longitude



latlong = pd.read_csv("/kaggle/input/world-cities-database/worldcitiespop.csv")

latlong.head()
# перемименуем колонку AccentCity в Location для объединения с нашей исходной базой

latlong.rename(columns={"AccentCity": "location"}, inplace=True)

latlong.head()
# Получим уникальные сочетания location и latitude / longitude для добавления в исходную базу, удалим дублированные строки и отсутствующие location

latlong_grouped = latlong[['location', 'Latitude', 'Longitude', 'Population']].drop_duplicates()

latlong_grouped = latlong_grouped[latlong_grouped.location != 'None']
# отсортируем базу сначала по алфавиту по location (по возрастанию) и затем по population (по убыванию)

latlong_grouped.sort_values(['location', 'Population'], ascending=[True, False], inplace=True)
latlong_grouped.shape
# удалим повторяющиеся города из базы, оставим только те города из дублей, для которых население максимально

# для этого сравним текущую строку и следующую (shift смещает на одну строку вперёд), если они повторяются, то мы их не включаем в базу

latlong_cleaned = latlong_grouped[latlong_grouped['location'] != latlong_grouped['location'].shift()]

# проверим размерность базы

latlong_cleaned.shape
# для базы скоординатами, в которой дубли не удалялись

latlong_grouped[latlong_grouped['location'] == 'Birmingham']
# для базы с координатами, в которой дубли локаций были очищены

latlong_cleaned[latlong_cleaned['location'] == 'Birmingham']
# Добавим координаты к нашей исходной базе твитов

df_latlong = pd.merge(df, latlong_cleaned, left_on='location', right_on='location', how='left')

#df_latlong = df.set_index('location').join(latlong.set_index('location'), how='left', on=['location'])

df_latlong.head()
df_latlong.shape
# Отобразим локации, для которых удалось определить координаты

df_latlong[~df_latlong['Latitude'].isna()]
# количество реальных твитов с координатами

df_latlong_real = df_latlong[(~df_latlong['Latitude'].isna()) & (df_latlong['target'] == 1)]

print( len( df_latlong_real ) )
# пример реальных твитов с координатами

df_latlong_real.head()
# Нанесём реальные твиты на карту

import folium

from folium.plugins import HeatMap



df_latlong_real.Latitude.fillna(0, inplace = True)

df_latlong_real.Longitude.fillna(0, inplace = True) 

twits = df_latlong_real[['Latitude', 'Longitude']]



RealTwitsMap=folium.Map(location=[0,0],zoom_start=2)

HeatMap(data=twits, radius=12).add_to(RealTwitsMap)

RealTwitsMap
# количество фейковых твитов с координатами

df_latlong_fake = df_latlong[(~df_latlong['Latitude'].isna()) & (df_latlong['target'] == 0)]

print( len( df_latlong_fake ) )
# пример фейковых твитов с координатами

df_latlong_fake.head()
# Нанесём фейковые твиты на карту

import folium

from folium.plugins import HeatMap



df_latlong_fake.Latitude.fillna(0, inplace = True)

df_latlong_fake.Longitude.fillna(0, inplace = True) 

twits = df_latlong_fake[['Latitude', 'Longitude']]



FakeTwitsMap=folium.Map(location=[0,0],zoom_start=2)

HeatMap(data=twits, radius=12).add_to(FakeTwitsMap)

FakeTwitsMap
# функция для отрисовки облака часто употребляемых слов

from wordcloud import WordCloud,STOPWORDS



def wordcloud_img(data):

    plt.figure(figsize = (20,20))

    wc = WordCloud(min_font_size = 3, 

                   background_color="white",  

                   max_words = 3000, 

                   width = 1000, 

                   height = 600, 

                   stopwords = STOPWORDS).generate(str(" ".join(data)))

    plt.imshow(wc,interpolation = 'bilinear')
wordcloud_img(df[df['target'] == 1]['text'])
wordcloud_img(df[df['target'] == 0]['text'])
from urllib.parse import unquote
# выведем все слова с urlencoded-символами (у них всегда стоит знак "%")

for phrase in df['keyword']:

    phrase  = str(phrase)

    if('%' in phrase):

        print(phrase)
# делаем замену

for i, phrase in enumerate(df['keyword']):

    phrase  = str(phrase)

    if('%20' in phrase):

        df.loc[i, 'keyword'] = df.loc[i, 'keyword'].replace('%20', ' ')



# проверим

# выведем все слова с urlencoded-символами (у них всегда стоит знак "%")

for phrase in df['keyword']:

    phrase  = str(phrase)

    if('%' in phrase):

        print(phrase)
# проверяем пропуски в cтолбцах

df.isna().sum()
def location_and_keyword_fill_nan(df):

    #пропуски для keywords заменим на None

    df.keyword.fillna('None', inplace = True)



    #пропуски для location заменим на None

    df.location.fillna('None', inplace = True)



location_and_keyword_fill_nan(df)



# проверим

df.isna().sum()
# вернём индексы тех элементов массива target (целевой переменной), где значение 0

target_np = df['target'].astype(int).to_numpy()

fake_twits_ids = np.argwhere(target_np == 0).flatten()

print('Всего fake_twits: ', len(fake_twits_ids))

fake_twits_ids
# перемешаем массив с id фейковых твитов

from sklearn.utils import shuffle

fake_twits_ids = shuffle(fake_twits_ids, random_state = 42)



# выберем в нем "лишние" id фейковых твитов

# т.е. возьмём все элементы после номера 3271 из перемешанного массива fake_twits_ids

fake_twits_ids_to_drop = fake_twits_ids[len(np.argwhere(target_np == 1).flatten()):]



# отображаем кол-во фейковых твитов, которые нужно выбросить из выборки для балансировки, а также их id

print(len(fake_twits_ids_to_drop))

fake_twits_ids_to_drop
# т.к. данные пермешаны, то после отбрасыания лишних элементов, выборка будет репрезентативной

df_balanced = df.drop(df.index[fake_twits_ids_to_drop])



# отобразим итоговый размер признаков датасета

df_balanced.shape
# теперь видим, что классы сбалансированы.

df_balanced['target'].value_counts()
# гистограмма

df_balanced[df_balanced['target'] == 0]['target'].astype(int).hist(label='Фейк', grid = False, bins=1, rwidth=0.8)

df_balanced[df_balanced['target'] == 1]['target'].astype(int).hist(label='Реальные', grid = False, bins=1, rwidth=0.8)

plt.xticks((0,1),('Фейк', 'Реальные'))

plt.show()
def location_and_keyword_to_bool_category(df):

    df.loc[df['location'] == 'None', 'location_bool'] = 0

    df.loc[df['location'] != 'None', 'location_bool'] = 1

    df.loc[df['keyword'] == 'None', 'keyword_bool'] = 0

    df.loc[df['keyword'] != 'None', 'keyword_bool'] = 1



location_and_keyword_to_bool_category(df_balanced)



# проверяем

df_balanced
# делаем подобные преобразования до разбивки на train и test,т.к. такие преобразование каждого текста происходят независимо отвыборки

# к тому же X_train и X_test, которые образуются после train_test_split являются копиями базы и на копиях такие преобразования делать сложнее (может возвращаться ошибкао том, что работы ведутся на копии и т.п.)

# regular expressions library

import re



# функция для преобразования единичного текста

def single_text_clean(text):

    # преобразуем текст в нижний регистр

    text = text.lower()

    # преобразуем https:// и т.п. адреса в текст "URL"

    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)

    # преобразуем имя пользователя @username в "AT_USER"

    text = re.sub('@[^\s]+','AT_USER', text)

    # преобразуем множественые пробелы в один пробел

    text = re.sub('[\s]+', ' ', text)

    # преобразуем хэштег #тема в "тема"

    text = re.sub(r'#([^\s]+)', r'\1', text)

    return text



# функция для преобразования текста в разных столбцах датафрейма 

# (применяет предыдущую функцию single_text_clean к разным столбцам)

def text_columns_clean(df):

    text_columns_to_clean = ['keyword', 'location', 'text']

    for column in text_columns_to_clean:

        df.loc[:, column] = df[column].apply(single_text_clean)



text_columns_clean(df_balanced)



# проверим

df_balanced
import nltk

from nltk.corpus import stopwords

import string



stop = set(stopwords.words('english'))

punctuation = list(string.punctuation)

stop.update(punctuation)
# функция для определения прилагательных (ADJ), глаголов (VERB), существительных (NOUN) и наречий (ADV)

from nltk.corpus import wordnet as wn



def get_simple_pos(tag):

    if tag.startswith('J'):

        return wn.ADJ

    elif tag.startswith('V'):

        return wn.VERB

    elif tag.startswith('N'):

        return wn.NOUN

    elif tag.startswith('R'):

        return wn.ADV

    else:

        return wn.NOUN



# Лемматизация 

from nltk.stem import WordNetLemmatizer

from nltk import pos_tag



# (отбрасываем всё лишнее в предложении, приводим слова к нормальной формеи получаем их список)

# 'Once upone a time a man walked into a door' -> ['upone', 'time', 'man', 'walk', 'door']

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):

    final_text = []

    for i in text.split():

        if i.strip().lower() not in stop:

            pos = pos_tag([i.strip()])

            word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))

            final_text.append(word.lower())

    return final_text   



# Объединяем лемматизированный список в предложение

# ['upone', 'time', 'man', 'walk', 'door'] -> 'upone time man walk door '

def join_text(text):

    string = ''

    for i in text:

        string += i.strip() +' '

    return string



# Запуск лемматизации и создание нового поля с лемматизированными предложениями 'text_lemma'

df_balanced.loc[:, 'text'] = df_balanced['text'].apply(lemmatize_words).apply(join_text)



# проверим

df_balanced
# удалим не нужное нам больше дополнительное поле id (оно есть в индексе)

df_balanced = df_balanced.drop(['id'], axis=1)



from sklearn.model_selection import train_test_split



# разделение выборки на X и y

y = df_balanced['target']

X = df_balanced.drop(['target'], axis=1)



# разделение выборки на train и test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)



# проверяем

X_train.shape
X_train
from sklearn.feature_extraction.text import TfidfVectorizer



# Присвоение весов словам с использованием TfidfVectorizer

tv = TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,2))

tv_X_train = tv.fit_transform(X_train['text'])

tv_X_test = tv.transform(X_test['text'])



print('TfidfVectorizer_train:', tv_X_train.shape)

print('TfidfVectorizer_test:', tv_X_test.shape)
# перевернём столбцы со словами из sparse матрицы с одним столбцом в numpy массив со многими столбцами

tv_X_train = tv_X_train.toarray()

tv_X_test = tv_X_test.toarray()
# Нормализуем данные в tv_X_train/test

from sklearn.preprocessing import StandardScaler



def scaling_train(tv_X_train):

    scaler_tv = StandardScaler(copy=False)

    tv_X_train_scaled = scaler_tv.fit_transform(tv_X_train)

    # трансформируем tv_X_train/test sparse numpy матрицы в pandas Data Frame

    tv_X_train_pd_scaled = pd.DataFrame(data=tv_X_train_scaled, 

                             index=X_train.index, 

                             columns=np.arange(0, np.size(tv_X_train_scaled,1)))

    return tv_X_train_pd_scaled, scaler_tv



def scaling_test(tv_X_test, scaler_tv):

    tv_X_test_scaled = scaler_tv.transform(tv_X_test)

    # трансформируем tv_X_train/test sparse numpy матрицы в pandas Data Frame

    tv_X_test_pd_scaled = pd.DataFrame(data=tv_X_test_scaled, 

                             index=X_test.index, 

                             columns=np.arange(0, np.size(tv_X_test_scaled,1)))

    return tv_X_test_pd_scaled



tv_X_train_pd_scaled, scaler_tv = scaling_train(tv_X_train)

tv_X_test_pd_scaled = scaling_test(tv_X_test, scaler_tv)



# проверяем

tv_X_train_pd_scaled.shape, tv_X_test_pd_scaled.shape
# преобразуем pandas dataframe в numpy array для pytorch модели

tv_X_train_pd_scaled_arr = tv_X_train_pd_scaled.to_numpy()

tv_X_test_pd_scaled_arr = tv_X_test_pd_scaled.to_numpy()
tv_X_test_pd_scaled_arr
import torch

from torch import nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset



# funtools - работа с функциями высшего порядка

# partial - изменяет количество аргументов для переданных функций и запускает функции одну за другой

from functools import partial



# делаем результаты случайных разделений выборок воспроизводимыми (это аналог random_state в sklearn)

torch.manual_seed(42)
# Определяем девайс, на котором будет происходить тренировка (если у нас CPU, то будет на CPU, если GPU, то - не GPU)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device
# класс для преобразования и загрузки данных в Pytorch

# он наследует класс Dataset

# df_tfidf - это df конвертированная в tf-idf (т.е. это матрица tv_X_train или tv_X_test)

# df - это сбалансированная база df_balanced



class TwitsDataset(Dataset):

    def __init__(self, tv_X_train_array, y_train):

        df = pd.DataFrame(index=y_train.index)

        

        # текст предварительно очищен, лематизирован и превращен в токены на предыдущих шагах

        df['tfidf_vector'] = [vector.tolist() for vector in tv_X_train_array]

        

        self.tfidf_vector = df.tfidf_vector.tolist()

        self.targets = y_train.tolist()

    

    def __getitem__(self, i):

        return (

            self.tfidf_vector[i],

            self.targets[i]

        )

    

    def __len__(self):

        return len(self.targets)
for i, vector in enumerate(tv_X_train):

    print(vector)

    vector.tolist()

    if(i ==  5): break
# загрузка данных в класс и их преобразование



# нормированные tf-idf

dataset = TwitsDataset(tv_X_train_pd_scaled_arr, y_train)
# Т.к. тестовую выборку мы уже ранее формировали, то на данном шаге не будем её выделять из текущих данных (тестовая выборка в этот датасет не входит)

from torch.utils.data.dataset import random_split



def split_train_valid_test(corpus, valid_ratio=0.1, test_ratio=0.1):

    """Split dataset into train, validation, and test."""

    test_length = int(len(corpus) * test_ratio)

    valid_length = int(len(corpus) * valid_ratio)

    train_length = len(corpus) - valid_length - test_length

    return random_split(

        corpus, lengths=[train_length, valid_length, test_length],

    )



train_dataset, valid_dataset, test_dataset = split_train_valid_test(dataset, valid_ratio=0.2, test_ratio=0.0)

len(train_dataset), len(valid_dataset), len(test_dataset)

# это не значит, что Test-выборка у нас = 0, просто она определена в другом месте и называется tv_X_test 

# (а на данном шаге мы просто не выделяли тестовую выборку из tv_X_train, т.к. у нас уже есть tv_X_test)
# проверим содержимое train_dataset

print('Число записей:', len(train_dataset), '\n')



import random

# генерация одних и тех же случайных величин

random.seed(a=42, version=2)



random_idx = random.randint(0,len(train_dataset)-1)

print('Случайны индекс из dataset:', random_idx, '\n')

tfidf_vector, sample_target = train_dataset[random_idx]

print('Размер вектора TF-IDF:', len(tfidf_vector), '\n')

print('Пример таргета:', sample_target, '\n')
# проверим содержимое valid_dataset

print('Число записей:', len(valid_dataset), '\n')



random.seed(a=42, version=2)

random_idx = random.randint(0,len(valid_dataset)-1)

print('Случайны индекс из dataset:', random_idx, '\n')

tfidf_vector, sample_target = valid_dataset[random_idx]

print('Размер вектора TF-IDF:', len(tfidf_vector), '\n')

print('Пример таргета:', sample_target, '\n')
# загрузчки данных для Pytorch (грузит данные по батчам)

BATCH_SIZE = 512



def collate(batch):

    tfidf = torch.FloatTensor([item[0] for item in batch])

    target = torch.LongTensor([item[1] for item in batch])

    return tfidf, target



train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate)

valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
# посмотрим содержимое train_loader

print('Число training batches:', len(train_loader), '\n')



random.seed(a=42, version=2)

batch_idx = random.randint(0, len(train_loader)-1)

example_idx = random.randint(0, BATCH_SIZE-1)



print("Batch index: ", batch_idx)

print("Example index: ", example_idx)



for i, fields in enumerate(train_loader):

    tfidf, target = fields

    if i == batch_idx:

        print('Размер вектора TF-IDF:', len(tfidf[example_idx]), '\n')

        #print('Случайный TF-IDF: ', tfidf[example_idx], '\n')

        print('Тип TF-IDF: ', type(tfidf[example_idx]), '\n')

        #print('Случайный таргет: ', target[example_idx], '\n')

        print('Тип таргета: ', type(target[example_idx]), '\n')
# строим нейронную сеть (Вариант 2 - CNN на нормализованных TF-IDF)



class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)



class Reorder(nn.Module):

    def forward(self, input):

        return input.permute((0, 2, 1))



class FeedfowardTextClassifier(nn.Module):

    def __init__(self, device, vocab_size, hidden1, hidden2, hidden3, hidden4, num_labels, batch_size):

        super(FeedfowardTextClassifier, self).__init__()

        self.device = device

        self.batch_size = batch_size



        self.convnet = nn.Sequential(

            nn.Conv1d(in_channels=vocab_size,

                      out_channels=hidden1, 

                      kernel_size=1),

            nn.ELU(),

            nn.Dropout(p=0.5),

            nn.Conv1d(in_channels=hidden1, 

                      out_channels=hidden2,

                      kernel_size=1, 

                      #stride=1

                     ),

            nn.ELU(),

            nn.Dropout(p=0.5),

        )

        self.fc = nn.Linear(hidden2, 2)

    

    def forward(self, x):

        batch_size = len(x)

        if batch_size != self.batch_size:

            self.batch_size = batch_size



        features = self.convnet(x)

        

        # после выполнения convnet удаляем 3-ее измерение (индекс = 2) размерностью 1

        features = features.squeeze(dim=2)

        

        prediction_vector = self.fc(features)

        return prediction_vector

        

# определяем размеры скрытых слоёв

HIDDEN1 = 512

HIDDEN2 = 128

HIDDEN3 = 128

HIDDEN4 = 128



tfidf_model = FeedfowardTextClassifier(

    vocab_size=len(tfidf_vector),

    hidden1=HIDDEN1,

    hidden2=HIDDEN2,

    hidden3=HIDDEN3,

    hidden4=HIDDEN4,

    num_labels=2,

    device=device,

    batch_size=BATCH_SIZE,

)
# переносим на GPU

tfidf_model = tfidf_model.to(device)

tfidf_model
# Loss-функция и оптимизатор



from torch import optim

from torch.optim.lr_scheduler import CosineAnnealingLR



LEARNING_RATE = 4e-2



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(

    filter(lambda p: p.requires_grad, tfidf_model.parameters()),

    lr=LEARNING_RATE,

)



scheduler = CosineAnnealingLR(optimizer, 1)
# определяем функцию и необходимые шаги для тренировки



def train_epoch(model, optimizer, train_loader):

    model.train()

    total_loss, total = 0, 0

    for i, (tfidf, target) in enumerate(train_loader):

        

        inputs = tfidf

        # добавляем новую размерность: канал для входа в cnn      

        inputs = inputs.unsqueeze(dim=2)

        

        #print(inputs.shape)

        

        # переносим на GPU

        inputs = inputs.to(device)

        target = target.to(device)

        

        # Reset gradient

        optimizer.zero_grad()

        

        # Forward pass

        output = model(inputs)

        

        # Compute loss

        loss = criterion(output, target)

        

        # Perform gradient descent, backwards pass

        loss.backward()



        # Take a step in the right direction

        optimizer.step()

        scheduler.step()



        # Record metrics

        total_loss += loss.item()

        total += len(target)



    return total_loss / total



# функция и шаги для валидации

def validate_epoch(model, valid_loader):

    model.eval()

    total_loss, total = 0, 0

    with torch.no_grad():

        for tfidf, target in valid_loader:

            inputs = tfidf

            

            # добавляем новую размерность: канал для входа в cnn

            inputs = inputs.unsqueeze(dim=2)

            

            # переносим на GPU

            inputs = inputs.to(device)

            target = target.to(device)

            

            #print("Val: ", inputs.shape)

            

            # Forward pass

            output = model(inputs)



            # Calculate how wrong the model is

            loss = criterion(output, target)



            # Record metrics

            total_loss += loss.item()

            total += len(target)



    return total_loss / total
# тренируем

from tqdm import tqdm



max_epochs = 50

n_epochs = 0

train_losses = []

valid_losses = []



for epoch_num in range(max_epochs):



    train_loss = train_epoch(tfidf_model, optimizer, train_loader)

    valid_loss = validate_epoch(tfidf_model, valid_loader)

    

    tqdm.write(

        f'эпоха #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.2e}\tvalid_loss: {valid_loss:.2e}\n',

    )

    

    # Early stopping (ранняя остановка) если текущий valid_loss (значение функции ошибки для валидаионной выборки) больше чем 10 последних valid losses

    if len(valid_losses) > 7 and all(valid_loss >= loss for loss in valid_losses[-8:]):

        print('Stopping early')

        break

    

    

    train_losses.append(train_loss)

    valid_losses.append(valid_loss)

    

    n_epochs += 1
# строим график loss-функции после тренировки

epoch_ticks = range(1, n_epochs + 1)

plt.plot(epoch_ticks, train_losses)

plt.plot(epoch_ticks, valid_losses)

plt.legend(['Train Loss', 'Valid Loss'])

plt.title('Losses') 

plt.xlabel('Epoch #')

plt.ylabel('Loss')

plt.xticks(epoch_ticks)

plt.show()
# загрузка данных в класс и их преобразование

test_dataset = TwitsDataset(tv_X_test, y_test)
# загрузчки данных для Pytorch (грузит данные по батчам)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
# проверим содержимое test_loader

print('Число training batches:', len(test_loader), '\n')



random.seed(a=42, version=2)

batch_idx = random.randint(0, len(test_loader)-1)

example_idx = random.randint(0, BATCH_SIZE-1)



for i, fields in enumerate(test_loader):

    tfidf, target = fields

    if i == batch_idx:

        print('Размер вектора TF-IDF:', len(tfidf[example_idx]), '\n')

        print('Случайный таргет: ', target[example_idx], '\n')
# Эффективность Pytorch CNN-модели на TFIDF

from sklearn.metrics import classification_report



tfidf_model.eval()

test_accuracy, n_examples = 0, 0

y_true, y_pred = [], []



with torch.no_grad():

    for tfidf, target in test_loader:

        # добавляем новую размерность: канал для входа в cnn

        tfidf = tfidf.unsqueeze(dim=2)

        

        inputs = tfidf.to(device)

        target = target.to(device)

        

        probs = tfidf_model(inputs)

        

        probs = probs.detach().cpu().numpy()

        predictions = np.argmax(probs, axis=1)

        target = target.cpu().numpy()

        

        y_true.extend(predictions)

        y_pred.extend(target)

        

print(classification_report(y_true, y_pred))