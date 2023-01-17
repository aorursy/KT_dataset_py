import re



import time

import datetime



import numpy as np

import pandas as pd
train = pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv", encoding='latin1')

test = pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv", encoding='latin1')
train.head()
train = train[['OriginalTweet', 'Sentiment']]

test = test[['OriginalTweet', 'Sentiment']]



test.head()
test.groupby('Sentiment').count()
label_mapper = {

    'Extremely Negative': 0,

    'Negative': 1,

    'Neutral': 2,

    'Positive': 3,

    'Extremely Positive': 4

}



train['Sentiment'] = train['Sentiment'].apply(lambda x: label_mapper[x])

test['Sentiment'] = test['Sentiment'].apply(lambda x: label_mapper[x])



test.head()
from gensim.utils import simple_preprocess



def preprocessing(X_train, X_test):

    url = re.compile(r'https?://\S+|www\.\S+')

    html = re.compile(r'<.*?>')

    number = re.compile(r'\d+')

    mention = re.compile(r'@\w+')

    tags = re.compile(r'#\w+')

    spaces = re.compile(r'\s+')

    

    def cleaning_pipe(X):

        out = []

        

        for x in X:

            x = url.sub('url_token', x)

            x = number.sub('num_token', x)

            x = mention.sub('mention_token', x)

            x = tags.sub('tag_token', x)

            

            x = html.sub(' ', x)

            x = spaces.sub(' ', x)

            

            out.append(x)

        

        return out  

        

    X_train = [simple_preprocess(x) for x in cleaning_pipe(X_train)]

    X_test = [simple_preprocess(x) for x in cleaning_pipe(X_test)]

    

    # any other steps

    

    with open('data.txt', 'w') as f:

        for x in X_train:

            f.write(' '.join(x) + '\n')

        

        for x in X_test:

            f.write(' '.join(x) + '\n')

    

    return X_train, X_test
def get_texts_repr(texts, model):

    global_pooling = lambda tokens: np.array([model[t] for t in tokens]).mean(axis=0)

    return [global_pooling(t) for t in texts]
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



from sklearn.preprocessing import StandardScaler



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



def eval_clf(X_train, y_train, X_test, y_test, model, rs=50):

    scaler = StandardScaler()

    

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.fit_transform(X_test)

    

    clf = model().fit(X_train, y_train)

    

    print(classification_report(y_train, clf.predict(X_train)))

    print(classification_report(y_test, clf.predict(X_test)))

    

    print(confusion_matrix(y_test, clf.predict(X_test)))
%%time



X_train = train['OriginalTweet']

y_train = train['Sentiment']

X_test = test['OriginalTweet']

y_test = test['Sentiment']



X_train, X_test = preprocessing(X_train, X_test)
emb_dim = 100
from gensim.models.fasttext import FastText
%%time



gensim_model = FastText(

    size=emb_dim,

    window=5, 

    min_count=5, 

    corpus_file='./data.txt',

    iter=5

)
%%time



eval_clf(

    X_train=get_texts_repr(X_train, gensim_model.wv),

    y_train=y_train,

    X_test=get_texts_repr(X_test, gensim_model.wv), 

    y_test=y_test,

    model=LogisticRegression, 

    rs=50

)
%%time



eval_clf(

    X_train=get_texts_repr(X_train, gensim_model.wv),

    y_train=y_train,

    X_test=get_texts_repr(X_test, gensim_model.wv), 

    y_test=y_test,

    model=GaussianNB,

    rs=50

)
import fasttext
facebook_model = fasttext.train_unsupervised(

    './data.txt', 

    model='cbow'

)
%%time



eval_clf(

    X_train=get_texts_repr(X_train, facebook_model),

    y_train=y_train,

    X_test=get_texts_repr(X_test, facebook_model), 

    y_test=y_test,

    model=LogisticRegression, 

    rs=50

)
%%time



eval_clf(

    X_train=get_texts_repr(X_train, facebook_model),

    y_train=y_train,

    X_test=get_texts_repr(X_test, facebook_model),

    y_test=y_test,

    model=GaussianNB,

    rs=50

)
import torch

import random



if torch.cuda.is_available():    

    device = torch.device("cuda")



    print('There are %d GPU(s) available.' % torch.cuda.device_count())



    print('We will use the GPU:', torch.cuda.get_device_name(0))



else:

    print('No GPU available, using the CPU instead.')

    device = torch.device("cpu")

    

seed_val = 50



random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)
train_test_border = len(X_train)



X = [' '.join(x) for x in X_train + X_test]

y = [int(y) for y in y_train.tolist() + y_test.tolist()]
from transformers import DistilBertTokenizer



tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
%%time



max_len = 0



for x in X:

    input_ids = tokenizer.encode(x, add_special_tokens=True)

    max_len = max(max_len, len(input_ids))



print('Max sentence length: ', max_len)
%%time



input_ids = []

attention_masks = []



for x in X:

    encoded_dict = tokenizer.encode_plus(

        x,                      

        add_special_tokens = True,

        max_length = 256,

        pad_to_max_length = True,

        return_attention_mask = True,

        return_tensors = 'pt',

    )

    

    input_ids.append(encoded_dict['input_ids'])

    attention_masks.append(encoded_dict['attention_mask'])



input_ids = torch.cat(input_ids, dim=0)

attention_masks = torch.cat(attention_masks, dim=0)

labels = torch.tensor(y)
from torch.utils.data import TensorDataset, random_split



# Combine the training inputs into a TensorDataset.

train_dataset = TensorDataset(

    input_ids[:train_test_border], 

    attention_masks[:train_test_border], 

    labels[:train_test_border]

)



train_size = int(0.9 * len(train_dataset))

val_size = len(train_dataset) - train_size



train_dataset, val_dataset = random_split(

    train_dataset, 

    [train_size, val_size]

)



test_dataset = TensorDataset(

    input_ids[train_test_border:], 

    attention_masks[train_test_border:], 

    labels[train_test_border:]

)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler



batch_size = 32



train_dataloader = DataLoader(

            train_dataset,

            sampler=RandomSampler(train_dataset),

            batch_size=batch_size

        )



validation_dataloader = DataLoader(

            val_dataset, 

            sampler=SequentialSampler(val_dataset),

            batch_size=batch_size

        )



test_dataloader = DataLoader(

            test_dataset, 

            sampler=SequentialSampler(test_dataset),

            batch_size=batch_size

        )
from transformers import DistilBertForSequenceClassification, AdamW

from transformers import get_linear_schedule_with_warmup



model = DistilBertForSequenceClassification.from_pretrained(

    "distilbert-base-uncased",

    num_labels = 5,

    output_attentions = False,

    output_hidden_states = False,

)



model.cuda()



optimizer = AdamW(

    model.parameters(),

    lr = 2e-5, 

    eps = 1e-8

)



epochs = 4



total_steps = len(train_dataloader) * epochs



scheduler = get_linear_schedule_with_warmup(

    optimizer, 

    num_warmup_steps = 0,

    num_training_steps = total_steps

)
# utils



from sklearn.metrics import classification_report



def update_classification_report(preds, labels, report):

    predicted_y = np.argmax(preds, axis=1).flatten()

    true_y = labels.flatten()

    

    metrics = classification_report(true_y, predicted_y, output_dict=True)

    

    report['precision'] += metrics['macro avg']['precision']

    report['f1-score']  += metrics['macro avg']['f1-score']

    report['accuracy']  += metrics['accuracy']

    report['recall']    += metrics['macro avg']['recall']

    

    return report





def format_time(elapsed):

    """Takes a time in seconds and returns a string hh:mm:ss"""

    return str(datetime.timedelta(seconds=int(round((elapsed)))))
import random



training_stats = []



# total training time for all epochs

total_t0 = time.time() 



for epoch_i in range(0, epochs):

    

    # ========================================

    #                  Train

    # ========================================

    

    print("")

    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))



    t0 = time.time() # epoch time tracking



    total_train_loss = 0



    model.train() # set train mode



    for step, batch in enumerate(train_dataloader):

        if step % 50 == 0 and not step == 0:

            elapsed = format_time(time.time() - t0) # elapsed time in minutes.

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))



        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch



        model.zero_grad()        



        loss, _ = model(

            b_input_ids, 

            attention_mask=b_input_mask, 

            labels=b_labels

        )



        total_train_loss += loss.item()



        # calc gradients

        loss.backward() 



        # avoid exploding gradients problem

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 



        # Update params

        optimizer.step()



        # Update the lr

        scheduler.step()



    # Calculate the average loss over all of the batches.

    avg_train_loss = total_train_loss / len(train_dataloader)            

    

    # Measure how long this epoch took.

    training_time = format_time(time.time() - t0)



    print("")

    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("  Training epcoh took: {:}".format(training_time))

        

    # ========================================

    #               Validation

    # ========================================

    

    print("")



    t0 = time.time()



    model.eval()



    # Tracking variables 

    report = {

        'precision': 0,

        'f1-score':  0,

        'accuracy':  0,

        'recall':    0

    }

    

    total_eval_loss = 0



    for batch in validation_dataloader:

        

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        

        with torch.no_grad():        

            loss, preds = model(

                b_input_ids, 

                attention_mask=b_input_mask,

                labels=b_labels

            )



        total_eval_loss += loss.item()



        preds = preds.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()



        # callculate metrics         

        report = update_classification_report(preds, label_ids, report)

    

    for k, v in report.items():

        report[k] = v / len(validation_dataloader)

        print("  {1}: {0:.2f}".format(report[k], k))



    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)

    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

    print("  Validation took: {:}".format(validation_time))



    # Record all statistics from this epoch.

    training_stats.append(

        {

            'epoch': epoch_i + 1,

            'Training Loss': avg_train_loss,

            'Valid. Loss': avg_val_loss,

            'Training Time': training_time,

            'Validation Time': validation_time,

            **report

        }

    )



print("Training complete!")



print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
pd.DataFrame(data=training_stats).set_index('epoch')
model.eval()



predictions , true_labels = [], []



for batch in test_dataloader:

    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_labels = batch



    with torch.no_grad():

        outputs = model(

            b_input_ids,

            attention_mask=b_input_mask

        )



    preds = outputs[0]



    preds = preds.detach().cpu().numpy()

    label_ids = b_labels.to('cpu').numpy()



    # Store predictions and true labels

    predictions.append(preds)

    true_labels.append(label_ids)



predicted_y = np.concatenate([np.argmax(x, axis=1) for x in predictions], axis=0)

true_y = np.concatenate(true_labels, axis=0)



print(classification_report(true_y, predicted_y))
np.concatenate(true_labels, axis=0)
dataset = """{"id": 1, "text": "чем занимаешься по жизни я вот бизнесмен", "meta": {}, "annotation_approver": null, "labels": [[31, 40, "activity"]]}

{"id": 2, "text": "а я вот учу детей работаю с начальными классами к свадьбе готовлюсь", "meta": {}, "annotation_approver": null, "labels": [[8, 17, "activity"]]}

{"id": 3, "text": "я люблю есть арбуз любишь готовить", "meta": {}, "annotation_approver": null, "labels": [[8, 18, "hobby"], [26, 34, "hobby"]]}

{"id": 4, "text": "люблю готовить пасту у меня классно получается", "meta": {}, "annotation_approver": null, "labels": [[6, 20, "hobby"]]}

{"id": 5, "text": "хочу быть психологом в газпроме а кем ты работаешь", "meta": {}, "annotation_approver": null, "labels": [[10, 20, "activity"]]}

{"id": 6, "text": "не поверишь я психолог в партии роста", "meta": {}, "annotation_approver": null, "labels": [[14, 22, "activity"]]}

{"id": 7, "text": "а у меня мама домохозяйка поэтому редко бываю дома одна в питере а ты", "meta": {}, "annotation_approver": null, "labels": [[58, 64, "place"], [14, 25, "activity"]]}

{"id": 8, "text": "а я в ростове-на-дону на 130-м шоссе в придорожном кафе", "meta": {}, "annotation_approver": null, "labels": [[6, 21, "place"]]}

{"id": 9, "text": "кто по профессии ты расскажи о себе, контракты перебираешь", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 10, "text": "лошадей люблю у нас были давно 10 лошадей вот следил за ними отцу помогал с тех пор так подрабатываю иногда как с трудоустройством в норвегии", "meta": {}, "annotation_approver": null, "labels": [[133, 141, "place"], [0, 7, "hobby"]]}

{"id": 11, "text": "живу в норвегии с 11 лет сейчас мне 21 уехали с родителями отцу надо было по работе остались периодически бываем в россии конечно", "meta": {}, "annotation_approver": null, "labels": [[7, 15, "place"], [115, 121, "place"]]}

{"id": 12, "text": "круто я бы тоже поехал люблю путешествовать думаю посетить страну когда последний раз был в россии", "meta": {}, "annotation_approver": null, "labels": [[92, 98, "place"], [29, 43, "hobby"]]}

{"id": 13, "text": "летом 2017 давно скучаю по россии ты из города на неве", "meta": {}, "annotation_approver": null, "labels": [[27, 33, "place"]]}

{"id": 14, "text": "я из тульской области чем увлекаешься", "meta": {}, "annotation_approver": null, "labels": [[5, 21, "place"]]}

{"id": 15, "text": "добрый день кирилл сто кило приятно, потому что вешу 250 фунтов познакомится", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 16, "text": "чем занимаетесь кирилл английскую музыку любите и английский язык наверно знаете", "meta": {}, "annotation_approver": null, "labels": [[23, 40, "hobby"], [50, 65, "hobby"]]}

{"id": 17, "text": "в декрете за жену сижу у меня три дочери - вот слежу, по спектаклям и в кино не хожу", "meta": {}, "annotation_approver": null, "labels": [[0, 9, "activity"], [57, 67, "hobby"], [70, 76, "hobby"]]}

{"id": 18, "text": "а я само леты испытываю сейчас много новых разработок так что постоянно в стрессе", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 19, "text": "на самом деле втягиваешься в домашнюю работу я кулинар - люблю готовить - вся семья довольна но конечно хочется обратно на работу все же", "meta": {}, "annotation_approver": null, "labels": [[63, 71, "hobby"], [123, 129, "activity"], [47, 54, "activity"]]}

{"id": 20, "text": "мне бы так расслабиться как вы но детей пока нет все с женой только хотим завести а кем работали до декрета", "meta": {}, "annotation_approver": null, "labels": [[100, 107, "activity"]]}

{"id": 21, "text": "сходите в бассейн - хорошо расслабляет я хорошо плаваю рекомендую", "meta": {}, "annotation_approver": null, "labels": [[10, 17, "hobby"]]}

{"id": 22, "text": "а я пою хорошо это мое расслабление мечтаю поучаствовать в шоу голос", "meta": {}, "annotation_approver": null, "labels": [[4, 7, "hobby"], [59, 68, "hobby"]]}

{"id": 23, "text": "представляешь летчик - испытатель в шоу бизнесе", "meta": {}, "annotation_approver": null, "labels": [[14, 33, "activity"]]}

{"id": 24, "text": "я помогаю людям правильно одеваться сейчас просто только онлайн могу это делать из - за декрета а так очень люблю шопинг", "meta": {}, "annotation_approver": null, "labels": [[88, 95, "activity"], [114, 120, "hobby"], [2, 35, "activity"]]}

{"id": 25, "text": "хорошо поющий лётчик это находка особенно во время восстания", "meta": {}, "annotation_approver": null, "labels": [[14, 20, "activity"]]}

{"id": 26, "text": "так сейчас многие онлайн из дома работают это удобно особенно для тех кто в декрете", "meta": {}, "annotation_approver": null, "labels": [[76, 83, "activity"]]}

{"id": 27, "text": "да так и получается так мечтаю поехать в париж опять но думаю скоро реализую мечту", "meta": {}, "annotation_approver": null, "labels": [[31, 46, "hobby"]]}

{"id": 28, "text": "пробовался в песеных конкурсах караоке например, далеко добираться, сначала на поезде, потом на автобусе или на машине, и в конце на пароме", "meta": {}, "annotation_approver": null, "labels": [[31, 38, "hobby"]]}

{"id": 29, "text": "может дашь пару советов по стилю что сейчас у мужиков в моде а то я постоянно в форме и редко выхожу в люди а так джинсы и пиджак моя основная домашняя одежда", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 30, "text": "ну на работе участвую в самодеятельности метные конкурсы так сказать но нигде на больших сценах не выступал меня уже достали на корпоративах просить спеть", "meta": {}, "annotation_approver": null, "labels": [[149, 154, "hobby"]]}

{"id": 31, "text": "хотя лепс тоже с ресторанов начинал не люблю лепса", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 32, "text": "отель 2 суток бесплатно третий день 270 ₽ далее 70 ₽ сутки скидка получается 75%", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 33, "text": "не надо tele2 не надо мне sim-карту она у вас за евро", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 34, "text": "а я встречаюсь с парнем надеюсь все серьёзно он у меня первый и единственный", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 35, "text": "свободное время посещаю животным в приюте собак ведь они такие умные", "meta": {}, "annotation_approver": null, "labels": [[16, 41, "hobby"]]}

{"id": 36, "text": "у тебя есть родные братья или сестры", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 37, "text": "сейчас реп слушаю а так из рока skillet nirvana слот ну и другие а ты а из репа всего по не многу очень люблю сёрфинг а ты пробовала", "meta": {}, "annotation_approver": null, "labels": [[7, 17, "hobby"], [110, 117, "hobby"]]}

{"id": 38, "text": "я программистом работаю программы пишу под 1 с", "meta": {}, "annotation_approver": null, "labels": [[2, 15, "activity"], [24, 46, "activity"]]}

{"id": 39, "text": "очень классно бухгалтерия я фрилансер", "meta": {}, "annotation_approver": null, "labels": [[28, 37, "activity"]]}

{"id": 40, "text": "бухгалтер классно я тоже дома работаю", "meta": {}, "annotation_approver": null, "labels": []}

{"id": 41, "text": "не фрилансер имела в виду 1с под бухгалтерию крутые программы", "meta": {}, "annotation_approver": null, "labels": [[3, 12, "activity"]]}"""
import json

import math



def split_train_test_for_ner(samples, train_size=0.8):

    """

    split input samples according to train_size and labels distribution

    """

    

    label2samples = {'no_label': set()}

    

    for i, sample_as_str in enumerate(samples):

        sample = json.loads(sample_as_str)

        labels = sample['labels']

        

        if len(labels) == 0:

            label2samples['no_label'].add(i)

        else:    

            for label_data in labels:

                label_name = label_data[2]



                if label_name not in label2samples:

                    label2samples[label_name] = set()

                

                label2samples[label_name].add(i)

    

    train_indexes = {x for x in range(len(samples))}

    test_indexes = set()

    

    for label, samples_id in label2samples.items():

        samples_id = samples_id - test_indexes     # remove selected indexes

        samples_id = list(samples_id)

        

        n_samples = len(samples_id)                # amount of samples that contain label

        n_test = math.ceil(n_samples * (1 - train_size)) # required samples for test

        

        for i in range(n_test):

            idx = np.random.randint(n_samples)     # select random sample

            sample_id = samples_id.pop(idx)  

            n_samples = n_samples - 1

            

            test_indexes.add(sample_id)

    

    

    return list(train_indexes - test_indexes), list(test_indexes)



dataset_to_split = dataset.split('\n')

train, test = split_train_test_for_ner(dataset_to_split)     



print(train)

print(test)



print(len(test) / len(dataset_to_split))