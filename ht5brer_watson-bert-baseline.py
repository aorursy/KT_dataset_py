import pandas as pd

import numpy as np





from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping



from transformers import BertConfig, TFBertForSequenceClassification

from transformers import BertTokenizer



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report







import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





import os
train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")

test  = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")

submit = pd.read_csv("../input/contradictory-my-dear-watson/sample_submission.csv")
plt.figure(figsize=(15, 10))

sns.countplot(x="label", hue="language", data=train)

plt.title("distribution of train language by label")
train["type"] = "train"

test["type"] = "test"

data = pd.concat([train, test], axis=0)



plt.figure(figsize=(15, 10))

sns.countplot(x="language", hue="type", data=data)

plt.title("distribution of train and test language")
def get_sample(train, label, lang='en'):

    print(f"label: {label}")

    sample = train[(train.label == label) & (train.lang_abv==lang)].sample(n=1)

    print(f"premise: {sample.premise.values[0]}")

    print(f"hypothesis: {sample.hypothesis.values[0]}")

    

get_sample(train, 0)

get_sample(train, 1)

get_sample(train, 2)
# data load and get feature, label

def load_dataset(data_path, is_train=True):

    df = pd.read_csv(data_path)

    features = list(zip(df.premise, df.hypothesis))

    if is_train:

        labels = df.label

    else:

        labels = None

    return  features, labels
def convert_examples_to_features(x, max_seq_length, tokenizer):

    features = {

        'input_ids': [],

        'attention_mask': [],

        'token_type_ids': [],

    }

        

    for pairs in x:

        # add [CLS]

        tokens = [tokenizer.cls_token]

        token_type_ids = []

        for i, sent in enumerate(pairs):

            word_tokens = tokenizer.tokenize(sent)

            tokens.extend(word_tokens)

            # add [SEP]

            tokens += [tokenizer.sep_token]

            len_sent = len(word_tokens) + 1

            token_type_ids += [i] * len_sent



        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)



        features['input_ids'].append(input_ids)

        features['attention_mask'].append(attention_mask)

        features['token_type_ids'].append(token_type_ids)



    for name in ['input_ids', 'attention_mask', 'token_type_ids']:

        features[name] = pad_sequences(features[name], padding='post', maxlen=max_seq_length)



    x = [features['input_ids'], features['attention_mask'], features['token_type_ids']]

    return x
os.environ["WANDB_API_KEY"] = "0"

def build_model(pretrained_model_name_or_path, num_labels):

    config = BertConfig.from_pretrained(

        pretrained_model_name_or_path,

        num_labels=num_labels

    )

    model = TFBertForSequenceClassification.from_pretrained(

        pretrained_model_name_or_path,

        config=config

    )

    model.layers[-1].activation = tf.keras.activations.softmax

    return model
def evaluate(model, features):

    label = model.predict(features)

    y_pred = np.argmax(label, axis=-1)

    return y_pred[0]
# Set hyper-parameters.

batch_size = 32

epochs = 100

model_path = '/'

pretrained_model_name_or_path = 'bert-base-multilingual-cased'

maxlen = 250



# data loading.

x, y = load_dataset("../input/contradictory-my-dear-watson/train.csv")

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)



# train valid split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

features_train = convert_examples_to_features(

    x_train,

    max_seq_length=maxlen,

    tokenizer=tokenizer

)

features_test = convert_examples_to_features(

    x_test,

    max_seq_length=maxlen,

    tokenizer=tokenizer

)



# Build model.

model = build_model(pretrained_model_name_or_path, len(set(y)))

model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')



# Preparing callbacks.

callbacks = [

    EarlyStopping(patience=3),

]



# Train the model.

model.fit(x=features_train,

          y=y_train,

          batch_size=batch_size,

          epochs=epochs,

          validation_split=0.1,

          callbacks=callbacks)

model.save_pretrained(model_path)



# Evaluation.

y_pred = evaluate(model, features_test)

print(classification_report(y_test, y_pred, digits=4))
result = y_test.to_frame()

result.columns = ["real"]

result["pred"] = y_pred

result = pd.concat([result, train], axis=1)

result.dropna(inplace=True)



fig = plt.figure(figsize=(20,10))

score = accuracy_score(result["label"], result["pred"])

sns.heatmap(pd.crosstab(result["label"], result["pred"]), annot=True, fmt="d", label=f"all {score}")

plt.title(f"all {score: .3f}")
lang = result.language.unique()



fig, axes = plt.subplots(3, 5, figsize=(20,10))

plt.subplots_adjust(wspace=0.4, hspace=0.6)

for i, l in enumerate(lang):

    lang_result = result[result.language == l]

    score = accuracy_score(lang_result["label"], lang_result["pred"])

    sns.heatmap(pd.crosstab(lang_result["label"], lang_result["pred"]), annot=True, fmt="d", ax=axes[i%3, i//3])

    axes[i%3, i//3].set_title(f"{l} {score: .3f}")
x_submit, _ = load_dataset("../input/contradictory-my-dear-watson/test.csv", is_train=False)

features_submit = convert_examples_to_features(

    x_submit,

    max_seq_length=maxlen,

    tokenizer=tokenizer

)



y_pred = evaluate(model, features_submit)



submit["prediction"] = y_pred

submit.to_csv("submission.csv", index=False)