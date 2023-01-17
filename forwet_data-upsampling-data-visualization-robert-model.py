# Installing nlp library for loading external dataset.
!pip install -q nlp
!pip install -q wordcloud
# Importing Libs
import nlp
import numpy as np
import pandas as pd
import tensorflow as tf
import random, os, math, cv2
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold, train_test_split
from transformers import BertTokenizer, TFBertModel, AutoTokenizer, TFAutoModel

# Text visualization
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
#PREPAIRING TPU
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU

print('Number of replicas:', strategy.num_replicas_in_sync)
# Loading original data
train_csv = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
test_csv  = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")
# CONFIGURATION

AUTO = tf.data.experimental.AUTOTUNE
MODEL_NAME = "jplu/tf-xlm-roberta-base"
REPLICAS  = strategy.num_replicas_in_sync
TOKENIZER  = AutoTokenizer.from_pretrained(MODEL_NAME)

# HYPER-PARAMS
BATCH_SIZE = 16 * REPLICAS
MAX_LEN = 192
EPOCHS = 8 # Due to running time for notebook. Please try to train model on atleast 5-10 epochs.
SEED = 48
FONT_DIR = "../input/font-dataset/FontScripts/"

np.random.seed(SEED)
random.seed(SEED)
def prepare_input_v2(sentences):
    """ Converts the premise and hypothesis to the input format required by model"""
    sen_enc = TOKENIZER.batch_encode_plus(sentences,
                                          pad_to_max_length=True,
                                          return_attention_mask=False,
                                          return_token_type_ids=False,
                                          max_length=MAX_LEN)
    return np.array(sen_enc["input_ids"])

def get_dataset(features, labels=None, labelled=True, batch_size=8, repeat=True, shuffle=True):
    """Generates a tf.data pipeline from the encoded sentences."""
    if labelled:
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(features)

    if repeat:
        ds = ds.repeat()
        
    if shuffle:
        ds = ds.shuffle(2048)
        
    ds = ds.batch(batch_size*REPLICAS)
    ds.prefetch(AUTO)
    return ds


def build_model():
    """Prepare the model for fine-tuning."""
    encoder = TFAutoModel.from_pretrained(MODEL_NAME)
    input_word_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
    
    # Passing input to pretrained model.
    embeddings = encoder(input_word_ids)[0]
    x = embeddings[:, 0, :]
    
    output = tf.keras.layers.Dense(3, activation="softmax")(x)
    
    model = tf.keras.models.Model(inputs=input_word_ids, outputs=output)
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
                  optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                  metrics=["accuracy"])
    return model

def ratio_languages(df):
    """Prints out the ratio of all the languages in the dataset"""
    languages = np.unique(df.language)
    total = df.language.value_counts().sum() 
    ratios = {}
    for e in languages:
        ratios[e] = round((df.language.value_counts().loc[e] / total), 2)*100
    
    ratios = sorted(ratios.items(), key=lambda x: (x[1],x[0]), reverse=True)
    
    languages = []
    values = []
    for e in ratios:
        languages.append(e[0])
        values.append(e[1])
    _, texts, _ = plt.pie(values, explode=[0.2]*(len(values)), labels=languages, autopct="%.2i%%", radius=2, 
                             rotatelabels=True)
    for e in texts:
        e.set_fontsize(15)
        e.set_fontfamily('fantasy')
    plt.show()

def get_lr_callback(batch_size):
    lr_start = 0.000001
    lr_max   = 0.00000125 * batch_size
    lr_min   = 0.00000001
    lr_sus_epoch = 0
    lr_decay = 0.80
    lr_ramp_ep = 5
    lr = lr_start
    
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max- lr_start)/lr_ramp_ep * epoch + lr_start
        elif epoch < (lr_ramp_ep + lr_sus_epoch):
            lr = lr_max
        else:
            lr = (lr_max - lr_min)*lr_decay**(epoch - lr_ramp_ep - lr_sus_epoch)+ lr_min
        return lr
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback

def plot_wordcloud(df, col):
    """Function to plot word cloud for multiple languages"""
    words = " "
    font_path = None

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(12, 12)

    res = []
    for i in range(2):
      for j in range(2):
        res.append([i,j])

    for i,lang in enumerate(["English", 
                             "Hindi", 
                             "Urdu",
                             "German" ,        
                            ]):
      
          for line in df[df.language==lang][col].values:
                tokens = line.split()

                tokens = [word.lower() for word in tokens]
                words += " ".join(tokens)+" "
        
          fig.add_subplot(ax[res[i][0]][res[i][1]])

          if lang=="Hindi":
            font_path = FONT_DIR + "Hindi.ttf"

          if lang=="French":
            font_path =  FONT_DIR + "French.ttf"

          if lang=="Russian":
            font_path= FONT_DIR + "Russian.ttf"

          if lang=="Arabic":
            font_path = FONT_DIR + "Arabic.ttf"

          if lang=="Chinese":
            font_path = FONT_DIR + "Chinese.otf"

          if lang=="Swahili":
            font_path = FONT_DIR + "Swahili.ttf"

          if lang=="Urdu":
            font_path = FONT_DIR + "Urdu.ttf"

          if lang=="Vietnamese":
            font_path = FONT_DIR + "Vietnamese.ttf"

          if lang=="Greek":
            font_path = FONT_DIR + "Greek.ttf"

          if lang=="Thai":
            font_path = FONT_DIR + "Thai.ttf"

          if lang=="Spanish":
            font_path = FONT_DIR + "Spanish.ttf"

          if lang=="German":
            font_path = FONT_DIR + "German.ttf"

          if lang=="Turkish":
            font_path = FONT_DIR + "Turkish.ttf"

          if lang=="Bulgarian":
            font_path = FONT_DIR + "Bulgarian.ttf"

          s_words = STOPWORDS

          wordcloud = WordCloud(font_path=font_path, width=800, height=800, 
                                background_color="black",
                                min_font_size=10,
                                stopwords=s_words).generate(words)

          ax[res[i][0]][res[i][1]].imshow(wordcloud)
          ax[res[i][0]][res[i][1]].axis("off")
          ax[res[i][0]][res[i][1]].set_title(f"Language: {lang}",  fontsize=14)  
# Value counts of samples for each language.
print(train_csv["language"].value_counts())

# Printing ratio of different language in the dataset
print()
ratio_languages(train_csv)
# Load xnli dataset
xnli = nlp.load_dataset(path="xnli")

# As this dataset does not contain direct 
# column name (premise, hypothesis) to sentence pair 
# and so we need to extract it out.
buff = {}
buff["premise"] = []
buff["hypothesis"] = []
buff["label"] = []
buff["language"] = []

# Making a set to map our dataset language abbreviations to 
# their complete names.
uniq_lang = set()
for e in xnli["test"]:
  for i in e["hypothesis"]["language"]:
    uniq_lang.add(i)

# Creating a dict that maps abv to their complete names. 
language_map = {}

# Taken test_csv just to use lang_abv column and nothing else.
for e in uniq_lang:
  language_map[e] = test_csv.loc[test_csv.lang_abv==e, "language"].iloc[0]

# Prepairing the dataset with the required columns.
for x in xnli['test']:
    label = x['label']
    for idx, lang in enumerate(x['hypothesis']['language']):
        
        # Skipping english samples as we don't want to upsample the samples
        # corresponding to english language.
        if lang=="en":
            continue
            
        hypothesis = x['hypothesis']['translation'][idx]
        premise = x['premise'][lang]
        buff['premise'].append(premise)
        buff['hypothesis'].append(hypothesis)
        buff['label'].append(label)
        buff['language'].append(language_map[lang])

# A pandas DataFrame for the prepared dataset.
xnli_df = pd.DataFrame(buff)
xnli_df["language"].value_counts()
# Extract columns which required from the dataset.
# Note: The columns in train_df and xnli_df must be same as we would be merging 
# them for upsampling.
train_df = train_csv[["premise", "hypothesis", "label", "language"]]
train_df.head()
# Concatenate the complete dataset
new_df = pd.concat([train_df, xnli_df], axis=0)
new_df.sample(5)
pd.merge(new_df, test_csv, how="inner")
new_df = new_df.merge(pd.merge(new_df, test_csv, how="inner"), how="left", indicator=True)
new_df = new_df[new_df._merge=="left_only"]
new_df = new_df.drop(["id", "lang_abv", "_merge"], axis=1)
new_df.info()
# No null instances in the dataset.
pd.merge(new_df, test_csv, how="inner")
ratio_languages(new_df)
new_df.language.value_counts()
# LOAD EXTRA DATA END
plot_wordcloud(new_df, "premise")
plot_wordcloud(new_df, "hypothesis")
X, y = new_df[["premise", "hypothesis"]], new_df.label
X["language_label"] = new_df.language.astype(str) + "_" + new_df.label.astype(str)
print("Splitting Data...")

# Using train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=X.language_label, test_size=0.2, random_state=SEED)

y_train = tf.one_hot(y_train, depth=3)
y_test  = tf.one_hot(y_test, depth=3)

print("Prepairing Input...")
train_input = prepare_input_v2(x_train[["premise", "hypothesis"]].values.tolist())
valid_input = prepare_input_v2(x_test[["premise", "hypothesis"]].values.tolist())

print("Preparing Dataset...")
train_dataset = get_dataset(train_input, y_train, labelled=True, batch_size=BATCH_SIZE, repeat=True, 
                            shuffle=True)
valid_dataset   = get_dataset(valid_input, y_test, labelled=True, batch_size=BATCH_SIZE//REPLICAS, repeat=False,
                            shuffle=False)

print("Downloading and Building Model...")
with strategy.scope():
    model  = build_model()

# Callbacks
#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.4, patience=3,
#                                                 verbose=1)

lr_callback = get_lr_callback(BATCH_SIZE)
checkpoint = tf.keras.callbacks.ModelCheckpoint("XLM-R-base.h5", save_weights_only=True,
                                                save_best_only=True, save_freq="epoch", monitor="val_loss",
                                                mode="min")

print("Training...")
model.fit(train_dataset, 
         steps_per_epoch= x_train.shape[0]/BATCH_SIZE,
         validation_data=valid_dataset,
         epochs=EPOCHS,
         callbacks=[lr_callback, checkpoint])
test_input = prepare_input_v2(test_csv[["premise", "hypothesis"]].values.tolist())
test_dataset = get_dataset(test_input, None, labelled=False, batch_size=BATCH_SIZE, repeat=False, shuffle=False) 
preds = model.predict(test_dataset)
preds = preds.argmax(axis=1)
submission = pd.read_csv("../input/contradictory-my-dear-watson/sample_submission.csv")
submission.head()
submission["prediction"] = preds
submission.sample(10)
submission.to_csv("submission.csv", header=True, index=False)
