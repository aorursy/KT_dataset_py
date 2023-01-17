import os; os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import keras
import keras.backend as K
train_df = pd.read_csv("../input/train.csv", sep=",")
print(train_df.shape)
train, dev = train_test_split(train_df, random_state=123, shuffle=True, test_size=0.1)
print("Training data shape:", train.shape)
print("Test data shape:", dev.shape)
%%time

def get_project_essay(df):
    return (df["project_essay_1"].fillna('') +
            ' ' + df["project_essay_2"].fillna('') +
            ' ' + df["project_essay_3"].fillna('') +
            ' ' + df["project_essay_4"].fillna(''))

def get_text(df):
    return df["project_title"].fillna('') + ' ' + get_project_essay(df)

#project_title_tokenizer = keras.preprocessing.text.Tokenizer()
#project_title_tokenizer.fit_on_texts(train["project_title"])

#project_essay_tokenizer = keras.preprocessing.text.Tokenizer()
#project_essay_tokenizer.fit_on_texts(get_project_essay(train))

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(get_text(train))

def preprocess_target(df):
    return df[["project_is_approved"]].copy()

def preprocess_data(df):
    processed_df = df[["teacher_number_of_previously_posted_projects"]].copy()

    #processed_df["project_title"] = project_title_tokenizer.texts_to_sequences(df["project_title"])
    processed_df["project_title"] = tokenizer.texts_to_sequences(df["project_title"])
    
    #processed_df["project_essay"] = project_essay_tokenizer.texts_to_sequences(get_project_essay(df))
    processed_df["project_essay"] = tokenizer.texts_to_sequences(get_project_essay(df))
    
    return processed_df

processed_train = preprocess_data(train)
y_train = preprocess_target(train)
print(processed_train.shape, y_train.shape)

processed_dev = preprocess_data(dev)
y_dev = preprocess_target(dev)
print(processed_dev.shape, y_dev.shape)
processed_train["project_title"].apply(lambda x: len(x)).hist(bins=10)
processed_train["project_essay"].apply(lambda x: len(x)).hist(bins=10)
MAX_PROJECT_TITLE_SEQ_LEN = 12
MAX_PROJECT_TITLE = processed_train["project_title"].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1

MAX_PROJECT_ESSAY_SEQ_LEN = 450
MAX_PROJECT_ESSAY = processed_train["project_essay"].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1

MAX_TEXT = max([MAX_PROJECT_TITLE, MAX_PROJECT_ESSAY])

def get_keras_data(df):
    return {
        "teacher_number_of_previously_posted_projects": np.array(df["teacher_number_of_previously_posted_projects"]),
        "project_title": keras.preprocessing.sequence.pad_sequences(df["project_title"], maxlen=MAX_PROJECT_TITLE_SEQ_LEN),
        "project_essay": keras.preprocessing.sequence.pad_sequences(df["project_essay"], maxlen=MAX_PROJECT_ESSAY_SEQ_LEN),
    }

X_train = get_keras_data(processed_train)
X_dev = get_keras_data(processed_dev)
def create_rnn_model():
    # Input layers
    teacher_number_of_previously_posted_projects = keras.layers.Input(shape=(1,), name="teacher_number_of_previously_posted_projects")
    project_title = keras.layers.Input(shape=(MAX_PROJECT_TITLE_SEQ_LEN,), name="project_title")
    project_essay = keras.layers.Input(shape=(MAX_PROJECT_ESSAY_SEQ_LEN,), name="project_essay")
    #project_resource_summary = keras.layers.Input(shape=(MAX_PROJECT_RESOURCE_SUMMARY_SEQ_LEN,), name="project_resource_summary")
    
    # Embedding layers
    #emb_project_title = keras.layers.Embedding(MAX_PROJECT_TITLE, 25)(project_title)
    #emb_project_essay = keras.layers.Embedding(MAX_PROJECT_ESSAY, 50)(project_essay)
    emb_layer = keras.layers.Embedding(MAX_TEXT, 50)
    emb_project_title = emb_layer(project_title)
    emb_project_essay = emb_layer(project_essay)
    
    # RNN layers
    rnn_project_title = keras.layers.GRU(8, activation="relu")(emb_project_title)
    rnn_project_essay = keras.layers.GRU(16, activation="relu")(emb_project_essay)
    #rnn_project_resource_summary = keras.layers.GRU(16, activation="relu")(emb_project_resource_summary)
    
    # Merge all layers into one
    x = keras.layers.concatenate([teacher_number_of_previously_posted_projects,
                                 rnn_project_title,
                                 rnn_project_essay,
                                 #rnn_project_resource_summary,
                                 ])
    
    # Dense layers
    #x = keras.layers.Dense(128, activation="relu")(x)

    # Output layers
    output = keras.layers.Dense(1, activation="sigmoid")(x)
    
    return keras.models.Model(
        inputs=[teacher_number_of_previously_posted_projects,
                project_title,
                project_essay,
                #project_resource_summary,
               ],
        outputs=output)

rnn_model = create_rnn_model()
rnn_model.summary()
optimizer = keras.optimizers.Adam(lr=0.001)
rnn_model.compile(optimizer=optimizer,
                  loss=keras.losses.binary_crossentropy,
                  metrics=["accuracy"])

for i in range(3):
    rnn_model.fit(X_train, y_train,
                 batch_size=(2 ** (i + 8)),
                 epochs=1,
                 validation_data=(X_dev, y_dev))
preds = rnn_model.predict(X_dev, batch_size=512)
auc_score = roc_auc_score(y_dev, preds)
print("AUC for validation data: %.4f" % (auc_score,))
test_df = pd.read_csv("../input/test.csv", sep=',')

processed_test = preprocess_data(test_df)

X_test = get_keras_data(processed_test)

preds = rnn_model.predict(X_test, batch_size=512)

submission = pd.DataFrame({
    "id": test_df["id"],
    "project_is_approved": preds.reshape(-1),
})

submission.to_csv("submission.csv", index=False)