# necessary installs

%pip install --upgrade tensorflow-gpu
# standard library imports

import os



from typing import List



# third party imports

import numpy as np

import pandas as pd

import spacy



from IPython.display import display, Markdown

from sklearn import metrics, set_config

from sklearn.model_selection import train_test_split



# some config settings

pd.set_option("display.max_colwidth", None)

set_config(display='diagram')



# some helper functions

def md(text: str):

    display(Markdown(text))

    



def benchmark(y_test, y_hat):

    print('_' * 80)

    print(f"f1 score: {metrics.f1_score(y_test, y_hat, average='macro'):.3f}")

    print("classification report:")

    print(metrics.classification_report(y_test, y_hat))
folder = "../input/nlp-getting-started"



test = pd.read_csv(os.path.join(folder, "test.csv"), index_col="id")

train = pd.read_csv(os.path.join(folder, "train.csv"), index_col="id")



X = train["text"]

y = train["target"]



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"We are using `{len(X_train)}` rows for model training, and `{len(X_valid)}` rows for validation")
X_train.head().to_frame()
from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from sklearn.pipeline import Pipeline



vectorizers = {

    "hashing": HashingVectorizer(stop_words="english", strip_accents="unicode", alternate_sign=False),

    "tf-idf": TfidfVectorizer(stop_words="english", strip_accents="unicode", sublinear_tf=True),

}



pipe = Pipeline([

    ("vectorizer", None),

    ("model", None)

])
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import RidgeClassifierCV, SGDClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

from sklearn.svm import LinearSVC, SVC



feature_options = {

    "vectorizer": [vectorizers["hashing"], vectorizers["tf-idf"]]

}



baseline_params = [

    {

        "model": [

            ComplementNB(),

        ],

        **feature_options,

    },

    {

        "model": [SGDClassifier()],

        "model__penalty": ["elasticnet", "l2"],

        **feature_options,

    },

    {

        "model": [LinearSVC()],

        "model__penalty": ["l1", "l2"],

        **feature_options

    }

]



grid = GridSearchCV(pipe, param_grid=baseline_params, cv=5, scoring="f1", n_jobs=-1, verbose=3)

_ = grid.fit(X_train, y_train)
print(grid.best_params_)



baseline = grid.best_estimator_



baseline
benchmark(y_valid, baseline.predict(X_valid))
failed = (

    X_valid

    .to_frame("text")

    .assign(

        y_hat=baseline.predict(X_valid),

        y_true=y_valid

    )

)



false_negatives = failed.pipe(lambda df: df[df["y_hat"].eq(0) & df["y_true"].eq(1)])

false_positives = failed.pipe(lambda df: df[df["y_hat"].eq(1) & df["y_true"].eq(0)])



false_negatives[:20]
false_positives[:20]
del false_positives

del failed
vectorizer = baseline.get_params()["vectorizer"]



def highlight_examples(vectorizer, examples: List[str]) -> None:

    for example in examples:

        print("-" * 80)

        print("original:", example)

        print()

        print("preprocessor:", vectorizer.build_preprocessor()(example))

        print()

        print("tokenizer:", vectorizer.build_tokenizer()(example))

        print()

        print("complete analyzer:", vectorizer.build_analyzer()(example))



example = false_negatives["text"].iloc[5]



highlight_examples(vectorizer, false_negatives["text"].iloc[:4])
del false_negatives
import re



from spacy.matcher import Matcher

from spacy.pipeline import EntityRuler

from spacy.tokens import Token

from spacy.tokenizer import Tokenizer

from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex, filter_spans





def custom_tokenizer(nlp):

    """

    Custom tokenizer to deal with common gotcha's in tweets.

    

    Prefix:

        - split off the hashtags to start

    

    Special cases:

        - Convert the "&amp" artefact into "&"

    

    Infix splits:

        - Split tokens if there was a missing sapce between a word and hashtag

        - Split tokens if there was two words linked by a slash e.g. "foo/bar"

        

    Ensure URLs don't get split up.

    

    Args:

        nlp: A spacy pipeline.

    

    Returns:

        A spacy tokenizer.

    """

    tokenizer = nlp.tokenizer

    

    special_cases = {

        ":)": [{"ORTH": ":)"}],

        "&amp": [{"ORTH": "&"}]

    }

    

    for text, change in special_cases.items():

        tokenizer.add_special_case(text, change)

    

    # infixes

    infixes = nlp.Defaults.infixes + (r"""[@:;+/\-~#*]""", r"""&amp""")

    infix_re = compile_infix_regex(infixes)

    tokenizer.infix_finditer = infix_re.finditer

    

    # prefixes

    prefixes = nlp.Defaults.prefixes + (r"""[\^\.]""",)

    prefix_re = spacy.util.compile_prefix_regex(prefixes)

    tokenizer.prefix_search = prefix_re.search

    

    # suffixes

    suffixes = nlp.Defaults.suffixes + (r"""[\^\.]""",)

    suffix_re = spacy.util.compile_suffix_regex(suffixes)

    tokenizer.suffix_search = suffix_re.search



    tokenizer.token_match = re.compile(r"""https?://""").match

    

    return tokenizer





def username_labeller(nlp):

    ruler = EntityRuler(nlp)

    patterns = [

        {"label": "USERNAME", "pattern": [{"TEXT": {"REGEX": "^@\w+$"}}]},

        {"label": "USERNAME", "pattern": [{"ORTH": "@"}, {"IS_ASCII": True}]}

    ]

    ruler.add_patterns(patterns)

    return ruler





class HashtagMerger:

    """

    Pipeline step to merge hashtags together.

    

    Code based on examples here: https://spacy.io/usage/rule-based-matching

    """

    name = "hashtag_merger"

    

    def __init__(self, nlp):

        Token.set_extension("is_hashtag", default=False, force=True)

        self.matcher = Matcher(nlp.vocab)

        self.matcher.add("HASHTAG", None, [{"ORTH": "#"}, {"IS_ASCII": True}])

    

    def __call__(self, doc) -> "spacy.tokens.doc.Doc":

        # This method is invoked when the component is called on a Doc

        matches = self.matcher(doc)

        spans = []  # Collect the matched spans here

        for match_id, start, end in matches:

            spans.append(doc[start:end])

            

        filtered_spans = filter_spans(spans)

        with doc.retokenize() as retokenizer:

            for span in filtered_spans:

                retokenizer.merge(span)

                for token in span:

                    token._.is_hashtag = True  # Mark token as a hashtag

        return doc

        



def create_nlp(model: str = "en_core_web_sm"):

    nlp = spacy.load(model, disable=["parser", "tagger", "ner"])

    nlp.tokenizer = custom_tokenizer(nlp)

    username = username_labeller(nlp)

    nlp.add_pipe(username)

    hashtag_merger = HashtagMerger(nlp)

    nlp.add_pipe(hashtag_merger, last=True)

    

    return nlp
nlp = create_nlp()

doc = nlp("^oo^ hello-world. :) http://t.co/ABC-DEF measures#blah #foo @user &amp friends/enemies")



pd.DataFrame({

    "text": [token.text for token in doc],

    "like_url": [token.like_url for token in doc],

    "entity_type": [token.ent_type_ for token in doc],

    "is_hashtag": [token._.is_hashtag for token in doc],

    "is_ascii": [token.is_ascii for token in doc],

    "is_punct": [token.is_punct for token in doc]

})
from typing import Callable, List



def is_valid_token(token: spacy.tokens.Token) -> bool:

    """

    Is not a hashtag, url or username, is ascii and not punctuation.

    

    We keep numbers in. It tends to be that models are able to use the useful ones and filter out the noise.

    """

    return (

        not token.like_url

        and not token._.is_hashtag

        and token.ent_type_ != "USERNAME"

        and not token.is_punct

        and token.text.strip() != "" # drop off any newlines

        and token.is_ascii # avoiding symbols like: Ûª

        # we don't filter out stopwords here. we do it as part of the scikit learn text vectorizers

        # and not token.is_stop

    )



def extract_text(

    tweet: spacy.tokens.Doc,

    include_text: bool = True,

    include_hashtags: bool = True,

    include_handles: bool = False

) -> str:

    cleaned_tokens = []

    for token in tweet:

        if (token._.is_hashtag and include_hashtags):

            cleaned_tokens.append(token.text.replace("#", ""))

        elif (token.ent_type_ == "USERNAME" and include_handles):

            cleaned_tokens.append(token.text.replace("@", ""))

        elif (is_valid_token(token) and include_text):

            cleaned_tokens.append(token.text.lower())



    return " ".join(cleaned_tokens)



def clean_tweets(tweets: List[str], **kwargs) -> List[str]:

    nlp = create_nlp()

    cleaned_tweets = []

    

    for doc in nlp.pipe(tweets, n_process=1):

        string = extract_text(doc, **kwargs)

        cleaned_tweets.append(string)

    return cleaned_tweets
from sklearn.base import clone

from sklearn.preprocessing import FunctionTransformer



cleaned_grid = clone(grid)



new_pipe = Pipeline([

    ("clean", FunctionTransformer(clean_tweets)),

    ("vectorizer", None),

    ("model", None)

])



cleaning_param_grid = {

    "clean__kw_args": [

        {"include_text": text, "include_hashtags": hashtags, "include_handles": handles}

        for text in [True]

        for hashtags in [True]

        for handles in [False]

        # you can try all combinations. For speed I've put in the best combo by default

    ]

}



old_param_grid = cleaned_grid.get_params()["param_grid"]

cleaned_grid.set_params(

    estimator=new_pipe,

    param_grid={**{key: [value] for key,value in grid.best_params_.items()}, **cleaning_param_grid}

)



cleaned_grid.get_params()["param_grid"]
cleaned_grid.fit(X_train, y_train)
print(cleaned_grid.best_params_)



cleaned_baseline = cleaned_grid.best_estimator_
benchmark(y_valid, cleaned_baseline.predict(X_valid))
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import tensorflow as tf

from transformers import RobertaTokenizer, TFRobertaModel

from tensorflow.keras.optimizers import Adam



# The maximum length of words we'll allow for the tweets

MAX_LENGTH = 64



# The bert pre-trained model to use

PRETRAINED = "roberta-base"





def construct_bert(bert_layer, max_length: int = MAX_LENGTH, lr=5e-5, dropout=0.2):

    attention_mask = tf.keras.Input(shape=(max_length,), dtype="int32", name="attention_mask")

    input_ids = tf.keras.Input(shape=(max_length,), dtype="int32", name="input_ids")



    output = bert_layer([input_ids, attention_mask])

    output = output[1]

    output = tf.keras.layers.Dense(32,activation='relu')(output)

    output = tf.keras.layers.Dropout(dropout)(output)

    output = tf.keras.layers.Dense(1,activation='sigmoid')(output)

    

    model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)

    

    model.compile(Adam(lr=lr), loss="binary_crossentropy", metrics=["accuracy"])



    return model





def bert_encode(tokenizer, texts: list, max_length: int = MAX_LENGTH):

    encoded = tokenizer.batch_encode_plus(

        batch_text_or_text_pairs=texts,

        max_length=max_length,

        pad_to_max_length=True,

        return_token_type_ids=False,

        return_attention_mask=True,

        return_tensors="tf"

    )

    

    return [encoded["input_ids"], encoded["attention_mask"]]
import optuna

from optuna.integration import TFKerasPruningCallback



# we run for 5 epochs with early stopping enabled

# most models reach their peak around 2 epochs.

EPOCHS = 5



# trains models with a range of different hyperparameters

# return the end score for the validation accuracy

def objective(trial: optuna.Trial):

    lr = trial.suggest_categorical("lr", [1e-5, 2e-5, 3e-5])

    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)

    clean = trial.suggest_categorical("clean_tweets", [True, False])



    monitor = "val_accuracy"

    tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED, do_lower_case=True)

    

    

    if clean:

        include_hashtags = trial.suggest_categorical("include_hashtags", [True, False])

        include_handles = trial.suggest_categorical("include_handles", [True, False])

        

        tweets = clean_tweets(X_train, include_hashtags=include_hashtags, include_handles=include_handles)

    else:

        tweets = X_train

    

    encoded_tweets = bert_encode(tokenizer, tweets)

    

    model = construct_bert(TFRobertaModel.from_pretrained(PRETRAINED), lr=lr, dropout=dropout_rate)

    

    fit_params = {

        "batch_size": batch_size,

        "epochs": EPOCHS,

        "callbacks": [

            tf.keras.callbacks.EarlyStopping(

                monitor=monitor,

                patience=1,

                restore_best_weights=True

            ),

            TFKerasPruningCallback(trial, monitor)

        ],

        "validation_split": 0.2

    }

    

    history = model.fit(encoded_tweets, y_train.to_numpy(), verbose=2, **fit_params)

    

    # since we are doing early stopping we want to report the best accuracy that

    # was found across the board.

    # Note, if you adjusted this to be a metric to minimize then you should adjust to `min()`

    return max(history.history[monitor])
# study = optuna.create_study(

#     direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)

# )



# study.optimize(objective, n_trials=10, timeout=1800, gc_after_trial=True)



# print(study.best_params)

# print(study.best_value)



# {'lr': 3e-05, 'batch_size': 32, 'dropout_rate': 0.3432504001345468, 'clean_tweets': False}

# 0.825944185256958
# params = study.best_params

params = {'lr': 3e-05, 'batch_size': 32, 'dropout_rate': 0.3432504001345468, 'clean_tweets': False}
if params["clean_tweets"]:

    tweets = clean_tweets(X_train, include_hashtags=params["include_hashtags"], include_handles=params["include_handles"])

else:

    tweets = X_train



tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED, do_lower_case=True)

encoded_tweets = bert_encode(tokenizer, tweets)



model = construct_bert(TFRobertaModel.from_pretrained(PRETRAINED), lr=params["lr"], dropout=params["dropout_rate"])

    

fit_params = {

    "batch_size": params["batch_size"],

    "epochs": 3,

    "callbacks": [

        tf.keras.callbacks.EarlyStopping(

            monitor="val_accuracy",

            patience=0,

            restore_best_weights=True

        )

    ],

    "validation_split": 0.1

}

    

history = model.fit(encoded_tweets, y_train.to_numpy(), **fit_params)
if params["clean_tweets"]:

    tweets_valid = clean_tweets(X_valid, include_hashtags=params["include_hashtags"], include_handles=params["include_handles"])

else:

    tweets_valid = X_valid

    

encoded_tweets_valid = bert_encode(tokenizer, tweets_valid)



preds = model.predict(encoded_tweets_valid)
benchmark(y_valid, preds > 0.5)
model = construct_bert(TFRobertaModel.from_pretrained(PRETRAINED), lr=params["lr"], dropout=params["dropout_rate"])



if params["clean_tweets"]:

    tweets_all = clean_tweets(X, include_hashtags=params["include_hashtags"], include_handles=params["include_handles"])

else:

    tweets_all = X

    

encoded_tweets_all = bert_encode(tokenizer, tweets_all)





history = model.fit(encoded_tweets_all, y.to_numpy(), **fit_params)
if params["clean_tweets"]:

    tweets_test = clean_tweets(test["text"], include_hashtags=params["include_hashtags"], include_handles=params["include_handles"])

else:

    tweets_test = test["text"]

    

encoded_tweets_test = bert_encode(tokenizer, tweets_test)
target = model.predict(encoded_tweets_test)
submission = pd.DataFrame({"target": target.reshape(-1).round().astype(int)}, index=test.index)



submission.head()
submission.to_csv("submission.csv")