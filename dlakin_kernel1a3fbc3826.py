!pip install spacymoji
import matplotlib.pyplot as plt

import lightgbm as lgb

import seaborn as sns

import pandas as pd

import numpy as np

import requests

import warnings

import tarfile

import random

import spacy

import json

import shap

import re



from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, average_precision_score

from sklearn.preprocessing import RobustScaler, OneHotEncoder

from emoji import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from requests.auth import HTTPBasicAuth

from sklearn.pipeline import Pipeline

from tarfile import ExFileObject

from textblob import TextBlob

from itertools import product

from spacymoji import Emoji

from pprint import pprint



%matplotlib inline

shap.initjs()



TWEET_COLUMNS = ["id", "combined_text", "language", "num_mentions", "num_hashtags", "gender"]

TWITTER_ROOT_URL = "https://api.twitter.com"

TWITTER_CONSUMER_KEY = "3RZxLkkQFDMnN3epDPOcP61hP"

TWITTER_CONSUMER_SECRET = "cwfoe7umOC4tAIJE4VmirEjmQE2NPWlnfCr91jS0EQUiOB4cNk"
def parse_julia_file(tarfile: ExFileObject):

    tar_string = tarfile.read().decode()

    return [json.loads(el) for el in tar_string.split("\n") if el != ""]





tar = tarfile.open("/kaggle/input/mso_ds_interview.tgz", "r")

manifest = tar.extractfile("ds_interview/manifest.jl")

manifest_details = parse_julia_file(manifest)

pprint(manifest_details)
tweets_data_list = []



# Authenticate to Twitter API

auth = HTTPBasicAuth("3RZxLkkQFDMnN3epDPOcP61hP", "cwfoe7umOC4tAIJE4VmirEjmQE2NPWlnfCr91jS0EQUiOB4cNk")

auth_response = requests.post("https://api.twitter.com/oauth2/token?grant_type=client_credentials", auth=auth).json()

headers = {"Authorization": f"Bearer {auth_response['access_token']}"}





def parse_tweet_text(tweet_obj: dict):

    try:

        text = tweet_obj["full_text"]

        if text == "":

            text = tweet_obj["text"]



    except KeyError:

        text = tweet_obj["text"]



    return text





with requests.session() as session:

    session.headers = headers



    for user in manifest_details:

        user_id = user["user_id_str"]

        print(f"Pulling tweets for user {user_id}")



        api_result = session.get(f"https://api.twitter.com/1.1/users/lookup.json?user_id={user_id}").json()

        if "errors" in api_result:

            profile_description = ""



        else:

            profile_description = api_result[0]["description"]



        tweet_file = tar.extractfile(f"ds_interview/tweet_files/{user_id}.tweets.jl")

        user_tweets = parse_julia_file(tweet_file)



        for t in user_tweets:

            tdoc = t["document"]

            tweet_id = tdoc["id"]

            combined_text = parse_tweet_text(tdoc) + " " + profile_description

            language = tdoc["lang"]

            num_mentions = len(tdoc["entities"]["user_mentions"])

            num_hashtags = len(tdoc["entities"]["hashtags"])



            tweets_data_list.append([

                tweet_id,

                combined_text,

                language,

                num_mentions,

                num_hashtags,

                user["gender_human"]

            ])
tweets_df = pd.DataFrame(tweets_data_list, columns=TWEET_COLUMNS)

tweets_df.set_index("id", inplace=True)

tweets_df.head(20)
language_counts = tweets_df.groupby("language").count()["combined_text"]

language_counts
SPACY_LANGS = ["de", "el", "en", "es", "fr", "it", "lt", "nb", "nl", "pt"]



# Throw out samples that spaCy can't parse

tweets_df = tweets_df[tweets_df["language"].isin(SPACY_LANGS)]



language_counts = tweets_df.groupby("language").count()["combined_text"]

language_counts
tweets_df = tweets_df.sample(frac=0.15, random_state=42)

tweets_df.shape
CONTRACTIONS = ["ain't", "aren't", "can't", "can't've", "'cause", "could've", "couldn't", "couldn't've", "didn't",

                "doesn't", "don't", "hadn't", "hadn't've", "hasn't", "haven't", "he'd", "he'd've", "he'll", "he'll've",

                "he's", "how'd", "how'd'y", "how'll", "how's", "I'd", "I'd've", "I'll", "I'll've", "I'm", "I've", "i'd",

                "i'd've", "i'll", "i'll've", "i'm", "i've", "isn't", "it'd", "it'd've", "it'll", "it'll've", "it's",

                "let's", "ma'am", "mayn't", "might've", "mightn't", "mightn't've", "must've", "mustn't", "mustn't've",

                "needn't", "needn't've", "o'clock", "oughtn't", "oughtn't've", "shan't", "sha'n't", "shan't've",

                "she'd", "she'd've", "she'll", "she'll've", "she's", "should've", "shouldn't", "shouldn't've", "so've",

                "so's", "that'd", "that'd've", "that's", "there'd", "there'd've", "there's", "they'd", "they'd've",

                "they'll", "they'll've", "they're", "they've", "to've", "wasn't", "we'd", "we'd've", "we'll",

                "we'll've", "we're", "we've", "weren't", "what'll", "what'll've", "what're", "what's", "what've",

                "when's", "when've", "where'd", "where's", "where've", "who'll", "who'll've", "who's", "who've",

                "why's", "why've", "will've", "won't", "won't've", "would've", "wouldn't", "wouldn't've", "y'all",

                "y'all'd", "y'all'd've", "y'all're", "y'all've", "you'd", "you'd've", "you'll", "you'll've", "you're",

                "you've"]

POS_MAP = {

    "ADJ": "num_adjectives",

    "ADV": "num_adverbs",

    "CONJ": "num_conjunctions",

    "NOUN": "num_nouns",

    "NUM": "num_numerals",

    "PART": "num_particles",

    "PRON": "num_pronouns",

    "PROPN": "num_proper_nouns",

    "PUNCT": "num_punctuation_mks",

    "VERB": "num_verbs"

}





def extract_linguistic_features(texts, tweet_ids, spacy_nlp):

    all_features = []



    for i, doc in enumerate(spacy_nlp.pipe(texts, disable=["tagger", "parser", "ner"], n_threads=16, batch_size=10000)):

        features = {

            "id": tweet_ids[i],

            "num_words": len(doc),

            "tweet_length": len(doc.text),

            "num_exclamation_pts": doc.text.count("!"),

            "num_question_mks": doc.text.count("?"),

            "num_periods": doc.text.count("."),

            "num_hyphens": doc.text.count("-"),

            "num_capitals": sum(1 for char in doc.text if char.isupper()),

            "num_emoticons": sum(

                1 for token in doc

                if token._.is_emoji

                or token in UNICODE_EMOJI

                or token in UNICODE_EMOJI_ALIAS

            ),

            "num_unique_words": len(set(token.text for token in doc)),

            "num_adjectives": 0,

            "num_nouns": 0,

            "num_pronouns": 0,

            "num_adverbs": 0,

            "num_conjunctions": 0,

            "num_numerals": 0,

            "num_particles": 0,

            "num_proper_nouns": 0,

            "num_verbs": 0,

            "num_contractions": 0,

            "num_punctuation_mks": 0

        }



        for token in doc:

            if token.text in CONTRACTIONS:

                features["num_contractions"] += 1



            if token.pos_ in POS_MAP:

                column_key = POS_MAP[token.pos_]

                features[column_key] += 1



        clean_tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+://\S+)", " ", doc.text).split())

        features["sentiment"] = parse_sentiment(clean_tweet)



        all_features.append(features)



    return all_features





def parse_sentiment(tweet):

    """

    Utility function to classify sentiment of passed tweet

    using textblob's sentiment method

    """



    # Create TextBlob object of tweet text

    parsed = TextBlob(tweet)

    # set sentiment

    if parsed.sentiment.polarity > 0:

        return 1

    elif parsed.sentiment.polarity == 0:

        return 0

    else:

        return -1





split_by_lang = [{"lang": lang, "df": tweets_df[tweets_df["language"] == lang]} for lang in SPACY_LANGS]

for item in split_by_lang:

    nlp = spacy.load(item["lang"])

    emoji = Emoji(nlp, merge_spans=False)

    nlp.add_pipe(emoji, first=True)



    tweet_ids = item["df"].index.tolist()

    texts = item["df"]["combined_text"].tolist()

    if len(texts):

        spacy_features = extract_linguistic_features(texts, tweet_ids, nlp)

        temp_df = pd.DataFrame.from_records(spacy_features)

        temp_df.set_index("id", inplace=True)



        item["df"] = item["df"].merge(temp_df, how="left", on="id")
X = pd.concat([lang_item["df"] for lang_item in split_by_lang], sort=False)

X.head()
y = np.array(X["gender"].map({"M": 0, "F": 1}))



X.drop(columns=["gender", "language", "combined_text"], inplace=True)

column_labels = pd.get_dummies(X, columns=["sentiment"]).columns.tolist()



# Scale numeric features

numeric_features = ["num_mentions", "num_hashtags", "num_nouns", "num_pronouns", "num_adjectives", "num_particles",

                    "num_words", "tweet_length", "num_exclamation_pts", "num_question_mks", "num_periods", "num_verbs",

                    "num_hyphens", "num_capitals", "num_emoticons", "num_unique_words", "num_conjunctions",

                    "num_numerals", "num_contractions", "num_adverbs", "num_proper_nouns", "num_punctuation_mks"]

numeric_transformer = Pipeline(steps=[

    ("robust", RobustScaler())

])



categorical_features = ["sentiment"]

categorical_transformer = Pipeline(steps=[

    ("onehot", OneHotEncoder(categories="auto"))

])



preprocessor = ColumnTransformer(

    transformers=[

        ("num", numeric_transformer, numeric_features),

        ("cat", categorical_transformer, categorical_features)

    ]

)

X = preprocessor.fit_transform(X)



# Split into train and testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
def eval_lgb_results(hyperparams, iteration):

    """

    Scoring helper for grid and random search. Returns CV score from the given

    hyperparameters

    """



    # Find optimal n_estimators using early stopping

    if "n_estimators" in hyperparams.keys():

        del hyperparams["n_estimators"]



    # Perform n_folds CV

    cv_res = lgb.cv(

        params=hyperparams,

        train_set=train_set,

        num_boost_round=1000,

        nfold=5,

        early_stopping_rounds=50,

        metrics=["auc", "accuracy"],

        seed=42

    )



    # Return CV results

    score = cv_res["auc-mean"][-1]

    estimators = len(cv_res["auc-mean"])

    hyperparams["n_estimators"] = estimators



    return [score, hyperparams, iteration]





def light_random_search(param_grid, max_evals=5):

    # Dataframe to store results

    results = pd.DataFrame(columns=["score", "params", "iteration"], index=list(range(max_evals)))



    # Select max_eval combinations of params to check

    for i in range(max_evals):

        iter_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

        results.loc[i, :] = eval_lgb_results(iter_params, i)



    # Sort by best score

    results.sort_values("score", ascending=False, inplace=True)

    results.reset_index(inplace=True)

    return results
train_set = lgb.Dataset(data=X_train, label=y_train)



param_grid = {

    'boosting_type': ['gbdt'],

    'num_leaves': list(range(20, 150)),

    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),

    'subsample_for_bin': list(range(20000, 300000, 20000)),

    'min_child_samples': list(range(20, 500, 5)),

    'reg_alpha': list(np.linspace(0, 1)),

    'reg_lambda': list(np.linspace(0, 1)),

    'colsample_bytree': list(np.linspace(0.6, 1, 10)),

    'subsample': list(np.linspace(0.5, 1, 100)),

    'is_unbalance': [False],

    "first_metric_only": [True]

}



random_results = light_random_search(param_grid, max_evals=15)

print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))

print('\nThe best hyperparameters were:')



pprint(random_results.loc[0, 'params'])
# Get the best params from the random search

rsearch_params = random_results.loc[0, "params"]



# Create, train, and test model with the derived params

model = lgb.LGBMClassifier(**rsearch_params, random_state=42)

model.fit(X_train, y_train)



preds = model.predict(X_test)

roc_auc = roc_auc_score(y_test, preds)

pr_score = average_precision_score(y_test, preds)



print("The best model from random search scores {:.5f} ROC-AUC on the test set and has an average precision of {:.5f}"

      .format(roc_auc, pr_score))
# Plot feature importances

importances = model.feature_importances_

tuples = sorted(zip(column_labels, importances), key=lambda x: x[1])



# Strip out features with zero importance

tuples = [x for x in tuples if x[1] > 0]

feature_names, values = zip(*tuples)



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

ax1.set_title("Random Search Feature Importances")

ax1.barh(np.arange(len(values)), values, align="center")

ax1.set_yticks(np.arange(len(values)))

ax1.set_yticklabels(feature_names)

ax1.set_xlim(0, max(values) * 1.1)

ax1.set_ylim(-1, len(values))

ax1.set_xlabel("Feature Importance")

ax1.set_ylabel("Features")



# Plot Confusion Matrix

cm = confusion_matrix(y_test, preds)

labels = ['Male', 'Female']

sns.heatmap(cm, xticklabels=labels, yticklabels=labels, annot=True, fmt='d', cmap="Blues", vmin=0.2)

ax2.set_title("Confusion Matrix")

ax2.set_ylabel("Ground Truth")

ax2.set_xlabel("Predictions")



plt.tight_layout()

plt.show()
high_probability_preds - X_test[preds >= 0.98]



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    explainer = shap.TreeExplainer(model)

    expected_value = explainer.expected_value

    shap_values = explainer.shap_values(T)[1]
shap.summary_plot(shap_values, features, feature_names=column_labels, class_names=["Male", "Female"])
shap.decision_plot(expected_value, shap_values, T, feature_order="hclust", link="logit", feature_names=column_labels)
def light_grid_search(param_grid):

    results = pd.DataFrame(columns=["score", "params", "iteration"])



    # Get every possible combination of params from the grid

    keys, grid_vals = zip(*param_grid.items())



    # Iterate over every possible combination of hyperparameters

    for i, v in enumerate(product(*grid_vals)):

        iter_params = dict(zip(keys, v))

        results.loc[i, :] = eval_lgb_results(iter_params, i)



    # Sort by best score

    results.sort_values("score", ascending=False, inplace=True)

    results.reset_index(inplace=True)

    return results