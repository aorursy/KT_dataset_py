from __future__ import unicode_literals, print_function

import pandas as pd

import os

import re

import string

from nltk.corpus import stopwords



import random

from pathlib import Path

import spacy

from spacy.util import minibatch, compounding

import datetime

from tqdm import tqdm



# spacy.prefer_gpu()





LABEL = "SELECTEDTEXT"





def read_data(datadir):

    train = pd.read_csv(os.path.join(datadir, "train.csv"))

    test = pd.read_csv(os.path.join(datadir, "test.csv"))

    sample_submission = pd.read_csv(os.path.join(datadir, "sample_submission.csv"))



    return (train, test, sample_submission)





def jaccard_similarity(string1, string2):

    wordset = lambda x: set(x.lower().split())

    a, b = wordset(string1), wordset(string2)



    c = a.intersection(b)



    return float(len(c)) / (len(a) + len(b) - len(c))





def clean_text(text):

    """Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.

    Source: https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model

    """

    text = str(text).lower()

    text = re.sub("\[.*?\]", "", text) # remove words in square brackets

    text = re.sub("https?://\S+|www\.\S+", "", text)

    text = re.sub("<.*?>+", "", text)

    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)

    text = re.sub("\n", "", text)

    text = re.sub("\w*\d\w*", "", text)

    return text





def remove_stopwords(words):

    return [word for word in words if word not in stopwords.words("english")]





def training_data(dataframe):

    data = (

        dataframe.dropna()

        .assign(

            text=lambda x: x.apply(lambda x: x.text.lower(), axis=1),

            selected_text=lambda x: x.apply(lambda x: x.selected_text.lower(), axis=1),

        )

        .assign(

            start=lambda x: x.apply(lambda x: x.text.find(x.selected_text), axis=1),

            end=lambda x: x.apply(lambda x: x.start + len(x.selected_text), axis=1),

        )

    )



    positive = []

    negative = []



    for i, row in data.iterrows():

        if row.end > row.start:

            train_row = (row.text, {"entities": [(row.start, row.end, LABEL)]})



            if row.sentiment == "positive":

                positive.append(train_row)

            elif row.sentiment == "negative":

                negative.append(train_row)

            else:

                pass



    print(f"Positive data size: {len(positive):,}")

    print(f"Negative data size: {len(negative):,}")



    return positive, negative



def run_model(data, positivemodel, negativemodel, outputpath=None):

    print("Loading models...")

    positive_nlp = spacy.load(positivemodel)

    negative_nlp = spacy.load(negativemodel)



    data = data.dropna()



    textIDs, selected_texts = [], []

    jaccards = {

        "positive": [],

        "negative": [],

        "neutral": []

    }



    input_train = "selected_text" in data.columns



    print("Iterating data...")

    for _, row in tqdm(data.iterrows()):

        if row.sentiment == "positive":

            if len(row.text.split()) <= 2:

                selected_text = row.text

            else: 

                ents = positive_nlp(row.text).ents

                selected_text = ents[0].text if len(ents) > 0 else row.text

        elif row.sentiment == "negative":

            if len(row.text.split()) <= 2:

                selected_text = row.text

            else:

                ents = negative_nlp(row.text).ents

                selected_text = ents[0].text if len(ents) > 0 else row.text

        else:

            selected_text = row.text



        textIDs += [row.textID]

        selected_texts += [selected_text]



        if input_train:

            jaccard = jaccard_similarity(row.text, selected_text)

            jaccards[row.sentiment] += [jaccard]



    output_df = pd.DataFrame({"textID": textIDs, "selected_text": selected_texts})

    

    if not input_train:

        if not outputpath:

            suffix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

            outputpath = os.path.join("submissions", "submission_" + suffix + ".csv")

        

        print("Saving submission...")

        output_df.to_csv(outputpath, index=False)



    if input_train:

        nums, dens = [], []

        for key in ("positive", "negative", "neutral"):

            num = sum(jaccards[key])

            den = len(jaccards[key])

            jaccard_score = num / den

            print(f"Jaccard score for {key}: {num / den:.3f}")



            nums.append(num)

            dens.append(den)



        print(f"Jaccard score for overall: {sum(nums) / sum(dens):.3f}")

        



def train_model(traindata, new_model_name, model=None, output_dir=None, n_iter=30):

    """Set up the pipeline and entity recognizer, and train the new entity."""

    random.seed(0)

    if model is not None:

        nlp = spacy.load(model)  # load existing spaCy model

        print("Loaded model '%s'" % model)

    else:

        nlp = spacy.blank("en")  # create blank Language class

        print("Created blank 'en' model")

    # Add entity recognizer to model if it's not in the pipeline

    # nlp.create_pipe works for built-ins that are registered with spaCy

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner)

    # otherwise, get it, so we can add labels to it

    else:

        ner = nlp.get_pipe("ner")



    ner.add_label(LABEL)  # add new entity label to entity recognizer

    # Adding extraneous labels shouldn't mess anything up

    # ner.add_label("VEGETABLE")

    if model is None:

        optimizer = nlp.begin_training()

    else:

        optimizer = nlp.resume_training()

    move_names = list(ner.move_names)

    # get names of other pipes to disable them during training

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*other_pipes):  # only train NER

        sizes = compounding(1.0, 4.0, 1.001)

        # batch up the examples using spaCy's minibatch

        for itn in range(n_iter):

            random.shuffle(traindata)

            batches = minibatch(traindata, size=sizes)

            losses = {}

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)

            print("Losses", losses)



    # test the trained model

    test_text = "i`d have responded, if i were going"

    doc = nlp(test_text)

    print("Entities in '%s'" % test_text)

    for ent in doc.ents:

        print(ent.label_, ent.text)



    # save model to output directory

    if output_dir is not None:

        output_dir = Path(output_dir)

        if not output_dir.exists():

            output_dir.mkdir()

        nlp.to_disk(output_dir)

        print("Saved model to", output_dir)



        # test the saved model

        print("Loading from", output_dir)

        nlp2 = spacy.load(output_dir)

        # Check the classes have loaded back consistently

        assert nlp2.get_pipe("ner").move_names == move_names

        doc2 = nlp2(test_text)

        for ent in doc2.ents:

            print(ent.label_, ent.text)
spacy.prefer_gpu()



n_iter = 20



modelsdir = "models"

datadir = "/kaggle/input/tweet-sentiment-extraction"



# preparing training dataset 

train, test, _ = read_data(datadir)

positive, negative = training_data(train)



# training positive model

# os.mkdir(modelsdir)

# print("Training positive model...")

# train_model(positive, "positive", output_dir=os.path.join(modelsdir, "positive"), n_iter=n_iter)



# #training negative model

# print("Training negative model...")

# train_model(negative, "negative", output_dir=os.path.join(modelsdir, "negative"), n_iter=n_iter)
# running model on test dataset



# using uploaded model files trained with 100 iterations

modelsdir = "/kaggle/input/tweetsentimentmodel/models"



run_model(test, 

        os.path.join(modelsdir, "positive"),

        os.path.join(modelsdir, "negative"),

        outputpath="submission.csv"

        )