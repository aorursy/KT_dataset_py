import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

train1 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')



valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

#valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv')



#test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv')



#subm = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
test
import spacy 

spacy.prefer_gpu() # or spacy.require_gpu()



nlp = spacy.load("en_core_web_sm")



# Create an empty model

#nlp = spacy.blank("en")



# Create the TextCategorizer with exclusive classes and "bow" architecture

textcat = nlp.create_pipe(

              "textcat",

              config={

                "exclusive_classes": True,

                "architecture": "bow"})



# Add the TextCategorizer to the empty model

nlp.add_pipe(textcat)

# Add labels to text classifier

textcat.add_label("toxic")

textcat.add_label("neutral")

#train=train[:1000]
train_texts = train['comment_text'].values

train_labels = [{'cats': {'toxic': label == 1,

                          'neutral': label == 0}} 

                for label in train['toxic']]
train_data = list(zip(train_texts, train_labels))

train_data[:3]
from spacy.util import minibatch



spacy.util.fix_random_seed(1)

optimizer = nlp.begin_training(n_threads=4)



# Create the batch generator with batch size = 8

batches = minibatch(train_data, size=12)

# Iterate through minibatches

for batch in batches:

    # Each batch is a list of (text, label) but we need to

    # send separate lists for texts and labels to update().

    # This is a quick way to split a list of tuples into lists

    texts, labels = zip(*batch)

    nlp.update(texts, labels, sgd=optimizer)
import random



random.seed(1)

spacy.util.fix_random_seed(1)

optimizer = nlp.begin_training(n_threads=4)



losses = {}

for epoch in range(3):

    random.shuffle(train_data)

    # Create the batch generator with batch size = 8

    batches = minibatch(train_data, size=500)

    # Iterate through minibatches

    for batch in batches:

        # Each batch is a list of (text, label) but we need to

        # send separate lists for texts and labels to update().

        # This is a quick way to split a list of tuples into lists

        texts, labels = zip(*batch)

        nlp.update(texts, labels, sgd=optimizer, losses=losses)

    print(losses)
texts = ["Are you ready for the tea party????? It's gonna be wild",

         "URGENT Reply to this message for GUARANTEED FREE TEA" ]

docs = [nlp.tokenizer(text) for text in texts]

    

# Use textcat to get the scores for each doc

textcat = nlp.get_pipe('textcat')

scores, _ = textcat.predict(docs)



print(scores)

docs = [nlp.tokenizer(text) for text in test.translated]

    

# Use textcat to get the scores for each doc

textcat = nlp.get_pipe('textcat')

scores, _ = textcat.predict(docs)



print(scores)

scores.shape,test.shape,scores[:,0].shape
submit=pd.DataFrame( test.id )

submit['toxic']=pd.DataFrame(scores[:,0].reshape(-1,1))

submit.to_csv('submission.csv',index=False)

submit