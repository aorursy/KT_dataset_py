# setup code checking

from learntools.core import binder

binder.bind(globals())

from learntools.nlp.ex2 import *

print("Setup is completed.")
# check your answer (run this code cell to receive credit!)

step_1.solution()
import pandas as pd

def load_data(csv_file, split=0.9):

    data = pd.read_csv(csv_file)

    

    # shuffle data, sampling with frac < 1, upsampling with frac > 1

    train_data = data.sample(frac=1, random_state=7)

    

    texts = train_data["text"].values

    labels = [

        {"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in train_data["sentiment"].values

    ]

    

    split = int(len(train_data) * split)

    train_labels = [{"cats": labels} for labels in labels[:split]]

    val_labels = [{"cats": labels} for labels in labels[split:]]

    

    return texts[:split], train_labels, texts[split:], val_labels



train_texts, train_labels, val_texts, val_labels = load_data('../input/nlp-course/yelp_ratings.csv')
print('Texts from training data\n', '-'*10)

print(train_texts[:2])

print('\n')

print('Labels from training data\n', '-'*10)

print(train_labels[:2])
# create an empty model

import spacy

nlp = spacy.blank("en")



# create the TextCategorizer with exclusive classes and Bag of Words (bow) architecture

textcat = nlp.create_pipe(

    "textcat",

    config={

        "exclusive_classes": True,

        "architecture": "bow"

    }

)



# add the TextCategorizer to the empty model

nlp.add_pipe(textcat)



# add labels to text classifier

textcat.add_label("NEGATIVE")

textcat.add_label("POSITIVE")



# check your answer

step_2.check()
# lines below will give you a hint or solution code

# step_2.hint()

# step_2.solution()
import random

from spacy.util import minibatch



nlp.begin_training()



def train(model, train_data, optimizer, batch_size=8):

    losses = {}

    random.seed(1)

    random.shuffle(train_data)

    

    # create the batch generator

    batches = minibatch(train_data, size=batch_size)

    for batch in batches:

        # split batch into texts and labels

        texts, labels = zip(*batch)

        

        # update model with texts and labels

        nlp.update(texts, labels, sgd=optimizer, losses=losses)

        

    return losses



# check your answer

step_3.check()
# lines below will give you a hint or solution code

# step_3.hint()

# step_3.solution()
# fix seed for reproducibility

spacy.util.fix_random_seed(1)

random.seed(1)



optimizer = nlp.begin_training()

train_data = list(zip(train_texts, train_labels))

losses = train(nlp, train_data, optimizer)

print(losses['textcat'])
text = "This tea cup was full of holes. Do not recommend."

doc = nlp(text)

print(doc.cats)
def predict(model, texts): 

    # Use the model's tokenizer to tokenize each input text

    docs = [model.tokenizer(text) for text in texts]

    

    # use textcat to get the scores for each doc

    textcat = model.get_pipe('textcat')

    scores, _ = textcat.predict(docs)

    

    # from the scores, find the class with the highest score/probability

    predicted_class = scores.argmax(axis=1)

    

    return predicted_class



# check your answer

step_4.check()
# lines below will give you a hint or solution code

# step_4.hint()

# step_4.solution()
texts = val_texts[34:38]

predictions = predict(nlp, texts)



for p, t in zip(predictions, texts):

    print(f"{textcat.labels[p]}: {t} \n")
predict(nlp, texts)
def evaluate(model, texts, labels):

    """

    Returns the accuracy of a TextCategorizer model. 

    

    Arguments

    ---------

    model: ScaPy model with a TextCategorizer

    texts: Text samples, from load_data function

    labels: True labels, from load_data function    

    """

    

    # get predictions from textcat model (using your predict method)

    predicted_class = predict(model, texts)

    

    # from labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)

    true_class = [int(label['cats']['POSITIVE']) for label in labels]

    

    # a boolean or int array indicating correct predictions

    correct_predictions = (predicted_class == true_class)

    

    # the accuracy, number of correct predictions divided by all predictions

    accuracy = correct_predictions.mean()

    

    return accuracy



# check your answer

step_5.check()
# lines below will give you a hint or solution code

# step_5.hint()

# step_5.solution()
accuracy = evaluate(nlp, val_texts, val_labels)

print(f"Accuracy: {accuracy:.4f}")
n_iters = 5

for i in range(n_iters):

    losses = train(nlp, train_data, optimizer)

    accuracy = evaluate(nlp, val_texts, val_labels)

    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")
# check your answer (run this code cell to receive credit!)

step_6.solution()