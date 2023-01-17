!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@nlp
import sys

sys.path.append('/kaggle/working')
import pandas as pd



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.nlp.ex2 import *

print("\nSetup complete")
import spacy



# Create an empty model

nlp = ____



# Create the TextCategorizer with exclusive classes and "bow" architecture

textcat = ____



# Add NEGATIVE and POSITIVE labels to text classifier

____



q_1.check()
# Uncomment if you need some guidance

# q_1.hint()

# q_1.solution()
#%%RM_IF(PROD)%%



import spacy



# Create an empty model

nlp = spacy.blank("en")



# Create the TextCategorizer with exclusive classes and "bow" architecture

textcat = nlp.create_pipe(

            "textcat",

            config={

                "exclusive_classes": True,

                "architecture": "bow"})

nlp.add_pipe(textcat)



# Add NEGATIVE and POSITIVE labels to text classifier

textcat.add_label("NEGATIVE")

textcat.add_label("POSITIVE")



q_1.assert_check_passed()
def load_data(csv_file, split=0.8):

    data = pd.read_csv(csv_file)

    

    # Shuffle data

    train_data = data.sample(frac=1, random_state=7)

    

    texts = train_data.text.values

    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)}

              for y in train_data.sentiment.values]

    split = int(len(train_data) * split)

    

    train_labels = [{"cats": labels} for labels in labels[:split]]

    val_labels = [{"cats": labels} for labels in labels[split:]]

    

    return texts[:split], train_labels, texts[split:], val_labels
train_texts, train_labels, val_texts, val_labels = load_data('../input/nlp-course/yelp_ratings.csv')
from spacy.util import minibatch

import random



def train(model, train_data, optimizer):

    losses = {}

    random.seed(1)

    # Shuffle the training data

    ____

    

    # Create batches with batch size = 8

    batches = "____"

    for batch in batches:

        # train_data is a list of tuples [(text0, label0), (text1, label1), ...]

        # Split batch into texts and labels

        ____

        

        # Update model with texts and labels

        ____

        

    return losses



q_2.check()
# Uncomment if you need some guidance

# q_2.hint()

# q_2.solution()
#%%RM_IF(PROD)%%



from spacy.util import minibatch

import random



def train(model, train_data, optimizer, batch_size=8):

    losses = {}

    #random.seed(1)

    random.shuffle(train_data)

    batches = minibatch(train_data, size=batch_size)

    for batch in batches:

        texts, labels = zip(*batch)

        model.update(texts, labels, sgd=optimizer, losses=losses)

    return losses



q_2.assert_check_passed()
# Fix seed for reproducibility

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

    docs = ____

    

    # Use textcat to get the scores for each doc

    ____

    

    # From the scores, find the class with the highest score/probability

    predicted_class = ____

    

    return predicted_class



q_3.check()
# Uncomment if you need some guidance

# q_3.hint()

# q_3.solution()
#%%RM_IF(PROD)%%



def predict(model, texts): 

    # Use the tokenizer to tokenize each input text example

    docs = [model.tokenizer(text) for text in texts]

    

    # Use textcat to get the scores for each doc

    textcat = model.get_pipe('textcat')

    scores, _ = textcat.predict(docs)

    

    # From the scores, find the class with the highest score/probability

    predicted_class = scores.argmax(axis=1)

    

    return predicted_class



q_3.assert_check_passed()
predictions = predict(nlp, val_texts[23:27])

texts = val_texts[23:27]



for p, t in zip(predictions, texts):

    print(f"{textcat.labels[p]}: {t} \n")
def evaluate(model, texts, labels):

    """ Returns the accuracy of a TextCategorizer model. 

    

        Arguments

        ---------

        model: ScaPy model with a TextCategorizer

        texts: Text samples, from load_data function

        labels: True labels, from load_data function

    

    """

    # Get predictions from textcat model (using your predict method)

    predicted_class = ____

    

    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)

    true_class = ____

    

    # A boolean or int array indicating correct predictions

    correct_predictions = ____

    

    # The accuracy, number of correct predictions divided by all predictions

    accuracy = ____

    

    return accuracy



q_4.check()
# Uncomment if you need some guidance

# q_4.hint()

# q_4.solution()
#%%RM_IF(PROD)%%



def evaluate(model, texts, labels):

    """ Returns the accuracy of a TextCategorizer model. 

    

        Arguments

        ---------

        model: ScaPy model with a TextCategorizer

        texts: Text samples, from load_data function

        labels: True labels, from load_data function

    

    """

    # Get predictions from textcat model

    predicted_class = predict(model, texts)

    

    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)

    true_class = [int(each['cats']['POSITIVE']) for each in labels]

    

    # A boolean or int array indicating correct predictions

    correct_predictions = predicted_class == true_class

    

    # The accuracy, number of correct predictions divided by all predictions

    accuracy = correct_predictions.mean()

    

    return accuracy



q_4.assert_check_passed()
accuracy = evaluate(nlp, val_texts, val_labels)

print(f"Accuracy: {accuracy:.4f}")
n_iters = 5

for i in range(n_iters):

    losses = train(nlp, train_data, optimizer)

    accuracy = evaluate(nlp, val_texts, val_labels)

    print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")
#q_5.solution()