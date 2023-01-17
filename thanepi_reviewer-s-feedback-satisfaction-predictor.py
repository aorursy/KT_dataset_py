import numpy as np

import pandas as pd

import spacy

from spacy.util import minibatch

import random
def measure(measurement, dataframe, targetcolumn):

    if 'measured_rate' in dataframe:

        overiding = 1

    else:

        overiding = 0

    dataframe['measured_rate'] = dataframe.apply (lambda row: measurement(row, targetcolumn), axis=1)

    if overiding == 1:

        result = "Measured complete and did overided"

    elif overiding == 0:

        result = "Measured complete"

    if -1 in dataframe['measured_rate'].unique():

        result = "Some or all record of this dataframe cannot be measure with this measurement, or user may make an incorrect call"

        del dataframe['measured_rate']

    return print(result)
def satisfaction(row, targetcolumn):

    if (row[targetcolumn] > 0) and (row[targetcolumn] < 4):

        return 0

    if (row[targetcolumn] > 3) and (row[targetcolumn] <= 5):

        return 1

    return -1
# Modified from https://www.kaggle.com/matleonard/text-classification exercise

def split_satisfaction_dataframe(dataframe, textcolumn, ratecolumn = "measured_rate", split = 0.7):

    #By default: ratecolumn will aim to "measured_rate" column in case of renamed from measure() process -> user must input new name, also as split rate by default was 70% : 30% | Train : Val

    data = dataframe

    

    # Shuffle data

    train_data = data.sample(frac=1, random_state=7) #frac=1 is return full amount of input size

    

    texts = train_data[textcolumn].values

    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} #Class name UPPERCASE sensitivity.

              for y in train_data[ratecolumn].values]

    split = int(len(train_data) * split)

    

    train_labels = [{"cats": labels} for labels in labels[:split]] #"cats" is fixed for further use, do not change.

    val_labels = [{"cats": labels} for labels in labels[split:]]

    

    return texts[:split], train_labels, texts[split:], val_labels
def train(model, train_data, optimizer):

    losses = {}

    random.seed(1)

    random.shuffle(train_data)

    

    # Learn more about batch size at: https://arxiv.org/abs/1711.00489

    batches = minibatch(train_data, size=16)

    

    for batch in batches:

        # train_data is a list of tuples [(text0, label0), (text1, label1), ...]

        # Split batch into texts and labels

        texts, labels = zip(*batch)

        

        # Update model with texts and labels

        model.update(texts, labels, sgd=optimizer, losses=losses)

        

    return losses
def predict(model, texts): 

    # Use the model's tokenizer to tokenize each input text

    docs = [model.tokenizer(text) for text in texts]

    

    # Use textcat to get the scores for each doc

    textcat = model.get_pipe('textcat')

    scores, _ = textcat.predict(docs)

    

    # From the scores, find the class with the highest score/probability

    predicted_class = scores.argmax(axis=1)

    

    return predicted_class
def evaluate(model, texts, labels):

    # Returns the accuracy of a TextCategorizer model. 

    # Get predictions from textcat model (using your predict method)

    predicted_class = predict(model, texts)

    

    # From labels, get the true class as a list of integers (POSITIVE -> 1, NEGATIVE -> 0)

    true_class = [int(each['cats']['POSITIVE']) for each in labels]

    

    # A boolean or int array indicating correct predictions

    correct_predictions = predicted_class == true_class

    

    # The accuracy, number of correct predictions divided by all predictions

    accuracy = correct_predictions.mean()

    

    return accuracy
def benchmark(model, optimizer, train_data, val_texts, val_labels, n_iters=10):

    sum_losses = 0

    sum_accuracy = 0

    for i in range(n_iters):

        losses = train(model, train_data, optimizer)

        sum_losses = sum_losses + losses['textcat']

        accuracy = evaluate(model, val_texts, val_labels)

        sum_accuracy = sum_accuracy + accuracy

        print(f"Loss: {losses['textcat']:.3f} \t Accuracy: {accuracy:.3f}")

    avg_losses = sum_losses / n_iters

    avg_accuracy = sum_accuracy / n_iters

    return print(f"---Average---\nLoss: {avg_losses:.3f} \t Accuracy: {avg_accuracy:.3f}")
#debuging tool - inspect to data record about its text and labeling

def inspect_txtlbl(train_texts, train_labels):

    print('Texts from training data\n------')

    print(train_texts[:2])

    print('\nLabels from training data\n------')

    print(train_labels[:2])
#Test Tool - Display label probability

#text = "If you are not use to using a large sustaining pedal while playing the piano, it may appear little awkward."

def lbl_prob(text, model):

    doc = model(text)

    return print(doc.cats)
#test tool - Do predict from validation set.

def val_predict(val_texts, model):

    texts = val_texts[10:15]

    predictions = predict(satisfaction_nlp, texts)

    for p, t in zip(predictions, texts):

        print(f"{textcat.labels[p]}: {t} \n")
#test tool - Demonstration

def demo(test_text, model):

    #test_text = "The guitar wouldn’t stay in tune for longer than 2 minutes. I tuned it string by string as soon as it arrived and then went back to the first string to make sure everything was fine but they were all out of tune again as if they were slowly slipping flat. Dont buy."

    # From real review that's not exist in dataset: https://www.amazon.com/gp/customer-reviews/RHQYAO44BFGOH/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B07NVWXTQQ

    test_data = {'text': test_text}

    test_df = pd.DataFrame(data=[test_data])

    test_subject = test_df[0:1].text.values

    text_predictions = predict(model, test_subject)

    for p, t in zip(text_predictions, test_subject):

        print(f"{textcat.labels[p]}: {t}")
#Uncomment below if - Loading from Kaggle notebook editor

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/amazon-music-reviews/Musical_instruments_reviews.csv")



#Uncomment and change directory if - Loading from your environment

#df = pd.read_csv("input/Musical_instruments_reviews.csv")
df.head()
df.info()
df.isna().sum()
df = df.fillna("")
t1df = df[['reviewText','overall']]
t1df.head()
measure(satisfaction, t1df, "overall")
t1df.head()
#buging tool - Adding noise

bugdf = pd.DataFrame([["Text0", 0], ["Text1", 0]], columns=['reviewText','overall'])

t1df = t1df.append(bugdf, ignore_index=True)

print("Noise Added")
t1df.tail()
#debuging tool - Removing noise

bugdf = pd.DataFrame([["Text0", 0], ["Text1", 0]], columns=['reviewText','overall'])

#Modified from original's one, Thank you code from https://datascience.stackexchange.com/posts/37229/revisions

cond = t1df['reviewText'].isin(bugdf['reviewText'])

t1df.drop(t1df[cond].index, inplace = True)

print("Clear! Its noise free now.")
t1df.tail()
train_texts, train_labels, val_texts, val_labels = split_satisfaction_dataframe(t1df, "reviewText")
inspect_txtlbl(train_texts, train_labels)
# Create an empty model

satisfaction_nlp = spacy.blank("en")



# Create the TextCategorizer with exclusive classes and "bow" architecture

textcat = satisfaction_nlp.create_pipe(

              "textcat",

              config={

                "exclusive_classes": True,

                "architecture": "bow"})



# Add the TextCategorizer to the empty model

satisfaction_nlp.add_pipe(textcat)



# Add labels to text classifier

textcat.add_label("NEGATIVE") #Class name UPPERCASE sensitivity.

textcat.add_label("POSITIVE") #Class name UPPERCASE sensitivity.
# Fix seed for reproducibility

spacy.util.fix_random_seed(1)

random.seed(1)
# This may take a while to run!

optimizer = satisfaction_nlp.begin_training()

train_data = list(zip(train_texts, train_labels))

losses = train(satisfaction_nlp, train_data, optimizer)

print(losses['textcat'])
text = "If you are not use to using a large sustaining pedal while playing the piano, it may appear little awkward."

lbl_prob(text, satisfaction_nlp)
val_predict(val_texts, satisfaction_nlp)
accuracy = evaluate(satisfaction_nlp, val_texts, val_labels)

print(f"Accuracy: {accuracy:.4f}")
benchmark(satisfaction_nlp, optimizer, train_data, val_texts, val_labels, n_iters=10)
input_text = "The guitar wouldn’t stay in tune for longer than 2 minutes. I tuned it string by string as soon as it arrived and then went back to the first string to make sure everything was fine but they were all out of tune again as if they were slowly slipping flat. Dont buy."

#This is real negative review that's not exist in dataset: https://www.amazon.com/gp/customer-reviews/RHQYAO44BFGOH/ref=cm_cr_arp_d_rvw_ttl?ie=UTF8&ASIN=B07NVWXTQQ



test_text = input_text

demo(test_text, satisfaction_nlp)
input_text = "*Edit your lovely review here!*"



test_text = input_text

demo(test_text, satisfaction_nlp)