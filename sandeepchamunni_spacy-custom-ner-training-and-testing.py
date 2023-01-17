import spacy

import random

import spacy.cli





#SPECIFY THE NER TRAINING DATA

TRAIN_DATA = [

        ("I have deposited an amount of $500 using my debit card.",{"entities":[(7,16,"action"),(30,34,"amount")]}),

        ("Send $500 to the merchant with account number 1234567890. ",{"entities":[(0,4,"action"),(5,9,"amount")]}),

        ("Transfer $20000 to my new bank account ending with the number 4567. ",{"entities":[(0,8,"action"),(9,15,"amount")]}),

        ("Please deposit $2000 in my account. ",{"entities":[(7,14,"action"),(15,20,"amount")]}),

        ("I would like to withdraw $10000 from my bank account. ",{"entities":[(16,24,"action"),(25,31,"amount")]})]



nlp = spacy.blank('en')

ner = nlp.create_pipe("ner")



nlp.add_pipe(ner, last=True)



#ADD THE CUSTOM NAMED ENTITIES HERE

nlp.entity.add_label('action')

nlp.entity.add_label('amount')





nlp.vocab.vectors.name = 'spacy_pretrained_vectors'

optimizer = nlp.begin_training()

for i in range(20):

    random.shuffle(TRAIN_DATA)

    for text, annotations in TRAIN_DATA:

        nlp.update([text], [annotations], sgd=optimizer)

#SAVE THE CUSTOM NER MODEL TO

nlp.to_disk("custom_ner_model")

print("Model saved")
#LOAD THE CUSTOM MODEL

nlp2 = spacy.load("custom_ner_model")

doc2 = nlp2("I have withdrawn an amount of $300 with my credit card.")

for ent in doc2.ents:

    print(ent.label_, ent.text)