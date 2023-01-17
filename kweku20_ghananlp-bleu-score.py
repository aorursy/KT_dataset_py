import nltk



generated = 'Mepawokyɛw, boa yɛn na yɛ nyɛ yɛn adwuma.'

actual = 'Yɛsrɛ wo boa yɛn ma yɛnyɛ yɛn adwuma no.'



generated = generated.split()

print(generated)

actual = actual.split()

print(actual)



#there may be several references

BLEUscore = nltk.translate.bleu_score.sentence_bleu([actual], generated)

print(BLEUscore)