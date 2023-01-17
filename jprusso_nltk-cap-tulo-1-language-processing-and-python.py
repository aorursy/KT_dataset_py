import nltk

from nltk.book import *
texts()
text1.concordance("monstrous")
text1.similar("monstrous")
text1.common_contexts(["monstrous", "doleful"])
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
len(text3)
sorted(set(text3))
print("Cantidad de palabras:",len(set(text3)))
print("Cantidad de vocabulario diferente:",(len(set(text3)) / len(text3))*100,"%")
print("Porcentaje de uso de la palabra a:",100 * text4.count('a') / len(text4), "%")
def lexical_diversity(text):

    return len(set(text)) / len(text)



def percentage(count, total):

    return 100 * count / total
sent1 = ['Call', 'me', 'Ishmael', '.']

lexical_diversity(sent1)
fdist1 = FreqDist(text1)

print(fdist1)

fdist1.most_common(10)
fdist1.plot(10, cumulative=True)
fdist1.hapaxes()
V = set(text1)

long_words = [w for w in V if len(w) > 15]

sorted(long_words)
list(nltk.bigrams(['more', 'is', 'said', 'than', 'done']))
text4.collocations()
fdist = FreqDist(len(w) for w in text1)

fdist

print("Frecuencias más comunes:", fdist.most_common())

print("Frecuencia más grande:", fdist.max(), "con una frecuencia de", fdist[fdist.max()], "siendo un porcentaje de", fdist.freq(fdist.max()))
