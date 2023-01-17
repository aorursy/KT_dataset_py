#Code to wordwrap result cells

from IPython.display import HTML, display

def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)
body = '''Eight people are dead following two shootings at shisha bars in the western German city of Hanau. At least five people were injured after gunmen opened fire at about 22:00 local time (21:00 GMT), police told the BBC. Police added that they are searching for the suspects, who fled the scene and are currently at large. The first shooting was at a bar in the city centre, while the second was in Hanau's Kesselstadt neighbourhood, according to local reports. Police officers and helicopters are patrolling both areas. An unknown number of gunmen killed three people at the first shisha bar, Midnight, before driving to the Arena Bar & Cafe and shooting dead another five victims, regional broadcaster Hessenschau reports. A dark-coloured vehicle was then seen leaving the scene.The motive for the attack is unclear, a police statement said. Can-Luca Frisenna, who works at a kiosk at the scene of one of the shootings said his father and brother were in the area when the attack took place. It's like being in a film, it's like a bad joke, that someone is playing a joke on us, he told Reuters.I can't grasp yet everything that has happened. My colleagues, all my colleagues, they are like my family - they can't understand it either. Hanau, in the state of Hessen, is about 25km (15 miles) east of Frankfurt. It comes four days after another shooting in Berlin, near a Turkish comedy show at the Tempodrom concert venue, which killed one person.'''
from gensim.summarization.summarizer import summarize
print(summarize(body, word_count=50, split=False))
print(summarize(body, word_count=100, split=True))
from gensim.summarization.summarizer import summarize_corpus
from gensim.summarization.textcleaner import split_sentences
from gensim import corpora
# Creates a corpus where each document is a sentence.
sentences = split_sentences(body)
tokens = [sentence.split() for sentence in sentences]
print(tokens)
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(sentence_tokens) for sentence_tokens in tokens]
print(corpus)
# Extracts the most important documents (shown here in BoW representation)
summary = summarize_corpus(corpus, ratio = 0.4)
print(summary)