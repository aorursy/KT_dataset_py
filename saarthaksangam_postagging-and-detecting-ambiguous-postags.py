from nltk.corpus import brown

brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')

tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)

tag_fd.most_common()
def findtags(tag_prefix, tagged_text):

    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text

                                  if tag.startswith(tag_prefix))

    return dict((tag, cfd[tag].most_common(5)) for tag in cfd.conditions())



tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))

for tag in sorted(tagdict):

     print(tag, tagdict[tag])
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')

data = nltk.ConditionalFreqDist((word.lower(), tag)

for (word, tag) in brown_news_tagged)

for word in sorted(data.conditions()):

    if len(data[word]) > 3:

        tags = [tag for (tag, _) in data[word].most_common()]

        print(word, ' '.join(tags))