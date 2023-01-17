from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import requests
from bs4 import BeautifulSoup
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict


def clean(text: str):
    """
    Cleans the passed in string by removing redundant new line and whitespace characters, then calling .strip() on the
    string
    :param text: A string to be cleaned
    :return: Returns the cleaned string
    """
    text_list = list(text)
    i = 0
    while i < len(text_list) - 1:
        if (text_list[i] == '\n' and text_list[i + 1] == '\n') or (text_list[i] == ' ' and text_list[i + 1] == ' '):
            text_list.pop(i)
            i -= 1
        i += 1
    return ''.join(text_list).strip()


def descendant_tags(soup: BeautifulSoup, parent_tag: str, descendant_tag: str):
    """
    Returns a list containing a parent tag's descendant tags
    :param soup: A BeautifulSoup object
    :param parent_tag: The parent tag type
    :param descendant_tag: The descendant tag type to search for
    :return: Returns a list containing all instances of the descendant tag found in the parent tag
    """
    descendants = []
    elements = soup.find_all(parent_tag)
    for element in elements:
        for child_element in element.findChildren():
            if child_element.name == descendant_tag:
                descendants.append(child_element)
    return descendants


def tags_to_text(tags):
    """
    Returns a string representing the contents of the passed in tag list separated by spaces
    :param tags: A list of tags
    :return: Returns the concatenated string
    """
    text = ''
    for tag in tags:
        text += ' ' + tag.text
    return text


def print_sentences(sentences):
    """
    Prints the passed in sentences.
    :param sentences: list
    :return:
    """
    for sentence in sentences:
        print(sentence)


def summarize(url: str, display_data: bool = False):
    """
    Summarizes an article using BeautifulSoup. Takes a URL string for the article, checks if the page has an article
    tag, then prints the summarized text
    :param url: The webpage URL
    :param display_data: Print internal data for review and debug purposes
    :return:
    """
    response = requests.get(url)
    response.encoding = 'utf-8'
    data = response.text
    soup = BeautifulSoup(data, features="html.parser")

    # text = ' '.join(map(lambda p: p.text, soup.find_all('article')))
    text = tags_to_text(descendant_tags(soup, 'article', 'p'))
    text = clean(text)

    # text.encode('ascii', 'ignore')

    sentences = sent_tokenize(text)
    if display_data:
        print('SENTENCES:', sentences, '\n')

    words = word_tokenize(text.lower())
    # print('WORDS:', words, '\n')

    _stopwords = set(stopwords.words('english') + list(punctuation))
    _stopwords.add("'s")

    if display_data:
        print('_STOPWORDS', _stopwords, '\n')

    # Filter out stopword
    words = [word for word in words if word not in _stopwords]
    if display_data:
        print('WORDS:', words, '\n')

    freq = FreqDist(words)
    if display_data:
        print('FREQ:', freq.__repr__(), '\n')

    if display_data:
        print('NLARGEST:', nlargest(10, freq, key=freq.get), '\n')

    # We want to create a significant score ordered by highest frequency
    ranking = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                ranking[i] += freq[word]

    if display_data:
        print('RANKING:', ranking, '\n')

    # Top 4 Sentences
    sentences_idx = nlargest(4, ranking, key=ranking.get)
    finalized_sentences = [sentences[j] for j in sorted(sentences_idx)]

    if display_data:
        print('SENTENCES_IDX:', finalized_sentences, '\n')

    print_sentences(finalized_sentences)


summarize("https://arstechnica.com/cars/2018/10/honda-will-use-gms-self-driving-technology-invest-2-75-billion/", True)
