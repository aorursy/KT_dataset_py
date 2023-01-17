!pip install nltk

!pip install spacy

!pip install cmudict

!pip install lexical-diversity



import nltk

nltk.download('punkt') #extra nltk resources we'll need to install
txt = ['''This news is about monkeys. It is about the smallest monkeys in the world. The smallest monkeys are only 100 grams heavy. There aren’t many of these monkeys in the wild. They are in danger because people are destroying forests.

A zoo in Sydney wants to help the monkeys. The zoo puts together two monkeys. It lets them have a family.

Two baby monkeys are born. They are beautiful and they are very, very small. They are just 15 grams heavy. They are smaller than your thumb.

'''.capitalize(), '''The smallest monkey in the world weighs only 100 grams. That is as much as an apple. The species is endangered because people are destroying the forests where this type of monkey lives.

A zoo in Sydney is trying to help the species survive. Zoo keepers put together two monkeys to start a monkey family. A year later, two lovely monkeys were born. They weigh just 15 grams and are smaller than a human thumb!

The monkeys are really cute. Check out their video!

'''.capitalize(),  '''Two tiny and absolutely adorable baby pygmy marmosets were born at Sydney’s Symbio Wildlife Park.

They weighed just 15 grammes at birth (and were the size of human thumbs) but giving birth to them was, from the mother’s perspective, comparable to a human giving birth to a ten-year-old child.

Their proud parents Gomez and IT were introduced to each other the previous year, when IT, the female, arrived at the zoo. Years of bachelorhood ended for Gomez and the two started a family. Apart from their baby monkeys being very cute, their birth is necessary for the ongoing survival of the endangered species.

They are members of the world’s smallest monkey species, with adults weighing in at around the same weight as an average apple, just 100 grammes. They are facing extinction due to deforestation and illegal pet trade.

'''.capitalize()]
import nltk



def sentence_count(txt):

  return len(nltk.sent_tokenize(txt))



for i in range(len(txt)):

    print("Text "+str(i)+" sentence count: "+str(sentence_count(txt[i])))
import re



def tokens_no_nums(txt):

  txt = re.sub('\d', '', txt)

  tokens = nltk.word_tokenize(txt)

  words = [word for word in tokens if word.isalpha()]

  return words



def token_count(txt):

  return len(tokens_no_nums(txt))



for i in range(len(txt)):

    print("Text "+str(i)+" token count: "+str(token_count(txt[i])))
from collections import Counter



def type_count(txt):

  counter = Counter(tokens_no_nums(txt))

  return len(counter.keys())



for i in range(len(txt)):

    print("Text "+str(i)+" type count: "+str(type_count(txt[i])))
def avg_sentence_length(txt):

  return token_count(txt)/sentence_count(txt)



for i in range(len(txt)):

    print("Text "+str(i)+" average sentence length: "+str(avg_sentence_length(txt[i])))
def type_token_ratio(txt):

  return type_count(txt)/token_count(txt)



for i in range(len(txt)):

    print("Text "+str(i)+" type/token ratio: "+str(type_token_ratio(txt[i])))
# Natural Language Toolkit: Tokenizers

#

# Copyright (C) 2001-2020 NLTK Project

# Author: Christopher Hench <chris.l.hench@gmail.com>

#         Alex Estes

# URL: <http://nltk.sourceforge.net>

# For license information, see LICENSE.TXT



"""

The Sonority Sequencing Principle (SSP) is a language agnostic algorithm proposed

by Otto Jesperson in 1904. The sonorous quality of a phoneme is judged by the

openness of the lips. Syllable breaks occur before troughs in sonority. For more

on the SSP see Selkirk (1984).



The default implementation uses the English alphabet, but the `sonority_hiearchy`

can be modified to IPA or any other alphabet for the use-case. The SSP is a

universal syllabification algorithm, but that does not mean it performs equally

across languages. Bartlett et al. (2009) is a good benchmark for English accuracy

if utilizing IPA (pg. 311).



Importantly, if a custom hiearchy is supplied and vowels span across more than

one level, they should be given separately to the `vowels` class attribute.



References:

- Otto Jespersen. 1904. Lehrbuch der Phonetik.

  Leipzig, Teubner. Chapter 13, Silbe, pp. 185-203.

- Elisabeth Selkirk. 1984. On the major class features and syllable theory.

  In Aronoff & Oehrle (eds.) Language Sound Structure: Studies in Phonology.

  Cambridge, MIT Press. pp. 107-136.

- Susan Bartlett, et al. 2009. On the Syllabification of Phonemes.

  In HLT-NAACL. pp. 308-316.

"""



import warnings



import re

from string import punctuation



from nltk.tokenize.api import TokenizerI

from nltk.util import ngrams





class SyllableTokenizer(TokenizerI):

    """

    Syllabifies words based on the Sonority Sequencing Principle (SSP).



        >>> from nltk.tokenize import SyllableTokenizer

        >>> from nltk import word_tokenize

        >>> SSP = SyllableTokenizer()

        >>> SSP.tokenize('justification')

        ['jus', 'ti', 'fi', 'ca', 'tion']

        >>> text = "This is a foobar-like sentence."

        >>> [SSP.tokenize(token) for token in word_tokenize(text)]

        [['This'], ['is'], ['a'], ['foo', 'bar', '-', 'li', 'ke'], ['sen', 'ten', 'ce'], ['.']]

    """



    def __init__(self, lang="en", sonority_hierarchy=False):

        """

        :param lang: Language parameter, default is English, 'en'

        :type lang: str

        :param sonority_hierarchy: Sonority hierarchy according to the

                                   Sonority Sequencing Principle.

        :type sonority_hierarchy: list(str)

        """

        # Sonority hierarchy should be provided in descending order.

        # If vowels are spread across multiple levels, they should be

        # passed assigned self.vowels var together, otherwise should be

        # placed in first index of hierarchy.

        if not sonority_hierarchy and lang == "en":

            sonority_hierarchy = [

                "aeiouy",  # vowels.

                "lmnrw",  # nasals.

                "zvsf",  # fricatives.

                "bcdgtkpqxhj",  # stops.

            ]



        self.vowels = sonority_hierarchy[0]

        self.phoneme_map = {}

        for i, level in enumerate(sonority_hierarchy):

            for c in level:

                sonority_level = len(sonority_hierarchy) - i

                self.phoneme_map[c] = sonority_level

                self.phoneme_map[c.upper()] = sonority_level



    def assign_values(self, token):

        """

        Assigns each phoneme its value from the sonority hierarchy.

        Note: Sentence/text has to be tokenized first.



        :param token: Single word or token

        :type token: str

        :return: List of tuples, first element is character/phoneme and

                 second is the soronity value.

        :rtype: list(tuple(str, int))

        """

        syllables_values = []

        for c in token:

            try:

                syllables_values.append((c, self.phoneme_map[c]))

            except KeyError:

                if c not in punctuation:

                    warnings.warn(

                        "Character not defined in sonority_hierarchy,"

                        " assigning as vowel: '{}'".format(c)

                    )

                    syllables_values.append((c, max(self.phoneme_map.values())))

                    self.vowels += c

                else:  # If it's a punctuation, assing -1.

                    syllables_values.append((c, -1))

        return syllables_values





    def validate_syllables(self, syllable_list):

        """

        Ensures each syllable has at least one vowel.

        If the following syllable doesn't have vowel, add it to the current one.



        :param syllable_list: Single word or token broken up into syllables.

        :type syllable_list: list(str)

        :return: Single word or token broken up into syllables

                 (with added syllables if necessary)

        :rtype: list(str)

        """

        valid_syllables = []

        front = ""

        for i, syllable in enumerate(syllable_list):

            if syllable in punctuation:

                valid_syllables.append(syllable)

                continue

            if not re.search("|".join(self.vowels), syllable):

                if len(valid_syllables) == 0:

                    front += syllable

                else:

                    valid_syllables = valid_syllables[:-1] + [

                        valid_syllables[-1] + syllable

                    ]

            else:

                if len(valid_syllables) == 0:

                    valid_syllables.append(front + syllable)

                else:

                    valid_syllables.append(syllable)



        return valid_syllables





    def tokenize(self, token):

        """

        Apply the SSP to return a list of syllables.

        Note: Sentence/text has to be tokenized first.



        :param token: Single word or token

        :type token: str

        :return syllable_list: Single word or token broken up into syllables.

        :rtype: list(str)

        """

        # assign values from hierarchy

        syllables_values = self.assign_values(token)



        # if only one vowel return word

        if sum(token.count(x) for x in self.vowels) <= 1:

            return [token]



        syllable_list = []

        syllable = syllables_values[0][0]  # start syllable with first phoneme

        for trigram in ngrams(syllables_values, n=3):

            phonemes, values = zip(*trigram)

            # Sonority of previous, focal and following phoneme

            prev_value, focal_value, next_value = values

            # Focal phoneme.

            focal_phoneme = phonemes[1]



            # These cases trigger syllable break.

            if focal_value == -1:  # If it's a punctuation, just break.

                syllable_list.append(syllable)

                syllable_list.append(focal_phoneme)

                syllable = ""

            elif prev_value >= focal_value == next_value:

                syllable += focal_phoneme

                syllable_list.append(syllable)

                syllable = ""



            elif prev_value > focal_value < next_value:

                syllable_list.append(syllable)

                syllable = ""

                syllable += focal_phoneme



            # no syllable break

            else:

                syllable += focal_phoneme



        syllable += syllables_values[-1][0]  # append last phoneme

        syllable_list.append(syllable)



        return self.validate_syllables(syllable_list)
import cmudict

d = cmudict.dict()



def nsyl(word):

    try:

        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]

    except:

        st = SyllableTokenizer()

        return len(st.tokenize(word))



def syl_count(txt):

  tokens = tokens_no_nums(txt)

  syl_tokens = [nsyl(t) for t in tokens]

  return sum(syl_tokens)



for i in range(len(txt)):

    print("Text "+str(i)+" syllable count: "+str(syl_count(txt[i])))
def more_2_syl(txt):

  count = 0

  tokens = tokens_no_nums(txt)

  syl_tokens = [nsyl(t) for t in tokens]

  for s in syl_tokens:

    if s > 2:

      count += 1

  return count



for i in range(len(txt)):

    print("Text "+str(i)+" difficult word count: "+str(more_2_syl(txt[i])))
def per_more_2_syl(txt):

  return 100*more_2_syl(txt)/token_count(txt)



for i in range(len(txt)):

    print("Text "+str(i)+" difficult word percentage: "+str(per_more_2_syl(txt[i])))
def avg_syl_sentence(txt):

  return syl_count(txt)/sentence_count(txt)



for i in range(len(txt)):

    print("Text "+str(i)+" average syllables per sentence: "+str(avg_syl_sentence(txt[i])))
def avg_syl_word(txt):

  return syl_count(txt)/token_count(txt)



for i in range(len(txt)):

    print("Text "+str(i)+" average syllables per word: "+str(avg_syl_word(txt[i])))
def summary(txt):

  print('- sentence count: '+str(sentence_count(txt)))

  print('- token count: '+str(token_count(txt)))

  print('- type count: '+str(type_count(txt)))

  print('- average sentence length: '+str(avg_sentence_length(txt)))

  print('- type/token ratio: '+str(type_token_ratio(txt)))

  print('- syllable count: '+str(syl_count(txt)))

  print('- words more than 2 syllables: '+str(more_2_syl(txt)))

  print('- percentage of words more than 2 syllables: '+str(per_more_2_syl(txt)))

  print('- average syllables sentence: '+str(avg_syl_sentence(txt)))

  print('- average syllables word: '+str(avg_syl_word(txt)))



for i in range(len(txt)):

    print("TEXT "+str(i))

    print(summary(txt[i]))

    print()
def flesch_reading_ease(txt):

  return 206.835 - 1.015 * (token_count(txt)/sentence_count(txt)) - 84.6 * (syl_count(txt)/token_count(txt))



for i in range(len(txt)):

    print("Text "+str(i)+" Flesch Reading Ease: "+str(flesch_reading_ease(txt[i])))
def flesch_kincaid_grade(txt):

  return 0.39 * (token_count(txt)/sentence_count(txt)) + 11.8 * (syl_count(txt)/token_count(txt)) - 15.59



for i in range(len(txt)):

    print("Text "+str(i)+" Flesch Kincaid Grade: "+str(flesch_kincaid_grade(txt[i])))
def gunning_fog_index(txt):

  return 0.4 * ((token_count(txt)/sentence_count(txt) + 100 * (more_2_syl(txt)/token_count(txt))))



for i in range(len(txt)):

    print("Text "+str(i)+" Gunning Fog Index: "+str(gunning_fog_index(txt[i])))
def readability_scores(txt):

  print('- flesch reading ease: '+str(flesch_reading_ease(txt)))

  print('- flesch kincaid grade: '+str(flesch_kincaid_grade(txt)))

  print('- gunning fog index: '+str(gunning_fog_index(txt)))



for i in range(len(txt)):

    print('Text '+str(i))

    readability_scores(txt[i])

    print()
import spacy

from lexical_diversity import lex_div as ld



def mtld(txt):

  nlp = spacy.load('en')

  doc = nlp(u""+txt)

  txt = ""

  for token in doc:

    txt += (" " + token.lemma_)

  txt = tokens_no_nums(txt)

  return ld.mtld_ma_wrap(txt)



for i in range(len(txt)):

    print("Text "+str(i)+" MLTD: "+str(mtld(txt[i])))
def hdd(txt):

  nlp = spacy.load('en')

  doc = nlp(u""+txt)

  txt = ""

  for token in doc:

    txt += (" " + token.lemma_)

  txt = tokens_no_nums(txt)

  return ld.hdd(txt)*100



for i in range(len(txt)):

    print("Text "+str(i)+" HDD: "+str(hdd(txt[i])))
def lex_diversity(txt):

  print('- HDD: '+str(vocD(txt)))

  print('- MTLD: '+str(mtld(txt)))



for i in range(len(txt)):

    print('Text '+str(i))

    lex_diversity(txt[i])

    print()