import  requests, re

from html.parser import HTMLParser

import urllib3

class GoogleSpellCheckCommand(object):

    

    def __init__(self):

        "Construtor"

        self.manager = urllib3.PoolManager()



    def correct(self, text):

        "Correct input text by using Google Spell Corrector"

        

        # grab html

        html = self.get_page('http://www.google.com/search?hl=en&q=' + requests.utils.quote(text) + "&meta=&gws_rd=ssl")

        html_parser = HTMLParser()



        # save html for debugging

        # open('page.html', 'w').write(html)



        # pull pieces out

        match = re.search(r'(?:Showing results for|Did you mean|Including results for)[^\0]*?<a.*?>(.*?)</a>', html)

        if match is None:

            fix = text

        else:

            fix = match.group(1)

            fix = re.sub(r'<.*?>', '', fix)

            fix = html_parser.unescape(fix)

        # return result

        return fix



    def get_page(self, url):

        # the type of header affects the type of response google returns

        # for example, using the commented out header below google does not 

        # include "Including results for" results and gives back a different set of results

        # than using the updated user_agent yanked from chrome's headers

        # user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

        user_agent = 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.116 Safari/537.36'

        headers = {'User-Agent':user_agent,}



        req =  self.manager.request('GET', url, headers)

        

        return str(req.data)
import re

import numpy as np

from collections import Counter

import enchant

from enchant.checker import SpellChecker



def words(text): return re.findall(r'\w+', text.lower())



WORDS =  Counter(words(open('big.txt').read()))



def P(word, N=sum(WORDS.values())):

    "Probability of `word`."

    return (WORDS[word])* np.exp(len(word)) / (N *20 )



def correction(word):

    "Most probable spelling correction for word."

    conditates = candidates(word)

    return max(conditates, key=P)



def candidates(word):

    "Generate possible spelling corrections for word."

    return (known([word])  or

            known(google_checker(word)) or

            known(edits0(word)) or

            known(edits0_1(word)) or

            known(edits0_2(word))  or

            known(edits0_3(word)) or

            known(edits1(word)) or

            known(edits2(word))  or

            known(enchant_checker(word)) or

            [word])



def known(words):

    "The subset of `words` that appear in the dictionary of WORDS."

    return set(w for w in words if w in WORDS)



def google_checker(word):

    google_corrector = GoogleSpellCheckCommand()

    

    return [google_corrector.correct(word)]



def enchant_checker(word):

    d1 = enchant.Dict('en_US')



    return set(d1.suggest(word))



def edits0_3(word):

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]



    return set(deletes)



def edits0_2(word):

    letters     = 'abcdefghijklmnopqrstuvwxyz'

    splits      = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    transposes0 = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    transposes1 = [L + R[0] + R[2:] for L, R in splits if len(R)>1]

    transposes2 = [L + R[1] + R[2:] for L, R in splits if len(R)>1]



    return set(transposes0 + transposes1 + transposes2 )



def edits0_1(word):

    volums = ['a','e','i','o','u','y']

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    replaces   = [L + c + R[1:]          for L, R in splits if R for c in volums]

    return set(replaces )





def edits0(word):

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(inserts)







def edits1(word):

    "All edits that are one edit away from `word`."

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes0    = [L + R[1:]               for L, R in splits if R]



    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]



    inserts    = [L + c + R               for L, R in splits for c in letters]



    return set(deletes0   + transposes+ replaces + inserts )



def edits2(word):

    "All edits that are two edits away from `word`."

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))