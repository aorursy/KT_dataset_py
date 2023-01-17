import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import wordpunct_tokenize

import pandas as pd

from wordcloud import WordCloud as wc

import matplotlib.pyplot as plt

import os

violations = pd.read_csv('../input/violations.csv')

violations.head()
viol_desc = violations.violation_description

viol_str = viol_desc.str.cat(sep = ' ')
stop = set(stopwords.words('english'))

list_of_words = [i.lower() for i in wordpunct_tokenize(viol_str) if i.lower() not in stop and i.isalpha()]

list_of_words[:15]
wordfreqdist = nltk.FreqDist(list_of_words)

mostcommon = wordfreqdist.most_common(30)

print(mostcommon)
plt.barh(range(len(mostcommon)),[val[1] for val in mostcommon], align='center')

plt.yticks(range(len(mostcommon)), [val[0] for val in mostcommon])

plt.show()
wc1 = wc().generate(' '.join(list_of_words))

 

plt.imshow(wc1)

plt.axis("off")

plt.show()
inspections = pd.read_csv('../input/inspections_train.csv')
inspections.head()
combinedDF = pd.merge(violations,inspections[['camis','passed']], right_on = 'camis', left_on = 'camis')
combinedDF.head()
passedDF = combinedDF[combinedDF['passed'] == 1]

failedDF = combinedDF[combinedDF['passed'] == 0]

viol_desc_passed = passedDF.violation_description

viol_desc_failed = failedDF.violation_description
viol_str_passed = viol_desc_passed.str.cat(sep = ' ')

viol_str_failed = viol_desc_failed.str.cat(sep = ' ')





list_of_words_passed = [i.lower() for i in wordpunct_tokenize(viol_str_passed) if i.lower() not in stop and i.isalpha()]

list_of_words_failed = [i.lower() for i in wordpunct_tokenize(viol_str_failed) if i.lower() not in stop and i.isalpha()]

wordfreqdistpassed = nltk.FreqDist(list_of_words_passed)

mostcommonpassed = wordfreqdistpassed.most_common(30)

print(mostcommonpassed)
plt.barh(range(len(mostcommonpassed)),[val[1] for val in mostcommonpassed], align='center')

plt.yticks(range(len(mostcommonpassed)), [val[0] for val in mostcommonpassed])

plt.show()
wordfreqdistfailed = nltk.FreqDist(list_of_words_failed)

mostcommonfailed = wordfreqdistfailed.most_common(30)

print(mostcommonfailed)
plt.barh(range(len(mostcommonfailed)),[val[1] for val in mostcommonfailed], align='center')

plt.yticks(range(len(mostcommonfailed)), [val[0] for val in mostcommonfailed])

plt.show()
wcpassed = wc().generate(' '.join(list_of_words_passed))

 

plt.imshow(wcpassed)

plt.axis("off")

plt.show()
wcfailed = wc().generate(' '.join(list_of_words_failed))

 

plt.imshow(wcfailed)

plt.axis("off")

plt.show()

 
failedwordlen = len(list_of_words_failed)

worddictfailed = dict(wordfreqdistfailed)

worddictfailednormalized = {k: float(v) / failedwordlen for k, v in worddictfailed.items()}

worddictfailednormalized

passedwordlen = len(list_of_words_passed)

worddictpassed = dict(wordfreqdistpassed)

worddictpassednormalized = {k: float(v) / passedwordlen for k, v in worddictpassed.items()}

worddictpassednormalized


worddictrelative = {k: worddictfailednormalized[k] - worddictpassednormalized[k] 

                    for k in worddictfailednormalized if k in worddictpassednormalized}



worddictrelative
wcrel = wc().generate_from_frequencies(worddictrelative)



plt.imshow(wcrel)

plt.axis("off")

plt.show()

 