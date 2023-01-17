# Under settings (right side pane), make sure Internet is set to 'on'.

!pip install textatistic #you will have to install new packages each time you run this kernel in Kaggle
from textatistic import Textatistic



text = """It was the best of times, it was the worst of times, 

it was the age of wisdom, it was the age of foolishness, 

it was the epoch of belief, it was the epoch of incredulity, 

it was the season of Light, it was the season of Darkness, 

it was the spring of hope, it was the winter of despair, 

we had everything before us, we had nothing before us, 

we were all going direct to Heaven, we were all going direct the other way – 

in short, the period was so far like the present period, 

that some of its noisiest authorities insisted on its being received, 

for good or for evil, in the superlative degree of comparison only."""



# Create a Textatistic object

s = Textatistic(text)
s.counts, s.scores, #s.dict() returns both
!pip install textstat
import textstat



text = (

    "Playing games has always been thought to be important to "

    "the development of well-balanced and creative children; "

    "however, what part, if any, they should play in the lives "

    "of adults has never been researched that deeply. I believe "

    "that playing games is every bit as important for adults "

    "as for children. Not only is taking time out to play games "

    "with our children and other adults valuable to building "

    "interpersonal relationships but is also a wonderful way "

    "to release built up tension."

)
#Returns the number of syllables present in the given text

textstat.syllable_count(text, lang='en_US') #lang='en_GB' for British English
# Calculates the number of words present in the text. 

# Optional removepunct specifies whether we need to take punctuation symbols into account while counting lexicons. Default value is True, which removes the punctuation before counting lexicon items.

textstat.lexicon_count(text, removepunct=True)
textstat.sentence_count(text)
textstat.flesch_reading_ease(text)
textstat.flesch_kincaid_grade(text)
# The Fog Scale (Gunning FOG Formula)

textstat.gunning_fog(text)
textstat.dale_chall_readability_score(text)
textstat.difficult_words(text), textstat.linsear_write_formula(text), textstat.text_standard(text)
import numpy as np 

import pandas as pd 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

merged1 = pd.read_csv("../input/mergedcsv/merged1.csv")