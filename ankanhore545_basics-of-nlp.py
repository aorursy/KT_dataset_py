# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import string

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
paragraph= """WHEN UNCLE STAN rose in the morning, he somehow managed
to wake the entire household. No one complained, as he was the
breadwinner in the family, and in any case he was cheaper and
more reliable than an alarm clock.
The first noise Harry would hear was the bedroom door
slamming. This would be followed by his uncle tramping along the
creaky wooden landing, down the stairs and out of the house. Then
another door would slam as he disappeared into the privy. If
anyone was still asleep, the rush of water as Uncle Stan pulled the
chain, followed by two more slammed doors before he returned to
the bedroom, served to remind them that Stan expected his
breakfast to be on the table by the time he walked into the kitchen.
He only had a wash and a shave on Saturday evenings before going
off to the Palais or the Odeon. He took a bath four times a year on

quarter-day. No one was going to accuse Stan of wasting his hard-
earned cash on soap.

Maisie, Harry’s mum, would be next up, leaping out of bed
moments after the first slammed door. There would be a bowl of
porridge on the stove by the time Stan came out of the privy.
Grandma followed shortly afterwards, and would join her daughter

in the kitchen before Stan had taken his place at the head of the
table. Harry had to be down within five minutes of the first
slammed door if he hoped to get any breakfast. The last to arrive in
the kitchen would be Grandpa, who was so deaf he often managed
to sleep through Stan’s early morning ritual. This daily routine in
the Clifton household never varied. When you’ve only got one
outside privy, one sink and one towel, order becomes a necessity.
By the time Harry was splashing his face with a trickle of cold
water, his mother would be serving breakfast in the kitchen: two
thickly sliced pieces of bread covered in lard for Stan, and four thin
slices for the rest of the family, which she would toast if there was
any coal left in the sack dumped outside the front door every
Monday. Once Stan had finished his porridge, Harry would be
allowed to lick the bowl.
A large brown pot of tea was always brewing on the hearth,
which Grandma would pour into a variety of mugs through a
silver-plated Victorian tea strainer she had inherited from her
mother. While the other members of the family enjoyed a mug of
unsweetened tea - sugar was only for high days and holidays - Stan
would open his first bottle of beer, which he usually gulped down
in one draught. He would then rise from the table and burp loudly
before picking up his lunch box, which Grandma had prepared
while he was having his breakfast: two Marmite sandwiches, a
sausage, an apple, two more bottles of beer and a packet of five
coffin nails. Once Stan had left for the docks, everyone began to
talk at once.

Grandma always wanted to know who had visited the tea shop
where her daughter worked as a waitress: what they ate, what they
were wearing, where they sat; details of meals that were cooked on
a stove in a room lit by electric light bulbs that didn’t leave any
candle wax, not to mention customers who sometimes left a
thruppenny-bit tip, which Maisie had to split with the cook.
Maisie was more concerned to find out what Harry had done at
school the previous day. She demanded a daily report, which
didn’t seem to interest Grandma, perhaps because she’d never been
to school. Come to think of it, she’d never been to a tea shop
either.
Grandpa rarely commented, because after four years of loading
and unloading an artillery field gun, morning, noon and night, he
was so deaf he had to satisfy himself with watching their lips move
and nodding from time to time. This could give outsiders the
impression he was stupid, which the rest of the family knew to
their cost he wasn’t.
The family’s morning routine only varied at weekends. On
Saturdays, Harry would follow his uncle out of the kitchen, always
remaining a pace behind him as he walked to the docks. On
Sunday, Harry’s mum would accompany the boy to Holy Nativity
Church, where, from the third row of the pews, she would bask in
the glory of the choir’s treble soloist.
But today was Saturday. During the twenty-minute walk to the
docks, Harry never opened his mouth unless his uncle spoke.

Whenever he did, it invariably turned out to be the same
conversation they’d had the previous Saturday.
‘When are you goin’ to leave school and do a day’s work,
young’un?’ was always Uncle Stan’s opening salvo.
‘Not allowed to leave until I’m fourteen,’ Harry reminded him.
‘It’s the law.’
‘A bloody stupid law, if you ask me. I’d packed up school and
was workin’ on the docks by the time I were twelve,’ Stan would
announce as if Harry had never heard this profound observation
before. Harry didn’t bother to respond, as he already knew what
his uncle’s next sentence would be. ‘And what’s more I’d signed
up to join Kitchener’s army before my seventeenth birthday.’
‘Tell me about the war, Uncle Stan,’ said Harry, aware that this
would keep him occupied for several hundred yards.
‘Me and your dad joined the Royal Gloucestershire Regiment on
the same day,’ Stan said, touching his cloth cap as if saluting a
distant memory. ‘After twelve weeks’ basic training at Taunton
Barracks, we was shipped off to Wipers to fight the Boche. Once
we got there, we spent most of our time cooped up in rat-infested
trenches waiting to be told by some toffee-nosed officer that when
the bugle sounded, we was going over the top, bayonets fixed,
rifles firing as we advanced towards the enemy lines.’ This would
be followed by a long pause, after which Stan would add, ‘I was
one of the lucky ones. Got back to Blighty all ship-shape and
Bristol fashion.’ Harry could have predicted his next sentence

word for word, but remained silent. ‘You just don’t know how
lucky you are, my lad. I lost two brothers, your uncle Ray and
your uncle Bert, and your father not only lost a brother, but his
father, your other grandad, what you never met. A proper man,
who could down a pint of beer faster than any docker I’ve ever
come across.’
If Stan had looked down, he would have seen the boy mouthing
his words, but today, to Harry’s surprise, Uncle Stan added a
sentence he’d never uttered before. ‘And your dad would still be
alive today, if only management had listened to me.’
Harry was suddenly alert. His dad’s death had always been the
subject of whispered conversations and hushed tones. But Uncle
Stan clammed up, as if he realized he’d gone too far. Maybe next
week, thought Harry, catching his uncle up and keeping in step
with him as if they were two soldiers on a parade ground.
‘So who are City playin’ this afternoon?’ asked Stan, back on
script.
‘Charlton Athletic,’ Harry replied.
‘They’re a load of old cobblers.’
‘They trounced us last season,’ Harry reminded his uncle.
‘Bloody lucky, if you ask me,’ said Stan, and didn’t open his
mouth again. When they reached the entrance to the dockyard, Stan
clocked in before heading off to the pen where he was working
with a gang of other dockers, none of whom could afford to be a

minute late. Unemployment was at an all-time high and too many
young men were standing outside the gates waiting to take their
place.
Harry didn’t follow his uncle, because he knew that if Mr
Haskins caught him hanging around the sheds he would get a clip
round the ear, followed by a boot up the backside from his uncle
for annoying the ganger. Instead, he set off in the opposite
direction.
Harry’s first port of call every Saturday morning was Old Jack
Tar, who lived in the railway carriage at the other end of the
dockyard. He had never told Stan about his regular visits because
his uncle had warned him to avoid the old man at all costs.
‘Probably hasn’t had a bath in years,’ said a man who washed
once a quarter, and then only after Harry’s mother complained
about the smell.
But curiosity had long ago got the better of Harry, and one
morning he’d crept up to the railway carriage on his hands and
knees, lifted himself up and peeped through a window. The old
man was sitting in first class, reading a book.
Old Jack turned to face him and said, ‘Come on in, lad.’ Harry
jumped down, and didn’t stop running until he reached his front
door.
The following Saturday, Harry once again crawled up to the
carriage and peered inside. Old Jack seemed to be fast asleep, but

then Harry heard him say, ‘Why don’t you come in, my boy? I’m
not going to bite you.’
Harry turned the heavy brass handle and tentatively pulled open
the carriage door, but he didn’t step inside. He just stared at the
man seated in the centre of the carriage. It was hard to tell how old

he was because his face was covered in a well-groomed salt-and-
pepper beard, which made him look like the sailor on the Players

Please packet. But he looked at Harry with a warmth in his eyes
that Uncle Stan had never managed.
‘Are you Old Jack Tar?’ Harry ventured.
‘That’s what they call me,’ the old man replied.
‘And is this where you live?’ Harry asked, glancing around the
carriage, his eyes settling on a stack of old newspapers piled high
on the opposite seat.
‘Yes,’ he replied. ‘It’s been my home for these past twenty
years. Why don’t you close the door and take a seat, young man?’
Harry gave the offer some thought before he jumped back out of
the carriage and once again ran away.
The following Saturday, Harry did close the door, but he kept
hold of the handle, ready to bolt if the old man as much as twitched
a muscle. They stared at each other for some time before Old Jack
asked, ‘What’s your name?’"""



text= re.sub(r'\[[0-9]*\]', ' ', paragraph)
text=text.lower()
text= re.sub(r'\d+', ' ', text)
text= text.translate(str.maketrans('','', string.punctuation))
text= re.sub(r'\s+', ' ', text)
# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
# Training the Word2Vec model
from gensim.models import Word2Vec
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab
words

# Finding Word Vectors
vector = model.wv['stupid']
#vector

# Most similar words
similar = model.wv.most_similar('door')
#similar
#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
X= cv.fit_transform(corpus).toarray()

#Visting the full array
#import sys
#import numpy
#numpy.set_printoptions(threshold=sys.maxsize)
#X
#Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tf= TfidfVectorizer()
X= tf.fit_transform(corpus).toarray()

#Visting the full array
#import sys
#import numpy
#numpy.set_printoptions(threshold=sys.maxsize)
#X