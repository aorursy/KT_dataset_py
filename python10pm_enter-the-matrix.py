! pip install tika
import numpy as np

import pandas as pd

import re

import os

from tika import parser

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import string

import requests

from PIL import Image as pil_image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from IPython.display import Image

from IPython.core.display import HTML 

import nltk

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVR, SVC

from xgboost import XGBClassifier



%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
def print_files():

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))
print_files()



PATH = "/kaggle/input/matrix-i-script/Matrix I Script.pdf"
def read_with_tika(PATH):

    raw = parser.from_file(PATH)

    return raw["content"]
raw_tika = read_with_tika(PATH)
raw_tika.split("\n\n")[40:100]
raw_tika.split("\n")[49:55]

print("................................................................")

raw_tika.split("\n")[177:180]
# this regex means: 

# ^\d --> the line must start with a digit

# +   --> can have one ore more digit and a space

re_extract_act = re.compile('^\d+ ') 
l_acts = []

for line in raw_tika.split("\n"):

    if re_extract_act.findall(line):

        # print(line)

        l_acts.append(line)

        

print("Example of some extracted acts.")

l_acts[10:15]
print("We have a total of {} acts".format(len(l_acts)))
len(set([act.split()[0] for act in l_acts]))
len(set([act.split()[0] for act in l_acts if "CONTINUE" in act]))
for act in l_acts:

    if act.split()[0] in ["135", "136"]:

        print(act)
unique_acts = {}

for line in l_acts:

    act_number = line.split()[0] # extract the act number

    if act_number not in unique_acts.keys(): # if act number is not in the dictionary keys, then

        unique_acts[act_number] = line # append the act number and the line



len(unique_acts)
unique_acts["49"]

unique_acts["120"]

unique_acts["135"]
for k,v in unique_acts.items():

    if "CONTINUE" in v:

        print(k ,v)
all_acts = list(unique_acts.values())

all_but_first_act = list(unique_acts.values())[1:]

print("The lenght of all_acts is {} and is bigger that the lenght of all_but_first_act {}".format(len(all_acts), len(all_but_first_act)))

all_acts[:10]

all_but_first_act[:10]
print("With our iterative aproach we will be able to extract {} % of the data. Not bad for such a simple solution".format(float(str(219/220)[:5])*100))
# this very simple trick consists of iterating through 2 acts (i and i + 1) and spliting our string data and assigning the text to the first act.

acts_data = {}



for starting_act, finish_act in zip(all_acts, all_but_first_act):

    current_act_data = raw_tika.split(starting_act)[1].split(finish_act)[0]

    acts_data[starting_act] = current_act_data

len(acts_data.keys())
all_acts[-1]
acts_data[all_acts[-1]] = raw_tika.split(all_acts[-1])[1]
print("This is the act {}".format(all_acts[10]))

acts_data[all_acts[10]]

print("----------------------------------------------------------------------------")

print("----------------------------------------------------------------------------")

print("----------------------------------------------------------------------------")

print("This is the act {}".format(all_acts[11]))

acts_data[all_acts[11]]

print("----------------------------------------------------------------------------")

print("----------------------------------------------------------------------------")

print("----------------------------------------------------------------------------")

print("This is the act {}".format(all_acts[12]))

acts_data[all_acts[12]]

print("----------------------------------------------------------------------------")

print("----------------------------------------------------------------------------")

print("----------------------------------------------------------------------------")

print("This is the act {}".format(all_acts[219]))

acts_data[all_acts[219]]
len(acts_data)
text_ = acts_data["1 ON COMPUTER SCREEN 1"]



for line in text_.split("\n\n"):

    print(line.split("\n"))
characters_list = []

for i, act in enumerate(acts_data.keys()):

    text_ = acts_data[act] # extract the act text

    for line in text_.split("\n"):

        line_ = line.replace(".", "").replace("(", "").replace(")", "").replace(" ", "") # replace (V.O.) with VO

        if bool(re.match(r'[A-Z]+$', line_)): # If any match returns True, enter the if and append the striped line to our list of caracthers_list

            characters_list.append(line.strip())



print("We found a total of {} caracthers and only {} unique.".format(len(characters_list), len(set(characters_list))))

# Here we have the "main characters", however we can clearly see that we have some junk in our set.

# Let's clean it manually



pop_list = [

'(CONTINUED)',

'(MORE)',

'BOOM.',

'CLICK.',

'FADE OUT.',

'FADE TO BLACK.',

'METAL.',

'MIRROR.',

'MONITOR.',

'MONITORS SNAP FLATLINE.',

'THE END'

]



unique_characters = sorted(list(set(characters_list)^set(pop_list))) # remove all elements that are common in 2 lists using set operations



full_characters_in_order = []

for character in characters_list:

    if character in pop_list:

        pass

    else:

        full_characters_in_order.append(character)
lt = []

i = 0

for act_name, act in acts_data.items():

    

    current_act = act.split("\n\n")

    ambientation_text = ""

    has_character = False

    

    for element in current_act: 

        if element == "":

            current_act.remove(element)

    

    

    if np.random.random() < 0.01: # randomly print some acts from the script

        print("-------------------------------------------")

        print()

        print(act_name)

        print()

        print(current_act)

        print()

        print("-------------------------------------------")

    

    for character in unique_characters:

        if character in " ".join(current_act):

            has_character = True

            

    # I have detected that act 197 is not parsed correctly.

    # This is because it starts with Agent Smith so has_character is marked as true.

    # But when you try to find the character_ it fails because the word Agent Smith appears as ambientation

    # That's why I have added this ugly line :(

    

    if has_character and act_name != "197 EXT. HEART O' THE CITY HOTEL - DAY 197": 

        

        for line in current_act:



            character_ = line.split("\n")[0].strip()



            if character_ in unique_characters:

                i += 1

                lt.append((i, act_name, "AMBIENTATION", ambientation_text))

                i += 1

                lt.append((i, act_name, character_, " ".join(line.split("\n")[1:])))    

                ambientation_text = ""

                

            else:

                ambientation_text = ambientation_text + " ".join(line.split("\n"))

                

        i += 1 # append the last line if it's not a character.

        lt.append((i, act_name, "AMBIENTATION", ambientation_text))

        ambientation_text = ""



    else: # has no character and we only append ambientation.

        for line in current_act:

            ambientation_text += line

            

        i += 1

        lt.append((i, act_name, "AMBIENTATION", ambientation_text))

df = pd.DataFrame(lt, columns = ["Numeration", "Act", "Ambientation/Character", "Text"])

df["Act_number"] = df["Act"].apply(lambda x: x.split()[0]) # create a new column so that we can easily acess the act by number

df = df[["Numeration", "Act_number", "Act", "Ambientation/Character", "Text"]] # rearrange the columns

df["Numeration"] = df["Numeration"].astype(int) # convert to int

df["Act_number"] = df["Act_number"].astype(int) # convert to in

df.head()

df.shape
df[df["Act_number"] == 197] # checking if this act has finally been parsed.
len(acts_data.keys())
len(df["Act"].unique())
len(acts_data.keys()) == len(df["Act"].unique()) # we have parsed all the acts. Great.
# no act is missing. Just another check.

for act_name in acts_data.keys():

    if act_name not in list(df["Act"].unique()):

        print(act_name) 
# Another check to reassure that we have all the characters.

# So far, everything seems okay.

for character in df["Ambientation/Character"].unique():

    if character not in unique_characters:

        print(character)
# Another check to reassure that we have all the characters.

# So far, everything seems okay.

for character in unique_characters:

    if character not in df["Ambientation/Character"].unique():

        print(character)
df.sample(10)
df[df["Numeration"] == 318]
text_ = df[df["Numeration"] == 318]["Text"].values

text_
df[df["Text"] == ""]
index_to_drop = df[df["Text"] == ""].index

index_to_drop



print("A total {} of rows will be dropped".format(len(index_to_drop)))
df.drop(index_to_drop, inplace = True)
df[df["Text"] == ""]
numeration = [i + 1 for i in range(df.shape[0])]

df["Numeration"] = numeration

df.reset_index(drop = True, inplace = True)
# A cleaning dictionary

# We want to always have the posibility to analyze all Neo's coversation regarding if it's V.O. or other type.



cleaning_characters = {

'AGENT BROWN': 'Agent Brown',

'AGENT JONES': 'Agent Jones',

'AGENT SMITH': 'Agent Smith',

'AMBIENTATION': 'Ambientation',

'APOC': 'Apoc',

'BIG COP': 'Big Cop',

'CHOI': 'Choi',

'CHOI (MAN)': 'Choi',

'CHOI (O.S)': 'Choi',

'COP': 'Cop',

'CYPHER': 'Cypher',

'CYPHER (MANV.O.)': 'Cypher',

'CYPHER (V.O.)': 'Cypher',

'DOZER': 'Dozer',

'DUJOUR': 'Dujour',

'FEDEX GUY': 'Fedex Guy',

'GUARD': 'Guard',

'LIEUTENANT': 'Lieutenant',

'MAN (BUSINESSMAN)': 'Man (Businessman)',

'MAN (V.O.)': 'Man (V.O.)',

'MORPHEUS': 'Morpheus',

'MORPHEUS (MANV.O.)':  'Morpheus',

'MORPHEUS (O.S.)':  'Morpheus',

'MORPHEUS (V.O.)':  'Morpheus',

'MOUSE': 'Mouse',

'NEO': 'Neo',

'NEO (O.S.)': 'Neo',

'NEO (V.O.)': 'Neo',

'ORACLE': 'Oracle',

'ORACLE (OLD WOMAN)': 'Oracle',

'PILOT': 'Pilot',

'PRIESTESS': 'Priestess',

'PRIESTESS (WOMAN)': 'Priestess',

'RHINEHEART': 'Rhineheart',

'SPOON BOY': 'Spoon Boy',

'SPOON BOY (SKINNY BOY)': 'Spoon Boy',

'SWITCH': 'Switch',

'TANK': 'Tank',

'TANK (V.O.)': 'Tank',

'TRINITY': 'Trinity',

'TRINITY (O.S.)': 'Trinity',

'TRINITY (V.O.)': 'Trinity',

'TRINITY (WOMANV.O.)': 'Trinity',

'VOICE (O.S.)': 'Fedex Guy',

'WOMAN (V.O.)': 'Trinity'

}
df["Char"] = df["Ambientation/Character"].map(cleaning_characters)
df[df["Act_number"] == 1]
df[df["Act_number"] == 2]
df[df["Act_number"] == 3]
puncts = string.punctuation

puncts = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~' # we delete the punct

numbers = "1234567890"



numbs_and_punct = numbers + puncts

numbs_and_punct
all_text = {}

for data in df["Text"]:

    for word in data.split():

        

        cleaned_word = ""

        

        for c in word.lower():

            if c in numbs_and_punct:

                pass

            else:

                cleaned_word += c



        if len(cleaned_word) > 0:



            if cleaned_word.lower() not in all_text.keys():

                all_text[cleaned_word.lower()] = 1



            else:

                current_value = all_text[cleaned_word.lower()]

                all_text[cleaned_word.lower()] = current_value + 1

all_text_ = {k: v for k, v in sorted(all_text.items(), key=lambda item: item[1], reverse=False)}

len(all_text_)
all_text = {}

for data in df["Text"]:

    for word in data.split():

        

        cleaned_word = ""

        

        for c in word.lower():

            if c in numbs_and_punct:

                pass

            else:

                cleaned_word += c

        

        # We have to be very careful with the text we extract from pdfs.

        # If we eliminate the dots from our words, then we might end with words like

        # glasstrinity, but it's because the words are extracted like glass.trinity

        # because of the pdf structure.

        

        if "." not in cleaned_word:

                

            if len(cleaned_word) > 0:



                if cleaned_word.lower() not in all_text.keys():

                    all_text[cleaned_word.lower()] = 1



                else:

                    current_value = all_text[cleaned_word.lower()]

                    all_text[cleaned_word.lower()] = current_value + 1

                    

        elif "." in cleaned_word:

            for word_ in cleaned_word.split("."):

                if word_.lower() not in all_text.keys():

                    all_text[word_.lower()] = 1

                else:

                    current_value = all_text[word_.lower()]

                    all_text[word_.lower()] = current_value + 1

                    
all_text = {k: v for k, v in sorted(all_text.items(), key=lambda item: item[1], reverse=True) if k != ""}

len(all_text)



i = 0

for k, v in all_text.items():

    if i < 10:

        print(k, v)

        i+=1

    

keys_i = [i for i in range(len(all_text.keys()))]

keys = list(all_text.keys())

values = list(all_text.values())
for k in all_text_.keys():

    for word_ in k.split("."):

        if word_ in all_text.keys() and word_ != "":

            pass

        elif word_ != "":

            print(word_, k, all_text_[k])
plt.figure(figsize = (12, 6))

plt.plot(values, label = "Distribution of the frequency of words")

plt.legend()
top_10_bottom_10_keys = keys[:15] + keys[-15:]

top_10_bottom_10_values = values[:15] + values[-15:]

plt.figure(figsize = (12, 6))

plt.bar(top_10_bottom_10_keys, top_10_bottom_10_values)

plt.title("Top 15 most common and less common words in the script.")

plt.xticks(rotation=90)

plt.tight_layout()

plt.legend();
most_used_words = '''1.     the

2.     be

3.     to

4.     of

5.     and

6.     a

7.     in

8.     that

9.     have

10.    I

11.    it

12.    for

13.    not

14.    on

15.    with

16.    he

17.    as

18.    you

19.    do

20.    at

21.    this

22.    but

23.    his

24.    by

25.    from

26.    they

27.    we

28.    say

29.    her

30.    she

31.     or

32.    an

33.    will

34.    my

35.    one

36.    all

37.    would

38.    there

39.    their

40.    what

41.     so

42.    up

43.    out

44.    if

45.    about

46.    who

47.    get

48.    which

49.    go

50.    me

51.     when

52.    make

53.    can

54.    like

55.    time

56.    no

57.    just

58.    him

59.    know

60.    take

61.    people

62.    into

63.    year

64.    your

65.    good

66.    some

67.    could

68.    them

69.    see

70.    other

71.     than

72.    then

73.    now

74.    look

75.    only

76.    come

77.    its

78.    over

79.    think

80.    also

81.     back

82.    after

83.    use

84.    two

85.    how

86.    our

87.    work

88.    first

89.    well

90.    way

91.    even

92.    new

93.    want

94.    because

95.    any

96.    these

97.    give

98.    day

99.    most

100.  us'''
most_used_verbs = '''1.      be

2.      have

3.      do

4.      say

5.      go

6.      can

7.      get

8.      would

9.      make

10.    know

11.     will

12.    think

13.    take

14.    see

15.    come

16.    could

17.    want

18.    look

19.    use

20.    find

21.     give

22.    tell

23.    work

24.    may

25.    should	

26.    call

27.    try

28.    ask

29.    need

30.    feel

31.    become

32.    leave

33.    put

34.    mean

35.    keep

36.    let

37.    begin

38.    seem

39.    help

40.    talk

41.     turn

42.    start

43.    might

44.    show

45.    hear

46.    play

47.    run

48.    move

49.    like

50.    live	

51.    believe

52.    hold

53.    bring

54.    happen

55.    must

56.    write

57.    provide

58.    sit

59.    stand

60.    lose

61.     pay

62.    meet

63.    include

64.    continue

65.    set

66.    learn

67.    change

68.    lead

69.    understand

70.    watch

71.     follow

72.    stop

73.    create

74.    speak

75.    read	

76.    allow

77.    add

78.    spend

79.    grow

80.    open

81.    walk

82.    win

83.    offer

84.    remember

85.    love

86.    consider

87.    appear

88.    buy

89.    wait

90.    serve

91.    die

92.    send

93.    expect

94.    build

95.    stay

96.    fall

97.    cut

98.    reach

99.    kill

100.  remain'''
most_used_nouns = '''1.      time

2.      year

3.      people

4.      way

5.      day

6.      man

7.      thing

8.      woman

9.      life

10.    child

11.     world

12.    school

13.    state

14.    family

15.    student

16.    group

17.    country

18.    problem

19.    hand

20.   part

21.    place

22.    case

23.    week

24.    company

25.    system	

26.    program

27.    question

28.    work

29.    government

30.    number

31.     night

32.    point

33.    home

34.    water

​35.    room

36.    mother

37.    area

38.    money

39.    story

40.    fact

41.     month

42.    lot

43.    right

44.    study

45.    book

46.    eye

47.    job

48.    word

49.    business

50.    issue	

51.     side

52.    kind

53.    head

54.    house

55.    service

56.    friend

57.    father

58.    power

59.    hour

60.    game

61.     line

62.    end

63.    member

64.    law

65.    car

66.    city

67.    community

68.    name

69.    president

70.    team

71.     minute

72.    idea

73.    kid

74.    body

75.    information	

76.    back

77.    parent

78.    face

79.    others

80.    level

81.     office

82.    door

83.    health

84.    person

85.    art

86.    war

87.    history

88.    party

89.    result

90.    change

91.     morning

92.    reason

93.    research

94.    girl

95.    guy

96.    moment

97.    air

98.    teacher

99.    force

100.  education'''
most_used_adjectives = '''1.     other

2.     new

3.     good

4.     high

5.     old

6.     great

7.     big

8.     American

9.     small

10.   large

11.    national

12.    young

13.    different

14.    black

15.    long

16.    little

17.    important

18.    political

19.    bad

20.   white

21.    real

22.   best

23.   right

24.   social

25.   only	

26.    public

27.    sure

28.    low

29.    early

30.    able

31.     human

32.    local

33.    late

34.    hard

35.    major

36.    better

37.    economic

38.    strong

39.    possible

40.    whole

41.     free

42.    military

43.    true

44.    federal

45.    international

46.    full

47.    special

48.    easy

49.    clear

50.    recent	

51.     certain

52.    personal

53.    open

54.    red

55.    difficult

56.    available

57.    likely

58.    short

59.    single

60.    medical

61.     current

62.    wrong

63.    private

64.    past

65.    foreign

66.    fine

67.    common

68.    poor

69.    natural

70.    significant

71.    similar

72.    hot

73.    dead

74.    central

75.    happy	

76.    serious

77.    ready

78.    simple

79.    left

80.    physical

81.     general

82.    environmental

83.    financial

84.    blue

85.    democratic

86.    dark

87.    various

88.    entire

89.    close

90.    legal

91.     religious

92.    cold

93.    final

94.    main

95.    green

96.    nice

97.    huge

98.    popular

99.    traditional

100.  cultural'''
lwords = [word.split()[1] for word in most_used_words.split("\n")]

lverbs = [word.split()[1] for word in most_used_verbs.split("\n")]

lnouns = [word.split()[1] for word in most_used_nouns.split("\n")]

ladjectives = [word.split()[1] for word in most_used_adjectives.split("\n")]



len(lwords) == len(lverbs) == len(lnouns) == len(ladjectives)
def check_words_used(list_, dict_, title, sort = True):

    d_ = {}

    for word in list_:

        word = word.lower()

        if word in dict_.keys():

            d_[word] = dict_[word]

        else:

            d_[word] = 0

            

    if sort:

        

        d_ = {k:v for k,v in sorted(d_.items(), key=lambda item: item[1], reverse=True)}

    

    # let's plot the data

    

    plt.figure(figsize = (12, 6))

    plt.bar(list(d_.keys()), list(d_.values()))

    plt.title(title)

    

    for k,v in d_.items():

        if v == 0:

            plt.axvline(x = k, c = "r", alpha = 0.3)

            

    plt.xticks(rotation = 90)

    plt.legend()

    plt.tight_layout()

    

    return d_

d_ = check_words_used(lwords, all_text, title = "Most used English words and their appearance in the Matrix", sort = False)
d_ = check_words_used(lverbs, all_text, title = "Most used English verbs and their appearance in the Matrix", sort = True)
d_ = check_words_used(lnouns, all_text, title = "Most used English nouns and their appearance in the Matrix", sort = True)
d_ = check_words_used(ladjectives, all_text, title = "Most used English adjectives and their appearance in the Matrix", sort = True)
def clean_text(text):

    '''

    It's pretty much the same code we used to find the unique words when doing the words analysis.

    '''

    

    cleaned_text = []

    

    for word in text.split():

        

        cleaned_word = ""

        

        for c in word.lower():

            if c in numbs_and_punct:

                pass

            else:

                cleaned_word += c

    

        if "." not in cleaned_word:

                

            if len(cleaned_word) > 0:



                cleaned_text.append(cleaned_word)

                    

        elif "." in cleaned_word:

            for word_ in cleaned_word.split("."):

#                 if word_ != ".":

                cleaned_text.append(word_)

                

    return " ".join(cleaned_text)

            

    
df["Cleaned_text"] = df["Text"].apply(clean_text)

df.head()
df[df["Numeration"] == 86]
neo_url ="https://vignette.wikia.nocookie.net/matrix/images/3/32/Neo.jpg/revision/latest?cb=20060715235228"

morpheus_url = "https://vignette.wikia.nocookie.net/matrix/images/9/9b/Morpheus1.jpg/revision/latest?cb=20090501203241"

trinity_url = "https://vignette.wikia.nocookie.net/matrix/images/6/67/Trinityfull.jpg/revision/latest?cb=20060803214449"

agent_smith_url = "https://vignette.wikia.nocookie.net/matrix/images/4/4d/Agent-smith-the-matrix-movie-hd-wallpaper-2880x1800-4710.png/revision/latest/scale-to-width-down/310?cb=20140504013834"



l_urls = [neo_url, morpheus_url, trinity_url, agent_smith_url]

l_names = ["Neo", "Morpheus", "Trinity", "Agent Smith"]



for url, name in zip(l_urls, l_names):

    

    text = " ".join(" ".join(df[df["Char"] == name]["Cleaned_text"].values).split())



    mask = np.array(pil_image.open(requests.get(url, stream=True).raw))



    wordcloud = WordCloud(stopwords = STOPWORDS, 

                          mask = mask,

                          max_font_size = 50, 

                          max_words = 100, 

                          background_color = "black",

                          contour_width=3, 

                          contour_color='gray').generate(text)

    

    plt.figure(figsize = (14, 7))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()

    

    Image(url = url);



    
import networkx as nx

from networkx.algorithms import community

import matplotlib.pyplot as plt
act_lats_of_characters = []



for act_number in df["Act_number"].unique():



    short_df = df[(df["Act_number"] == act_number) & (df["Ambientation/Character"] != "AMBIENTATION")]

    

    if len(short_df["Ambientation/Character"].unique()) >= 4:

        act_lats_of_characters.append(act_number)
act_lats_of_characters


for act_number in act_lats_of_characters:



    short_df = df[(df["Act_number"] == act_number) & (df["Ambientation/Character"] != "AMBIENTATION")][["Text", "Char"]]

    

    if short_df.shape[0] > 0:



        nodes = list(short_df["Char"].values)

        edges = list(short_df["Text"].apply(len))



        G = nx.Graph()



        for start_node, edge, end_node in zip(nodes, edges, nodes[1:]):



#             print(start_node, "--------->", edge, "--------->", end_node)

            G.add_edge(start_node, end_node)



        plt.figure(figsize=(10, 7))

        nx.draw_networkx(G, with_label = True);

short_df = df[(df["Ambientation/Character"] != "AMBIENTATION")][["Text", "Char"]]



if short_df.shape[0] > 0:



    nodes = list(short_df["Char"].values)

    edges = list(short_df["Text"].apply(len))



    G = nx.Graph()



    for start_node, edge, end_node in zip(nodes, edges, nodes[1:]):



#         print(start_node, "--------->", edge, "--------->", end_node)

        G.add_edge(start_node, end_node)

        

    plt.figure(figsize=(20, 10))

    nx.draw_networkx(G, with_label = True) 

short_df = df[(df["Ambientation/Character"] != "AMBIENTATION")][["Act_number", "Text", "Char"]]
short_df[short_df["Act_number"] == 1]
l_interactions = []



for act in short_df["Act_number"].unique():

    

    act_df = short_df[short_df["Act_number"] == act]

    

    if act_df.shape[0] == 1:

        pass # dont extract the data where there is just a single character.

    

    else:

        characters = list(act_df["Char"]) # get all the characters



        characters_but_first = characters[1:] # create a copy with the all the characters except the first one



        penultimate_character = characters[len(characters)-2] # extract the penultimate character, since we assume that the last words of a character are ment for the previous

        characters_but_first.append(penultimate_character) # append to the list



        for i, (c1, c2) in enumerate(zip(characters, characters_but_first)): # iterate over all the characters in a dialogue

            l_interactions.append((c1, c2)) # add all the interactions
d_ = {}

for t in l_interactions:

    

    t_swaped = (t[1], t[0]) # create a temporary tuple with swaped values. In this case (Neo, Trinity) == (Trinity, Neo)

    

    if t[0] == t[1]:

        pass # don't parse the same characters. 

    

    elif t not in d_.keys() and t_swaped not in d_.keys():

        d_[t] = 1

        

    elif t in d_.keys() or t_swaped in d_.keys():

        try:

            current_interactions = d_[t]

            d_[t] = current_interactions + 1

        except:

            current_interactions = d_[t_swaped]

            d_[t_swaped] = current_interactions + 1

d_


G = nx.Graph()



for k ,v in d_.items():

    

    G.add_edge(k[0], k[1], weight = v)



max_edge = max(d_.values()) # scale the weight, either the graph will be vey ugly

edges = G.edges()

weights = [G[u][v]['weight']/(max_edge/2) for u,v in edges]



plt.figure(figsize=(20, 10))



nx.draw_networkx(G,  node_size = 1000, edges=edges, width=weights)

    


G = nx.Graph()



for k ,v in d_.items():

    

    G.add_edge(k[0], k[1], weight = v)



max_edge = max(d_.values()) # scale the weight, either the graph will be vey ugly

edges = G.edges()

weights = [G[u][v]['weight']/(max_edge/4) for u,v in edges]



pos = nx.circular_layout(G)



plt.figure(figsize=(20, 10))



nx.draw_networkx(G,pos,  edges=edges, width=weights)

    
color_dict = {

'Man (V.O.)': '#9bbff4', # good

'Woman (V.O.)': '#9bbff4',

'Trinity': '#9bbff4',

'Morpheus': '#9bbff4',

'Neo': '#9bbff4',

'Switch': '#9bbff4',

'Apoc': '#9bbff4',

'Dozer': '#9bbff4',

'Tank': '#9bbff4',

'Mouse': '#9bbff4',

'Priestess': '#9bbff4',

'Spoon Boy': '#9bbff4',

'Oracle': '#9bbff4',

'Choi': '#bbdaa4', # neutral

'Dujour': '#bbdaa4',

'Rhineheart': '#bbdaa4',

'Fedex Guy': '#bbdaa4',

'Cypher': '#f18d00', # bad

'Agent Smith': '#f18d00',

'Lieutenant': '#f18d00',

'Agent Jones': '#f18d00',

'Agent Brown': '#f18d00',

'Pilot': '#f18d00'

}


G = nx.Graph()



for k ,v in d_.items():

    

    G.add_edge(k[0], k[1], weight = v)



max_edge = max(d_.values()) # scale the weight, either the graph will be vey ugly

edges = G.edges()

weights = [G[u][v]['weight']/(max_edge/7) for u,v in edges]



colors = [color_dict[node] for node in G.nodes()]



plt.figure(figsize=(20, 10))



nx.draw_networkx(G, node_size = 1000, edges=edges, width=weights, node_color  = colors)

    


G = nx.Graph()



for k ,v in d_.items():

    

    G.add_edge(k[0], k[1], weight = v)



max_edge = max(d_.values()) # scale the weight, either the graph will be vey ugly

edges = G.edges()

weights = [G[u][v]['weight']/(max_edge/7) for u,v in edges]



colors = [color_dict[node] for node in G.nodes()]



pos = nx.circular_layout(G)



plt.figure(figsize=(20, 10))



nx.draw_networkx(G, pos, node_size = 1000, edges=edges, width=weights, node_color  = colors)

    
df[df["Numeration"].isin([1268, 1269, 1270, 1271])]
In = nx.closeness_centrality(G)

In
In = {k: v for k, v in sorted(In.items(), key=lambda item: item[1])}

keys = [value for value in In.keys()][::-1] # reverse since the sorting is done from smallest to largest

values = [name for name in In.values()][::-1] # reverse since the sorting is done from smallest to largest

colors = [color_dict[name] for name in In.keys()][::-1]

mean = np.mean(values)
plt.figure(figsize = (20, 10))

plt.bar(keys, values, color = colors)

plt.hlines(xmin = 0, xmax = len(keys), y = mean, label = "Mean centrality", color = "r")



plt.annotate('Mean value is {}'.format(round(mean, 2)),

xy=(10, 0.55),

xycoords='data',

xytext=(-150, 50), 

textcoords='offset points',

arrowprops=dict(arrowstyle="->", color = "r"),

color = "r")



for index, value in enumerate(values):

    plt.text(index, value, str(round(value, 2)))



good_guys = mpatches.Patch(color='#9bbff4', label='Good guys')

neutral_guys = mpatches.Patch(color='#bbdaa4', label='Neutral guys')

bad_guys = mpatches.Patch(color='#f18d00', label='Bad guys')



plt.legend(handles=[good_guys, neutral_guys, bad_guys])



plt.title("Centrality for each character")

plt.tight_layout();
stemmer = SnowballStemmer("english")
df.sample(10)
def stemmer_text(text, stemmer, STOPWORDS):

    return " ".join([stemmer.stem(word) for word in text.split() if word not in STOPWORDS])
df["Stemmed_text"] = df["Cleaned_text"].apply(stemmer_text, 

                                              args = [stemmer, STOPWORDS] # this way you can pass extra arguments when using apply function

                                             )
df.head()

df[df["Char"] != "Ambientation"][["Char"]].groupby(by = "Char").size()
# create the vectorizer

vectorizer = TfidfVectorizer()



# separate between X (features) and Y (target)

train_test = df[~df["Char"].isin(["Ambientation", "Man (Businessman)", "Pilot"])][["Char", "Stemmed_text"]]

X = vectorizer.fit_transform(train_test['Stemmed_text']).toarray()

le = LabelEncoder()

Y = le.fit_transform(train_test['Char'].values)

Y = pd.Series(Y).to_frame()

Y.rename(columns = {0:"Target"}, inplace = True)



# concatenate into train_test df

train_test = pd.concat([train_test, pd.DataFrame(data=X[:,:]), Y], axis = 1, join = "inner")

target_label = train_test["Char"]

train_test.drop(["Char", "Stemmed_text"], inplace = True, axis = 1)



# train test split

X_train, X_test, y_train, y_test = train_test_split(train_test.drop("Target", axis = 1), 

                                                    train_test["Target"], 

                                                    test_size = 0.25

#                                                     stratify = train_test["Target"]

                                                   )



# fit our models

list_of_clfs = [RandomForestClassifier(), LogisticRegression(), MultinomialNB(), SVC()]



for clf in list_of_clfs:



    clf.fit(X_train, y_train)



    prediction = clf.predict(X_test)

    score = np.mean([prediction == y_test]), 

    

    print(round(score[0],3))

    

    print(classification_report(y_test, prediction))

    cm = pd.DataFrame(confusion_matrix(y_test, prediction))

    index = le.inverse_transform(cm.index.values)

    labels = le.inverse_transform(cm.columns.values)

    cm.set_index(index, inplace = True)

    cm.columns = labels

    cm
