# !pip install spacy
import spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy validate
# Load the installed model "en_core_web_sm"

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a example text")
# Import the English language class

from spacy.lang.en import English



# Create the nlp object

nlp = English()



# Process a text

doc = nlp("Progress to Contributor to make your voice count!")



# Print the document text

print(doc.text)
# Import the German language class

from spacy.lang.de import German



# Create the nlp object

nlp = German()



# Process a text (this is German for: "Kind regards!")

doc = nlp("Liebe Grüße!")



# Print the document text

print(doc.text)
# Import the Spanish language class

from spacy.lang.es import Spanish



# Create the nlp object

nlp = Spanish()



# Process a text (this is Spanish for: "How are you?")

doc = nlp("¿Cómo estás?")



# Print the document text

print(doc.text)
# Import the English language class and create the nlp object

from spacy.lang.en import English

nlp = English()



# Process the text

doc = nlp("I like tree kangaroos and narwhals.")



# Select the first token

first_token = doc[0]



# Print the first token's text

print(first_token.text)
nlp = spacy.load("en_core_web_sm")

doc = nlp("This is a example text")

# Token texts

[token.text for token in doc]
doc = nlp("This is a example text")

span = doc[2:4]

span.text
# Import the English language class and create the nlp object

from spacy.lang.en import English

nlp = English()



# Process the text

doc = nlp("I like tree kangaroos and narwhals.")



# A slice of the Doc for "tree kangaroos"

tree_kangaroos = doc[2:4]

print(tree_kangaroos.text)



# A slice of the Doc for "tree kangaroos and narwhals" (without the ".")

tree_kangaroos_and_narwhals = doc[2:6]

print(tree_kangaroos_and_narwhals.text)
# Import the Span object

from spacy.tokens import Span

# Create a Doc object

doc = nlp("I live in Guwahati Assam")

# Span for "Guwahati" with label GPE (geopolitical)

span = Span(doc, 3, 5, label="GPE")

span.text
# Process the text

doc = nlp("In 1990, more than 60% of people in East Asia were in extreme poverty. Now less than 4% are.")



# Iterate over the tokens in the doc

for token in doc:

    # Check if the token resembles a number

    if token.like_num:

        # Get the next token in the document

        next_token = doc[token.i + 1]

        # Check if the next token's text equals '%'

        if next_token.text == '%':

          print('Percentage found:', token.text)
nlp = spacy.load("en_core_web_sm")

doc = nlp("This is an another example text.")

# Coarse-grained part-of-speech tags

[token.pos_ for token in doc]
# Fine-grained part-of-speech tags

[token.tag_ for token in doc]
doc = nlp("This is a simple text example.")

# Dependency labels

[token.dep_ for token in doc]
# Syntactic head token (governor)

[token.head.text for token in doc]
doc = nlp("Steve Jobs founded Apple")

# Text and label of named entity span

[(ent.text, ent.label_) for ent in doc.ents]
# Load the small English model

nlp = spacy.load('en_core_web_sm')

text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"



# Process the text

doc = nlp(text)



# Print the document text

print(doc.text)
text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"



# Process the text

doc = nlp(text)



for token in doc:

    # Get the token text, part-of-speech tag and dependency label

    token_text = token.text

    token_pos = token.pos_

    token_dep = token.dep_

    # This is for formatting only

    print('{:<12}{:<10}{:<10}'.format(token_text, token_pos, token_dep))
text = "It’s official: Apple is the first U.S. public company to reach a $1 trillion market value"



# Process the text

doc = nlp(text)



# Iterate over the predicted entities

for ent in doc.ents:

    # Print the entity text and its label

    print(ent.text, ent.label_)
text = "New iPhone X release date leaked as Apple reveals pre-orders by mistake"



# Process the text

doc = nlp(text)



# Iterate over the entities

for ent in doc.ents:

    # print the entity text and label

    print(ent.text, ent.label_)
text = "New iPhone X release date leaked as Apple reveals pre-orders by mistake"



# Process the text

doc = nlp(text)



# Iterate over the entities

for ent in doc.ents:

    # print the entity text and label

    print(ent.text, ent.label_)



# Get the span for "iPhone X"

iphone_x = doc[1:3]



# Print the span text

print('Missing entity:', iphone_x.text)
doc = nlp("This a sentence. This is another one.")

# doc.sents is a generator that yields sentence spans

[sent.text for sent in doc.sents]
doc = nlp("I have a brown car")

# doc.noun_chunks is a generator that yields spans

[chunk.text for chunk in doc.noun_chunks]
spacy.explain("NN")
spacy.explain("GPE")
from spacy import displacy
doc = nlp("I live in Guwahati, Assam")

displacy.render(doc, style="dep")
doc = nlp("Bill Gates founded Microsoft")

displacy.render(doc, style="ent")
doc1 = nlp("I like cats")

doc2 = nlp("I like dogs")

# Compare 2 documents

doc1.similarity(doc2)
# Compare 2 tokens

doc1[2].similarity(doc2[2])
# Compare tokens and spans

doc1[0].similarity(doc2[1:3])
# Vector as a numpy array

doc = nlp("I like cats")

# The L2 norm of the token's vector

doc[2].vector
doc[2].vector_norm
nlp = spacy.load("en_core_web_sm")

nlp.pipe_names
nlp.pipeline
# Function that modifies the doc and returns it

def custom_component(doc):

    print("Do something to the doc here!")

    return doc



# Add the component first in the pipeline

nlp.add_pipe(custom_component, first=True)
from spacy.tokens import Doc, Token, Span

doc = nlp("The sky over Guwahati is blue")
# Register custom attribute on Token class

Token.set_extension("is_color", default=False)

# Overwrite extension attribute with default value

doc[5]._.is_color = True
# Register custom attribute on Doc class

get_reversed = lambda doc: doc.text[::-1]

Doc.set_extension("reversed", getter=get_reversed)

# Compute value of extension attribute with getter

doc._.reversed
# Register custom attribute on Span class

has_label = lambda span, label: span.label_ == label

Span.set_extension("has_label", method=has_label)

# Compute value of extension attribute with method

doc[3:5]._.has_label("GPE")
# Matcher is initialized with the shared vocab

from spacy.matcher import Matcher

# Each dict represents one token and its attributes

matcher = Matcher(nlp.vocab)

# Add with ID, optional callback and pattern(s)

pattern = [{"LOWER": "new"}, {"LOWER": "york"}]

matcher.add('CITIES', None, pattern)

# Match by calling the matcher on a Doc object

doc = nlp("I live in New York")

matches = matcher(doc)

# Matches are (match_id, start, end) tuples

for match_id, start, end in matches:

     # Get the matched span by slicing the Doc

     span = doc[start:end]

     print(span.text)
doc = nlp("After making the iOS update you won't notice a radical system-wide redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of iOS 11's furniture remains the same as in iOS 10. But you will discover some tweaks once you delve a little deeper.")



# Write a pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")

pattern = [{'TEXT': 'iOS'}, {'IS_DIGIT': True}]



# Add the pattern to the matcher and apply the matcher to the doc

matcher.add('IOS_VERSION_PATTERN', None, pattern)

matches = matcher(doc)

print('Total matches found:', len(matches))



# Iterate over the matches and print the span text

for match_id, start, end in matches:

    print('Match found:', doc[start:end].text)
doc = nlp("i downloaded Fortnite on my laptop and can't open the game at all. Help? so when I was downloading Minecraft, I got the Windows version where it is the '.zip' folder and I used the default program to unpack it... do I also need to download Winzip?")



# Write a pattern that matches a form of "download" plus proper noun

pattern = [{'LEMMA': 'download'}, {'POS': 'PROPN'}]



# Add the pattern to the matcher and apply the matcher to the doc

matcher.add('DOWNLOAD_THINGS_PATTERN', None, pattern)

matches = matcher(doc)

print('Total matches found:', len(matches))



# Iterate over the matches and print the span text

for match_id, start, end in matches:

    print('Match found:', doc[start:end].text)
doc = nlp("Features of the app include a beautiful design, smart search, automatic labels and optional voice responses.")



# Write a pattern for adjective plus one or two nouns

pattern = [{'POS': 'ADJ'}, {'POS': 'NOUN'}, {'POS': 'NOUN', 'OP': '?'}]



# Add the pattern to the matcher and apply the matcher to the doc

matcher.add('ADJ_NOUN_PATTERN', None, pattern)

matches = matcher(doc)

print('Total matches found:', len(matches))



# Iterate over the matches and print the span text

for match_id, start, end in matches:

    print('Match found:', doc[start:end].text)
# "love cats", "loving cats", "loved cats"

pattern1 = [{"LEMMA": "love"}, {"LOWER": "cats"}]

# "10 people", "twenty people"

pattern2 = [{"LIKE_NUM": True}, {"TEXT": "people"}]

# "book", "a cat", "the sea" (noun + optional article)

pattern3 = [{"POS": "DET", "OP": "?"}, {"POS": "NOUN"}]