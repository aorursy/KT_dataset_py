def sanitizeWord(word):
    import re
    
    word = word.replace("\n", "") # Replace all newlines with empty space
    word = re.sub(r'[^a-zA-Z]', '', word) # Regex Expression to replace all non alphabetic characters with empty space
    word = word.lower()
    return word



def getDialogue(book): # Function that returns list of all words (as strings) that appear in the book. (Strings have been cleaned.)
    import re
    lexicon = [] # AKA Vocab
    
    for sentence in book:
        words = sentence.split(" ")
        
        for word in words:
            cleanedWord = sanitizeWord(word)
            lexicon.append(cleanedWord)
    
    lexicon = [x for x in lexicon if x != ""] # Remove all occurances of "" in lexicon
    return lexicon



def getFrequencies(allWords):
    frequencies = {} # I typed out 'frequencies' wrong so many times I almost lost it.
    for word in allWords: # Adds / Increments occurances (AKA frequencies) of a specific word.
        if(word in frequencies):
            frequencies[word] = frequencies[word] + 1
        
        elif(word not in frequencies):
            frequencies[word] = 1
        
        else:
            print("This seriously should not have happened.") # There's honestly no scenario I can think of where this occurs.

    return frequencies



def orderFrequencies(frequencies):
    tupledFrequencies = sorted(frequencies.items(), key=lambda x: x[1], reverse=True) # Nifty one liner to sort dictionary into ordered tuples.
    
    orderedFrequencies = {}
    for tup in tupledFrequencies: # Make the tuples BACK into a dictionary
        orderedFrequencies[tup[0]] = int(tup[1])
    
    return orderedFrequencies



def getTopKFrequencies(frequencies, k): # Function that returns top K keys as sorted by their integer values. 
    orderedFrequencies = orderFrequencies(frequencies)
    
    i = 0
    keys = []
    values = []

    for x, y in orderedFrequencies.items():
        keys.append(x)
        values.append(y)
        i += 1
        if(i >= k): # Hence, the top K values.
            break 

    return keys, values
with open("/kaggle/input/FellowshipOfTheRings.txt", "r", encoding="latin-1") as f:
    book = f.readlines()

allWords = getDialogue(book)
#print(len(allWords)) # Prints number of words in entire book
#print(len(set(allWords))) # Prints vocabulary (ie, number of unique words used)
frequencies = getFrequencies(allWords)

topWords, topFrequencies = getTopKFrequencies(frequencies, 10)
def writeAndShowFrequencies(topWords, topFrequencies, filename):
    with open(filename, "w") as f:
        for x in range(len(topWords)):
            line = "The word was '{0}', with a frequency of {1}\n".format(topWords[x], topFrequencies[x]) # String formatting is pretty cool
            print(line)
            f.write(line)
        
writeAndShowFrequencies(topWords, topFrequencies, "wordsAndFreqs.txt")    
import plotly.graph_objects as go

fig = go.Figure([go.Bar(x=topWords, y=topFrequencies)])
fig.update_layout(
    title='Word Frequencies in Provided Book',
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Word Frequency',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title='Word',
        titlefont_size=16,
        tickfont_size=14,
    )
)
    
fig.show()


