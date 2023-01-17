import re
text =  "    i'd: i would, i'd've: i would have, i': i will, :-)       The film 12 Pulp Fiction  :>? was released in it's 20   year 1994...... That's born in 19th century. You can search on www.go99ogle.com"
result = re.sub(r"\d", "", text)     #  Remove digits

result = re.sub(r"http\S+", "", result) #  Remove urls

result = re.sub(r"www\S+", "", result) #  Remove urls

result = re.sub(r"\s+", " ", result) #  Remove extra spaces





print(result)
#Clean up the comment text

def clean_text(g_text):

    

    # Contractions

    g_text = re.sub(r"he's", "he is", g_text)

    g_text = re.sub(r"there's", "there is", g_text)

    g_text = re.sub(r"We're", "We are", g_text)

    g_text = re.sub(r"That's", "That is", g_text)

    g_text = re.sub(r"won't", "will not", g_text)

    g_text = re.sub(r"they're", "they are", g_text)

    g_text = re.sub(r"Can't", "Cannot", g_text)

    g_text = re.sub(r"wasn't", "was not", g_text)

    g_text = re.sub(r"don\x89Ûªt", "do not", g_text)

    g_text = re.sub(r"aren't", "are not", g_text)

    g_text = re.sub(r"isn't", "is not", g_text)

    g_text = re.sub(r"What's", "What is", g_text)

    g_text = re.sub(r"haven't", "have not", g_text)

    g_text = re.sub(r"hasn't", "has not", g_text)

    g_text = re.sub(r"There's", "There is", g_text)

    g_text = re.sub(r"He's", "He is", g_text)

    g_text = re.sub(r"It's", "It is", g_text)

    g_text = re.sub(r"You're", "You are", g_text)

    g_text = re.sub(r"I'M", "I am", g_text)

    g_text = re.sub(r"shouldn't", "should not", g_text)

    g_text = re.sub(r"wouldn't", "would not", g_text)

    g_text = re.sub(r"i'm", "I am", g_text)

    g_text = re.sub(r"I\x89Ûªm", "I am", g_text)

    g_text = re.sub(r"I'm", "I am", g_text)

    g_text = re.sub(r"Isn't", "is not", g_text)

    g_text = re.sub(r"Here's", "Here is", g_text)

    g_text = re.sub(r"you've", "you have", g_text)

    g_text = re.sub(r"you\x89Ûªve", "you have", g_text)

    g_text = re.sub(r"we're", "we are", g_text)

    g_text = re.sub(r"what's", "what is", g_text)

    g_text = re.sub(r"couldn't", "could not", g_text)

    g_text = re.sub(r"we've", "we have", g_text)

    g_text = re.sub(r"it\x89Ûªs", "it is", g_text)

    g_text = re.sub(r"doesn\x89Ûªt", "does not", g_text)

    g_text = re.sub(r"It\x89Ûªs", "It is", g_text)

    g_text = re.sub(r"Here\x89Ûªs", "Here is", g_text)

    g_text = re.sub(r"who's", "who is", g_text)

    g_text = re.sub(r"I\x89Ûªve", "I have", g_text)

    g_text = re.sub(r"y'all", "you all", g_text)

    g_text = re.sub(r"can\x89Ûªt", "cannot", g_text)

    g_text = re.sub(r"would've", "would have", g_text)

    g_text = re.sub(r"it'll", "it will", g_text)

    g_text = re.sub(r"we'll", "we will", g_text)

    g_text = re.sub(r"wouldn\x89Ûªt", "would not", g_text)

    g_text = re.sub(r"We've", "We have", g_text)

    g_text = re.sub(r"he'll", "he will", g_text)

    g_text = re.sub(r"Y'all", "You all", g_text)

    g_text = re.sub(r"Weren't", "Were not", g_text)

    g_text = re.sub(r"Didn't", "Did not", g_text)

    g_text = re.sub(r"they'll", "they will", g_text)

    g_text = re.sub(r"they'd", "they would", g_text)

    g_text = re.sub(r"DON'T", "DO NOT", g_text)

    g_text = re.sub(r"That\x89Ûªs", "That is", g_text)

    g_text = re.sub(r"they've", "they have", g_text)

    g_text = re.sub(r"i'd", "I would", g_text)

    g_text = re.sub(r"should've", "should have", g_text)

    g_text = re.sub(r"You\x89Ûªre", "You are", g_text)

    g_text = re.sub(r"where's", "where is", g_text)

    g_text = re.sub(r"Don\x89Ûªt", "Do not", g_text)

    g_text = re.sub(r"we'd", "we would", g_text)

    g_text = re.sub(r"i'll", "I will", g_text)

    g_text = re.sub(r"weren't", "were not", g_text)

    g_text = re.sub(r"They're", "They are", g_text)

    g_text = re.sub(r"Can\x89Ûªt", "Cannot", g_text)

    g_text = re.sub(r"you\x89Ûªll", "you will", g_text)

    g_text = re.sub(r"I\x89Ûªd", "I would", g_text)

    g_text = re.sub(r"let's", "let us", g_text)

    g_text = re.sub(r"it's", "it is", g_text)

    g_text = re.sub(r"can't", "cannot", g_text)

    g_text = re.sub(r"don't", "do not", g_text)

    g_text = re.sub(r"you're", "you are", g_text)

    g_text = re.sub(r"i've", "I have", g_text)

    g_text = re.sub(r"that's", "that is", g_text)

    g_text = re.sub(r"i'll", "I will", g_text)

    g_text = re.sub(r"doesn't", "does not", g_text)

    g_text = re.sub(r"i'd", "I would", g_text)

    g_text = re.sub(r"didn't", "did not", g_text)

    g_text = re.sub(r"ain't", "am not", g_text)

    g_text = re.sub(r"you'll", "you will", g_text)

    g_text = re.sub(r"I've", "I have", g_text)

    g_text = re.sub(r"Don't", "do not", g_text)

    g_text = re.sub(r"I'll", "I will", g_text)

    g_text = re.sub(r"I'd", "I would", g_text)

    g_text = re.sub(r"Let's", "Let us", g_text)

    g_text = re.sub(r"you'd", "You would", g_text)

    g_text = re.sub(r"It's", "It is", g_text)

    g_text = re.sub(r"Ain't", "am not", g_text)

    g_text = re.sub(r"Haven't", "Have not", g_text)

    g_text = re.sub(r"Could've", "Could have", g_text)

    g_text = re.sub(r"youve", "you have", g_text)  

    g_text = re.sub(r"donå«t", "do not", g_text)   

            

    # Character entity references

    g_text = re.sub(r"&gt;", ">", g_text)

    g_text = re.sub(r"&lt;", "<", g_text)

    g_text = re.sub(r"&amp;", "&", g_text)

    

    # Remove digits

    g_text = re.sub(r"\d", "", g_text)     #  Remove digits

           

    # Urls

    g_text = re.sub(r"http\S+", "", g_text) #  Remove urls

    g_text = re.sub(r"www\S+", "", g_text) #  Remove urls



    # Words with punctuations and special characters

    g_text = re.sub(r'\W', " ", g_text)



    # Replace multiple fullstops with single fullstop



    g_text = re.sub(r'\.+', ".", g_text)

    

    # Remove extra spaces

    g_text = re.sub(r"\s+", " ", g_text) #  Remove extra spaces

    

    

    

    

    return g_text
output = clean_text(text)

print(output)