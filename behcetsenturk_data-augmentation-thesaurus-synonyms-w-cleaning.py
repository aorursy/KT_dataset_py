import pandas as pd, numpy as np
# Get the data

train_df=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

train_df['text']=train_df['text'].str.lower().astype(str)

train_df['selected_text']=train_df['selected_text'].str.lower().astype(str)
# Keys = Unsual/Wrong words

# Values = Fixed words



td = {

    "u":"you",

    "ur":"you are",

    "n":"and",

    "aww":"cute",

    "sooo":"so",

    "r":"are",

    "cuz":"because",

    "til":"till",

    "lil":",little",

    "b":"be",

    "ppl":"people",

    "yay":"cheer",

    "nite":"night",

    "lmao":"haha",

    "tho":"though",

    "btw":"by the way",

    "yr":"year",

    "dm":"message",

    "idk":"i do not know",

    "outta":"out of",

    "jus":"just",

    "thru":"through",

    "wtf":"what the fuck",

    "wit":"with",

    "gettin":"getting",

    "dnt":"dont",

    "mum":"mom",

    "mums":"moms",

    "hun":"honey",

    "luv":"love",

    "hrs":"hours",

    "chillin":"chilling",

    "abt":"about",

    "tha":"that",

    "ahh":"ah",

    "feelin":"feeling",



    "tho.":"though",

    "w/":"with",

    "u?":"you?",

    "s":"is",



    ":O":"suprized",

    ":p":"lol",

    "(:":":)",

    ":S":":("

}



def cleaning_function(string):

    # Take tweet clean and return

    

    cleaned_words = []

    for word in string.split():

        word = td.get(word, word)

        cleaned_words.append(word)

    

    return " ".join(cleaned_words)
import json

with open('../input/englishengen-synonyms-json-thesaurus/eng_synonyms.json') as json_file:  

    synonyms_dict = json.load(json_file)





print("\/  EXAMPLES  \/ ")    

print("system   =", synonyms_dict["system"])

print("data     =", synonyms_dict["data"])    

print("weapon   =", synonyms_dict["weapon"])

print("i        =", synonyms_dict["i"])
def synonym_function(text, selected_text):

    words = text.split()

    

    if len(words) >= 5:  # Dont do anything on short texts

        # Select three word

        word1 = words[0]

        word2 = words[2]

        word3 = words[4]

        

        # Try to get synonymed words for selected words

        synonymed_word1 = synonyms_dict.get(word1, [])

        synonymed_word2 = synonyms_dict.get(word2, [])

        synonymed_word3 = synonyms_dict.get(word3, [])

        

        # If synonymed words is empty changed it to original word

        if synonymed_word1 == []:

            synonymed_word1 = [word1]

        if synonymed_word2 == []:

            synonymed_word2 = [word2]

        if synonymed_word3 == []:

            synonymed_word3 = [word3]

            

    else:

        return text, selected_text

    

    # There can be more than one synonym for any word so we have to select one

    synonymed_word1 = synonymed_word1[0]

    synonymed_word2 = synonymed_word2[len(synonymed_word2)//2]

    synonymed_word3 = synonymed_word3[-1]

    

    # Change text's words

    changed_text = []

    for word in text.split():

        if word == word1:

            word = synonymed_word1

        elif word == word2:

            word = synonymed_word2

        elif word == word3:

            word = synonymed_word3

        changed_text.append(word)

    

    # Change selected_text's words

    changed_selected_text = []

    for word in selected_text.split():

        if word == word1:

            word = synonymed_word1

        elif word == word2:

            word = synonymed_word2

        elif word == word3:

            word = synonymed_word3

        changed_selected_text.append(word)

            

    # Return text, selected_text

    return " ".join(changed_text), " ".join(changed_selected_text)
text_ids = []

texts = []

selected_texts = []

sentiments = []



for i,row in train_df.iterrows():

    text_id = "007"

    text = row["text"]

    selected_text = row["selected_text"]

    sentiment = row["sentiment"]

    

    text = cleaning_function(text)

    selected_text = cleaning_function(selected_text)

    

    text, selected_text = synonym_function(text, selected_text)

    

    text_ids.append(text_id)

    texts.append(text)

    selected_texts.append(selected_text)

    sentiments.append(sentiment)

    

augmented = {'textID': text_ids, 'text': texts, 'selected_text': selected_texts, 'sentiment': sentiments}

train_df_aug = pd.DataFrame(data=augmented)



train_df_aug
pd.DataFrame(data={'Original': train_df["text"], 'Augmentation': texts}).head(50)
train_df = train_df.append(train_df_aug)

train_df.reset_index(inplace=True,drop=True)



train_df