# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('/kaggle/input/customer-support-on-twitter/twcs/twcs.csv',  
                 usecols=[
                     'author_id', 
                     'inbound', 
                     'text', 
                     'in_response_to_tweet_id'],
                 dtype={
                     'author_id': str,
                     'inbound': np.bool_,
                     'text': str,
                     'in_response_to_tweet_id': pd.Int64Dtype()
                 })
df.head(10)
start_discussion_indices = df.index[df['in_response_to_tweet_id'].isna()].tolist()

conversations = []
last_end_of_conversation = 0

# Split the dataframe at every indices in the start_discussion_indices
for index in start_discussion_indices:
    conversations.append(df[last_end_of_conversation : index + 1].drop('in_response_to_tweet_id', axis=1)) # Don't need the in response to tweet id column anymore
    last_end_of_conversation = index + 1

for conversation in conversations:
    lastInbound = None
    rows_to_remove = []
    messages_to_append = {}
    for index, row in conversation.iterrows():
        inbound = row['inbound']
        message = row['text']
        
        # Two concecutive messages from the same person
        if lastInbound is not None and (lastInbound == inbound):
            # Remove this row 
            rows_to_remove.append(index)
            if not index - 1 in messages_to_append.keys():
                messages_to_append[index - 1] = []
                
            messages_to_append[index - 1].append(message)
            pass

        lastInbound = inbound
        

    for index, message_to_add in messages_to_append.items():
        conversation.at[index, 'text'] = conversation.loc[index]['text'] + '. ' + message_to_add[0]
        
    conversation.drop(rows_to_remove, axis=0, inplace=True)
import re
import string
from langdetect import detect

tagging_regex = re.compile(r"@\S*")
url_pattern = re.compile(r'https?://\S+|www\.\S+')
signature_pattern = re.compile(r"-\S*")
weird_thing_pattern = re.compile(r"\^\S*")
new_line_pattern = re.compile(r"\n+\S*")

chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "ILU: I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My Ass Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait",
    "IMMA": "I am going to",
    "2NITE": "tonight",
    "DMED": "mesaged",
    'DM': "message",
    "SMH": "I am dissapointed"
}

# Thanks to https://stackoverflow.com/a/43023503/3971619
contractions = {
    "ain't": "are not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he shall have / he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'd've": "I would have",
    "i'll": "I will",
    "i'll've": "I will have",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that had",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Reference : https://stackoverflow.com/a/49986645/3971619
def remove_emoji(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

# Thanks to user sudalairajkumar
def remove_url(string):
    return url_pattern.sub(r'', string)

def remove_chat_words_and_contractions(string):
    new_text = []
    for word in string.split(' '):
        if word.upper() in chat_words.keys():
            new_text += chat_words[word.upper()].lower().split(' ')
        if word.lower() in contractions.keys():
            new_text += contractions[word.lower()].split(' ')
        else:
            new_text.append(word)
            
    return ' '.join(new_text)

def remove_signature(text):
    return signature_pattern.sub(r'', text)
    

# Thanks to user sudalairajkumar
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def clean_message(message):
    # Remove user taggings
    message = re.sub(tagging_regex, '', message) # Replace by you. Good idea?
    
    # Remove the emojis
    message = remove_emoji(message)
    
    # Remove urls
    message = remove_url(message)
    
    # Remove signatures
    message = remove_signature(message)
    
    # Remove the chat words and contractions
    message = remove_chat_words_and_contractions(message)
    
    # Remove weird things
    message = weird_thing_pattern.sub(r'', message)

    # Change new line to dot
    message = new_line_pattern.sub(r'.', message)
    
    # Remove punctuation
    # message = remove_punctuation(message)
    
    # Remove start and end whitespace
    message = message.strip()
    
    # Make multiple spaces become a single space
    message = ' '.join(message.split())
    
    # Lower case the message
    # message = message.lower()
    
    # If not in english, return empty string
#     if message and len(message) > 15:
#         if detect(message) != 'en':
#             return ""
    
    return message

for conversation in conversations:
    conversation['cleaned_text'] = conversation.apply(lambda row: clean_message(row['text']), axis=1)
    
# Keep only the conversations with more than 2 nonempty messages
# Also keep only the conversations with even number of messages
conversations = [x for x in conversations if len(x[x['cleaned_text'] == '']) == 0 and len(x) % 2 == 0]

for index, conversation in enumerate(conversations):
    conversation.drop(['text'], axis=1, inplace=True)
    # Add an ID to each conversation because I will put them all back together
    conversation["ID"] = index
    
    # Remove unececery columns
    conversation.drop(['author_id', 'inbound'], inplace=True, axis=1)
    
    # Reverse the conversation
    conversation.sort_index(axis=0 ,ascending=False, inplace=True)
    
full_conversations = pd.concat(conversations)
full_conversations = full_conversations[["ID", "cleaned_text"]]


questions = full_conversations.iloc[::2]['cleaned_text'].tolist()
ids = full_conversations.iloc[::2]['ID'].tolist()
responses = full_conversations.iloc[1::2]['cleaned_text'].tolist()

# Make all the fields the same size
min_length = min(len(questions), len(responses), len(ids))
questions = questions[:min_length]
ids = ids[:min_length]
responses = responses[:min_length]

new_full_df = pd.DataFrame({
    'ID': ids,
    'question': questions,
    'response': responses
})
# todo: Save
new_full_df.to_csv("/kaggle/working/cleaned_data_upper.csv", index=False)
