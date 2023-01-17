import os

import re 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import shutil

        

shutil.copyfile(src = "../input/bad-words-for-tweets/bad_words.py", dst = "../working/bad_words.py")

from bad_words import whole_words
all_data = pd.read_csv("/kaggle/input/emotion/text_emotion.csv")

train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

val = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")



# we use train + test to test the mapping 

train = pd.concat([train, val])

train.dropna(how="any", subset=["text"], inplace=True)

train.reset_index(inplace=True, drop=True)



all_data = all_data.rename(columns={"tweet_id" : "textID", "content" : "text"})
def remove_html(text):

	text = re.sub("&quot;", "'", text)

	text = re.sub("&gt;", ">",  text)

	text = re.sub("&lt;", "<", text)

	text = re.sub("&le;", "≤", text)

	text = re.sub("&ge;", "≥", text)

	text = re.sub("&amp;", "&", text)

	return text



def find_all(input_str, search_str):

    l1 = []

    length = len(input_str)

    index = 0

    while index < length:

        i = input_str.find(search_str, index)

        if i == -1:

            return l1

        l1.append(i)

        index = i + 1

    return l1



inchars = "abcdefghijklmnopqrstuvwxyzåä*'ö0123456789"

def clean_text(text):

    text = re.sub( "'", "`", text)

    text = remove_html(text)

    text = re.sub( "@[a-zA-Z0-9]+", '', str(text))  # sloppy regex to remove @user 

    for word in whole_words:

        old_txt = text

        if word.lower() in text.lower():

            starts = find_all(text.lower(), word.lower())

            while len(starts) != 0:

                start = starts[0]

                end = start+len(word)

                # skip the word if the preceding or end character is a number or in the alphabet

                if len(text[:start]) != 0 and text[:start][-1].lower() in inchars:

                    starts.remove(start)

                    continue

                elif len(text[end:]) != 0 and text[end:][0].lower() in inchars:

                    starts.remove(start)

                    continue

                

                text = text[:start] + "****" + text[end:]

                starts = find_all(text.lower(), word.lower())

   

    # only edge case

    text = re.sub(" x x ", ' **** ',  text)    

    return text

all_data["old_text"] = all_data.text

all_data.text = all_data.text.map(clean_text)
added = 0

unmapped = 0

all_texts = train.text.tolist()

all_ids = train.textID.astype(str).tolist()

for idx in range(len(all_data)):

    text = all_data.text[idx]

    

    if text in all_texts:

        index = all_texts.index(text)

        all_texts.pop(index)

        text_id = all_ids.pop(index)

        all_data.loc[idx, "aux_id"] = text_id

        added += 1

    else:

        unmapped += 1



print(f"Unmapped:{unmapped} Total:{len(all_data)} Prop: {(added)/len(all_data)}")
# Get the unique Id's 

auxes = all_data.aux_id.unique().tolist()

# remove "nan" which is the first index 

auxes.pop(0) 

# show the unmapped train+test texts

train[~train.textID.isin(auxes)]
all_data = all_data.replace(r'^\s*$', np.nan, regex=True)

all_data = all_data.where(pd.notnull(all_data), None)



index = 1000000000

for idx in range(len(all_data)):

    if all_data.aux_id[idx] == None:

        all_data.loc[idx, "aux_id"] = f"p{index}" 

        index += 1



all_data.rename(columns={"aux_id" : "textID"}, inplace=True)

all_data.to_csv("all_data.csv", index=False)