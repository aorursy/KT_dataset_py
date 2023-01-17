INPUT_CSV_FILE = "../input/All-seasons.csv"
OUTPUT_TEXT_FILE = "./vector.txt"
OUTPUT_JSON_FILE = "./vector.json"
print("Ready")
# Any results you write to the current directory are saved as output.
import pandas as pd

southpark_df = pd.read_csv(INPUT_CSV_FILE)
southpark_df.info()
unique_characters = southpark_df.Character.unique()
print("Found ",len(unique_characters)," unique characters")
print(unique_characters)
grouped_by_character = southpark_df.groupby(['Character']).count().reset_index()
sorted_characters = grouped_by_character.sort_values('Line', ascending=False)
sorted_characters.head(4)
CHARACTER_NAME = "Cartman"
character_lines = southpark_df.loc[southpark_df.Character == CHARACTER_NAME]["Line"]
print(character_lines.describe())
character_lines.head()
import string
import re
def tokenize(s) :
    lower_case = s.lower();
    without_punctuation = re.sub(r'[^\w\s]','',lower_case)
    return without_punctuation.split()

tokenized_lines = character_lines.apply(lambda row: tokenize(row))
tokenized_lines.tail(4)
from gensim.models import Word2Vec
model = Word2Vec(tokenized_lines, size=100, window=5, min_count=5, workers=4)
model.wv.save_word2vec_format(OUTPUT_TEXT_FILE, binary=False)
import os
print(os.listdir('./'))


import json
def to_json(input, output): 
    f = open(input)
    v = {"vectors": {}}
    for line in f:
        w, n = line.split(" ", 1)
        v["vectors"][w] = list(map(float, n.split()))
    with open(output, "w") as out:
        json.dump(v, out)

to_json(OUTPUT_TEXT_FILE, OUTPUT_JSON_FILE)
import os
print(os.listdir('./'))