import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv("../input/simpsons_script_lines.csv",error_bad_lines=False,warn_bad_lines=False  )



#Filter out Homer's lines and get count of total lines

homer = df[df["raw_character_text"] == "Homer Simpson"]

homer.shape[0]
doh = homer[homer["normalized_text"].str.contains("doh",na=False)]

doh.shape[0]
for line in doh["raw_text"]:

    print(line)
#Use grunt as short for annoyed grunt

annoyed_grunt = homer[homer["normalized_text"].str.contains("grunt",na=False)]

annoyed_grunt.shape[0]
for line in annoyed_grunt["raw_text"][0:10]:

    print(line)
#Convert raw text to lowercase and search for "annoyed grunt"

raw_annoyed_grunt = homer[homer["raw_text"].str.lower().str.contains("annoyed grunt")]

raw_annoyed_grunt.shape[0]
for line in raw_annoyed_grunt["raw_text"][0:10]:

    print(line)
#Could easily calculate 12 + 385 im my head, but this is just coding practise

total_doh = raw_annoyed_grunt.shape[0] + doh.shape[0]

print(str(total_doh) + " of Homer's " + str(homer.shape[0]) + " lines contain the word D'oh!")