import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "../input/quora-question-pairs"]).decode("utf8"))

print(check_output(["ls", "../input/movehub-city-rankings"]).decode("utf8"))
dataset = "test" # Obviously you want to run this on the test set as well



df = pd.read_csv("../input/quora-question-pairs/{}.csv".format(dataset))

locations = pd.read_csv("../input/movehub-city-rankings/cities.csv")
# There's lots of room to add more locations, but start with just countries

countries = set(locations['Country'].dropna(inplace=False).values.tolist())

all_places = countries



# Turn it into a Regex

regex = "|".join(sorted(set(all_places)))
from tqdm import tqdm



subset = 10000 # Remove the subsetting 



results = []

print("processing:", df[0:subset].shape)

for index, row in tqdm(df[0:subset].iterrows()):

    q1 = str(row['question1'])

    q2 = str(row['question2'])



    rr = {}



    q1_matches = []

    q2_matches = []



    if (len(q1) > 0):

        q1_matches = [i.lower() for i in re.findall(regex, q1, flags=re.IGNORECASE)]



    if (len(q2) > 0):

        q2_matches = [i.lower() for i in re.findall(regex, q2, flags=re.IGNORECASE)]



    rr['z_q1_place_num'] = len(q1_matches)

    rr['z_q1_has_place'] =len(q1_matches) > 0



    rr['z_q2_place_num'] = len(q2_matches) 

    rr['z_q2_has_place'] = len(q2_matches) > 0



    rr['z_place_match_num'] = len(set(q1_matches).intersection(set(q2_matches)))

    rr['z_place_match'] = rr['z_place_match_num'] > 0



    rr['z_place_mismatch_num'] = len(set(q1_matches).difference(set(q2_matches)))

    rr['z_place_mismatch'] = rr['z_place_mismatch_num'] > 0



    results.append(rr)     



out_df = pd.DataFrame.from_dict(results)

#out_df.to_csv("../features/{}_place_matches.csv".format(dataset), index=False, header=True)

out_df.to_csv("{}_place_matches.csv".format(dataset))
print(check_output(["ls", "./"]).decode("utf8"))