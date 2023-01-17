#@title Set up Directory { run: "auto"}

import os

from IPython.display import clear_output

from google.colab import drive 

from IPython.display import clear_output

drive.mount('/content/gdrive')

working_directory = 'My Drive/Research/form' #@param {type:"string"}

wd="/content/gdrive/"+working_directory

os.chdir(wd)



dirpath = os.getcwd()

print("current directory is : " + dirpath)

%pip install twarc

%pip install jsonlines

clear_output()
#Check if TWARC was installed correctly on the Virtual Machine

%pip show twarc

%pip show jsonlines
#@title Insert API Keys here { run : "auto"}

from twarc import Twarc



consumer_key = "" #@param {type:"string"}

consumer_secret = "" #@param {type:"string"}

access_token = "" #@param {type:"string"}

access_token_secret = "" #@param {type:"string"}



t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
!if cd COVID19_Tweets_Dataset; then git pull; else git clone https://github.com/lopezbec/COVID19_Tweets_Dataset.git COVID19_Tweets_Dataset; fi
#@title Check Keywords to Hydrate { run: "auto" }

coronavirus = True #@param {type:"boolean"}

virus = False #@param {type:"boolean"}

covid = False #@param {type:"boolean"}

ncov19 = False #@param {type:"boolean"}

ncov2019 = False #@param {type:"boolean"}

keyword_dict = {"coronavirus": coronavirus, "virus": virus, "covid": covid, "ncov19": ncov19, "ncov2019": ncov2019}
#@title Enter range of dates to Hydrate { run: "auto" }

start_date = '2020-01-22' #@param {type:"date"}

end_date = '2020-01-23' #@param {type:"date"}





import datetime as dt

files = []

covid_loc = "COVID19_Tweets_Dataset"

for folder in os.listdir(covid_loc):

    foldername = os.fsdecode(folder)

    if keyword_dict.get(foldername.split()[0].lower()) == True:

        folderpath = os.path.join(covid_loc, foldername)

        for file in os.listdir(folderpath):

            filename = os.fsdecode(file)

            date = filename[filename.index("_")+1:filename.index(".")]

            if (dt.datetime.strptime(start_date, "%Y-%m-%d").date() 

            <= dt.datetime.strptime(date, '%Y_%m_%d').date()

             <= dt.datetime.strptime(end_date, "%Y-%m-%d").date()):

                files.append(os.path.join(folderpath, filename))

ids = set()

for filename in files:

    with open(filename) as f:

        for i in f.readline().strip('][').replace(" ", "").split(","):

            ids.add(i) 

print(round((len(ids)/1000000), 3), "million unique tweets.")
#@title Enter ID output file {run: "auto"}

final_tweet_ids_filename = "final_ids.txt" #@param {type: "string"}

with open(final_tweet_ids_filename, "w+") as f:

    for id in ids:

        f.write('%s\n' % id)
#@title Set up Directory { run: "auto"}

final_tweet_ids_filename = "final_ids.txt" #@param {type: "string"}

output_filename = "output.csv" #@param {type: "string"}
import jsonlines, json

output_json_filename = output_filename[:output_filename.index(".")] + ".txt"

ids = []

with open(final_tweet_ids_filename, "r") as ids_file:

    ids = ids_file.read().split()

hydrated_tweets = []

ids_to_hydrate = set(ids)

if os.path.isfile(output_json_filename):

    with jsonlines.open(output_json_filename, "r") as reader:

        for i in reader.iter(type=dict, skip_invalid=True):

            hydrated_tweets.append(i)

            ids_to_hydrate.remove(i["id_str"])

print("Total IDs: " + str(len(ids)) + ", IDs to hydrate: " + str(len(ids_to_hydrate)))

print("Hydrated: " + str(len(hydrated_tweets)))



count = len(hydrated_tweets)

start_index = count;

num_save  = 1000



for tweet in t.hydrate(ids_to_hydrate):

    hydrated_tweets.append(tweet)

    count += 1

    if (count % num_save) == 0:

        with jsonlines.open(output_json_filename, "a") as writer:

            print("Started IO")

            for hydrated_tweet in hydrated_tweets[start_index:]:

                writer.write(hydrated_tweet)

            print("Finished IO")

        print("Saved " + str(count) + " hydrated tweets.")

        start_index = count

if count != start_index:

    print("Here with start_index", start_index)

    with jsonlines.open(output_json_filename, "a") as writer:

        for hydrated_tweet in hydrated_tweets[start_index:]:

           writer.write(hydrated_tweet)   
# Convert jsonl to csv

import csv, jsonlines

output_json_filename = output_filename[:output_filename.index(".")] + ".txt"

keyset = ["created_at", "id", "id_str", "full_text", "source", "truncated", "in_reply_to_status_id",

          "in_reply_to_status_id_str", "in_reply_to_user_id", "in_reply_to_user_id_str", 

          "in_reply_to_screen_name", "user", "coordinates", "place", "quoted_status_id",

          "quoted_status_id_str", "is_quote_status", "quoted_status", "retweeted_status", 

          "quote_count", "reply_count", "retweet_count", "favorite_count", "entities", 

          "extended_entities", "favorited", "retweeted", "possibly_sensitive", "filter_level", 

          "lang", "matching_rules", "current_user_retweet", "scopes", "withheld_copyright", 

          "withheld_in_countries", "withheld_scope", "geo", "contributors", "display_text_range",

          "quoted_status_permalink"]

hydrated_tweets = []

with jsonlines.open(output_json_filename, "r") as reader:

    for i in reader.iter(type=dict, skip_invalid=True):

        hydrated_tweets.append(i)

with  open(output_filename, "w+") as output_file:

    d = csv.DictWriter(output_file, keyset)

    d.writeheader()

    d.writerows(hydrated_tweets)