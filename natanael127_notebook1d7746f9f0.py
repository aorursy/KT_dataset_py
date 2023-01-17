import json



DATA_FILE_PATH = "/kaggle/input/most-visited-websites-by-hierachycal-categories/Alexa Rank By Categories.json"



fp = open(DATA_FILE_PATH, "r")

my_dict = json.load(fp)

fp.close()



macro_categories = list(my_dict["topsites"]["category"]["Top"].keys())



for k in range(len(macro_categories)):

    print(str(k + 1).zfill(2) + " - " + macro_categories[k])
most_popular_in_sports = my_dict["topsites"]["category"]["Top"]["Sports"]["_Most_popular_"]



for k in range(len(most_popular_in_sports)):

    print(str(k + 1).zfill(2) + " - " + most_popular_in_sports[k])
random_subject = my_dict["topsites"]["category"]["Top"]["Sports"]["Soccer"]["UEFA"]["Italy"]["Clubs"]["Milan"]



print("Most famous sites about Italian soccer team Milan: ")

for k in range(len(random_subject)):

    print(str(k + 1).zfill(2) + " - " + random_subject[k])