'''

This is the third homework of NLP course.



The aim of the homework is the classifying the documents into defined classes. 

They will be classified by keywords of each class.

There are 100 documents with 5 different class(topic)

These are the classes(topics)

'''

# 0 - 20 Business

# 21 - 40 Entertainment

# 41 - 60 Politics

# 61 - 80 Sport

# 81 - 100 Tech
import pandas as pd

from collections import Counter

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer
# Followings are the keywords of the classes(topics)
sport = ["competition", "game", "tennis", "basketball", "player", "team", "football", "referee", "baseball", "position",

         "volleyball", "coach", "defence", "kick", "competitive", "sportsman", "event", "draw", "manager", "quarter"]
entertainment = ["treat", "entertain", "amusement", "show", "entertaining", "festival", "film", "movie", "book",

                 "theater", "award", "artist", "actor", "audience", "role", "novel", "concert", "musical", "hit",

                 "cinema"]
politics = ["election", "vote", "political", "politician", "liberal", "liberalism", "president", "party", "public",

            "country", "government", "democratic", "official", "protesters", "terror", "office", "demonstration", "voters",

            "civil", "law"]
business = ["buyer", "company", "tax", "economic", "economy", "dollar", "business", "profit", "euro", "finance",

            "market", "bank", "economist", "marketing", "exchange", "growth", "customer", "price", "bid", "money"]
tech = ["technology", "download", "click", "software", "digital", "tech", "robotics", "computer", "game", "video",

        "phone", "e-mail", "application", "algorithm", "bug", "website", "web", "mobile", "laptop", "code"]



subject = pd.DataFrame({'Business': business, 'Entertainment': entertainment, 'Politics': politics, 'Sport': sport,

                        'Technology': tech})

display(subject)
# Take the 100 documents from input

filenames = ['../input/001.txt', '../input/002.txt', '../input/003.txt', '../input/004.txt', '../input/005.txt', '../input/006.txt',

             '../input/007.txt', '../input/008.txt', '../input/009.txt', '../input/010.txt', '../input/011.txt', '../input/012.txt',

             '../input/013.txt', '../input/014.txt', '../input/015.txt', '../input/016.txt', '../input/017.txt', '../input/018.txt',

             '../input/019.txt', '../input/020.txt', '../input/021.txt', '../input/022.txt', '../input/023.txt', '../input/024.txt',

             '../input/025.txt', '../input/026.txt', '../input/027.txt', '../input/028.txt', '../input/029.txt', '../input/030.txt',

             '../input/031.txt', '../input/032.txt', '../input/033.txt', '../input/034.txt', '../input/035.txt', '../input/036.txt',

             '../input/037.txt', '../input/038.txt', '../input/039.txt', '../input/040.txt', '../input/041.txt', '../input/042.txt',

             '../input/043.txt', '../input/044.txt', '../input/045.txt', '../input/046.txt', '../input/047.txt', '../input/048.txt',

             '../input/049.txt', '../input/050.txt', '../input/051.txt', '../input/052.txt', '../input/053.txt', '../input/054.txt',

             '../input/055.txt', '../input/056.txt', '../input/057.txt', '../input/058.txt', '../input/059.txt', '../input/060.txt',

             '../input/061.txt', '../input/062.txt', '../input/063.txt', '../input/064.txt', '../input/065.txt', '../input/066.txt',

             '../input/067.txt', '../input/068.txt', '../input/069.txt', '../input/070.txt', '../input/071.txt', '../input/072.txt',

             '../input/073.txt', '../input/074.txt', '../input/075.txt', '../input/076.txt', '../input/077.txt', '../input/078.txt',

             '../input/079.txt', '../input/080.txt', '../input/081.txt', '../input/082.txt', '../input/083.txt', '../input/084.txt',

             '../input/085.txt', '../input/086.txt', '../input/087.txt', '../input/088.txt', '../input/089.txt', '../input/090.txt',

             '../input/091.txt', '../input/092.txt', '../input/093.txt', '../input/094.txt', '../input/096.txt', '../input/097.txt',

             '../input/098.txt', '../input/099.txt', '../input/100.txt']





listOfColumnNames = list(subject) # Class names as a list

count = 0

mod = 0

change = 0

for fname in filenames:  # Take files one by one

    if mod % 20 == 0:

        print("\n\nThe class must be " + listOfColumnNames[change] + "\n\n")

        change += 1

    count += 1

    file_content = open(fname).read()  # Open and read file

    lower_text = file_content.lower()

    tokenizer = RegexpTokenizer(r'\w+')  # Remove the punctuations

    tokens = tokenizer.tokenize(file_content) # Tokenize

    stop_words = set(stopwords.words('english'))

    filtered_tokens = [w for w in tokens if not w in stop_words]  # Remove the stopwords

    counted_file = dict(Counter(filtered_tokens))

    counted_df = pd.DataFrame.from_dict(counted_file, orient='index').reset_index()

    counted_df = counted_df.rename(columns={counted_df.columns[0]: 'Words', counted_df.columns[1]: 'Frequencies'})

    my_list = []

    for k in subject.columns:

        hold = 0

        a = 0

        for row in counted_df.index:



            for i in range(20):

                if counted_df["Words"][row] == subject[k][i]:  # Search for each word in the document and count

                    hold = hold + counted_df["Frequencies"][row] 

                    # aa = counted_df[words].iloc[select_indices]

            # print(hold)



        my_list.append(hold) 



    print("File number ", count, " is ", listOfColumnNames[my_list.index(max(my_list))]) # Take the maximum element's index and find it from the names of classes

    mod += 1