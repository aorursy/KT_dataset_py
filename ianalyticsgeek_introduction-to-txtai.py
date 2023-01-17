# ! pip install txtai
# import the required packages and function

import numpy as np

from txtai.embeddings import Embeddings

embed = Embeddings({"method": "transformers", "path": "sentence-transformers/bert-base-nli-mean-tokens"})
# Please provide your own API in the below input text box:

import wandb

api = wandb.Api()



# you can register for free on https://wandb.ai/ 

# and then can find unique API under "settings"
# some random examples

TxtSections = ["India is on second spot with close to million confirmed corona virus cases",

               "US President Donald Trump on Sunday declared himself immune from Covid-19 as he prepares to return to the campaign trail in a fight to regain ground against surging White House rival Joe Biden",

               "If climate change was a somewhat abstract notion a decade ago, today it is all too real for Californians fleeing wildfires and smothered in a blanket of smoke, the worst year of fires on record",

               "Apple Reaches $2 Trillion. Apple is the first U.S. company to hit that value, a staggering ascent that began in the pandemic",

               "Firefighters in Australia are trying to take advantage of cooler and damper weather to slow the spread of the devastating fires",

               "Oscar for Parasite, the film received Academy Awards for best picture and best director, the honors set off cheers and a burst of pride in a country fearful of being overlooked"]
TxtSections
print("%-20s %s" % ("Input Query", "Best Match"))

print("-" * 100)



for input_query in ("film awards", "happy story","tech news","weather report","health","tragedy", "asia"):

    # Get index of best text section that best matches the input query

    indx = np.argmax(embed.similarity(input_query, TxtSections))

    # print the input query and the best match

    print("%-20s %s" % (input_query, TxtSections[indx]))

# Create an index for the list of sections

embed.index([(idx, text, None) for idx, text in enumerate(TxtSections)])



print("%-20s %s" % ("Input Query", "Best Match"))

print("-" * 100)





# Run an embeddings search for each query

for input_query in ("film awards", "happy story","tech news","weather report","health","tragedy", "asia"):

    # Extract uid of first result

    # search result format: (uid, score)

    indx = embed.search(input_query, 1)[0][0]

    

    # print the input query and the best match

    print("%-20s %s" % (input_query, TxtSections[indx]))

    

# save the index

embed.save("index")



# view

!ls
# load the index

embeddings = Embeddings()

embeddings.load("index")
indx = embeddings.search("climate change", 1)[0][0]

print(TxtSections[indx])