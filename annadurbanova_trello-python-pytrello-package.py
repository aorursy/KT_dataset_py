# Author: Anna Durbanova & Nicholas Del Grosso

# Date: 18.09.2020
!pip install py-trello

## Learn more here: https://pypi.org/project/py-trello/
import pandas as pd

from trello import TrelloClient ## Py-Trello Package

import random
client = TrelloClient(

    api_key='API KEY', ## about 32-characters long (see the #1)

    token='TOKEN', ## about 50-60 characters long (see the #1)

) 
all_boards = client.list_boards() ## Access all boards in your Trello

mss=all_boards[7] ## Choose the one you need to access

mss ## Print
allcards=mss.get_cards() ## Access all cards in your boards

allcards
first_card=allcards[0] ## Acess the first card

print(first_card)
all_dfs = []

for card in mss.get_cards(): ## Loop every card in all cards

    for checklist in card.checklists: ## Loop through all checklists

        df = pd.DataFrame(checklist.items) ## Create a DataFrame with checklist items

        all_dfs.append(df) ## Append the empty data frame

        

checklist_items = pd.concat(all_dfs) ## Put together

checklist_items.head() ## Show
item_names = checklist_items.name.tolist() ## Convert to the list

item_names[:10] ## Show first 10
problems=[]

for card in allcards:

    problems.append(card.name[9:])

problems
sentences = item_names + problems

assert len(sentences) == len(item_names) + len(problems) ## Check if we have everything in the list

sentences[:10] ## Show first 10 sentences
random.shuffle(sentences) ## Shuffle the order

print(sentences[:10])
third=len(sentences)//3+1

third
big_list = [sentences[:third], sentences[third:third * 2], sentences[third * 2:]]

big_list;
data=pd.DataFrame(big_list).T

data
data.to_excel("Sentences.xlsx", header=False, index=False)