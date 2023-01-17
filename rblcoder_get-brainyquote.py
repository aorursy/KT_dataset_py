import pandas as pd
!git clone https://github.com/rblcoder/brainscrape.git
import brainscrape.brainscrape as quotescrape
insp_quotes = quotescrape.getQuotes('inspiration', 2)

# for q, author in insp_quotes:

#     print(q)

#     print(author)



confucius_quotes = quotescrape.getQuotesByAuthor('confucius', 2)

# for q, author in confucius_quotes:

#     print(q)

#     print(author)
df_insp_quotes = pd.DataFrame(insp_quotes, columns = ['quote', 'author']) 



df_insp_quotes 
df_insp_quotes['key_word'] = 'inspiration'
df_insp_quotes.to_csv('quotes_by_keyword'+'.csv')
!ls
!cat 'quotes_by_keyword.csv'
df_confucius_quotes = pd.DataFrame(confucius_quotes, columns = ['quote', 'author']) 



df_confucius_quotes

df_confucius_quotes.to_csv('quotes_by_author'+'.csv')
!ls
!cat quotes_by_author.csv
!rm -rf brainscrape
!ls