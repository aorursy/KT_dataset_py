import advertools as adv

import pandas as pd



cx = 'YOUR_CUSTOM_SEARCH_ENGINE'

key = 'YOUR_KEY'

adv.__version__
wikipedia_urls = ['https://www.wikipedia.org/',  # search for this domain as a keyword

                  'https://www.wikipedia.org/wrong_page',  # search for this domain as a keyword (does not exist)

                  'site:https://www.wikipedia.org/', # search for this site/page (exists)

                  'site:https://www.wikipedia.org/wrong_again'] # search for this site/page (does not exist)
# wikipedia = adv.serp_goog(cx=cx, key=key, q=wikipedia_urls)
wikipedia = pd.read_csv('../input/wikipedia_serps.csv')

wikipedia[['searchTerms', 'rank', 'title', 'displayLink', 'formattedTotalResults']]