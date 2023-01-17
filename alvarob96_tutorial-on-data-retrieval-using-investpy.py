import investpy

import pandas as pd



import random



from pprint import pprint
# This function retrieves all the available equities indexed on es.Investing.com

available_equities = investpy.get_equities_list()

pprint(available_equities)
# As a test, you can either pick a random one or just pass as parameter a specific one

random_equity = random.choice(available_equities)

print(random_equity)
selected_equity = 'bbva'



# Once we select an equity, we can get its recent historical data or specify a range of 

# time to retrieve historical data from using the following functions respectively. In this

# case we will be retrieving the information as a pandas.DataFrame, but we can also retrieve

# it as a json file, for more information type in terminal: help(get_recent_data) or

# help(get_historical_data)



# Retrieved values are displayed in Euros (€) as it is the currency used in Spain
recent_df = investpy.get_recent_data(equity=selected_equity,

                                     as_json=False,

                                     order='ascending')

print(recent_df.head())
historical_df = investpy.get_historical_data(equity=random_equity,

                                             start='01/01/2010',

                                             end='01/01/2019',

                                             as_json=False,

                                             order='ascending')

print(historical_df.head())
# The Company Profile from an equity can also be retrieved using investpy,

# the equity name to retrieve the data from and the language (spanish or english)

# need to be specified as it follows.

company_profile = investpy.get_equity_company_profile(equity=selected_equity, language='english')

pprint(company_profile)