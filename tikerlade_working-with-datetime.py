import numpy as np

import pandas as pd
# Creating DataFrame

data = pd.DataFrame(columns = ['Date'])



# Adding some data

data = data.append({'Date':'Jan-2013'}, ignore_index=True)

data = data.append({'Date':'Mar-2017'}, ignore_index=True)

data = data.append({'Date':'Feb-2011'}, ignore_index=True)

data = data.append({'Date':'Sep-2016'}, ignore_index=True)



# Show data

data.head()
# Let's show, that our column data type is not simething related to data

data.info()
# Changes

data['Date'] = pd.to_datetime(data['Date'])



# Now our column data type is datetime

data.info()
# Final changes

data['Month'] = data['Date'].apply(lambda x: x.month)

data['Year'] = data['Date'].apply(lambda x: x.year)



# Remove useless stuff and show data

data = data.drop(['Date'], axis = 1)

data.head()