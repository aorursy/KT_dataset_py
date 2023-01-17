import json

import pandas



with open('/kaggle/input/gfootball-playerdata/playerdata.json') as file:

    dictionary = json.load(file)



dataframe = pandas.json_normalize(dictionary)
dataframe.iloc[:,11:18]
dataframe.iloc[:,18:27]
dataframe.iloc[:,27:]
dataframe.iloc[:,:11]