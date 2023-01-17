import numpy as np

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# date parsing function

parser = lambda x: pd.to_datetime(x, format='%m/%d/%y %H:%M:%S')
# load data

df = pd.read_csv('../input/wowah_data.csv', parse_dates=[' timestamp'], date_parser=parser)
# group logs by character

avatars = df.groupby('char')



# number of unique characters

len(avatars)
# count logs per characters

log_number = avatars.count()



# number of characters with a single log

len(avatars.filter(lambda x: len(x) == 1))
# clean data from single logs

df = avatars.filter(lambda x: len(x) > 1)



# number of remaining avatars

avatars = df.groupby('char')

len(avatars)
races = avatars[' race'].unique().value_counts()

races.head(n=10)
# let's look at this avatar

avatars.get_group(65856).sort_index()[' race'].value_counts()
# clean characters with multiple races

df = df.groupby('char').filter(lambda x: len(x[' race'].unique()) == 1)

df.groupby('char')[' race'].unique().value_counts().head(n=10)