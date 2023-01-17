import pandas as pd
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
data=pd.read_csv('../input/NA Candidate List.csv')
parties=pd.read_csv('../input/parties.csv')
candidates_count=data.groupby('Constituency_title').count()
candidates_count.Party.mean()
data.groupby(['Constituency_title'])['Seat'].count().nlargest(10)
data.groupby(['Constituency_title'])['Seat'].count().nsmallest(10)
def resolveParty(x):
    max=0
    match=''
    for i in parties['Name of Political Party']:
        r=fuzz.ratio(x,i)
        if r>max:
            max=r
            match=i
    return match
extracted_parties=data.Party.apply(lambda x: resolveParty(x))
data.Party=extracted_parties
seats=data.groupby(['Party'])['Seat'].count()
party_position=seats.nlargest(11)
party_position
party_position.iloc[1:].plot(kind='bar')
plt.show()
seats.nsmallest(10)
