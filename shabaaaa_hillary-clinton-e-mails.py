%matplotlib inline
#RUDIMENTARY SENTIMENT ANALYSIS
# W I P

from textblob import TextBlob
import pandas as pd
import sqlite3
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

con = sqlite3.connect('../input/database.sqlite')

allmail = pd.read_sql_query("""
        SELECT * FROM Emails
        """, con)
#Some countries involved in Libyan crisis
COIs = ['Libya', 'Israel', 'Syria', 'Iran', 'Saudi Arabia', 'Algeria', 
        'Qatar', 'Turkey', 'Russia', 'United Kingdom', 'United States']

#Selecting only mails whose body contains at least one country from the list above 
filtered = []
for mail in allmail['ExtractedBodyText']:
    if any(word in mail for word in COIs):
        filtered.append(mail)
    else:
        filtered.append('')

#Get e-mails sent only by Hillary with countries above
mail = allmail[allmail['ExtractedBodyText'].isin(filtered)]
mail = mail[mail['MetadataFrom'] == 'H']

print("How many e-mails were left?")
print(mail['ExtractedBodyText'].size)
#Get text sentiments
empolarity = []
emsubject = []
for row in mail['ExtractedBodyText']:
    toput = TextBlob(row)
    empolarity.append(toput.sentiment.polarity)
    emsubject.append(toput.sentiment.subjectivity)

mail['Polarity'] = empolarity
mail['Subjectivity'] = emsubject
# Top three most (positively) polarized bodies of text
top3p = mail.sort_values(by = 'Polarity', ascending = False).head(3)[['MetadataDateSent', 'Polarity', 'Subjectivity', 'ExtractedBodyText', 'RawText']]

print("The topmost positive e-mail is as follows:")
print("'" + top3p['ExtractedBodyText'].iloc[0] + "'")
print("It has a polarity of {0}".format(top3p['Polarity'].iloc[0]) + ",")
print("with a subjectivity of {0}".format(top3p['Subjectivity'].iloc[0]) + ",")
print("but the statement is more neutral than anything else.")
print("The text in its original form shows this.")
print("'" + top3p['RawText'].iloc[0] + "'")
# Top three most (negatively) polarized bodies of text
top3n = mail.sort_values(by = 'Polarity', ascending = True).head(3)[['MetadataDateSent', 'Polarity', 'Subjectivity', 'ExtractedBodyText', 'RawText']]

print("The topmost negative e-mail is as follows:")
print("'" + top3n['ExtractedBodyText'].iloc[0] + "'")
print("It has a polarity of {0}".format(top3n['Polarity'].iloc[0]) + ",")
print("with a subjectivity of {0}".format(top3n['Subjectivity'].iloc[0]) + ".")
# Datetime objects
mail['MetadataDateSent'] = pd.to_datetime(mail['MetadataDateSent'])
mail = mail.set_index('MetadataDateSent')

# 0 for Monday, 6 for Sunday
mail['dayofweek'] = mail.index.dayofweek
sns.set_style('white')
t_labels = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
ax = sns.barplot(x=np.arange(0,7), y=mail.groupby('dayofweek').Polarity.mean(),\
     label=t_labels, palette="Spectral")
sns.despine(offset=10)
ax.set_xticklabels(t_labels)
ax.set_ylabel('Polarity')
ax.set_title('HRC\'s Sent Emails')
plt.savefig('polarity.png', bbox='tight')


