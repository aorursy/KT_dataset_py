import pandas as pd



df = pd.read_csv('/kaggle/input/list-of-roman-emperors-from-wikipedia/romans 2.csv')

print(f"Total number of emperors: {df.shape[0]}")

print(f"Total number of raw features: {df.shape[1]}")

print(f"Features available: {list(df.columns)}")
df.head()
# Make all columns lowercase except the names

for col in df.columns:

    if col != "Name":

        df[col] = df[col].str.lower()

    

# Drop Succession column

if 'Succession' in df:

    df = df.drop(['Succession'], axis=1)



# Extract age of death and replace missing values

death_age = df['Deaths'].str.extract('\((.*?)\)').apply(lambda x: x.str.strip())

df['AgeOfDeath'] = death_age[0].astype(str).map(lambda x: x.lstrip('aged ').strip())



df['AgeOfDeath'].replace(to_replace ="?", value ="0", inplace=True) 

df['AgeOfDeath'].replace(to_replace ="nan", value ="0", inplace=True) 



# Extract Reign in years

df['ReignYears'] = df['Time'].str.extract('(\d+).(?=years|year)').fillna(0)



# Extract Reign in months

df['ReignMonths'] = df['Time'].str.extract('(\d+).(?=months|month)').fillna(0)



# Extract Reign in days

df['ReignDays'] = df['Time'].str.extract('(\d+).(?=days|day)').fillna(0)



# Extract birth name

df['BirthName'] = df['Name'].str.extract('(.[a-z]+)')



# Extract title

df['Title'] = df['Name'].str.extract('(.[A-Z]{2,})')



# Extract cause of death

df['CauseOfDeath'] = df['Deaths'].str.extract('\).(.*)').fillna('Unknown')

df['CauseOfDeath'] = df['CauseOfDeath'].apply(lambda x: x.replace('  ', ' ').strip())



# Extract birth year (caring for AD/BC)

df['BirthYearBC'] = df['Birth'].str.extract('(\d+.(?=bc))').fillna(0).astype(str)

df['BirthYearAD'] = df['Birth'].str.extract('(\d+.(?=,|ad))').fillna(0).astype(str)

df['BirthYearNum'] = df['Birth'].str.extract('(\d+(?:\.\d+)?)').fillna(0).astype(str)

df['BirthYearAD'] = df['BirthYearAD'].apply(lambda x: x.replace(',', '').replace('/', '')).astype(int)



def add_year_prefix(row):

    year_bc = int(row['BirthYearBC'])

    if year_bc > 0:

        return -year_bc

    else:

        return int(row['BirthYearAD'])

    

def add_num_year(row):

    if row['BirthYear'] is 0:

        return row['BirthYearNum']

    else:

        return row['BirthYear']

    

df['BirthYear'] = df.apply(add_year_prefix, axis=1)

df['BirthYear'] = df.apply(add_num_year, axis=1)

df = df.drop(['BirthYearNum', 'BirthYearBC', 'BirthYearAD'], axis=1)



# Extract birth day

df['BirthDay'] = df['Birth'].str.extract('(\d+)').fillna(0)



# Extract birth month

df['BirthMonth'] = df['Birth'].str.split(',', 1).str[0].str.extract('([a-zA-Z]{5,})').fillna('Unknown')

df['BirthMonth'] = df['BirthMonth'].str.capitalize()



# Extract reign start and end year (manual handle the BC case)

df['ReignStartYear'] = df['Reign'].str.split('–', 1).str[0].str.extract('(\d+.(?=ad|bc)|\d{4})')

df['ReignEndYear'] = df['Reign'].str.split('–', 1).str[1].str.extract('(\d+.(?=ad|bc)|\d{4})')

df.at[0, 'ReignStartYear'] = int('-' + df['ReignStartYear'].iloc[0].strip())



# Extract reign day

df['ReignStartDay'] = df['Reign'].str.split('–', 1).str[0].str.extract('(\d+).(?=[a-zA-Z])')

df['ReignEndDay'] = df['Reign'].str.split('–', 1).str[1].str.extract('(\d+).(?=[a-zA-Z])')



# Extract reign start month

df['ReignStartMonth'] = df['Reign'].str.split('–', 1).str[0].str.extract('([a-zA-Z]+)')

df['ReignStartMonth'] = df['ReignStartMonth'].str.capitalize()



# Extract reign end month

df['ReignEndMonth'] = df['Reign'].str.split('–', 1).str[1].str.extract('([a-zA-Z]+)')

df['ReignEndMonth'] = df['ReignEndMonth'].str.capitalize()



# Drop handled columns

df = df.drop(['Name', 'Birth', 'Time', 'Deaths', 'Reign'], axis=1)
# Show new df



df.head()
# Print a few causes of death



for i in range(0,10,2):

    print(f'{df.BirthName.loc[i]} died of {df.CauseOfDeath.loc[i]}')
# Look at the most occuring causes



import spacy

from collections import Counter



nlp = spacy.load("en_core_web_sm")



complete_text = df.CauseOfDeath.str.strip().str.cat(sep=' ')

complete_doc = nlp(complete_text)



words = [token.text for token in complete_doc

         if not token.is_stop and not token.is_punct]

word_freq = Counter(words)



# Print 20 most common words

common_words = word_freq.most_common(20)

print (f'20 most common words: \n{common_words}\n')



# Print all unique words

unique_words = [word for (word, freq) in word_freq.items() if freq == 1]

print (f'Unique words in death notes: \n{unique_words}\n')