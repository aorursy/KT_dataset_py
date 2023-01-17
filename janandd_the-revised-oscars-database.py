import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read the entire .csv file into a single dataframe, df0

fl = '../input/database.csv'

df0 = pd.read_csv(fl, sep=',', encoding='ISO-8859-1')
# Awards filed under non-uniform names; have to be cleaned

award0 = df0['Award'].unique()



# Sort the array alphabetically; for ease in cleaning

award1 = np.sort(award0)



# Make a copy of the dataframe

df1 = df0.copy()
# Create series where Year is composed of two years; e.g. 1927/1928

nines = df0['Year'] < '1934'

# In reality, would have liked to impose condition on length of

# df0['Year'] instead of specifying the contentious years



# Extract those years given by nines as a list

years0 = df0.loc[nines, ['Year']].values.tolist()



# Flatten the list of lists into a single list

years1 = [y1 for y0 in years0 for y1 in y0]



# Extract the later year from the consecutive year pair

years2 = years1

for i in range(len(years1)):

    years2[i] = years1[i][-4:]



# Assign years2 to the appropriate Year from rows in df1

df1.loc[nines, ['Year']] = years2



# Convert 'Year' from string to int

df1 = df1.astype({'Year':int})



# Get all the Oscar years as a list

YYYY = np.sort(df1['Year'].unique().tolist())
# In Winner column convert nominees from NaN to 0.0

# Convert datatype of Winner from float to integer

df1.loc[np.isnan(df1['Winner']), ['Winner']] = 0

df1['Winner'] = df1['Winner'].astype(int)
# 'Actor in a Leading Role'

# 'Actor'

df1.loc[df0['Award']=='Actor', ['Award']] = 'Actor in a Leading Role'
# 'Actor in a Leading Role'

# 'Actress'

df1.loc[df0['Award']=='Actress', ['Award']] = 'Actress in a Leading Role'
# 'Director'

# 'Directing' and 'Directing (Dramatic Picture)'

df1.loc[df0['Award']=='Directing', ['Award']] = 'Director'

df1.loc[df0['Award']=='Directing (Dramatic Picture)', ['Award']] = 'Director'
# 'Best Picture'

# 'Outstanding Production', 'Outstanding Picture', 'Best Motion Picture', and 'Outstanding Motion Picture' 

df1.loc[df0['Award']=='Outstanding Production', ['Award']] = 'Best Picture'

df1.loc[df0['Award']=='Outstanding Picture', ['Award']] = 'Best Picture'

df1.loc[df0['Award']=='Best Motion Picture', ['Award']] = 'Best Picture'

df1.loc[df0['Award']=='Outstanding Motion Picture', ['Award']] = 'Best Picture'
# 'Music (Original Score)'

# 'Music (Original Dramatic Score)', 'Music (Music Score of a Dramatic or Comedy Picture)', 

# 'Music (Original Score, for a Motion Picture [Not a Musical])', 'Music (Music Score, Substantially Original)', 'Music (Scoring)', 

# 'Music (Music Score of a Dramatic Picture)', and 'Music (Original Music Score)'

df1.loc[df0['Award']=='Music (Original Dramatic Score)', ['Award']] = 'Music (Original Score)'

df1.loc[df0['Award']=='Music (Music Score of a Dramatic or Comedy Picture)', ['Award']] = 'Music (Original Score)'

df1.loc[df0['Award']=='Music (Original Score, for a Motion Picture ' '[Not a Musical])', ['Award']] = 'Music (Original Score)'

df1.loc[df0['Award']=='Music (Music Score, Substantially Original)', ['Award']] = 'Music (Original Score)'

df1.loc[df0['Award']=='Music (Scoring)', ['Award']] = 'Music (Original Score)'

df1.loc[df0['Award']=='Music (Music Score of a Dramatic Picture)', ['Award']] = 'Music (Original Score)'

df1.loc[df0['Award']=='Music (Original Music Score)', ['Award']] = 'Music (Original Score)'
# 'Writing (Story)            # this one is no longer used

# Change 'Writing (Motion Picture Story)' and 'Writing (Original Motion Picture Story)'

df1.loc[df0['Award']=='Writing (Motion Picture Story)', ['Award']] = 'Writing (Story)'

df1.loc[df0['Award']=='Writing (Original Motion Picture Story)', ['Award']] = 'Writing (Story)'
# 'Writing (Original Screenplay)'

# 'Writing (Original Motion Picture Story)', 'Writing (Screenplay, Original)', 'Writing (Screenplay Written Directly for the Screen)', 

# 'Writing (Story and Screenplay Written Directly for the Screen)', 'Writing (Story and Screenplay, Written Directly for the Screen)', 

# 'Writing (Screenplay Written Directly for the Screen, Based on Factual Material or on Story Material Not Previously Published or Produced)', 

# 'Writing (Story and Screenplay, Based on Factual Material or Material Not Previously Published or Produced)', and 

# 'Writing (Story and Screenplay, Based on Material Not Previously Published or Produced)', 

df1.loc[df0['Award']=='Writing (Screenplay, Original)', ['Award']] = 'Writing (Original Screenplay)'

df1.loc[df0['Award']=='Writing (Story and Screenplay)', ['Award']] = 'Writing (Original Screenplay)'

df1.loc[df0['Award']=='Writing (Screenplay Written Directly for the Screen)', ['Award']] = 'Writing (Original Screenplay)'

df1.loc[df0['Award']=='Writing (Story and Screenplay Written Directly for the Screen)', ['Award']] = 'Writing (Original Screenplay)'

df1.loc[df0['Award']=='Writing (Story and Screenplay, Written Directly for the Screen)', ['Award']] = 'Writing (Original Screenplay)'

df1.loc[df0['Award']=='Writing (Screenplay Written Directly for the Screen, Based on Factual Material or on Story Material ' \

            'Not Previously Published or Produced)', ['Award']] = 'Writing (Original Screenplay)'

df1.loc[df0['Award']=='Writing (Story and Screenplay, Based on Factual Material or Material Not Previously Published or Produced)', \

            ['Award']] = 'Writing (Original Screenplay)'

df1.loc[df0['Award']=='Writing (Story and Screenplay, Based on Material Not Previously Published or Produced)', \

            ['Award']] = 'Writing (Original Screenplay)'
# 'Writing (Adapted Screenplay)'

# 'Writing', 'Writing (Adaptation)', 'Writing (Screenplay, Adapted)', 'Writing (Screenplay Adapted from Other Material)', 

# 'Writing (Screenplay Based on Material Previously Produced or Published)', 'Writing (Screenplay Based on Material from Another Medium)', 

# 'Writing (Screenplay, Based on Material from Another Medium)', and 'Writing (Screenplay)' 

df1.loc[df0['Award']=='Writing', ['Award']] = 'Writing (Adapted Screenplay)'

df1.loc[df0['Award']=='Writing (Adaptation)', ['Award']] = 'Writing (Adapted Screenplay)'

df1.loc[df0['Award']=='Writing (Screenplay, Adapted)', ['Award']] = 'Writing (Adapted Screenplay)'

df1.loc[df0['Award']=='Writing (Screenplay Adapted from Other Material)', ['Award']] = 'Writing (Adapted Screenplay)'

df1.loc[df0['Award']=='Writing (Screenplay Based on Material Previously Produced or Published)', ['Award']] = 'Writing (Adapted Screenplay)'

df1.loc[df0['Award']=='Writing (Screenplay, Based on Material Previously Produced or Published)', ['Award']] = 'Writing (Adapted Screenplay)'

df1.loc[df0['Award']=='Writing (Screenplay Based on Material from Another Medium)', ['Award']] = 'Writing (Adapted Screenplay)'

df1.loc[df0['Award']=='Writing (Screenplay, Based on Material from Another Medium)', ['Award']] = 'Writing (Adapted Screenplay)'

df1.loc[df0['Award']=='Writing (Screenplay)', ['Award']] = 'Writing (Adapted Screenplay)'
# The significant categories of Award

categ = ['Actor in a Leading Role', 'Actress in a Leading Role', 'Actor in a Supporting Role', 'Actress in a Supporting Role', \

            'Director', 'Best Picture', 'Writing (Original Screenplay)', 'Writing (Adapted Screenplay)']

# short forms for the above award categories

cateS = ['OLR', 'ELR', 'OSR', 'ESR', 'Dir', 'Pic', 'WOS', 'WAS']
# The name of the film is 'M*A*S*H' and not 'MA*S*H'

df1.loc[(df1['Year']==1970) & (df1['Name'].str.contains('M\*?(?=A)\*?')), ['Name']] = 'M*A*S*H'



# Sally Kellerman did not win Actress in a Supporting Role for M*A*S*H ... www.oscars.org

# M*A*S*H did not win Best Picture ... www.oscars.org

# Robert Altman did not win Director for M*A*S*H ... www.oscars.org

df1.loc[(df1['Year']==1970) & (df1['Award']=='Actress in a Supporting Role') & (df1['Name']=='Sally Kellerman'), ['Winner']] = 0

df1.loc[(df1['Year']==1970) & (df1['Award']=='Best Picture') & (df1['Name']=='M*A*S*H'), ['Winner']] = 0

df1.loc[(df1['Year']==1970) & (df1['Award']=='Director') & (df1['Name']=='M*A*S*H'), ['Winner']] = 0
# Best Picture

# For years 1928-1929, Film contains name of the movie and Name contains name of the producer, rightly so.

# For all subsequent years, it is reversed.

cat_film = df1.loc[(df1['Award']=='Best Picture') & (df1['Year']>1929)]['Film']

cat_name = df1.loc[(df1['Award']=='Best Picture') & (df1['Year']>1929)]['Name']

df1.loc[(df1['Award']=='Best Picture') & (df1['Year']>1929), ['Film']] = cat_name

df1.loc[(df1['Award']=='Best Picture') & (df1['Year']>1929), ['Name']] = cat_film
# Director

# For years 1928-1930, Film contains name of the movie and Name contains name of Director, rightly so. 

# For all subsequent years, this is reversed.

cat_film = df1.loc[(df1['Award']=='Director') & (df1['Year']>1930)]['Film']

cat_name = df1.loc[(df1['Award']=='Director') & (df1['Year']>1930)]['Name']

df1.loc[(df1['Award']=='Director') & (df1['Year']>1930), ['Film']] = cat_name

df1.loc[(df1['Award']=='Director') & (df1['Year']>1930), ['Name']] = cat_film
# Writing (Original Screenplay)

# Film and Name are reversed for all years

cat_film = df1.loc[df1['Award']=='Writing (Original Screenplay)']['Film']

cat_name = df1.loc[df1['Award']=='Writing (Original Screenplay)']['Name']

df1.loc[df1['Award']=='Writing (Original Screenplay)', ['Film']] = cat_name

df1.loc[df1['Award']=='Writing (Original Screenplay)', ['Name']] = cat_film
# Writing (Adapted Screenplay)

# Film and Name are reversed for all years

cat_film = df1.loc[df1['Award']=='Writing (Adapted Screenplay)']['Film']

cat_name = df1.loc[df1['Award']=='Writing (Adapted Screenplay)']['Name']

df1.loc[df1['Award']=='Writing (Adapted Screenplay)', ['Film']] = cat_name

df1.loc[df1['Award']=='Writing (Adapted Screenplay)', ['Name']] = cat_film
# Inspect whether more than one (same as not equal to 1) awards 

# were given for any category in any year

print('Year --- Award category with more than 1 Winner')

for yy in YYYY:

    dfyy = df1.loc[df1['Year']==yy]['Award']

    b1 = np.sort(dfyy.unique().tolist())

    #

    for j, wrd in enumerate(b1):

        dfww = df1[(df1['Year']==yy) & (df1['Award']==wrd)]['Winner'].tolist()

        if (sum(dfww) > 1):

            if (wrd in categ):

                print(yy, '---', wrd)
# Finally, after all the cleaning up, we make a new .csv file

df1.to_csv(path_or_buf='database_clean.csv')