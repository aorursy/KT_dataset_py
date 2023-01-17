import unicodedata

import re



def removeURL(text):

    text = re.sub("http\\S+\\s*", "", text)

    return text



# 'NFKD' is the code of the normal form for the Unicode input string.

def remove_accentuation(text):

    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII','ignore')

    return text.decode("utf-8")



def remove_punctuation(text):

    # re.sub(replace_expression, replace_string, target)

    new_text = re.sub(r"\.|,|;|!|\?|\(|\)|\[|\]|\$|\:|\\|\/", "", text)

    return new_text



def remove_numbers(text):

    # re.sub(replace_expression, replace_string, target)

    new_text = re.sub(r"[0-9]+", "", text)

    return new_text



# Conver a text to lower case

def lower_case(text):

    return text.lower()
# Remove stop words from a text

from nltk.corpus import stopwords

nltk_stop = set(stopwords.words('portuguese'))

#for word in ["2018","claro","oi","tim","vivo","dia","e","pois","r$"]:

#    nltk_stop.add(word)



def remove_stop_words(text, stopWords=nltk_stop):

    for sw in stopWords:

        text = re.sub(r'\b%s\b' % sw, "", text)

        

    return text
def pre_process(text):

    new_text = lower_case(text)

    new_text = removeURL(new_text)

    new_text = remove_stop_words(new_text)

    new_text = remove_numbers(new_text)

    new_text = remove_punctuation(new_text)

    new_text = remove_accentuation(new_text)

    return new_text
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD

from datetime import datetime



file_co = "consumidor.tsv"

file_ra = "reclameAqui.tsv"



df_co = pd.read_csv("../input/oi-complaints/"+file_co, sep="\t", encoding="utf-8")#, na_values=['-'])

df_ra = pd.read_csv("../input/oi-complaints/"+file_ra, sep="\t", encoding="utf-8")#, na_values=['-'])



df_co.drop(['Empresa'], axis=1, inplace=True)

#df_co['Nota'] = df_co['Nota'].astype(float)



df_ra.drop(['title', 'hash', 'userreply', 'status', 'wouldBuyAgain'], axis=1, inplace=True)

df_ra.rename(columns={'dataTime':'Data','citystate':'Cidade','score':'Nota','complaint':'Relato','companyreply':'Resposta','finalreply':'Avaliacao'},inplace=True)



df_co = df_co[['Data', 'Cidade', 'Nota', 'Relato']]#, 'Resposta', 'Avaliacao'

df_ra = df_ra[['Data', 'Cidade', 'Nota', 'Relato']]#, 'Resposta', 'Avaliacao'

df_ra['Data'] = df_ra['Data'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S').strftime('%d/%m/%Y') )

df_co.head()
df_ra.head()
# ### Removing registers without Note



#df_co.drop(df_co[ df_co['Nota'] == '-' ].index, inplace=True)#df_co_validNote = 

#df_ra.drop(df_ra[ df_ra['Nota'] == '-' ].index, inplace=True)#df_ra_validNote = 

#df_ra = df_ra.dropna()

#df_co = df_co.dropna()



# ### Merging Note ranging



#df_ra['Nota'] = df_ra['Nota'].astype(float)



# #df_ra['Nota'] = round(df_ra['Nota']/2)

# #df_ra['Nota'] = df_ra['Nota'].astype(int)



# #OldRange = (OldMax - OldMin)  

# #NewRange = (NewMax - NewMin)  

# #NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin



#df_ra['Nota'] = df_ra['Nota'].apply( lambda x: round( ((x * 4) / 10) + 1 ) )

#df_ra['Nota'] = df_ra['Nota'].astype(int)



#df_co['Nota'] = df_co['Nota'].astype(int)



#df_ra.head()

#df_co.head()



# ### Counting the amount of Notes given



#import matplotlib.pyplot as plt

#plt.hist(dataset['Nota'])

#plt.xlabel('Nota')

#plt.ylabel('Quantidade')

#plt.show()





#df_all = pd.DataFrame(pd.concat([df_co, df_ra]))

#df_samples = df_all.iloc[:100,:]

#df_samples["Relato"] = df_samples["Relato"].apply(pre_process)

#df_samples.head()
import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



df_all = pd.DataFrame(pd.concat([df_co, df_ra]))

#df_samples = df_all.iloc[:100,:]

df_samples = df_all



df_samples["num_of_words"] = df_samples["Relato"].apply(lambda x: len(str(x).split())) # Generate the column with amounts of words



count_str = df_samples["num_of_words"].value_counts() # Getting the pairs (values and amounts)

count_str = count_str[ count_str.values > 1 ] # Considering only those with amount >1



#Plot it!

plt.figure(figsize=(12,6))

sns.barplot(count_str.index, count_str.values, alpha=0.8, color=color[0])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Number of words in the complaint', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
print( "Lista de valores num_of_words que incidem em mais de um registro (sem preprocessamento): \n" , count_str.index.tolist() )
#print(count_str.values)

#print(count_str.index)

#print(count_str.index.tolist())

#print(3.5 in range(1,10))

#print([i for i in count_str.index.tolist()])

#print(df_samples.loc[count_str.index.tolist(),:])

#print(count_str.index.tolist())

#print(25 in count_str.index.tolist())



df_show = df_samples[ df_samples["num_of_words"].isin( count_str.index.tolist() ) ]

df_show = df_show[ ["Relato", "num_of_words"] ]#.drop(['Nota'], axis=1, inplace=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    print(df_show)
# Now a second graph with the amount counted with the preprocessed texts:

df_preprocessed = df_samples

df_preprocessed["Relato"] = df_preprocessed["Relato"].apply(pre_process)



df_preprocessed["num_of_words"] = df_preprocessed["Relato"].apply(lambda x: len(str(x).split()))



count_str_preprocessed = df_preprocessed["num_of_words"].value_counts()

count_str_preprocessed = count_str_preprocessed[ count_str_preprocessed.values > 1 ]



#Plot it!

plt.figure(figsize=(12,6))

sns.barplot(count_str_preprocessed.index, count_str_preprocessed.values, alpha=0.8, color=color[0])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Number of words in the pre processed complaint', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()

print( "Lista de valores num_of_words que incidem em mais de um registro (com preprocessamento): \n" , count_str_preprocessed.index.tolist() )
df_show = df_preprocessed[ df_preprocessed["num_of_words"].isin( count_str_preprocessed.index.tolist() ) ]

df_show = df_show[ ["Relato", "num_of_words"] ]#.drop(['Nota'], axis=1, inplace=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    print(df_show)
df_show = df_samples[ df_samples["num_of_words"] == 38 ]

df_show = df_show[ ["Relato", "num_of_words"] ]#.drop(['Nota'], axis=1, inplace=True)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    print(df_show)

    

df_samples[ df_samples["num_of_words"] == 38 ]