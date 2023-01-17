import pandas as pd

from unidecode import unidecode
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')



kagg = pd.concat([train, test], sort=False)

kagg.head()
wiki1 = pd.read_html('https://en.wikipedia.org/w/index.php?title=Passengers_of_the_RMS_Titanic&oldid=883859055', header=0, encoding='unicode')[1]

wiki2 = pd.read_html('https://en.wikipedia.org/w/index.php?title=Passengers_of_the_RMS_Titanic&oldid=883859055', header=0, encoding='unicode')[2]

wiki3 = pd.read_html('https://en.wikipedia.org/w/index.php?title=Passengers_of_the_RMS_Titanic&oldid=883859055', header=0, encoding='unicode')[3]
wiki1.head()
wiki1['Class'] = 1

wiki1.head()
wiki2.head()
wiki2['Class'] = 2

wiki2.head()
wiki3.head()
wiki3['Hometown'] = wiki3['Hometown'] + ', ' + wiki3['Home country']

wiki3 = wiki3.drop('Home country', axis=1, errors='ignore')



wiki3['Class'] = 3



wiki3.head()
wiki = pd.concat([wiki1, wiki2, wiki3], ignore_index=True)

wiki.head()
kagg.sort_values(['Pclass', 'Name']).reset_index(drop=True).head()
kagg_corr = kagg.copy()



# Sorting by class and name

kagg_corr = kagg_corr.sort_values(['Pclass', 'Name']).reset_index(drop=True)



# Extracting surnames and names using regular expressions

temp = kagg_corr.Name.str.extract(r'(?P<Surname>.*), Mrs\. (?P<Husband_name>.*)\((?P<Wife_name>.*)\)')

temp2 = kagg_corr.Name.str.extract(r'(?P<Surname>.*), (?P<Title>.*)\. (?P<Name>.*)')



# Adding Kaggle surname codes

surname = temp.Surname

surname2 = temp2.Surname

surname = surname.fillna(surname2)

surname = surname.str.title()

surname_code = surname.str[0:3]

kagg_corr['Surname_code'] = surname_code



# Adding Kaggle name codes

name = temp.Wife_name

name2 = temp2.Name

name = name.fillna(name2)

name = name.str.title()

name_code = name.str[0:3]

kagg_corr['Name_code'] = name_code



kagg_corr.head()
wiki_corr = wiki.copy()



# Adding WikiId

wiki_corr.insert(0, 'WikiId', wiki_corr.index + 1)



# Correcting a typo

wiki_corr.loc[339, 'Name'] = 'Beane, Mrs. Ethel (née Clarke)'



# Replacing -- with NaN

wiki_corr = wiki_corr.replace('–', float('nan'))



# Extracting surnames and names using regular expressions

temp = wiki_corr.Name.str.extract(r'and (?P<Profession>.*), (?P<Title>.*?)\.* (?P<Name>.*) (?P<Surname>.*)')

temp2 = wiki_corr.Name.str.extract(r'(?P<Surname>.*), (?P<Title>.*?)\.* (?P<Name>.*)')



# Adding Wikipedia surname codes

surname = temp.Surname

surname2 = temp2.Surname

surname = surname.fillna(surname2)

surname = surname.str.title()

surname = surname.apply(unidecode)

surname_code = surname.str[0:3]

wiki_corr['Surname_code'] = surname_code



# Adding Wikipedia name codes

name = temp.Name

name2 = temp2.Name

name = name.fillna(name2)

name = name.str.title()

name = name.apply(unidecode)

name_code = name.str[0:3]

wiki_corr['Name_code'] = name_code



# Converting age type to float

months = wiki_corr.Age.str.extract(r'(?P<Months>\d*) mo.', expand=False).astype('float64')

age = months / 12.0

age = age.fillna(wiki_corr.Age)

wiki_corr['Age'] = age.astype('float64').round(2)



# Adding Wikipedia suffixes

wiki_corr = wiki_corr.rename(columns={'Name': 'Name_wiki', 'Age': 'Age_wiki', 'Surname_code': 'Surname_code_wiki', 'Name_code': 'Name_code_wiki'})



wiki_corr.head()
merg = pd.merge(kagg_corr, wiki_corr, left_on=['Surname_code', 'Name_code', 'Age'],

                right_on=['Surname_code_wiki', 'Name_code_wiki', 'Age_wiki'])



def merge_report(df, kagg):

    dupl = df.PassengerId.duplicated(keep=False)

    dupl2 = df.WikiId.duplicated(keep=False)

    dupl_num = (dupl | dupl2).sum()

    print(f'''Matched: {df.shape[0]} ({dupl_num} duplicates)

Unmatched: {kagg.shape[0] - df.shape[0]}

Total: {kagg.shape[0]}''')

    

merge_report(merg, kagg_corr)

merg.head()
def dupl_drop(df):

    df_corr = df.drop_duplicates(subset=['PassengerId'], keep=False)

    df_corr = df_corr.drop_duplicates(subset=['WikiId'], keep=False)

    return df_corr



merg_corr = dupl_drop(merg)



merge_report(merg_corr, kagg_corr)
def df_rest(df, kagg, wiki):

    kagg_rest = kagg[~kagg.PassengerId.isin(df.PassengerId)].copy()

    wiki_rest = wiki[~wiki.WikiId.isin(df.WikiId)].copy()

    return kagg_rest, wiki_rest



kagg_rest, wiki_rest = df_rest(merg_corr, kagg_corr, wiki_corr)



merg_rest = pd.merge(kagg_rest, wiki_rest, left_on=['Surname_code', 'Name_code'],

                     right_on=['Surname_code_wiki', 'Name_code_wiki'])

merg_rest = dupl_drop(merg_rest)



merge_report(merg_rest, kagg_rest)

merg_rest.head()
kagg_rest2, wiki_rest2 = df_rest(merg_rest, kagg_rest, wiki_rest)



merg_rest2 = pd.merge(kagg_rest2, wiki_rest2, left_on=['Surname_code', 'Age'],

                      right_on=['Surname_code_wiki', 'Age_wiki'])

merg_rest2 = dupl_drop(merg_rest2)



merge_report(merg_rest2, kagg_rest2)

merg_rest2.head()
kagg_rest3, wiki_rest3 = df_rest(merg_rest2, kagg_rest2, wiki_rest2)



merg_rest3 = pd.merge(kagg_rest3, wiki_rest3, left_on=['Surname_code'],

                      right_on=['Surname_code_wiki'])

merg_rest3 = dupl_drop(merg_rest3)



merge_report(merg_rest3, kagg_rest3)

merg_rest3.head()
kagg_rest4, wiki_rest4 = df_rest(merg_rest3, kagg_rest3, wiki_rest3)



kagg_rest4 = kagg_rest4.sort_values('Name')

wiki_rest4 = wiki_rest4.sort_values('Name_wiki')



kagg_rest4.to_csv('kagg_rest4.csv', index=False)

wiki_rest4.to_csv('wiki_rest4.csv', index=False)
wiki_id_match = [622,float('nan'),822,661,662,671,672,670,669,667,852,1188,960,311,

                 697,698,853,696,741,717,720,float('nan'),1002,402,789,791,float('nan'),

                 float('nan'),798,1205,603,722,804,319,802,859,981,434,879,961,1202,

                 1308,902,908,612,629,989,948,float('nan'),1021,999,1000,1053,1006,

                 1008,1027,float('nan'),1045,1046,1044,1311,1057,1055,1056,519,1072,

                 1071,1075,1082,1084,782,1085,229,668,1139,702,703,float('nan'),1137,

                 1138,552,555,553,312,float('nan'),183,1063,1181,1182,1190,1189,1191,

                 1199,1222,1223,270,275,1250,1248,1249,1059,1265,893,602,604,181,1310,

                 1291,1309,607,884,885,354]



kagg_rest4_corr = kagg_rest4.reset_index(drop=True).copy()

kagg_rest4_corr['WikiId'] = wiki_id_match

kagg_rest4_corr.head()
merg_rest4 = pd.merge(kagg_rest4_corr, wiki_rest4, on=['WikiId'])



merge_report(merg_rest4, kagg_rest4)

merg_rest4.head()
merg_all = pd.concat([merg_corr, merg_rest, merg_rest2, merg_rest3, merg_rest4],

                     ignore_index=True)



merge_report(merg_all, kagg_corr)
kagg_rest5, wiki_rest5 = df_rest(merg_all, kagg_corr, wiki_corr)



kagg_rest5
wiki_rest5
wiki_corr[wiki_corr.Name_wiki.str.contains('Peters')]
merg_all[merg_all.WikiId == 1128]
merg_all_corr = merg_all[(merg_all.PassengerId != 1146) &

                         (merg_all.PassengerId != 569) &

                         (merg_all.PassengerId != 534) &

                         (merg_all.PassengerId != 919) &

                         (merg_all.PassengerId != 508)]



kagg_rest5_corr, wiki_rest5_corr = df_rest(merg_all_corr, kagg_corr, wiki_corr)



kagg_rest5_corr.loc[:, 'WikiId'] = float('nan')

kagg_rest5_corr.loc[kagg_rest5_corr.PassengerId == 147, 'WikiId'] = 1293

kagg_rest5_corr.loc[kagg_rest5_corr.PassengerId == 1146, 'WikiId'] = 980

kagg_rest5_corr.loc[kagg_rest5_corr.PassengerId == 6, 'WikiId'] = 785

kagg_rest5_corr.loc[kagg_rest5_corr.PassengerId == 681, 'WikiId'] = 1128

kagg_rest5_corr.loc[kagg_rest5_corr.PassengerId == 534, 'WikiId'] = 701

kagg_rest5_corr.loc[kagg_rest5_corr.PassengerId == 569, 'WikiId'] = 750

kagg_rest5_corr.loc[kagg_rest5_corr.PassengerId == 919, 'WikiId'] = 1203

kagg_rest5_corr.loc[kagg_rest5_corr.PassengerId == 508, 'WikiId'] = 41



merg_rest5 = pd.merge(kagg_rest5_corr, wiki_rest5_corr, on=['WikiId'])

merg_rest5
merg_all2 = pd.concat([merg_all_corr, merg_rest5], ignore_index=True)



merge_report(merg_all2, kagg_corr)
kagg_rest6, wiki_rest6 = df_rest(merg_all2, kagg_corr, wiki_corr)

kagg_rest6
wiki_rest6
merg_all3 = pd.concat([merg_all2, kagg_rest6], ignore_index=True, sort=False)



merge_report(merg_all3, kagg_corr)
merg_all3_corr = merg_all3.drop(['Surname_code', 'Name_code', 'Surname_code_wiki', 'Name_code_wiki'], axis=1)

merg_all3_corr = merg_all3_corr.sort_values('PassengerId').reset_index(drop=True)

merg_all3_corr.head()
full = merg_all3_corr.copy()



train = full[:891]



test = full[891:]

test = test.drop('Survived', axis=1)
full.to_csv('full.csv', index=False)

train.to_csv('train.csv', index=False)

test.to_csv('test.csv', index=False)