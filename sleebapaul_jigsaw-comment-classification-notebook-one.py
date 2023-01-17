# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_PATH = "/kaggle/input/jigsaw-multilingual-toxic-comment-classification/"

os.listdir(DATA_PATH)

TEST_PATH = DATA_PATH + "test.csv"

VAL_PATH = DATA_PATH + "validation.csv"

TRAIN_PATH = DATA_PATH + "jigsaw-toxic-comment-train.csv"



val_data = pd.read_csv(VAL_PATH)

test_data = pd.read_csv(TEST_PATH)

train_data = pd.read_csv(TRAIN_PATH)
train_data.head()
val_data.head()
test_data.head()
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt 



def clean_comment(x):

    if type(x) == str:

        return x.replace("\n", " ")

    else:

        return ""
text = train_data.apply(lambda row: clean_comment(row["comment_text"]), axis=1).to_string()

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1200, stopwords=set(STOPWORDS)).generate(text)

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
text = val_data.apply(lambda row: clean_comment(row["comment_text"]), axis=1).to_string()

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1200, stopwords=set(STOPWORDS)).generate(text)

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
text = test_data.apply(lambda row: clean_comment(row["content"]), axis=1).to_string()

wordcloud = WordCloud(max_font_size=None, background_color='black', collocations=False,

                      width=1200, height=1200, stopwords=set(STOPWORDS)).generate(text)

plt.figure(figsize = (7, 7), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
!apt-get install -y python-numpy libicu-dev



!pip install pyicu==2.3.1



!pip install morfessor



!pip install pycld2

from polyglot.detect import Detector

from polyglot.detect.langids import isoLangs



def detect_lang(comment):

    return Detector("".join(x for x in comment if x.isprintable()), quiet=True).languages[0].name

    

def get_lang_name(code):

    try:

        return isoLangs[code]["name"]

    except:

        return "Unknown" 
train_data["lang_code"] = train_data.apply(lambda row: detect_lang(row["comment_text"]), axis=1)

train_data["lang_name"] = train_data["lang_code"].apply(lambda row: get_lang_name(row))
print("\n------ Training data -------\n")

print("-> Unknown language code: ", train_data["lang_code"].loc[

    train_data["lang_name"] == "Unknown"].unique())

print("-> Unknown language examples count: ",train_data["lang_code"].loc[

    train_data["lang_name"] == "Unknown"].count() )

print("-> Unknown language value counts: \n", train_data["lang_code"].loc[

    train_data["lang_name"] == "Unknown"].value_counts())
print("\n------ Validation data -------\n")

  

val_data["lang_name"] = val_data["lang"].apply(lambda row: get_lang_name(row))

print("-> Unknown language code: ", val_data["lang"].loc[

    val_data["lang_name"] == "Unknown"].unique())

print("-> Unknown language examples count: ",val_data["lang"].loc[

    val_data["lang_name"] == "Unknown"].count() )

print("-> Unknown language value counts: \n", val_data["lang"].loc[

    val_data["lang_name"] == "Unknown"].value_counts())
print("\n------ Testing data -------\n")

  

test_data["lang_name"] = test_data["lang"].apply(lambda row: get_lang_name(row))

print("-> Unknown language code: ", test_data["lang"].loc[

    test_data["lang_name"] == "Unknown"].unique())

print("-> Unknown language examples count: ",test_data["lang"].loc[

    test_data["lang_name"] == "Unknown"].count() )

print("-> Unknown language value counts: \n", test_data["lang"].loc[

    test_data["lang_name"] == "Unknown"].value_counts())
train_eng_count = train_data["lang_name"].loc[train_data["lang_name"] == "English"].count()

train_non_eng_count = train_data["lang_name"].loc[~train_data["lang_name"].isin(["English", "Unknown"])].count()



val_eng_count = val_data["lang_name"].loc[val_data["lang_name"] == "English"].count()

val_non_eng_count = val_data["lang_name"].loc[~val_data["lang_name"].isin(["English", "Unknown"])].count()



test_eng_count = test_data["lang_name"].loc[test_data["lang_name"] == "English"].count()

test_non_eng_count = test_data["lang_name"].loc[~test_data["lang_name"].isin(["English", "Unknown"])].count()

x = ['English', 'Non-English']

lang_count = [[train_eng_count, train_non_eng_count], [val_eng_count, val_non_eng_count], 

          [test_eng_count, test_non_eng_count]]

lang_count_titles =  [ "Training data", "Validation data", "Testing data"]

x_pos = np.arange(len(x))



fig, ax = plt.subplots(1, 3, figsize=(15,5))



for i, col in enumerate(ax):

    tmp_data = lang_count[i]

    col.bar(x_pos, tmp_data)

#     plt.bar(x_pos, tmp_data)

    for j, v in enumerate(tmp_data):

        col.text(x_pos[j] - 0.1, v + .01, "{:.2f}%".format(v*100/sum(tmp_data)))

    col.set_title(lang_count_titles[i])

    col.set_xlabel("Language")

    col.set_ylabel("Count")

    col.set_xticks(x_pos)

    col.set_xticklabels(x)

    col.plot()





fig.suptitle("Language Count Across Datasets", fontsize = 33)

fig.tight_layout()

fig.subplots_adjust(top=0.8)



fig.show()
train_non_eng_count = train_data["lang_name"][~train_data["lang_name"].isin(

    ['English', 'Unknown'])].value_counts().rename_axis('lang').reset_index(name='lang_count')



val_non_eng_count = val_data["lang_name"][~val_data["lang_name"].isin(

    ['English', 'Unknown'])].value_counts().rename_axis('lang').reset_index(name='lang_count')



test_non_eng_count = test_data["lang_name"][~test_data["lang_name"].isin(

    ['English', 'Unknown'])].value_counts().rename_axis('lang').reset_index(name='lang_count')
df_set = [train_non_eng_count, val_non_eng_count, test_non_eng_count]

non_eng_count_titles =  [ "Training data", "Validation data", "Testing data"]



fig, ax = plt.subplots(1, 3, figsize=(18,5))



for i, col in enumerate(ax):

    tmp_data = df_set[i]

    

    if i == 0:

        mean_count = tmp_data.lang_count.mean()

        idx = tmp_data[tmp_data['lang_count'] > int(mean_count)].lang

        vals = tmp_data[tmp_data['lang_count'] > int(mean_count)].lang_count

    else:

        idx = tmp_data.lang

        vals = tmp_data.lang_count

    col.pie(vals, labels=idx, autopct='%1.1f%%')

    col.axis('equal')

    col.set_title(non_eng_count_titles[i])

    col.plot()



fig.suptitle("Non English Count Across Datasets", fontsize = 33)

fig.tight_layout()

fig.subplots_adjust(top=0.8)

plt.show()
country_map = {

    'Unknown': 'Unknown',

    'English': 'United Kingdom',

    'Azerbaijani': 'Azerbaijan',

    'Basque': 'Spain',

    'Interlingua': 'Italy',

    'German': 'Germany',

    'Volapük': 'Germany',

    'Norwegian Nynorsk': 'Norway',

    'Scots': 'Scotland',

    'Dutch': 'Netherlands',

    'Polish': 'Poland',

    'Greek, Modern': 'Greece',

    'Manx': 'Ireland',

    'Portuguese': 'Portugal',

    'Luganda': 'Uganda',

    'Hungarian': 'Hungary',

    'Kinyarwanda': 'Rwanda',

    'Danish': 'Denmark',

    'Latin': 'Italy',

    'Western Frisian': 'Netherlands',

    'Galician': 'Spain',

    'Italian': 'Italy',

    'Fijian': 'Fiji',

    'Sanskrit (Saṁskṛta)': 'India',

    'Occitan': 'Spain',

    'Xhosa': 'South Africa',

    'Quechua': 'Peru',

    'Welsh': 'Wales',

    'Maltese': 'Malta',

    'Chichewa; Chewa; Nyanja': 'Zimbabwe',

    'Czech': 'Czech Republic',

    'Oromo': 'Ethiopia',

    'Tagalog': 'Philippines',

    'Romansh': 'Switzerland',

    'Turkish': 'Turkey',

    'Irish': 'Ireland',

    'Venda': '	South Africa',

    'Indonesian': 'Indonesia',

    'Icelandic': 'Iceland',

    'Uzbek': 'Uzbekistan',

    'Interlingue': 'Italy',

    'Malagasy': 'Madagascar',

    'Swedish': 'Sweden',

    'Breton': 'France',

    'Somali': 'Somalia',

    'Luxembourgish, Letzeburgesch': 'Luxembourg',

    'Swahili': 'Kenya',

    'Faroese': 'Denmark',

    'Wolof': 'Senegal',

    'Spanish; Castilian': 'Spain',

    'Tsonga': 'Mozambique',

    'Shona': 'Zambia',

    'French': 'France',

    'Tonga (Tonga Islands)': 'Tonga Islands',

    'Southern Sotho': 'South Africa',

    'Norwegian': 'Norway',

    'Lithuanian': 'Lithuania',

    'Malay': 'Malaysia',

    'Tamil': 'India',

    'Corsican': 'France',

    'Japanese': 'Japan',

    'Turkmen': 'Turkmenistan',

    'Scottish Gaelic; Gaelic': 'Scotland',

    'Nauru': 'Nauru',

    'Samoan': 'Samoa',

    'Estonian': 'Estonia',

    'Waray-Waray': 'Philippines',

    'Latvian': 'Latvia',

    'Albanian': 'Albania',

    'Slovak': 'Slovakia',

    'Haitian; Haitian Creole': 'Haiti',

    'Esperanto': 'Brazil',

    'Māori': 'New Zealand',

    'Bulgarian': 'Bulgaria',

    'Sundanese': 'Indonesia',

    'Finnish': 'Finland',

    'Tatar': 'Russia',

    'Afar': 'Ethiopia',

    'Romanian, Moldavian, Moldovan': 'Romania',

    'Chinese': 'China',

    'Tswana': 'South Africa',

    'Zhuang, Chuang': '',

    'Serbian': 'Serbia',

    'Cebuano': 'China',

    'Lingala': 'Democratic Republic of the Congo',

    'Catalan; Valencian': 'Spain',

    'Ukrainian': 'Ukraine',

    'Persian': 'Iran',

    'Marathi (Marāṭhī)': 'India',

    'Guaraní': 'Paraguay',

    'Korean': 'South Korea',

    'Arabic': 'UAE',

    'Bosnian': 'Bosnia',

    'Vietnamese': 'Vietnam',

    'Urdu': 'India',

    'Thai': 'Thailand',

    'Croatian': 'Croatia',

    'Bengali': 'India',

    'Kurdish': 'Iraq',

    'Malayalam': 'India',

    'Hindi': 'India',

    'Macedonian': 'Macedonia',

    'Aymara': 'Bolivia',

    'Afrikaans': 'Australia',

    'Georgian': 'Georgia',

    'Oriya': 'India',

    'Kannada': 'India',

    'Russian': 'Russia',

    'Tibetan Standard, Tibetan, Central': 'Tibet',

    'Gujarati': 'India',

    'Mongolian': 'Mangolia',

    'Khmer': 'Vietnam',

    'Kirundi': 'Tanzania',

    'Nepali': 'Nepal',

    'Sinhala, Sinhalese': '',

    'Burmese': 'Burma',

    'Kalaallisut, Greenlandic': '',

    'Panjabi, Punjabi': 'India',

    'Swati': 'South Africa',

    'Yoruba': 'Nigeria',

    'Kazakh': 'Kazakhstan',

    'Hausa': 'Nigeria',

    'Slovene': 'Slovenia',

    'Tigrinya': 'Ethiopia',

    'Pashto, Pushto': 'Pakistan',

    'Akan': 'Ghana',

    'Telugu': 'India',

    'Bislama': 'Republic of Vanuatu',

    'Igbo': 'Nigeria',

    'Belarusian': 'Belarus'

}
train_data["country"] = train_data.apply(lambda row: country_map[row["lang_name"]], axis=1)

val_data["country"] = val_data.apply(lambda row: country_map[row["lang_name"]], axis=1)

test_data["country"] = test_data.apply(lambda row: country_map[row["lang_name"]], axis=1)
import plotly.express as px
country_data_train = train_data.country.value_counts().rename_axis('country').reset_index(name='count')

fig = px.choropleth(country_data_train.query("country != 'United Kingdom' and country != 'Unknown'"), locations="country", hover_name="country",

                     projection="natural earth", locationmode="country names", title="Training data - geographical distribution", color="count",

                     template="plotly", color_continuous_scale="agsunset")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.2

fig.show()
country_data_val = val_data.country.value_counts().rename_axis('country').reset_index(name='count')

fig = px.choropleth(country_data_val.query("country != 'United Kingdom' and country != 'Unknown'"), locations="country", hover_name="country",

                     projection="natural earth", locationmode="country names", title="Validation data - geographical distribution", color="count",

                     template="plotly", color_continuous_scale="agsunset")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.2

fig.show()
country_data_test = test_data.country.value_counts().rename_axis('country').reset_index(name='count')

fig = px.choropleth(country_data_test.query("country != 'United Kingdom' and country != 'Unknown'"), locations="country", hover_name="country",

                     projection="natural earth", locationmode="country names", title="Test data - geographical distribution", color="count",

                     template="plotly", color_continuous_scale="agsunset")

fig.data[0].marker.line.color = 'rgb(0, 0, 0)'

fig.data[0].marker.line.width = 0.2

fig.show()
import seaborn as sns
train_data['label'] = 0

train_data.loc[(train_data.toxic == 1) | (train_data.obscene == 1) | 

           (train_data.insult == 1) | (train_data.identity_hate == 1) | 

           (train_data.severe_toxic == 1) | (train_data.threat == 1), 'label'] = 1
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7), )

ax = sns.countplot(x="label", data=train_data)

for i, p in enumerate(ax.patches):

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2., height + 0.7,

        "{:.2f}%".format(train_data['label'].value_counts(normalize=True)[i]*100),

            ha="center", rotation=10)

ax.set_xticklabels(['Non-Toxic', 'Toxic'])

ax.set_xlabel("Comments")

ax.set_ylabel("Count")

ax.set_title('Toxic and Non-Toxic comments in training data', fontsize=14)

fig.show()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7), )

ax = sns.countplot(x="toxic", data=val_data)

for i, p in enumerate(ax.patches):

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2., height + 0.7,

        "{:.2f}%".format(val_data['toxic'].value_counts(normalize=True)[i]*100),

            ha="center", rotation=10)

ax.set_xticklabels(['Non-Toxic', 'Toxic'])

ax.set_xlabel("Comments")

ax.set_ylabel("Count")

ax.set_title('Toxic and Non-Toxic comments in validation data', fontsize=14)

fig.show()
comment_cat_df = pd.melt(train_data[['toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']])


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7), )

ax = sns.countplot(data=comment_cat_df[comment_cat_df.value == 1], 

                   x='variable', 

                  order=comment_cat_df[comment_cat_df.value == 1].variable.value_counts().index)

for i, p in enumerate(ax.patches):

    height = p.get_height()

    ax.text(p.get_x() + p.get_width()/2., height + 0.7,

        "{:.2f}%".format(comment_cat_df[comment_cat_df.value==1].variable.value_counts(normalize=True)[i]*100),

            ha="center", rotation=10)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=10, ha="right")

    ax.set_xlabel("Toxic comments")

    ax.set_ylabel("Count")





ax.set_title('Categories within toxic comments', fontsize=14)

fig.show()
train_data.to_csv("notebook_one_train_data.csv", index = False)

val_data.to_csv("notebook_one_val_data.csv", index = False)

test_data.to_csv("notebook_one_test_data.csv", index = False)