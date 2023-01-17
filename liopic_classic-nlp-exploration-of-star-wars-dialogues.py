# Basic data loading
import pandas as pd

df = pd.read_table('../input/SW_EpisodeIV.txt',
                   delim_whitespace=True, header=0, escapechar='\\')
df.sample(10)
df.shape
df.groupby('character').count().sort_values('dialogue', ascending=False)
def character_group(name: str) -> str:
    rebel = ('BASE VOICE', 'CONTROL OFFICER', 'MAN', 'PORKINS', 'REBEL OFFICER', 'RED ELEVEN',
             'RED TEN', 'RED SEVEN', 'RED NINE', 'RED LEADER', 'BIGGS', 'GOLD LEADER',
             'WEDGE', 'GOLD FIVE', 'REBEL', 'DODONNA', 'CHIEF', 'TECHNICIAN', 'WILLARD',
             'GOLD TWO', 'MASSASSI INTERCOM VOICE')
    imperial = ('CAPTAIN', 'CHIEF PILOT', 'TROOPER', 'OFFICER', 'DEATH STAR INTERCOM VOICE',
                'FIRST TROOPER', 'SECOND TROOPER', 'FIRST OFFICER', 'OFFICER CASS', 'TARKIN',
                'INTERCOM VOICE', 'MOTTI', 'TAGGE', 'TROOPER VOICE', 'ASTRO-OFFICER',
                'VOICE OVER DEATH STAR INTERCOM', 'SECOND OFFICER', 'GANTRY OFFICER', 
                'WINGMAN', 'IMPERIAL OFFICER', 'COMMANDER', 'VOICE')
    neutral = ('WOMAN', 'BERU', 'CREATURE', 'DEAK', 'OWEN', 'BARTENDER', 'CAMIE', 'JABBA',
               'AUNT BERU', 'GREEDO', 'NEUTRAL', 'HUMAN', 'FIXER')

    if name in rebel:
        return 'rebels'
    elif name in imperial:
        return 'imperials'
    elif name in neutral:
        return 'neutrals'
    else:
        return name


df['character'] = df['character'].apply(character_group)
df.groupby('character').count().sort_values('dialogue', ascending=False)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(max_df=0.1, max_features=200, stop_words='english')

features = tfidf_vec.fit_transform(df.dialogue)
X = pd.DataFrame(data=features.toarray(), 
                 index=df.character, 
                 columns=tfidf_vec.get_feature_names())
X.sample(10)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

df_reduced = pd.DataFrame(X_reduced)
df_reduced['character'] = X.index
df_reduced.head(10)
import matplotlib.pyplot as plt
%matplotlib inline


def character_to_color(name: str):
    color = {'LUKE': 'b', 'HAN': 'b', 'THREEPIO': 'b', 'BEN': 'b', 'LEIA': 'b',
             'VADER': 'r', 'imperials': 'm', 'rebels': 'c', 'neutrals': 'k'}
    return color[name]


df_reduced['color'] = df_reduced['character'].apply(character_to_color)

plt.figure(figsize=(10, 10))
plt.scatter(x=df_reduced[0], y=df_reduced[1],
            color=df_reduced['color'], alpha=0.5)
df_reduced[(df_reduced[0]>0.1) & (df_reduced[1]>0.55) & (df_reduced[1]<0.6)]
df.loc[714]