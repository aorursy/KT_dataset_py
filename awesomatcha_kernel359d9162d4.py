import numpy as np
import pandas as pd
import json
import itertools as it
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
#import winsound

all_sets = pd.read_json("AllSets-x.json", orient="index")
all_sets.cards = all_sets.cards.apply(lambda x: pd.read_json(json.dumps(x),orient="records"))

setSizeCol=all_sets.apply(lambda x: x.cards.shape[0],axis=1)
all_sets = all_sets.assign(setSize = setSizeCol)

invalid_sets = ["UGL", "UNH","pCEL", "pHHO", "VAN"]
def remover(set0,set1):
    print("Before: "+str(set0.shape))
    for thing in set1:
        print(thing)
        set0=set0.drop(index=thing)
    print("After: " + str(set0.shape))
    return(set0)

def test_invalid_setcode(s,invalid_sets):
    for setname in invalid_sets:
        if s==setname:
            return True
    return False
all_sets = all_sets.loc[~all_sets.code.map(lambda x: test_invalid_setcode(x,invalid_sets))]

card_layouts = ["double-faced", "flip", "leveler", "meld", "normal", "split"]
all_sets.cards = all_sets.cards.apply(lambda x: x.loc[x.layout.map(lambda y: y in card_layouts)])
all_sets.cards = all_sets.cards.apply(lambda x: x.loc[x.layout.map(lambda y: y!=["Conspiracy"])])

def fix_pts(c):
    col_list = list(c.columns)
    if "power" in col_list and "toughness" in col_list:
        c.loc[:,"power"] = pd.to_numeric(c.loc[:,"power"],errors ="coerce")
        c.loc[:,"toughness"] = pd.to_numeric(c.loc[:,"toughness"], errors = "coerce")
    return c
all_sets.cards = all_sets.cards.apply(lambda x: fix_pts(x))

cols_to_remove = ["multiverseid", "imageName", "border", "mciNumber", "foreignNames", "originalText", "originalType", "source"]

all_sets.cards = all_sets.cards.apply(lambda x: x.loc[ :, list(set(x.columns) - set(cols_to_remove))])
bla=[]
for thing in all_sets.cards.apply(lambda x: x.columns):
    bla = list(set().union(bla,thing))
bla.sort()
all_cards_columns = bla

all_cards = pd.DataFrame(data = None, columns = all_cards_columns)
all_cards.rename_axis("name", inplace =True)
def convert_printings(x, set_name):
    x["printings"] = dict.fromkeys(x["printings"])
    x["printings"].update({set_name : x["rarity"]})
    
    return x

def convert_row(row):
    row["cards"] = row["cards"].apply(lambda x: convert_printings(x, row['code']),axis=1).set_index("name")
    
    return row

def filter_columns(row,all_cards_cols):
    set_cols=list(row.columns)
    intersection = list(set(set_cols)& set(all_cards_cols))
    
    return row.filter(intersection)

only_cards = all_sets.apply(lambda x: convert_row(x), axis = 1)["cards"]
only_cards = only_cards.apply(lambda x: filter_columns(x,all_cards_columns))
all_cards = pd.concat(list(only_cards))

all_cards = all_cards.loc[~(all_cards.printings.map(lambda x: bool(set(invalid_sets) &set(x)))
                           & all_cards.supertypes.map(lambda x: x != ["Basic"]))]
def merge_dicts(dicts):
    merged_dicts = {}
    for d in dicts:
        for k, v in d.items():
            if bool(v):
                merged_dicts.update({k:v})             
    return merged_dicts

for cardname in all_cards.index.unique():
    reprints = all_cards.loc[cardname]
    print(cardname)
    if type(reprints) == pd.core.frame.DataFrame:
        merged_dicts = merge_dicts((reprints.printings))
        reprints.iat[0,list(reprints.columns).index("printings")].update(merged_dicts)
        
all_cards = all_cards[~all_cards.index.duplicated(keep="first")]

cmcs = all_cards.loc[:,"cmc"].dropna()
def plot_hist(df_, title, x_axis, y_axis, fig_x, fig_y):
    df = df_.dropna()
    num_bins = len(np.unique(df.values))
    
    fig, ax = plt.subplots(figsize = (fig_x, fig_y))
    
    n, bins, patches = ax.hist(df, num_bins, normed = True)
    
    df_mean = df.mean()
    df_std = df.std()
    y= mlab.normpdf(bins, df_mean, df_std)
    
    ax.plot(bins, y, "--")
    ax.set_ylabel(y_axis)
    ax.set_xlabel(x_axis)
    ax.set_title(title)
    plt.text(10,0.20,"mean = " + str(round(df_mean, 5)))
    plt.text(10,0.18,"stdev = " + str(round(df_std, 5)))
    
    fig.tight_layout()
    plt.show()
def plotter(jack, title = "your card set"): 
    plot_int_hist(jack.cmc,title = "Distribution of Converted Mana Cost - " + title, x_axis = "CMC", y_axis="Percentage", fig_x=12, fig_y=8)


def ptheat(cardset):
    print("Mean P: " + str(np.average(cardset.power.mean())))
    print("Mean T: " + str(np.average(cardset.toughness.mean())))
    print("StDev T: " + str(np.std(cardset.loc[:,"toughness"])))
    print("StDev P: " + str(np.std(cardset.loc[:,"power"])))
    lm_pt_cmc = cardset.loc[:,["power","toughness","cmc"]]
    lm_pt_cmc = lm_pt_cmc.loc[lm_pt_cmc.power.notnull() | lm_pt_cmc.toughness.notnull()]
    hm.hist2d(lm_pt_cmc.power, lm_pt_cmc.toughness, bins = np.arange(-1.5, np.max(cardset.loc[:,"power"] + 1.5)), range = ((-1,  np.max(cardset.loc[:,"power"] + 2)), (-1, 16)),
          cmap = "summer" , norm = matplotlib.colors.LogNorm())

    hm.set_xlabel("Power")
    hm.set_ylabel("Toughness")
    hm.set_xticks(np.arange(-1, np.max(cardset.loc[:,"power"] + 1)))
    hm.set_yticks(np.arange(-1, np.max(cardset.loc[:,"toughness"] + 1)))
    hm.set_title("Power/Toughness Heatmap")




def subtypeget(cardset,sub="Trap"):
    allthis = cardset.loc[cardset.subtypes.notnull()]
    allthis = allthis.loc[allthis.subtypes.map(lambda x: sub in x)]
    print(list(allthis.index))
    return allthis

def typesget(cardset,types="Creature"):
    allthis = cardset.loc[cardset.types.notnull()]
    allthis = allthis.loc[allthis.types.map(lambda x: types in x)]
    print(list(allthis.index))
    return allthis

subtypeget(all_cards,"Goblin")
def CreatureMusic(cardlist):
    ptlist = cardlist.power.add(cardlist.toughness, fill_value=0)
    ptlist = ptlist.fillna(0)
    ptlist = list(ptlist)
    length=(cardlist.cmc)
    length = length.fillna(0)
    length = list(length)
    maxi = 60
    mini = -2
    player(ptlist,maxi,mini,length)
    
def player(listed,maxi,mini,length):
    ind=0
    for thing in listed: 
        noot = int(((thing)/(maxi-mini))*5920+32)
        winsound.Beep(noot, length[ind] * 100 +25)
        ind +=1
        print(noot,length[ind], thing)
def CreatureMusicM(cardlist, accel=False):
    ptlist = cardlist.power.add(cardlist.toughness, fill_value=0)
    ptlist = ptlist.fillna(0)
    ptlist = list(ptlist)
    length=(cardlist.cmc)
    length = length.fillna(0)
    length = list(length)
    maxi = 60
    mini = -2
    listed= [196,220,247,262,294,330,349,392,440,494,523,587,659,698,784,880,988,1047,1175,1319,1397,1568,1760,1976,2093, 2349,2637,2794,3136, 3322,3520,3729,3951,4186,4435]
    lsted = []
    for bop in ptlist:
        lsted.append(listed[int(bop)])
    playermap(lsted,maxi,mini,length, accel)
    
def playermap(listed,maxi,mini,length, accel=False):
    meter = 8
    ind=0
    print(meter)
    if accel ==True:
        for ind, thing in enumerate(listed): 
            noot = int(thing)
            winsound.Beep(noot, int((length[ind] / meter * 500 + 500/meter/2)/2))
    else:    
        for ind, thing in enumerate(listed): 
            noot = int(thing)
            winsound.Beep(noot, int((length[ind] / meter * 500 + 500/meter/2)))
drgon = CreatureMusicM(subtypeget(all_cards,"Dragon"))
train = all_cards.loc[:, ["text",'cmc'] ]

train.head(3)


train['text']=train['text'].fillna("")
train.text.isnull().values.any()
print(train.text.head())
train['text']=train['text'].apply(lambda x: " ".join(x.splitlines()))
print(train.text.head())


train['wordcount'] = train['text'].apply(lambda x: len(str(x).split(" ")))
train['charcount'] = train['text'].str.len()

def avg_word(sentence):
    words = sentence.split()
    if len(words) == 0:
        ans =0
    else:
        ans=sum(len(word)for word in words)/len(words)
    return(ans)

train['averageword'] = train['text'].apply(lambda x: avg_word(x))
def getto(y):
    g = y
    if y.charcount == 0:
        y.wordcount = 0
    return y
train=train.apply(lambda x: getto(x), axis=1)
train.loc[train.wordcount ==0].count()
train['text'] = train['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

train['text']=train['text'].str.replace('[^\w\s]','')

from nltk.corpus import stopwords
stop = stopwords.words('english')

train['stopwords'] = train['text'].apply(lambda x: len([x for x in x.split() if x in stop]))

train['SpecialChar'] = train['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

train['numerics'] = train['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

train['upper'] = train['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
frequents = pd.Series(''.join(train['text']).split()).value_counts()[:15]
rarewords = pd.Series(''.join(train['text']).split()).value_counts()[-15:]
freq = list(frequents.index)
rare = list(rarewords.index)
train2=train
train2['text'] = train2['text'].apply(lambda x:' '.join(x for x in x.split() if ((x not in freq) or (x not in rare)) ))


'''from textblob import TextBlob

train['text'][:5].apply(apply(lambda x: str(TextBlob(x).correct())))'''
Eldrazic = CreatureMusicM(subtypeget(all_cards,"Eldrazi"))




    




















