import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
cannabis = pd.read_csv("../input/cannabis.csv")
cannabis.info()
top_strains = cannabis.Rating >= 4.8
cannabis[top_strains].sort_values('Rating', ascending=False)[:50]
fig, ax = plt.subplots(figsize=(9, 9))
shap = cannabis.Type.value_counts(dropna=False)
labels = 'Hybrid','Indica','Sativa'
explode = (0.09, 0.09, 0.09)
ax.pie(shap, explode=explode, labels=labels, autopct='%1.1f%%', colors=['blue','purple','red'], shadow=True)
plt.title('Types of strains')
plt.show()
def get_effects(dataframe):
    ret_dict = {}
    for list_ef in dataframe.Effects:
        effects_list = list_ef.split(',')
        for effect in effects_list:
            if not effect in ret_dict:
                ret_dict[effect] = 1
            else:
                ret_dict[effect] += 1
    return ret_dict

def plot_effects(cannabis_effects, strain_type):
    fig, ax = plt.subplots(figsize=(10, 10))
    shap = list(cannabis_effects.values())[:50]
    labels = list(cannabis_effects.keys())[:50]
    ax.pie(shap, labels=labels, autopct='%1.1f%%', shadow=True)
    plt.title('top effects of {} type of strain'.format(strain_type))
    plt.show()

hybrids = cannabis[cannabis.Type == 'hybrid']
indicas = cannabis[cannabis.Type == 'indica']
sativas = cannabis[cannabis.Type == 'sativa']

hybrid_effects = get_effects(hybrids)
indica_effects = get_effects(indicas)
sativa_effects = get_effects(sativas)

plot_effects(hybrid_effects, 'Hybrid')
plot_effects(indica_effects, 'Indica')
plot_effects(sativa_effects, 'Sativa')
def get_flavors(dataframe):
    ret_dict = {}
    for list_ef in dataframe.Flavor.dropna():
        flavors_list = list_ef.split(',')
        for flavor in flavors_list:
            if not flavor in ret_dict:
                ret_dict[flavor] = 1
            else:
                ret_dict[flavor] += 1
    return ret_dict

def plot_flavors(cannabis_flavors, strain_type):
    fig, ax = plt.subplots(figsize=(10, 10))
    shap = list(cannabis_flavors.values())[:50]
    labels = list(cannabis_flavors.keys())[:50]
    ax.pie(shap, labels=labels, autopct='%1.1f%%', shadow=True)
    plt.title('top flavors of {} type of strain'.format(strain_type))
    plt.show()

hybrids = cannabis[cannabis.Type == 'hybrid']
indicas = cannabis[cannabis.Type == 'indica']
sativas = cannabis[cannabis.Type == 'sativa']

hybrid_flavors = get_flavors(hybrids)
indica_flavors = get_flavors(indicas)
sativa_flavors = get_flavors(sativas)

plot_flavors(hybrid_flavors, 'Hybrid')
plot_flavors(indica_flavors, 'Indica')
plot_flavors(sativa_flavors, 'Sativa')
wordcld = pd.Series(cannabis.Description.tolist()).astype(str)
stopwords = ('with', 'and', 'the', 'this', 'it', 'an', 'of', 'in', 'or', 'like', 'that', 'a', 'but')
cloud = WordCloud(width=1250, height=900,
                  stopwords=stopwords, 
                  colormap='hsv').generate(''.join(wordcld.astype(str)))
plt.figure(figsize=(15, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()
