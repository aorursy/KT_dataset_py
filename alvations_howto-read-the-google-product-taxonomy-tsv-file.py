import pandas as pd
df = pd.read_csv('../input/google-product-taxonomy.tsv', sep='\t', header=None)

df = df.rename(columns={0:'lang', 1:'idx', 2:'date', 3:'category'})
df.head()
code2lang = {

 'cs-CZ': 'Czech',

 'da-DK': 'Danish',

 'de-CH': 'Swiss German',

 'de-DE': 'Germany German',

 'en-AU': 'Australian English',

 'en-GB': 'British English',

 'en-US': 'American English',

 'es-ES': 'Spanish Spanish',

 'fr-CH': 'Swiss French',

 'fr-FR': 'France French',

 'it-CH': 'Swiss Italian',

 'it-IT': 'Italy Italian',

 'ja-JP': 'Japanese',

 'nl-NL': 'Dutch',

 'no-NO': 'Norwegian',

 'pl-PL': 'Polish',

 'pt-BR': 'Brazillian Portuguese',

 'ru-RU': 'Russian',

 'sv-SE': 'Swedish',

 'tr-TR': 'Turkish',

 'zh-CN': 'Chinese'

}
en_us = df[df['lang'] == 'en-US']

en_us.head()
zh_cn = df[df['lang'] == 'zh-CN']

zh_cn.head()
df2 = pd.merge(en_us, zh_cn, on='idx', how='outer').dropna()

df2.head()
en2zh = {}

for idx, row in df2.iterrows():

    en_cat, zh_cat = row['category_x'], row['category_y']

    for en_word, zh_word in zip(en_cat.split(' > '), zh_cat.split(' > ')):

        en2zh[en_word] = zh_word
en2zh