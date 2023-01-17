from datetime import datetime 

start_real = datetime.now()

#Importing libraries

import pandas as pd

import numpy as np

import scipy as sci

import seaborn as sns

import matplotlib.pyplot as plt

import multiprocessing

%matplotlib inline
train = pd.read_csv("../input/train.tsv", sep='\t')

test = pd.read_csv("../input/test.tsv", sep='\t')
#Getting rid of outliers

train['bigger_than_200'] = train['price'].map(lambda x: 1 if x >200 else 0)

train = train[train['bigger_than_200'] ==0]

del train['bigger_than_200']
print(train.shape)

print(test.shape)
#Checking any missing values,

import missingno as msno

msno.bar(train,sort=True,figsize=(10,5))

msno.bar(test,sort=True,figsize=(10,5))
#Getting the length of item description

train['length'] = train['item_description'].map(lambda x: len(str(x)))

test['length'] = test['item_description'].map(lambda x: len(str(x)))

#Merging data

data = pd.concat([train,test])

#Defining a variable

data['train_or_not'] = data['train_id'].map(lambda x: 1 if x.is_integer() else 0)
#lowering letters

data['brand_name'] = data['brand_name'].map(lambda x: str(x).lower())

data['category_name'] = data['category_name'].map(lambda x: str(x).lower())

data['item_description'] = data['item_description'].map(lambda x: str(x).lower())

data['name'] = data['name'].map(lambda x: str(x).lower())
data['no_of_words'] = data['item_description'].map(lambda x: len(str(x).split()))
#Nan values in brand

%%time

data['brand_nan'] = data['brand_name'].map(lambda x: 1 if x =="nan" else 0)
##Brand names

#Number of unique brand names

print(len(set(data['brand_name'])))

print('brand_name in train',len(set(train['brand_name'])))

print('brand_name in test',len(set(test['brand_name'])))
train_cat_names= list(set(train['brand_name']))

test_cat_names= list(set(test['brand_name']))



in_test_not_in_train = [x for x in test_cat_names if x not in train_cat_names]

print(len(in_test_not_in_train))



in_train_not_in_test = [x for x in train_cat_names if x not in test_cat_names]

print(len(in_train_not_in_test))
#category

data['categories'] = data['category_name'].map(lambda x: list(str(x).split('/')))
#no descriptions

data['no_description'] = data['item_description'].map(lambda x: 1 if str(x) =='no description yet' else 0)

print(len(data[data['no_description']==1]))
print('brand_name = nan & no description',len(data[(data['brand_name']=='nan') & (data['no_description'] ==1)]))
#No brand name and no desc

no_desc_no_brand = data[(data['brand_name']=='nan') & (data['no_description'] ==1)]

no_desc_no_brand['test'] = no_desc_no_brand['test_id'].map(lambda x: 1 if x.is_integer() else 0)

no_desc_no_brand = no_desc_no_brand[no_desc_no_brand['test'] ==0]
plt.style.use('fivethirtyeight')

plt.subplots(figsize=(15,5))

no_desc_no_brand['price'].hist(bins=150,edgecolor='black',grid=False)

plt.xticks(list(range(0,100,5)))

plt.title('Price vs no brand&no_description')

plt.show() 
#No of rows whose price is bigger than 100

print("No of rows whose price is bigger than 200 in no_brand&no_description",len(no_desc_no_brand[no_desc_no_brand['price'] >200]))



no_desc_no_brand['price'].describe()

del no_desc_no_brand
from ggplot import *

p = ggplot(aes(x='price'), data=train[train['price']<200]) + geom_histogram(binwidth=10)+ theme_bw() + ggtitle('Histogram of price in train data')

print(p)
data['price'].describe().apply(lambda x: format(x, 'f'))
#Length of categories

data['len_categories'] = data['categories'].map(lambda x: len(x))
#Value_counts for item_condition_id

temp1=data['item_condition_id'].value_counts()[:5].to_frame()

sns.barplot(temp1.index,temp1['item_condition_id'],palette='inferno')

plt.title('Item condition id')

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()
#Making binary 'item_condition_id'

ic_list = list(set(data['item_condition_id']))



for i in ic_list:

    data['item_condition_id'+str(i)] = data['item_condition_id'].map(lambda x: 1 if x==i else 0)



del data['item_condition_id']
#Correlation between no_of_words and price

corr = data[['no_of_words','price','shipping','len_categories','length']].corr()



# Set up the matplot figure

f,ax = plt.subplots(figsize=(12,9))



#Draw the heatmap using seaborn

sns.heatmap(corr, cmap='inferno', annot=True)
#Determined via XGBoost

most_imp = ['cat3_full-length', 'cat2_jewelry', 'cat3_tracksuits & sweats', 'item_description_case', 'name_michael', 'name_ring', 'name_nike', 'item_description_price', 'name_pink', 'cat3_headphones', 'no_of_words', 'item_description_[rm]', 'cat1_electronics', 'cat3_sticker', 'length', 'item_description_silver', 'cat3_consoles', 'item_condition_id5', 'item_description_-', 'brand_old navy', 'item_condition_id4', 'brand_forever 21', 'name_palette', 'cat2_cat2_other', 'name_bracelet', 'item_description_set', 'cat3_hoodie', 'name_boys', 'cat3_makeup palettes', 'name_purse', 'name_bundle', 'brand_lululemon', 'cat2_makeup', 'brand_beats', 'cat1_home', 'item_description_high', 'cat2_home appliances', 'item_description_full', 'cat2_dresses', 'brand_apple', 'brand_ugg australia', 'cat1_beauty', 'name_sleeve', 'brand_kate spade', 'brand_gymshark', "cat2_women's accessories", 'cat2_diapering', 'item_description_bag', 'name_air', 'name_eagle', "cat2_men's accessories", 'name_one', 'brand_lularoe', 'cat3_socks', 'name_jacket', 'name_coach', 'item_description_super', 'name_jordan', "brand_victoria's secret", 'brand_air jordan', 'cat1_women', 'item_description_fits', 'name_lace', 'item_description_new', 'name_top', 'brand_louis vuitton', 'brand_chanel', 'cat3_shoes', 'cat2_bags and purses', 'item_description_shipping', 'cat2_underwear', 'item_condition_id1', 'name_lularoe', 'item_description_comes', 'item_description_item', 'item_description_bundle', 'brand_samsung', 'name_disney', 'cat1_handmade', 'brand_supreme', 'brand_lilly pulitzer', 'item_description_see', 'shipping', 'name_case', 'cat2_cell phones & accessories', 'name_funko', 'name_silver', 'item_description_firm', 'cat3_shipping supplies', 'brand_tiffany & co.', 'item_description_8', 'brand_michael kors', 'brand_tory burch', 'cat2_sweaters', 'brand_kendra scott', 'name_vs', 'name_adidas', 'name_reserved', 'cat3_backpack style', 'brand_adidas', 'item_description_secret', 'item_condition_id2', 'item_description_ship', 'item_description_red', 'cat2_coats & jackets', 'cat3_hair styling tools', 'cat3_fleece jacket', 'cat2_shoes', 'item_description_free', 'name_shorts', 'brand_american eagle', 'len_categories', 'name_kors', 'brand_senegence', 'name_girls', 'item_description_x', 'name_makeup', 'name_shirt', 'item_description_box', 'item_description_body', 'cat3_pants, tights, leggings', 'name_set', 'item_description_gold', 'item_description_condition', 'brand_puma', 'brand_birkenstock', 'name_lot', 'item_description_cute', 'cat2_trading cards', 'cat1_men', 'item_description_7', 'item_description_beautiful', 'brand_miss me', 'name_gold', "cat2_women's handbags", 'cat3_necklaces', 'brand_other_brand', 'cat3_dining & entertaining', 'cat2_tops & blouses', 'brand_dooney & bourke', 'item_description_save', 'item_condition_id3', 'brand_rae dunn', 'brand_pink', 'item_description_plus', 'cat3_jerseys', 'cat3_cosmetic bags', 'cat3_flats', 'cat3_athletic', 'brand_beats by dr. dre', 'brand_nan', 'cat1_kids', 'brand_free people', 'cat2_computers & tablets', 'cat3_cases, covers & skins', 'cat2_cameras & photography', 'cat3_boots', 'item_description_original']

print(len(most_imp))
itemdesc_imp = [x[17:] for x in most_imp if 'item_description_' in x]

name_imp = [x[5:] for x in most_imp if 'name_' in x]

brand_imp = [x[6:] for x in most_imp if 'brand_' in x]

cat1_imp= [x[5:] for x in most_imp if 'cat1_' in x]

cat2_imp= [x[5:] for x in most_imp if 'cat2_' in x]

cat3_imp= [x[5:] for x in most_imp if 'cat3_' in x]

other_imp = ['no_of_words', 'length', 'item_condition_id5', 'item_condition_id4', 'item_condition_id1', 'shipping', 'item_condition_id2', 'len_categories', 'item_condition_id3']
print("length of name",len(name_imp))

print("length of name",len(itemdesc_imp))

print("length of name",len(brand_imp))

print("length of name",len(cat1_imp))

print("length of name",len(cat2_imp))

print("length of name",len(cat3_imp))

print("length of name",len(other_imp))
"""

##Name

import nltk

import collections as co

stopWords =co.Counter( nltk.corpus.stopwords.words() )

words = list(data['name'])

#Merging in a big string

big_string=" ".join(words)

#Splitting them via blank

name_list = big_string.split()

#Omitting splitwords

name_list = [x for x in name_list if x not in stopWords]

#Getting unique words

unique_names = list(set(name_list))

#Counting them

c = co.Counter(name_list)

most_common_100 = c.most_common(100)

most_common_100_2 = [x[0] for x in most_common_100]

"""

#Making them a column

for i in name_imp:

    data['name_'+str(i)] = data['name'].map(lambda x: 1 if i in x else 0)



print("name completed")
"""

##Description

words1 = list(data['item_description'])

big_string1=" ".join(words1)

name_list1 = big_string1.split()



name_list1 = [x for x in name_list1 if x not in stopWords]

unique_names1 = list(set(name_list1))

c = co.Counter(name_list1)

most_common_100_desc = c.most_common(100)

most_common_100_2_desc = [x[0] for x in most_common_100_desc]



"""
for i in itemdesc_imp:

    data['item_description_'+str(i)] = data['item_description'].map(lambda x: 1 if i in x else 0)

print("description completed")
"""

##First common 200 brands

most_common_brands = data['brand_name'].value_counts().sort_values(ascending=False)[:150]

"""

most_common_brands = brand_imp

#If a brand not in common brands, it was labeled as other_brand

other_brand = "other_brand"

data['brand_name'] = data['brand_name'].map(lambda x: x if x in most_common_brands else other_brand)
empty_df = pd.get_dummies(data['brand_name'])

emp_list = list(empty_df.columns.values)

emp_list = ['brand_'+str(x) for x in emp_list]

empty_df.columns = emp_list

print(emp_list)
data2 = pd.concat([data,empty_df],axis=1)

data = data2

del data2,empty_df

print("brand completed")
print(list(data.columns.values))
#categories

data['categories']= data['categories'].map(lambda x: list(x)+[0,0,0,0])

data['cat1']=data['categories'].map(lambda x: x[0])

data['cat2']=data['categories'].map(lambda x: x[1])

data['cat3']=data['categories'].map(lambda x: x[2])

data['cat4']=data['categories'].map(lambda x: x[3])

data['cat5']=data['categories'].map(lambda x: x[4])



most_common_cat1=data['cat1'].value_counts().sort_values(ascending=False)[:11]

most_common_cat2=data['cat2'].value_counts().sort_values(ascending=False)[:70]

most_common_cat3=data['cat3'].value_counts().sort_values(ascending=False)[:90]

#most_common_cat4=data['cat4'].value_counts().sort_values(ascending=False)[:100]

#most_common_cat5=data['cat5'].value_counts().sort_values(ascending=False)[:100]
#Bucketing the features(cat1)

cat1_b1 = ['women','vintage & collectibles','sports & outdoors','nan','home']

cat1_b2 = ['other','beauty','handmade']

cat1_b3 = ['men','electronics','beauty']

data['cat1_fe'] = data['cat1'].map(lambda x: 1 if x in cat1_b1 else 2 if x in cat1_b2 else 3)
#Putting tablet as a feature

data['cat4_tablet'] = data['cat4'].map(lambda x: 1 if x =='tablet' else 0)
ebook = ['ebook access','ebook readers']

data['cat5_ebook'] = data['cat5'].map(lambda x: 1 if x in ebook else 0)
most_common_cat1=cat1_imp

most_common_cat2=cat2_imp

most_common_cat3=cat3_imp

#Categories, we fill focus on first 3 categories

cat1_list = list(most_common_cat1)

cat2_list = list(most_common_cat2)

cat3_list = list(most_common_cat3)
#If a category not in cat1, it was labeled as 'cat1_other'

cat1_other = "cat1_other"

data['cat1'] = data['cat1'].map(lambda x: x if x in cat1_list else cat1_other)

#If a category not in cat2, it was labeled as 'cat2_other'

cat2_other = "cat2_other"

data['cat2'] = data['cat2'].map(lambda x: x if x in cat2_list else cat2_other)

#If a category not in cat3, it was labeled as 'cat3_other'

cat3_other = "cat3_other"

data['cat3'] = data['cat3'].map(lambda x: x if x in cat3_list else cat3_other)
cat1_exp = ['electronics']

data['cat1_exp'] = data['cat1'].map(lambda x: 1 if x in cat1_exp else 0)



cat2_exp = ["women's handbags","cell phones & accessories","shoes"]

data['cat2_exp'] = data['cat2'].map(lambda x: 1 if x in cat2_exp else 0)



cat3_exp = ["cell phones & smartphones","shoulder bag","athletic","totes & shoppers","messenger & crossbody"]

data['cat3_exp'] = data['cat3'].map(lambda x: 1 if x in cat3_exp else 0)

good_brands = ['forever 21', 'american eagle', 'under armour', 'old navy', 'hollister', "carter's", 'brandy melville', 'gap', 'charlotte russe', 'ralph lauren', 'converse', 'h&m', 'express', 'abercrombie & fitch', 'nyx', 'hot topic', 'calvin klein', "levi's®", 'anastasia beverly hills', 'torrid', 'tommy hilfiger', 'mossimo', 'aeropostale', 'columbia', 'guess', 'urban outfitters', 'target', 'xhilaration', 'maybelline', 'american apparel', 'maurices', 'elmers', 'rue21', "l'oreal", 'smashbox', 'champion', 'fashion nova', 'lucky brand', 'wet n wild', 'banana republic', 'toms', 'popsockets', 'wet seal', 'ann taylor loft', 'colourpop cosmetics', 'hello kitty', 'it cosmetics', 'merona', "osh kosh b'gosh", 'crocs', 'rue', 'e.l.f.', 'avon', 'revlon', "the children's place", 'starbucks', 'stila', 'jessica simpson', 'new york & company', 'lane bryant', 'pacific sunwear', 'skechers', 'motherhood maternity', 'nine west', "children's place", 'no boundaries', 'simply southern', 'athleta', 'roxy', 'fox racing', 'covergirl', 'bareminerals', 'aldo', 'gildan', 'new era', 'bare escentuals', 'silver jeans co.', 'yankee candle', 'bullhead', 'lacoste', 'lc lauren conrad', 'faded glory', 'hollister co.', 'hot wheels', 'billabong', 'laura mercier', 'tupperware', 'white house black market', 'affliction', 'stride rite', 'mac cosmetic', 'crest', 'sally hansen', 'nickelodeon', 'cacique', 'aéropostale', 'bobbi brown', "candie's", 'gillette', 'tobi', 'volcom', 'sperrys', 'mudd', 'gerber', 'leap frog', 'diamond supply co.', 'cato', 'nautica', 'laura geller', 'my little pony', 'disney princess', 'danskin', 'cherokee', 'mossimo supply co.', 'lime crime', 'vtech', 'sperry', 'dc shoes', 'daytrip', 'kenneth cole new york', 'dickies', 'stussy', 'pampered chef', 'cotton on', 'the limited', 'neutrogena', 'inc international concepts', 'ardell', 'hanna anderson', 'liz lange', 'so', 'comfort colors', 'liz claiborne', 'hurley', 'eddie bauer', 'bcbgeneration', "burt's bees", 'ann taylor', "chico's", "dr. brown's", 'nerf', 'thebalm', 'garnier', 'papaya', 'aden & anais', 'bongo', 'melissa & doug', 'fila', 'dove', 'make up for ever', 'american rag', 'ed hardy', 'sonoma', 'beautyblender®', 'aerie', 'petsmart', 'huggies', 'sesame street', 'ikea', 'anne klein', 'febreze', 'origins', 'pier one', 'worthington', 'munchkin', 'ivory ella', 'floam', 'bonne bell', 'ambiance apparel', 'avent', 'converse shoes', 'full tilt', 'dkny', 'vanity', 'shiseido', 'wrangler', 'lokai', 'arizona', 'the body shop', 'spanx', 'apt.', 'jumping beans', 'hourglass cosmetics', 'hard candy', 'a.n.a', 'obey', 'sperry top-sider', 'boppy', 'schick', 'rock & republic', 'simply vera vera wang', 'ben nye', 'almay', 'thrasher magazine', "lands' end", 'jennifer lopez', 'infantino', 'bke', "o'neill", 'rimmel', 'chaps', 'disney pixar cars', 'croft & barrow', 'op', "gilligan & o'malley", 'colgate', 'bdg', 'eos', 'rvca', 'pampers', 'dermablend', 'wilton', 'delia*s', 'modcloth', 'fabletics', 'ymi', 'venus', 'la hearts', 'dressbarn', 'disney pixar', "kiehl's", 'style&co.', 'soffe', 'playtex', 'tommee tippee', 'xoxo', 'vigoss', 'speedo', 'hanes', 'rave', 'paper mate', 'tommy bahama', 'sinful by affliction', 'derek heart', 'refuge', 'sanuk', 'talbots', 'elizabeth arden', 'olay', 'zella', 'lalaloopsy', 'avenue', 'pokemon usa', 'pampers swaddlers', "francesca's collections", 'gilly hicks', 'kendall & kylie', 'zumba', 'la idol', 'bumbo', 'arizona jean company', 'decree', 'huggies snug & dry', 'glade', 'dreamworks', 'franco sarto', "st. john's bay", 'nivea', 'chinese laundry', 'incipio', 'us polo assn', "claire's", 'boohoo', "lulu's", 'kotex', 'cabi', 'obey clothing', 'jones new york', 'crayola', 'disney jr.', 'everlast', 'lee', 'material girl', 'catalina', 'art', 'levi strauss & co.', 'bic', 'nick jr.', 'l.e.i.', "tilly's", 'dockers', 'russell athletic', 'capezio', 'kiplling', 'avia', 'charming charlie', 'air wick', 'nike golf', 'kimchi blue', 'hydraulic', 'dollhouse', 'geneva', 'justfab', 'kardashian kollection', 'pur minerals', 'nollie', 'lucy activewear', 'partylite', 'bobbie brooks', 'the hundreds', "dr. scholl's", 'hamilton beach', 'young & reckless', 'always', 'life is good', 'reef', 'southern marsh', 'brooks brothers', 'unionbay', 'izod', 'elle', 'fila sport', 'playskool', 'lenox', 'aerosoles', 'coty', 'baby phat', 'danskin now', 'moda international', 'bravado', 'sharpie', 'george', 'kodak', 'loft', 'belkin', 'apt. 9', 'red cherry', 'huf', 'a pea in the pod', 'john deere', 'new directions', 'robeez', 'twenty one', 'premier designs', 'ivanka trump', 'accessory workshop', 'ecko unltd.', 'safety st', 'max studio', 'hot kiss', 'jj cole collections', 'baby einstein', 'madden girl', 'tek gear', 'cynthia rowley', 'xersion', 'nostalgia electrics', 'bisou bisou', 'mam baby', 'huggies little snugglers', 'thrasher', 'arden b', 'angie', "summer's eve", 'nuk', 'quiksilver', 'precious moments', 'neff', 'degree', 'luvs', 'keds', 'maidenform', 'm.i.a.', 'pillow pets', 'celebrity pink', 'tahari', 'bass', "a'gaci", 'a. byer', 'one clothing', 'carbon', 'corningware', 'bright starts', 'scott paper', 'daisy fuentes', 'nhl', 'kut from the kloth', 'jaclyn smith', 'white stag', 'bandolino', 'cartoon network', 'silver jeans', 'missguided', 'fashion bug', 'jakks pacific', 'antonio melani', 'stance', 'rainbow shops', 'stüssy', 'qupid', 'belly bandit®', 'brita', 'browning', 'aveeno', 'lrg', "cabela's", 'oster', '% pure', 'huggies pull-ups', 'silence + noise', 'liz lange for target', 'ashley stewart', 'monopoly', 'k-swiss', 'willow', 'nascar', 'southpole', 'nicole miller', 'manic panic', 'mally beauty', 'homedics', 'charter club', 'minnetonka', 'angels', 'okie dokie', 'tonka', '47 brand', 'nuby', 'dress barn', 'gloria vanderbilt', 'love culture', 'aroma', 'rewind', 'realtree', 'machine', 'top paw', 'rampage', 'on the byas', 'rocawear', 'crooks & castles', 'ferasali', 'eyeshadow', 'coldwater creek', 'hasbro games', 'divided', 'jada toys', 'jockey', 'kenneth cole reaction', 'body central', 'body glove', 'jerzees', 'empire', 'fruit of the loom', 'guy harvey', 'bebop', 'my michelle', 'axe', 'poetry', 'marmot', 'dana buchman', 'perry ellis', 'tampax', 'urban pipeline', 'jolt', 'ball', 'next level', 'kirra', 'deb', 'soda', 'jansport', 'helly hansen', 'gund', 'van heusen', 'madame alexander', 'j. jill', 'thermos', 'kim rogers', 'sunbeam', 'just my size', 'speechless', 'frontline', 'wubbanub', 'the first years', 'mr. coffee', 'play-doh', 'primitive', 'anchor hocking', 'gianni bini', 'studio y', 'michael stars', 'mighty fine', 'marika', 'garage', 'self esteem', 'charlotte tilbury', 'trixxi', 'lysol', 'conair', "frederick's of hollywood", 'a plus child supply', 'flying monkey', 'city triangles', 'harajuku lovers', 'jane iredale', 'suave', 'fun world', 'g by guess', 'the sak', 'kensie', 'jamberry', 'kyodan', 'christopher & banks', 'breathablebaby', 'custom accessories', 'sean john', 'lucky brand jeans', 'enfagrow', 'romeo & juliet couture', 'farberware', 'matchbox', 'esprit', 'baby bullet', 'cowgirl tuff', 'dragon ball z', 'anvil', 'mossy oak', 'c by champion', 'nokia', 'pro keds', 'dr. seuss', 'energie', 'cottonelle', 'rubbermaid', 'boon', 'stayfree', 'lauren conrad', 'element', 'kong', 'covington', 'buffalo', 'ivivva', 'bcbg', 'sugarpill', 'huggies little movers', 'paris blues', 'tultex', 'baublebar', 'hue', 'cable & gauge', 'soma', 'vocal', 'mitchum', 'teva', 'zion rootswear', 'scotch', 'gaiam', 'tomtom', 'cello jeans', 'pilot', 'style & co', "altar'd state", 'miley cyrus', 'bcx', 'blizzard', 'venezia', 'isaac mizrahi', 'ellen tracy', 'keen', 'fox', 'rival', 'zeroxposur', 'zoo york', 'rocket dog', 'discovery kids', 'melissa', 'blue asphalt', 'furminator', 'nick & nora', 'shoe dazzle', 'vanilla star', 'as seen on tv', 'creativity for kids', 'dial', 'soprano', 'george foreman', 'en focus studio', 'mango', 'laura ashley', 'andrew christian', 'pinkblush']
data['good_brand_or_not'] = data['brand_name'].map(lambda x: 1 if x in good_brands else 0)
#Making binary for cat1

empty_df1 = pd.get_dummies(data['cat1'])

emp_list1 = list(empty_df1.columns.values)

emp_list1 = ['cat1_' + str(x) for x in emp_list1]

empty_df1.columns = emp_list1

#Making binary for cat2

empty_df2 = pd.get_dummies(data['cat2'])

emp_list2 = list(empty_df2.columns.values)

emp_list2 = ['cat2_' + str(x) for x in emp_list2]

empty_df2.columns = emp_list2

#Making binary for cat3

empty_df3 = pd.get_dummies(data['cat3'])

emp_list3 = list(empty_df3.columns.values)

emp_list3 = ['cat3_' + str(x) for x in emp_list3]

empty_df3.columns = emp_list3

#Merging them

data2 = pd.concat([data,empty_df1,empty_df2,empty_df3],axis=1)

data = data2

#Deleting unnecessary things

del data2,empty_df1,empty_df2,empty_df3

del data['cat1'],data['cat2'],data['cat3'],data['cat4'],data['cat5'],data['item_description'],data['name'],data['categories'],data['category_name'],data['brand_name']
print("category completed")
print(data.shape)
test_id = data['test_id']

train_id = data['train_id']

del data['train_id'],data['test_id']

data_head = data.head()

#Separating the merged data into train and test

training = data[data['train_or_not'] ==1]

testing = data[data['train_or_not'] ==0]
del training['train_or_not']

del testing['train_or_not']
y = training['price'].values

y = np.log(y+1)

#Deleting unnecessary columns

del training['price']

del testing['price']

train_size = len(list(training.columns.values))

train_names = list(training.columns.values)
print(train_names)
"""

training = training.values

testing = testing.values

start = datetime.now()

import xgboost as xgb

model = xgb.XGBRegressor(n_estimators=50)

model.fit(training,y)

ending = datetime.now()

print(ending-start)

print (model)





from xgboost import plot_importance

fig, ax = plt.subplots(figsize=(20, 15))

plot_importance(model, ax=ax)



training = pd.DataFrame(training)

testing= pd.DataFrame(testing)



temp = pd.DataFrame(model.feature_importances_)

temp2 = list(temp[temp[0]>0].index)

"""
#temp3 = ['cat3_full-length', 'cat2_jewelry', 'cat3_tracksuits & sweats', 'item_description_case', 'name_michael', 'name_ring', 'name_nike', 'item_description_price', 'name_pink', 'cat3_headphones', 'no_of_words', 'item_description_[rm]', 'cat1_electronics', 'cat3_sticker', 'length', 'item_description_silver', 'cat3_consoles', 'item_condition_id5', 'item_description_-', 'brand_old navy', 'item_condition_id4', 'brand_forever 21', 'name_palette', 'cat2_cat2_other', 'name_bracelet', 'item_description_set', 'cat3_hoodie', 'name_boys', 'cat3_makeup palettes', 'name_purse', 'name_bundle', 'brand_lululemon', 'cat2_makeup', 'brand_beats', 'cat1_home', 'item_description_high', 'cat2_home appliances', 'item_description_full', 'cat2_dresses', 'brand_apple', 'brand_ugg australia', 'cat1_beauty', 'name_sleeve', 'brand_kate spade', 'brand_gymshark', "cat2_women's accessories", 'cat2_diapering', 'item_description_bag', 'name_air', 'name_eagle', "cat2_men's accessories", 'name_one', 'brand_lularoe', 'cat3_socks', 'name_jacket', 'name_coach', 'item_description_super', 'name_jordan', "brand_victoria's secret", 'brand_air jordan', 'cat1_women', 'item_description_fits', 'name_lace', 'item_description_new', 'name_top', 'brand_louis vuitton', 'brand_chanel', 'cat3_shoes', 'cat2_bags and purses', 'item_description_shipping', 'cat2_underwear', 'item_condition_id1', 'name_lularoe', 'item_description_comes', 'item_description_item', 'item_description_bundle', 'brand_samsung', 'name_disney', 'cat1_handmade', 'brand_supreme', 'brand_lilly pulitzer', 'item_description_see', 'shipping', 'name_case', 'cat2_cell phones & accessories', 'name_funko', 'name_silver', 'item_description_firm', 'cat3_shipping supplies', 'brand_tiffany & co.', 'cat3_cat3_other', 'item_description_8', 'brand_michael kors', 'brand_tory burch', 'cat2_sweaters', 'brand_kendra scott', 'name_vs', 'name_adidas', 'name_reserved', 'cat3_backpack style', 'brand_adidas', 'item_description_secret', 'item_condition_id2', 'item_description_ship', 'item_description_red', 'cat2_coats & jackets', 'cat3_hair styling tools', 'cat3_fleece jacket', 'cat2_shoes', 'item_description_free', 'name_shorts', 'brand_american eagle', 'len_categories', 'name_kors', 'brand_senegence', 'name_girls', 'item_description_x', 'name_makeup', 'name_shirt', 'item_description_box', 'item_description_body', 'cat3_pants, tights, leggings', 'name_set', 'item_description_gold', 'cat3_other', 'item_description_condition', 'brand_puma', 'brand_birkenstock', 'name_lot', 'item_description_cute', 'cat2_trading cards', 'cat1_men', 'item_description_7', 'item_description_beautiful', 'brand_miss me', 'name_gold', "cat2_women's handbags", 'cat3_necklaces', 'brand_other_brand', 'cat3_dining & entertaining', 'cat2_tops & blouses', 'brand_dooney & bourke', 'item_description_save', 'item_condition_id3', 'brand_rae dunn', 'brand_pink', 'item_description_plus', 'cat3_jerseys', 'cat3_cosmetic bags', 'cat3_flats', 'cat3_athletic', 'brand_beats by dr. dre', 'brand_nan', 'cat1_kids', 'brand_free people', 'cat2_computers & tablets', 'cat3_cases, covers & skins', 'cat2_cameras & photography', 'cat3_boots', 'item_description_original']
#Preparing model for ANN

testing.columns = train_names

training.columns = train_names

#Getting important columns

training_last = training

testing_last = testing

print(training_last.shape)

print(testing_last.shape)
input_node = len(list(training_last.columns.values))

print("there are ",input_node," nodes in input layer")

#Makin ndarray

training_last = training_last.values

testing_last = testing_last.values
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

training_last = sc_X.fit_transform(training_last)

testing_last = sc_X.transform(testing_last)
#part 2 :Let'S make ANN

# importing the keras library

import keras

# required to initialize NN

from keras.models import Sequential

#Required to build layers of NN

from keras.layers import Dense

from keras.layers import Dropout

#Initializing the ANN

classifier = Sequential()

from keras.optimizers import RMSprop

rmsprop = RMSprop(lr =0.0001)
from keras import backend as K

def root_mean_squared_log_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(K.log(y_pred+1) - K.log(y_true+1)), axis=-1)) 
#adding the input layer and first hidden layer (160 nodes on Input layer, 70 nodes on Hidden Layer 1) and RELU

classifier.add(Dense(output_dim = input_node , init ='he_normal', activation ='relu',input_dim = input_node))

#Adding the second layer(70 nodes on Hidden layer 1, 20 nodes on Hidden Layer 2) and RELU

classifier.add(Dense(output_dim = 51 , init ='he_normal', activation ='relu'))

classifier.add(Dropout(p=0.15))

#adding the output layer- 

classifier.add(Dense(output_dim = 1 , init ='uniform'))

#compiling ANN- optimizer for weights on ANN 

classifier.compile( optimizer=rmsprop , loss='mean_squared_logarithmic_error', metrics = ['mse']  )
start = datetime.now()

classifier.fit(training_last, y ,batch_size=16,nb_epoch=5)

stop = datetime.now()

execution_time = stop-start 

print(execution_time)
#Preparing the submission file

our_pred = classifier.predict(testing_last)

our_pred = np.exp(our_pred) - 1

our_pred = pd.DataFrame(our_pred)

ourpred = pd.DataFrame(our_pred).rename(columns={0:'price'})



test_id = test_id[len(train):len(data)]

test_id = test_id.map(lambda x: int(x))

test_id = test_id.reset_index(drop=True)

test_id = pd.DataFrame(test_id)
output_file = pd.concat([test_id,ourpred],axis=1)

output_file.head()
print("average of test predictions = ",np.mean(output_file['price']))



output_file.to_csv('18-01-2018-mercari-scaled-ANN-161feature.csv',index=False)
stop_real = datetime.now()

execution_time_real = stop_real-start_real 

print(execution_time_real)