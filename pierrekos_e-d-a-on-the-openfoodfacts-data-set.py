# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import seaborn as sns

from sklearn.linear_model import LinearRegression

from IPython.display import display

#import testmodul

%pwd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
d=pd.read_csv('../input/en.openfoodfacts.org.products.tsv', delimiter='\t', encoding='utf-8')

d.info()

pd.set_option('display.max_columns', 500)

d.sample(5)
"""Eliminate duplicates if there are any."""

print(d.shape)

d=d.drop_duplicates()

print(d.shape)

d.describe()
col_vides=[]

for col in d.columns:

    if d[col].isnull().all():

        col_vides.append(col)

#print (df.sample(5).to_html())

print ('empty columns are :\n\n')

display(col_vides)

t=pd.DataFrame(index=['number of missing values', 'number of values','fill_rate'], columns=d.columns )

for col in d.columns:

    nb_nan=d[col].isnull().sum()

    t.loc['number of missing values',col]=nb_nan              

    t.loc['number of values', col]= d.shape[0]-nb_nan 

    t.loc['fill_rate', col]=(d.shape[0]-nb_nan)/(d.shape[0]*1.0)

display(t)#pd.set

#t.to_pickle('./tableuxValeursManquantes.pkl')

print(d.shape[0]*0.3/100)

#t[t[col]>df.shape[0]*0.2/100]



"""Selection criterion"""

p=0.95 #0.05

for col in t.columns:

    if t.loc['number of missing values',col]>(d.shape[0]*(100-p)/100):

    #if f=t[t[col]>d.shape[0]*0.2/100]

        #display(t[col])

        print('{} deleted -------> only {} values.'.format(col, d.shape[0]-t.loc['number of missing values', col]))

        d=d.drop(col, axis=1)

display(d.shape) 



l=4 #(l*(df.shape[1]//5)+1, l*5)

i=0

j=0

f, ax = plt.subplots((d.shape[1]//5)+1, 5, figsize=(l*5, l*(d.shape[1]//5)))

f.subplots_adjust(left=0, right=0.9, top= 0.95, bottom= 0., wspace=0.9, hspace=0.9)

f.tight_layout(rect=(0, 0, 0.95,0.95 ), h_pad=0.5, w_pad=0.5)

for col in d.columns:

    if i < 5:

        ax[j,i].pie(t.loc[['number of missing values', 'number of values'], col], 

                        autopct='%0.1f')

        ax[j,i].set_title(col)

        i+=1

    else:

        i=0

        j+=1

        ax[j,i].pie(t.loc[['number of missing values', 'number of values'], col], 

                        autopct='%0.1f')

        ax[j,i].set_title(col)

        i+=1

f.legend(loc='best', labels=['missing values', 'reals values'] , fontsize=15) 

f.suptitle('ratio : reals values / missing values', fontsize=40, fontweight=30)        

      

plt.show()    
dico = {'columns': d.columns, 'fill_rate': t.loc['fill_rate', d.columns]*100, 'missing_values': 100-t.loc['fill_rate',d.columns]*100}



tr = pd.DataFrame(dico)

#display(tr)







preserved = [c for c in d.columns if not((c.endswith('_100g'))|

                                          (c.endswith('_en'))|

                                          (c.endswith('_tags'))|

                                          (c.endswith('_fr'))|

                                          (c.endswith('_datetime'))|

                                          (c.endswith('_t')))]



preserved+=['nutrition_grade_fr', 'nutrition-score-uk_100g', 'nutrition-score-fr_100g']

nutri_info = [c for c in d.columns if c.endswith('_100g') and c!='energy_100g' and c!='energy-from-fat_100g'

                and c!='nutrition-score-fr_100g' and c!='nutrition-score-uk_100g' and c!='carbon-footprint_100g']

d_no_nutri =d[d[nutri_info].isnull().all(axis=1)]

nutri_empty = 100*d_no_nutri.shape[0]/d.shape[0]

nutri_info = {'columns' : ['**have_nutri_info**'], 'fill_rate' : [100-nutri_empty], 'missing_values': [nutri_empty] }

tr_nutri = pd.DataFrame(nutri_info)

tr = tr.loc[preserved,:]

tr = pd.concat([tr, tr_nutri])

tr=tr.sort_values(by='fill_rate', ascending=False)

#display(tr)

r = range(tr.shape[0])



tr.drop(index=['image_small_url', 'creator'], inplace=True)#Remove any columns





barWidth=0.85

plt.figure(figsize=(8,10))

plt.gca().invert_yaxis()

#plt

plt.barh(tr['columns'], tr['fill_rate'], color='#a3acff')

plt.barh(tr['columns'], tr['missing_values'], left=tr['fill_rate'], color ='#b5ffb9')

plt.title('fill rate representation', fontsize=30)

plt.yticks(fontsize=10)

plt.ylabel('columns', fontdict={'fontsize' : 20})

plt.xlabel('fill rate(%)', fontdict={'fontsize' : 20})



plt.axvline(x=80, color='b')

plt.text(82, -1, '>80%', color='b')

plt.axvline(x=20, color='r')

plt.text(12, -1, '<20%', color='r')

plt.grid(True)



d=d.dropna(axis=0, how='all')

print(d.shape)
NUTRI_COL = [c for c in d.columns if c.endswith('_100g')]

dftemp=d[NUTRI_COL].isnull().all(axis=1)

print("{} products have no nutritional indications ie {}% of the data".format(dftemp.sum(), (dftemp.sum()*100/d.shape[0]))) 
numeric = ['int32', 'int64', 'float32', 'float64']

newdf=d.select_dtypes(include=numeric)

colnum=newdf.columns

colnum[-2:]
dftest=pd.DataFrame(d)



for col in newdf.columns:

    if col!='nutrition-score-fr_100g' and col!='nutrition-score-uk_100g':

        for i in dftest[dftest[col].fillna(0)<0].index:

            dftest.loc[i, col] = np.nan
d2=dftest.copy()



quantile=0.001

qmax=d.quantile(q=1-quantile, axis=0, interpolation='higher')

qmin=d.fillna(0).quantile(q=quantile, interpolation='lower')

#display(qmin)

#print('qmax: \n {} \n qmin: \n {}'.format(qmax[qmax>100], qmin[qmin<0]))





for lab in colnum[:-2]:#here, i cut on every numericals columns except in nutrition_score_** columns wich are already OK

    valmax=qmax[lab]

    valmin=qmin[lab]

    #print(lab, valmax, valmin)

    d2=d2[d2[lab].fillna(0)<=valmax]

    d2=d2[d2[lab].fillna(0)>=valmin]

    

   

d2.describe()
"""Below we are interested in the words present in the database"""

from collections import Counter

from wordcloud import WordCloud
def wordclouding(data, label='product_name', sep=' '):

    """To return a wordcloud present in the column 'label', the separation of the word is the argument 'sep' '"""

    words = []

    

    for string in data[label]:

        listwords= str(string).split(sep)

        for w in listwords:

            if (w!=' ')and (w!='nan'):

                words.append(w)

    count=Counter(words)

        



    wordcloud = WordCloud(width=1080, height=920, colormap='PuBuGn').fit_words(count)

    plt.figure(figsize=(25,15))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis('off')

    plt.margins(x=0, y=0)

    plt.show()

        



wordclouding(d2, label='main_category_en', sep=',')
def most_common_word(labels, sep=","):

    words=[]

    for lab in labels:

        words+=str(lab).split(sep)

    count=Counter(words)

    for word in count.most_common(100):

        print(word)
most_common_word(d['main_category'], sep=',')



d[d['main_category']=='en:farming-products']

d[d['main_category']=='en:sweeteners']

d[d['main_category']=='en:dried-products']

d[d['main_category']=='en:dairies']

d[d['main_category']=='en:labeled-products']

d[(d['main_category']=='en:plant-based-foods-and-beverages')&(d['nutrition-score-fr_100g'].dropna()!=d['nutrition-score-uk_100g'].dropna())]

d[d['main_category']=='en:spreads']

d[d['main_category']=='en:desserts']

d[d['main_category']=='en:sweeteners']

d[d['main_category']=='en:pie-dough']

d[d['main_category']=='en:fats']

d[d['main_category']=='en:fresh-foods'].head(5)

wordclouding(d2, label='pnns_groups_2', sep=',')

most_common_word(d['pnns_groups_2'], sep=',')
CATEGS ={

    'cheese' : ['Cheese'],

    'juice' : ['Fruit juices', 'Fruit nectars'],

    'plant' : [ 'fruits', 'Fruits', 'Vegetables'],

    'legume' : ['legumes', 'Legumes'],

    'cake' : ['Biscuits and cakes', ],

    'feculent' : ['Cereals', 'Bread', 'pastries', 'Potatoes' ],

    'beverage' : ['Non-sugared beverages', 'Artificially sweetened beverages', 'Alcoholic beverages'],

    #'sea_food' : [],

    'meat_fish' : ['Tripe dishes', 'Meat','Fish and seafood'],

    'fats' : ['Fats'],

    'milk' : ['Milk and yogurt', 'Ice cream'],

}

#'Soups' has been removed from 'vegetables' as it includes soups of noodles and other




def compare_word(word):

    """Check if 'word' can be referenced to one of our clusters. Each category is a key of our dico 'CATEGS'"""

    if word == ' 'or word == 'nan':

        pass

    for key, val in CATEGS.items():

        if word in val:

            return key 

    pass





def new_categ (data, lab='main_category'):

    """defines a new 'simple_categ' column in 'data'. This allows a cluster according to the catergory of the product."""

    #i=0

    list_column=[]

    for lab in data[lab]:

        list_prov= []

        for w in str(lab).split("en:"):

            s=compare_word(w)

            if s != None:

                break

                        

        list_column.append(s)

        

    data['simple_categ'] = list_column

 
new_categ(d2, lab='pnns_groups_2')


def nutri_bar(data, categ='vegetables', sep=0.2):

    """displays two diagrams representing the average nutrient content in the category 'categ'.

     The first diagram corresponds to the main nutrients and the second one to the secondary nutrients.

     The 'sep' argument separates the main nutrients from the secondary nutrients (secondary nutrients <'sep' grams)"""

    data_categ = data[data['simple_categ']==categ]

    data_mean=data_categ.mean()

    nutri_col = [c for c in data.columns if c.endswith('_100g') and c!='energy_100g' and c!='energy-from-fat_100g'

                and c!='nutrition-score-fr_100g' and c!='nutrition-score-uk_100g' and c!='carbon-footprint_100g']

    data_mean=data_mean.loc[nutri_col]

    

    data_mean=data_mean[data_mean>0]

    data_mean_main=data_mean[data_mean>=sep]

    data_mean_second=data_mean[data_mean<sep]

    

    ind_main = np.arange(len(data_mean_main))

    ind_second = np.arange(len(data_mean_second))

    width=0.55

    

    

    

    plt.title('Average nutrient present in the category {}'.format(categ), fontsize=40)

    fig, ax = plt.subplots(1, 1, figsize=(8,5),dpi=75, num=1)

    

    

    plt.yticks(ind_main, (data_mean_main.index), fontsize=20)

    plt.xticks(fontsize=15)

    ax.invert_yaxis()

    plt.barh(ind_main, data_mean_main, width, align='center')

    

    plt.subplots(1, 1, figsize=(10,15), dpi=50, num=2)

    plt.title('Secondary Nutriments (<{}g)'.format(sep), fontsize=20)

    plt.yticks(ind_second, (data_mean_second.index), fontsize=15)

    plt.xticks(fontsize=15)

    ax.invert_yaxis()

    plt.barh(ind_second, data_mean_second, width, align='center')

    

    

    

    plt.show()

    
"""Improved products recognition without nutritional information"""

nutri_col = [c for c in d.columns if c.endswith('_100g') and c!='energy_100g' and c!='energy-from-fat_100g'

                and c!='nutrition-score-fr_100g' and c!='nutrition-score-uk_100g' and c!='carbon-footprint_100g']



d_no_nutri =d2[d2[nutri_col].isnull().all(axis=1)]



c_no_nutri = d_no_nutri.loc[:,['code', 'simple_categ']].groupby('simple_categ').count()

c_no_nutri = c_no_nutri.rename(columns={ 'code' : 'nb_no_nutri'})

c_categ = d2.loc[:,['code','simple_categ']].groupby('simple_categ').count()#compte le nombre de produit repertorie dans chaques categories

c_categ = c_categ.rename(columns={ 'code' : 'nb'})

c_no_nutri['nb'] = c_categ['nb']



c_no_nutri['percent_no_nutri']=pd.DataFrame(c_no_nutri['nb_no_nutri']*100/c_categ['nb'])

c_no_nutri
nutri_bar(d2, categ='plant')
nutri_bar(d2, categ='cheese')
nutri_bar(d2[d2['simple_categ']=='cheese'].fillna(0), categ='cheese')
#import matplotlib.pyplot as plt

x=pd.DataFrame(d2['nutrition-score-fr_100g'].fillna(0), columns=['nutrition-score-fr_100g'])

y=pd.DataFrame(d2['nutrition-score-uk_100g'].fillna(0), columns=['nutrition-score-uk_100g']) 

plt.figure(figsize=(9,9))

plt.title('relationship between nutrition-score-fr and nutrition-score-uk')

plt.xlabel('nutrition-score-fr_100g')

plt.ylabel('nutrition-score-uk_100g')

plt.plot(x, y, '.')





              



reg = LinearRegression()

droite = reg.fit(x, y)





print('y=ax with a={}\n score : {}'.format(droite.coef_[0], droite.score(x,y)))

plt.plot([-10,40], [droite.coef_[0]*(-10),droite.coef_[0]*40])

plt.show()



#sns.scatterplot(x, y)#, plot_kws={'s' :80}, palette='husl')

plt.figure(figsize=(10,10))

sns.scatterplot(x=d2['nutrition-score-fr_100g'], 

                y=d2['nutrition-score-uk_100g'], 

                hue=d2['simple_categ'],

                legend='full',

                s=90) 

             
"""Under these conditions, we find again fats products"""

d_fr_indulgent = d2[(d2['main_category']=='en:plant-based-foods-and-beverages')&(d2['nutrition-score-fr_100g'].dropna()!=d2['nutrition-score-uk_100g'].dropna())]

wordclouding(d_fr_indulgent)
d_uk_indulgent = d2[(d2['main_category']=='en:beverages')&(d2['nutrition-score-fr_100g'].dropna()>d2['nutrition-score-uk_100g'].dropna())]

wordclouding(d_uk_indulgent)
"""Let keep the traditionnal color code for 'nutrition_grade_fr'"""

palette = sns.color_palette("RdYlGn", 5)

legend = {'e': palette[0], 'd': palette[1], 'c': palette[2], 'b':palette[3], 'a':palette[4]}





nutri_grade=d.loc[:,['nutrition_grade_fr', 'energy_100g']].groupby('nutrition_grade_fr').agg(['mean', 'count'])

#nutri_grade = d.groupby('nutrition_grade_fr').agg(['mean', 'count'])

plt.title('Average energy according to nutrition_grade_fr')

plt.bar(nutri_grade.index, nutri_grade['energy_100g']['mean'], color=palette[::-1])
ds=d2.sample(150000)

x = pd.DataFrame(ds['energy_100g'], columns=['energy_100g'])

fig=plt.figure(figsize=(30,15))

plt.style.use('ggplot')

plt.title('Energy according to mains nutrients' )



plt.plot(ds['sugars_100g'], x, 'o', label='sugars_100g')

plt.plot(ds['fat_100g'], x, 'o', label='fat_100g')

plt.plot(ds['saturated-fat_100g'], x, 'o', label='saturated-fat_100g')

plt.plot(ds['carbohydrates_100g'], x, 'o', label='carbohydrates_100g')

plt.plot(ds['proteins_100g'], x, 'o', label='proteins_100g')

plt.plot(ds['fiber_100g'], x, 'o', label='fiber_100g')

plt.ylabel('energy_100g')

plt.plot([0, 100], [0, 3765.6], linewidth=5, linestyle='-', color='pink')# 9 Kcal = 37.656 kJ

plt.plot([0, 100], [0, 1673.6], linewidth=7, color= 'blue', linestyle='-')# 4 Kcal = 1973.6 kJ



plt.legend(loc='upper left', prop={'size':32})
nb_sample=10000

#marker_size=nb_sample*5/10000

ds=d2.sample(10000)

sns.set(font_scale=1)

plt.figure(figsize=(4, 4))



sns.pairplot(ds.loc[:,['fat_100g', 'saturated-fat_100g', 'fiber_100g', 'carbohydrates_100g', 'proteins_100g', 'nutrition_grade_fr' ]], 

             hue='nutrition_grade_fr',

             hue_order=['e','d','c','b','a'],

             height=2.5,

             plot_kws = {'s': 12},

             palette=legend)



plt.title('Mains nutriments according to nutriment_grade_fr', loc='right')
ds=d2.sample(50000)

plt.figure(figsize=(8,8))

sns.scatterplot(x=ds['carbohydrates_100g'], 

                y=ds['proteins_100g'], 

                hue=ds['nutrition_grade_fr'],

                hue_order=['e','d','c','b','a'],

                legend='full',

                palette=legend) 

plt.title('Carbohydrate and protein levels according to nutri-grade')
d_a = d2[(d2['carbohydrates_100g']>56)&(d2['proteins_100g']>20)&(d2['nutrition_grade_fr']=='a')]

wordclouding(d_a)

d_a.shape
#pal_bgrd = sns.color_palette("RdYlGn", 5, desat=0.35)

da=d2[d2['nutrition_grade_fr']=='a']

plt.figure(figsize=(8,8))

plt.style.use('default')

#plt.set_facecolor(pal_bgrd[0])



sns.scatterplot(x=da['carbohydrates_100g'], 

                y=da['proteins_100g'], #'fiber_100g'

                hue=da['simple_categ'],

                hue_order=['plant', 'cake', 'feculent', 'meat_fish', 'beverage', 'milk',

       'cheese', 'fats', 'juice', 'legume'],

                legend='full')

                #palette=legend) 

plt.title("the distribution of products 'A' according to the categories", color=palette[-1])
l=['b', 'c', 'd', 'e']

pal_bgrd=['#FADBD8', '#FDEBD0', '#FCF3CF', '#D5F5E3', '#A9DFBF']

fig = plt.figure(figsize=(8,16))

plt.title('product comparison')  

plt.style.use('default')

ax = plt.subplot(2,1, 1)

sns.set(font_scale=0.7)

ax.set_facecolor(pal_bgrd[-1])

sns.scatterplot(x=da['carbohydrates_100g'], 

                y=da['proteins_100g'], #'fiber_100g'

                hue=da['simple_categ'],

                hue_order=['plant', 'cake', 'feculent', 'meat_fish', 'beverage', 'milk',

       'cheese', 'fats', 'juice', 'legume'],

                legend='full') 

ax.set_title("the distribution of products 'A' according to the categories", fontsize=10)

for note in l:

    sns.set(font_scale=0.5)

    ds=d2[d2['nutrition_grade_fr']==note]

    ind=l.index(note)

    ax = plt.subplot(4, 2, (ind+5))

    ax.set_facecolor(pal_bgrd[-ind+3])

    sns.scatterplot(x=ds['carbohydrates_100g'], 

                y=ds['proteins_100g'], #'fiber_100g'

                hue=ds['simple_categ'],

                hue_order=['plant', 'cake', 'feculent', 'meat_fish', 'beverage', 'milk',

       'cheese', 'fats', 'juice', 'legume'],

                legend='full',)

                #s=27)

                

    ax.set_title("the distribution of products '{}'".format(note.upper()), 

                 fontsize=8)

                

      
wordclouding(d2, label='packaging_tags', sep=',')
def organic_class(data, bio=True):

    """organic product clustering attempt'"""

    organic_product = []

    not_organic_product = []

    i=0

    for name in data['product_name']:

        words = str(name).split(" ")

        words = [w.lower() for w in words]

        if ('organic' in words) or ('bio' in words) or ('органический'in words):

            

            organic_product.append(name)

            i+=1

        else:

            words = str(data.iloc[i]['labels_tags']).split('en:')

            words = [w.lower() for w in words]

            if ('organic,' in  words) or ('bio' in words):

                organic_product.append(name)

                i+=1

            else:        

                not_organic_product.append(name)

                i+=1

    organic = pd.DataFrame(organic_product, columns=['product_name'])

    not_organic = pd.DataFrame(not_organic_product, columns=['product_name'])

    print('We have class {} organic products against {} not organic\n {}% are organic'.format(organic.shape[0],

                                                                            not_organic.shape[0],

                                                                            organic.shape[0]*100/(organic.shape[0]+not_organic.shape[0]) ))

    if bio==True:

        return(pd.merge(data, organic))

    else:

        return(pd.merge(data, not_organic))







organic = organic_class(d2)

organic.head(2)
d2.shape