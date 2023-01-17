import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from urllib.request import urlopen

from bs4 import BeautifulSoup

import re



main_url = 'https://www.cartier.com'



html = urlopen(main_url + '/en-us/collections/jewelry.html')

bsObj = BeautifulSoup(html)



categories_url_list = []



for link in bsObj.findAll('a', href=re.compile('(.)*(\/categories\/)(.)*(viewall\.html)')):

    if 'href' in link.attrs:

        categories_url_list.append(link.attrs['href'])
products_url_list = []



for categorie in categories_url_list:

    match = re.match(r'([\w\-\/]+).*', categorie)



    html = urlopen(main_url + categorie)

    bsObj = BeautifulSoup(html)



    for link in bsObj.findAll('a', href=re.compile(match.group(1)+'/')):

        if 'href' in link.attrs:

            products_url_list.append(link.attrs['href'])
# define panda dataframe

product_df = pd.DataFrame(columns=['ref', 'categorie', 'title', 'price', 'tags', 'description', 'image'])



for product_url in products_url_list:



    html = urlopen(main_url + product_url)

    bsObj = BeautifulSoup(html)



    try:

        html = urlopen(main_url + product_url)

    except HTTPError as e:

        print(e)

        #return null, break, or do some other "Plan B"

    else:        

        #REF

        ref = bsObj.find('span', {'class':'local-ref'}).get_text().strip()



        #Title 

        title = bsObj.find('h1', {'class':'c-pdp__cta-section--product-title js-pdp__cta-section--product-title'}).get_text().strip()



        #Price

        price = bsObj.find('div', {'class':'price js-product-price-formatted hidden'}).get_text().strip()



        #Tags

        tags = bsObj.find('div', {'class':'c-pdp__cta-section--product-tags js-pdp__cta-section--product-tags'}).get_text().strip()

        

        #Description

        description = bsObj.find('div', {'class':'tabbed-content__content-column'}).p.get_text().strip()



        #Image Link

        image = bsObj.find('div', {'class':'c-pdp__zoom-wrapper js-pdp-zoom-wrapper'}).img.attrs['src']

        

        #Categorie

        categorie = re.findall(r'\/categories\/(\w*)\/',product_url)[0]



        product = pd.Series([ref, categorie, title, price, tags, description, image], index=product_df.columns)

        product_df = product_df.append(product, ignore_index=True)
product_df['ref'] = product_df['ref'].str.replace('REF.:','')
product_df['price'] = product_df['price'].str.replace('$','')

product_df['price'] = product_df['price'].str.replace(',','')

product_df['price'] = product_df['price'].str.replace('from','')



product_df[product_df['price']=='(1)'] = np.nan

product_df[product_df['price']=='0'] = np.nan



product_df['price'] = product_df['price'].astype(float)
product_df['tags'] = product_df['tags'].str.lower()

product_df['tags'].unique()[:20]
# Pie chart

x_axis = product_df['categorie'].dropna().unique()

y_axis = product_df['categorie'].value_counts()



print(pd.concat([product_df['categorie'].value_counts(), product_df['categorie'].value_counts(normalize=True)], keys=['counts', 'normalized_counts'], axis=1))



import seaborn as sns



sns.set_style('darkgrid')

f, ax = plt.subplots(figsize=(20, 8))

sns.set(font_scale=1.5)



ax = sns.barplot(y=product_df.categorie.value_counts().index, x=product_df.categorie.value_counts(), orient='h')

ax.set_title('Product Distribution',fontsize=20);
metal_list = {'yellow gold':0,'white gold':0,'pink gold':0, 'platinum':0}



tag_list = {}



for tags in product_df['tags'].dropna():

    for tag in tags.split(', '):

        if tag[-1] == 's':

            tag = tag[:-1]

        if tag not in tag_list:

            tag_list[tag] = 1

        else:

            tag_list[tag] += 1

            

for metal in metal_list:

    metal_list[metal] = tag_list[metal]



print('metal\t\t: counts\t: normalized_counts')

for metal, value in metal_list.items():

    print('{}\t: {}\t\t: {:0.2f}'.format(metal, value, value/len(product_df['tags'].dropna())))

    

    

pd_metal = pd.DataFrame(list(metal_list.items()))

pd_metal.columns =['Metal','Count']

pd_metal = pd_metal.sort_values('Count',ascending=False).reset_index()



f, ax = plt.subplots(figsize=(20, 8))

ax = sns.barplot(y='Metal', x='Count',data=pd_metal, order=pd_metal['Metal'],orient='h')

ax.set_title('Metal Distribution',fontsize=20);
material_list = tag_list

for metal in metal_list:

    if metal in material_list:

        del material_list[metal]



pd_material = pd.DataFrame(list(material_list.items()))

pd_material.columns =['Material','Count']

pd_material = pd_material.sort_values('Count',ascending=False).reset_index()

      

f, ax = plt.subplots(figsize=(20, 8))

ax = sns.barplot(y='Material', x='Count',data=pd_material, order=pd_material['Material'],orient='h')

ax.set_title('Material Distribution',fontsize=20);
print(product_df['price'].describe())



f, ax = plt.subplots(figsize=(20, 8))

ax = sns.distplot(product_df['price'], kde=False, norm_hist=False);

ax.set_title('Price Distribution',fontsize=20)



plt.xlabel('Price', fontsize=20)

plt.ylabel('Frequency', fontsize=20)



plt.xlim(0, product_df['price'].max())

plt.show()
plt.figure(figsize=(20, 8))

ax = sns.boxplot(y='categorie', x='price', orient='h',data=product_df, showfliers=False)

ax.set_xticks(np.arange(0, 110000, 10000))

ax.set(ylabel='Categories', xlabel='Price')

plt.show()
metal_df = pd.DataFrame(columns=['title', 'metal', 'price'])



for index,row in product_df.dropna().iterrows():

    for tag in row['tags'].split(', '):

        if tag in metal_list:

            metal_series = pd.Series([row['title'], tag, row['price']], index=metal_df.columns)

            metal_df = metal_df.append(metal_series, ignore_index=True)



plt.figure(figsize=(20, 8))

ax = sns.boxplot(y='metal', x='price', orient='h',data=metal_df, showfliers=False)

ax.set_xticks(np.arange(0, 90000, 10000))

ax.set(ylabel='Categories', xlabel='Price')

plt.show()
metal_df[metal_df['metal']=='white gold'].sort_values(by='price', ascending=False).head(20)
metal_df[metal_df['metal']=='yellow gold'].sort_values(by='price', ascending=False).head(20)
product_df.dropna().to_csv('/kaggle/working/cartier_catalog.csv',index=False)