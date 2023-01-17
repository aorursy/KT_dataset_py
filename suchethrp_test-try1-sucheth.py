# -*- coding: utf-8 -*-

##  removed cat1 and removed missing values fill



import pandas as pd

import statsmodels.formula.api as sm



cat3_list =['Consoles',

    'Desktops & All-In-Ones',

    'Hobo',

    'Laptops & Netbooks',

    'Satchel',

    'Shoulder Bag',

    'Sweeping',

    'Totes&Shoppers',

    'unknown'

]



brands_list = ['Abbott',

'Acacia Swimwear',

'adidas Originals',

'Air Jordan',

'Alexander McQueen',

'Alexander Wang',

'Betty Boop',

'Bose',

'Bottega Veneta',

'Brahmin',

'Breville',

'Burberry',

'Canon',

'Cartier',

'Celine',

'Chanel',

'Christian Louboutin',

'David Yurman',

"David's Bridal",

'DeWALT',

'Faviana',

'Fendi',

'Gateway',

'gDiapers',

'Gucci',

'Hunter',

'Jovani',

'Logitech',

'Louis Vuitton',

'MCM Worldwide',

'Mezco',

'MICHELE',

'Nikon',

'Orbit Baby',

'Physicians Formula',

'Pioneer',

'Rick Owens',

'Rolex',

'Saint Laurent',

'Spin Master',

'Tieks',

'Tiffany & Co.',

'Toshiba',

'Valentino',

'Western Digital',

'YSL Yves Saint Laurent',

'unknown',

]





def brandnew_flag(data):

    flag = 0

    if 'never used' in data.lower() or 'brand new' in data.lower():

        flag = 1

    return flag



def used_flag(data):

    flag = 0

    if 'never used'  not in data.lower() or 'used' in data.lower():

        flag = 1

    return flag



## Read train data

train_df = pd.read_csv(

        '../input/train.tsv',

        sep='\t',

        skipinitialspace=True,

        usecols=[

                'train_id',

                'name',

                'item_condition_id',

                'category_name',

    	        'brand_name',

                'price',

                'shipping',

                'item_description'

                ],

        nrows=20000,

        )

train_df.set_index('train_id', inplace=True)



## Fill blank values with unknown

train_df[['brand_name']] = train_df[['brand_name']].fillna(value='unknown')

train_df[['category_name']] = train_df[['category_name']].fillna(value='unknown/unknown/unknown')



## Split the category into seperate columns

foo = lambda x: pd.Series([i for i in x.split('/')])

cat_split = train_df['category_name'].apply(foo)

cat_split.rename(columns={0:'cat1',1:'cat2',2:'cat3'},inplace=True)

cat_split = cat_split[['cat1','cat2','cat3']]

train_df = train_df.join(cat_split)





train_df['brand_name'] = train_df['brand_name'].map(lambda s: 'unknown' if s not in brands_list else s)

train_df['cat3'] = train_df['cat3'].map(lambda s: 'unknown' if s not in cat3_list else s)

train_df['new_flag'] = train_df['item_description'].map(brandnew_flag)



## Build multi linear regression model

result = sm.ols(formula="price ~  shipping + item_condition_id + brand_name + cat3 + new_flag ", data=train_df).fit()

print (result.summary())



def uniqueValues(seq):

   keys = {}

   for e in seq:

       keys[e] = 1

   return keys.keys()



cat1s = uniqueValues(train_df['cat1'].tolist())

cat2s = uniqueValues(train_df['cat2'].tolist())



##read test data

test_df = pd.read_csv(

        '../input/test.tsv',

        sep='\t',

        skipinitialspace=True,

        usecols=[

                'test_id',

                'name',

                'item_condition_id',

                'category_name',

    	        'brand_name',

                'shipping',

                'item_description'

                ],

        )



test_df[['brand_name']] = test_df[['brand_name']].fillna(value='unknown')

test_df[['category_name']] = test_df[['category_name']].fillna(value='unknown/unknown/unknown')

test_df.set_index('test_id', inplace=True)



test_cat_split = test_df['category_name'].apply(foo)

test_cat_split.rename(columns={0:'cat1',1:'cat2',2:'cat3'},inplace=True)

test_cat_split = test_cat_split[['cat1','cat2','cat3']]

test_df = test_df.join(test_cat_split)





## Take care of new data values which are not in train data

test_df['brand_name'] = test_df['brand_name'].map(lambda s: 'unknown' if s not in brands_list else s)

# test_df['cat1'] = test_df['cat1'].map(lambda s: 'unknown' if s not in cat1s else s)

# test_df['cat2'] = test_df['cat2'].map(lambda s: 'unknown' if s not in cat2s else s)

test_df['cat3'] = test_df['cat3'].map(lambda s: 'unknown' if s not in cat3_list else s)



test_df['new_flag'] = test_df['item_description'].map(brandnew_flag)

# test_df['used_flag'] = test_df['item_description'].map(used_flag)

## Predictions for test data

predictions = result.predict(test_df)

output_df = pd.DataFrame({'test_id': predictions.index, 'price': predictions.values})

output_df['price'] = output_df['price'].map(lambda s : -s if s < 0 else s )



## Output the test result to csv

output_df[['test_id','price']].to_csv('output.csv', index=False)
