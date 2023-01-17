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
! mkdir dataExtracted
! unzip /kaggle/input/walmart-recruiting-trip-type-classification/sample_submission.csv.zip -d dataExtracted
! unzip /kaggle/input/walmart-recruiting-trip-type-classification/train.csv.zip -d dataExtracted
! unzip -P Work4WalmarT /kaggle/input/walmart-recruiting-trip-type-classification/test.csv.zip -d dataExtracted

#importng libraries
from __future__ import division
import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder

def dataPreprocessing(df) :
   
    label_encoder = LabelEncoder()
    
    df.Weekday = label_encoder.fit_transform(df.Weekday)
    df.DepartmentDescription = df.DepartmentDescription.replace({"MENS WEAR" : "MENSWEAR"})

    df['ItemReturned'] = pd.Series(np.zeros(df.Weekday.shape), index=df.index)
    df.loc[df.ScanCount < 0, 'ItemReturned'] = 1
    df.loc[df.ScanCount < 0, 'ScanCount'] = 0


    desc = pd.get_dummies(df.DepartmentDescription)
    df = pd.concat([df, desc], axis=1)

    del df['DepartmentDescription']
    del df['Upc']
    del df['FinelineNumber']

    grouped = df.groupby("VisitNumber")
    grouped = grouped.agg({'TripType' : np.max,'Weekday' : np.max, 'ScanCount' : np.sum, 'ItemReturned' : np.sum,
           '1-HR PHOTO' : np.sum, 'ACCESSORIES' : np.sum, 'AUTOMOTIVE' : np.sum, 'BAKERY' : np.sum, 'BATH AND SHOWER' : np.sum,
           'BEAUTY' : np.max, 'BEDDING' : np.max, 'BOOKS AND MAGAZINES' : np.max, 'BOYS WEAR' : np.max,
           'BRAS & SHAPEWEAR' : np.sum, 'CAMERAS AND SUPPLIES' : np.sum, 'CANDY, TOBACCO, COOKIES' : np.sum,
           'CELEBRATION' : np.sum, 'COMM BREAD' : np.sum, 'CONCEPT STORES' : np.sum, 'COOK AND DINE' : np.sum, 'DAIRY' : np.sum,
           'DSD GROCERY' : np.sum, 'ELECTRONICS' : np.sum, 'FABRICS AND CRAFTS' : np.sum,
           'FINANCIAL SERVICES' : np.sum, 'FROZEN FOODS' : np.sum, 'FURNITURE' : np.sum,
           'GIRLS WEAR, 4-6X  AND 7-14' : np.sum, 'GROCERY DRY GOODS' : np.sum, 'HARDWARE' : np.sum,
           'HEALTH AND BEAUTY AIDS' : np.sum, 'HOME DECOR' : np.sum, 'HOME MANAGEMENT' : np.sum,
           'HORTICULTURE AND ACCESS' : np.sum, 'HOUSEHOLD CHEMICALS/SUPP' : np.sum,
           'HOUSEHOLD PAPER GOODS' : np.sum, 'IMPULSE MERCHANDISE' : np.sum, 'INFANT APPAREL' : np.sum,
           'INFANT CONSUMABLE HARDLINES' : np.sum, 'JEWELRY AND SUNGLASSES' : np.sum, 'LADIES SOCKS' : np.sum,
           'LADIESWEAR' : np.sum, 'LARGE HOUSEHOLD GOODS' : np.sum, 'LAWN AND GARDEN' : np.sum,
           'LIQUOR,WINE,BEER' : np.sum, 'MEAT - FRESH & FROZEN' : np.sum, 'MEDIA AND GAMING' : np.sum,
           'MENSWEAR' : np.sum, 'OFFICE SUPPLIES' : np.sum, 'OPTICAL - FRAMES' : np.sum, 'OPTICAL - LENSES' : np.sum,
           'OTHER DEPARTMENTS' : np.sum, 'PAINT AND ACCESSORIES' : np.sum, 'PERSONAL CARE' : np.sum,
           'PETS AND SUPPLIES' : np.sum, 'PHARMACY OTC' : np.sum, 'PHARMACY RX' : np.sum,
           'PLAYERS AND ELECTRONICS' : np.sum, 'PLUS AND MATERNITY' : np.sum, 'PRE PACKED DELI' : np.sum,
           'PRODUCE' : np.sum, 'SEAFOOD' : np.sum, 'SEASONAL' : np.sum, 'SERVICE DELI' : np.sum, 'SHEER HOSIERY' : np.sum,
           'SHOES' : np.sum, 'SLEEPWEAR/FOUNDATIONS' : np.sum, 'SPORTING GOODS' : np.sum,
           'SWIMWEAR/OUTERWEAR' : np.sum, 'TOYS' : np.sum, 'WIRELESS' : np.sum})
    df = grouped[['TripType', 'Weekday', 'ScanCount', 'ItemReturned',
           '1-HR PHOTO', 'ACCESSORIES', 'AUTOMOTIVE', 'BAKERY', 'BATH AND SHOWER',
           'BEAUTY', 'BEDDING', 'BOOKS AND MAGAZINES', 'BOYS WEAR',
           'BRAS & SHAPEWEAR', 'CAMERAS AND SUPPLIES', 'CANDY, TOBACCO, COOKIES',
           'CELEBRATION', 'COMM BREAD', 'CONCEPT STORES', 'COOK AND DINE', 'DAIRY',
           'DSD GROCERY', 'ELECTRONICS', 'FABRICS AND CRAFTS',
           'FINANCIAL SERVICES', 'FROZEN FOODS', 'FURNITURE',
           'GIRLS WEAR, 4-6X  AND 7-14', 'GROCERY DRY GOODS', 'HARDWARE',
           'HEALTH AND BEAUTY AIDS', 'HOME DECOR', 'HOME MANAGEMENT',
           'HORTICULTURE AND ACCESS', 'HOUSEHOLD CHEMICALS/SUPP',
           'HOUSEHOLD PAPER GOODS', 'IMPULSE MERCHANDISE', 'INFANT APPAREL',
           'INFANT CONSUMABLE HARDLINES', 'JEWELRY AND SUNGLASSES', 'LADIES SOCKS',
           'LADIESWEAR', 'LARGE HOUSEHOLD GOODS', 'LAWN AND GARDEN',
           'LIQUOR,WINE,BEER', 'MEAT - FRESH & FROZEN', 'MEDIA AND GAMING',
           'MENSWEAR', 'OFFICE SUPPLIES', 'OPTICAL - FRAMES', 'OPTICAL - LENSES',
           'OTHER DEPARTMENTS', 'PAINT AND ACCESSORIES', 'PERSONAL CARE',
           'PETS AND SUPPLIES', 'PHARMACY OTC', 'PHARMACY RX',
           'PLAYERS AND ELECTRONICS', 'PLUS AND MATERNITY', 'PRE PACKED DELI',
           'PRODUCE', 'SEAFOOD', 'SEASONAL', 'SERVICE DELI', 'SHEER HOSIERY',
           'SHOES', 'SLEEPWEAR/FOUNDATIONS', 'SPORTING GOODS',
           'SWIMWEAR/OUTERWEAR', 'TOYS', 'WIRELESS']]
    return df;

from sklearn.model_selection import train_test_split
train_data = dataPreprocessing(pd.DataFrame(pd.read_csv('./dataExtracted/train.csv')))
print(train_data)



from sklearn import preprocessing
x = train_data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled)
x_scaled.index = train_data.index
x_scaled.columns = train_data.columns
train_data = x_scaled

from sklearn.model_selection import train_test_split
train_data = dataPreprocessing(pd.DataFrame(pd.read_csv('./dataExtracted/train.csv')))
train, test = train_test_split(train_data, test_size = 0.3)
train = train.dropna()
test = test.dropna()

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
clf = clf.fit(np.asarray(train.iloc[:,1:]), np.asarray(train.TripType))
predictions = clf.predict(np.asarray(test.iloc[:,1:]))
print(accuracy_score(test.TripType,predictions ))


df = pd.DataFrame(pd.read_csv('./dataExtracted/test.csv'))
#print(df)
desc = pd.get_dummies(df.DepartmentDescription)
df = pd.concat([df, desc], axis=1)

label_encoder = LabelEncoder()
    
df.Weekday = label_encoder.fit_transform(df.Weekday)

df.DepartmentDescription = df.DepartmentDescription.replace({"MENS WEAR" : "MENSWEAR"})

df['ItemReturned'] = pd.Series(np.zeros(df.Weekday.shape), index=df.index)
df.loc[df.ScanCount < 0, 'ItemReturned'] = 1
df.loc[df.ScanCount < 0, 'ScanCount'] = 0
del df['DepartmentDescription']
del df['Upc']
del df['FinelineNumber']
feature1 = train.columns
print(feature1)
my_set = set()
for e in feature1 :
    my_set.add(e)
feature2 = df.columns
print(feature2)
my_set1 = set()
for e in feature2 :
    my_set1.add(e)
for e in my_set :
    if e not in my_set1 :
        df[e] = pd.Series(np.zeros(df.Weekday.shape), index=df.index)
print(df)
print(df.columns.shape)
print(train.columns.shape)

grouped = df.groupby("VisitNumber")
grouped = grouped.agg({'VisitNumber' : np.max,'TripType' : np.max,'Weekday' : np.max, 'ScanCount' : np.sum, 'ItemReturned' : np.sum,
       '1-HR PHOTO' : np.sum, 'ACCESSORIES' : np.sum, 'AUTOMOTIVE' : np.sum, 'BAKERY' : np.sum, 'BATH AND SHOWER' : np.sum,
       'BEAUTY' : np.max, 'BEDDING' : np.max, 'BOOKS AND MAGAZINES' : np.max, 'BOYS WEAR' : np.max,
       'BRAS & SHAPEWEAR' : np.sum, 'CAMERAS AND SUPPLIES' : np.sum, 'CANDY, TOBACCO, COOKIES' : np.sum,
       'CELEBRATION' : np.sum, 'COMM BREAD' : np.sum, 'CONCEPT STORES' : np.sum, 'COOK AND DINE' : np.sum, 'DAIRY' : np.sum,
       'DSD GROCERY' : np.sum, 'ELECTRONICS' : np.sum, 'FABRICS AND CRAFTS' : np.sum,
       'FINANCIAL SERVICES' : np.sum, 'FROZEN FOODS' : np.sum, 'FURNITURE' : np.sum,
       'GIRLS WEAR, 4-6X  AND 7-14' : np.sum, 'GROCERY DRY GOODS' : np.sum, 'HARDWARE' : np.sum,
       'HEALTH AND BEAUTY AIDS' : np.sum, 'HOME DECOR' : np.sum, 'HOME MANAGEMENT' : np.sum,
       'HORTICULTURE AND ACCESS' : np.sum, 'HOUSEHOLD CHEMICALS/SUPP' : np.sum,
       'HOUSEHOLD PAPER GOODS' : np.sum, 'IMPULSE MERCHANDISE' : np.sum, 'INFANT APPAREL' : np.sum,
       'INFANT CONSUMABLE HARDLINES' : np.sum, 'JEWELRY AND SUNGLASSES' : np.sum, 'LADIES SOCKS' : np.sum,
       'LADIESWEAR' : np.sum, 'LARGE HOUSEHOLD GOODS' : np.sum, 'LAWN AND GARDEN' : np.sum,
       'LIQUOR,WINE,BEER' : np.sum, 'MEAT - FRESH & FROZEN' : np.sum, 'MEDIA AND GAMING' : np.sum,
       'MENSWEAR' : np.sum, 'OFFICE SUPPLIES' : np.sum, 'OPTICAL - FRAMES' : np.sum, 'OPTICAL - LENSES' : np.sum,
       'OTHER DEPARTMENTS' : np.sum, 'PAINT AND ACCESSORIES' : np.sum, 'PERSONAL CARE' : np.sum,
       'PETS AND SUPPLIES' : np.sum, 'PHARMACY OTC' : np.sum, 'PHARMACY RX' : np.sum,
       'PLAYERS AND ELECTRONICS' : np.sum, 'PLUS AND MATERNITY' : np.sum, 'PRE PACKED DELI' : np.sum,
       'PRODUCE' : np.sum, 'SEAFOOD' : np.sum, 'SEASONAL' : np.sum, 'SERVICE DELI' : np.sum, 'SHEER HOSIERY' : np.sum,
       'SHOES' : np.sum, 'SLEEPWEAR/FOUNDATIONS' : np.sum, 'SPORTING GOODS' : np.sum,
       'SWIMWEAR/OUTERWEAR' : np.sum, 'TOYS' : np.sum, 'WIRELESS' : np.sum})
df = grouped[['VisitNumber','TripType', 'Weekday', 'ScanCount', 'ItemReturned',
       '1-HR PHOTO', 'ACCESSORIES', 'AUTOMOTIVE', 'BAKERY', 'BATH AND SHOWER',
       'BEAUTY', 'BEDDING', 'BOOKS AND MAGAZINES', 'BOYS WEAR',
       'BRAS & SHAPEWEAR', 'CAMERAS AND SUPPLIES', 'CANDY, TOBACCO, COOKIES',
       'CELEBRATION', 'COMM BREAD', 'CONCEPT STORES', 'COOK AND DINE', 'DAIRY',
       'DSD GROCERY', 'ELECTRONICS', 'FABRICS AND CRAFTS',
       'FINANCIAL SERVICES', 'FROZEN FOODS', 'FURNITURE',
       'GIRLS WEAR, 4-6X  AND 7-14', 'GROCERY DRY GOODS', 'HARDWARE',
       'HEALTH AND BEAUTY AIDS', 'HOME DECOR', 'HOME MANAGEMENT',
       'HORTICULTURE AND ACCESS', 'HOUSEHOLD CHEMICALS/SUPP',
       'HOUSEHOLD PAPER GOODS', 'IMPULSE MERCHANDISE', 'INFANT APPAREL',
       'INFANT CONSUMABLE HARDLINES', 'JEWELRY AND SUNGLASSES', 'LADIES SOCKS',
       'LADIESWEAR', 'LARGE HOUSEHOLD GOODS', 'LAWN AND GARDEN',
       'LIQUOR,WINE,BEER', 'MEAT - FRESH & FROZEN', 'MEDIA AND GAMING',
       'MENSWEAR', 'OFFICE SUPPLIES', 'OPTICAL - FRAMES', 'OPTICAL - LENSES',
       'OTHER DEPARTMENTS', 'PAINT AND ACCESSORIES', 'PERSONAL CARE',
       'PETS AND SUPPLIES', 'PHARMACY OTC', 'PHARMACY RX',
       'PLAYERS AND ELECTRONICS', 'PLUS AND MATERNITY', 'PRE PACKED DELI',
       'PRODUCE', 'SEAFOOD', 'SEASONAL', 'SERVICE DELI', 'SHEER HOSIERY',
       'SHOES', 'SLEEPWEAR/FOUNDATIONS', 'SPORTING GOODS',
       'SWIMWEAR/OUTERWEAR', 'TOYS', 'WIRELESS']]
print(df)
del df['TripType']

result = np.zeros((95674 ,39), dtype=int)
j = 0
for i in  df.VisitNumber :
    result[j][0] = i
    j = j+1
    
print(df.columns.shape)
print(train.columns.shape)
del df['VisitNumber']

predictions = clf.predict(np.asarray(df))
print(predictions)


a = [3,4,5,6,7,8,9,12,14,15,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,999]
dict = {}
j = 1
for i in a :
    dict[i] = j
    j = j+1
for i in range(0,95674) :
    result[i][dict[predictions[i]]] = 1

result = pd.DataFrame(data = result, columns = ['VisitNumber','TripType_3','TripType_4','TripType_5','TripType_6','TripType_7','TripType_8','TripType_9','TripType_12','TripType_14','TripType_15','TripType_18','TripType_19','TripType_20','TripType_21','TripType_22','TripType_23','TripType_24','TripType_25','TripType_26','TripType_27','TripType_28','TripType_29','TripType_30','TripType_31','TripType_32','TripType_33','TripType_34','TripType_35','TripType_36','TripType_37','TripType_38','TripType_39','TripType_40','TripType_41','TripType_42','TripType_43','TripType_44','TripType_999'])

print(result)
result.to_csv('walmart_submission_final.csv',encoding='utf-8', index=False)
print('done')