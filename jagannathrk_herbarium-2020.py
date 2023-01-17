import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if filename.endswith('.jpg'):

            break

        print(os.path.join(dirname, filename))
sample_sub = pd.read_csv('../input/herbarium-2020-fgvc7/sample_submission.csv')

display(sample_sub)
import json, codecs

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/train/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    train_meta = json.load(f)

    

with codecs.open("../input/herbarium-2020-fgvc7/nybg2020/test/metadata.json", 'r',

                 encoding='utf-8', errors='ignore') as f:

    test_meta = json.load(f)
display(train_meta.keys())
train_df = pd.DataFrame(train_meta['annotations'])

display(train_df)
train_cat = pd.DataFrame(train_meta['categories'])

train_cat.columns = ['family', 'genus', 'category_id', 'categort_name']

display(train_cat)
train_img = pd.DataFrame(train_meta['images'])

train_img.columns = ['file_name', 'height', 'image_id', 'license', 'width']

display(train_img)
train_reg = pd.DataFrame(train_meta['regions'])

train_reg.columns = ['region_id', 'region_name']

display(train_reg)
train_df = train_df.merge(train_cat, on='category_id', how='outer')

train_df = train_df.merge(train_img, on='image_id', how='outer')

train_df = train_df.merge(train_reg, on='region_id', how='outer')
print(train_df.info())



display(train_df)

na = train_df.file_name.isna()

keep = [x for x in range(train_df.shape[0]) if not na[x]]

train_df = train_df.iloc[keep]
dtypes = ['int32', 'int32', 'int32', 'int32', 'object', 'object', 'object', 'object', 'int32', 'int32', 'int32', 'object']

for n, col in enumerate(train_df.columns):

    train_df[col] = train_df[col].astype(dtypes[n])

print(train_df.info())

display(train_df)
test_df = pd.DataFrame(test_meta['images'])

test_df.columns = ['file_name', 'height', 'image_id', 'license', 'width']

print(test_df.info())

display(test_df)
#train_df.to_csv('full_train_data.csv', index=False)

#test_df.to_csv('full_test_data.csv', index=False)
print(len(train_df.category_id.unique()))
sub = pd.DataFrame()

sub['Id'] = test_df.image_id

sub['Predicted'] = list(map(int, np.random.randint(1, 32093, (test_df.shape[0]))))

display(sub)

sub.to_csv('submission.csv', index=False)