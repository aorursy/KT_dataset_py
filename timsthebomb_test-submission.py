import os

import random

import pandas as pd



df = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')

print(f"Length of test.csv: {len(df)}\n")





print(df.head())

print('')





for i in df['image_id'][:2]:

    print(f'{i} - {os.path.exists(f"../input/prostate-cancer-grade-assessment/train_images/{i}.tiff")}')
for i in os.walk('/kaggle/'):

    if 'test_images' in i:

        print(i)


if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    print('inference!')

    images = os.listdir('../input/prostate-cancer-grade-assessment/test_images')



else:

    print('not inference')

    images = os.listdir('../input/prostate-cancer-grade-assessment/train_images')



with open('./submission.csv', 'wb') as f:

    f.write(bytes('image_id,isup_grade\n', encoding='utf8'))

    for image in images:

        f.write(bytes(f'{image.replace(".tiff", "")},{random.choice(range(6))}\n', encoding='utf8'))




