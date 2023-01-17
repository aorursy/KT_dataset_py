import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_excel('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/3. Technical Performance/Vision/Image Classification/ImageNet/ImageNet.xlsx')

df['year'] = df['Date'].dt.year

df = df[df['Method'] != 'Inception ResNet V1'] #think this result might be an error. Performance looks off and couldn't find this model in https://paperswithcode.com/sota/image-classification-on-imagenet

idx = df.groupby('year')['Top 1 Accuracy'].transform(max) == df['Top 1 Accuracy']

df_top1 = df[idx]

df_top1.index = df_top1['year']

df_top1[['Method','Top 1 Accuracy','Number of params']].sort_values(by='Top 1 Accuracy')