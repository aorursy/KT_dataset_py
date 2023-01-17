import pandas as pd
train_df = pd.read_csv("/kaggle/input/imaterialist-fashion-2020-fgvc7/train.csv")
from tqdm import tqdm
areas = []
for index,row in tqdm(train_df.iterrows(), total = len(train_df)):
#     sub_mask = np.full(row['Height']*row['Width'], 0, dtype=np.uint8) 
    fashion_rle = [int(x) for x in row["EncodedPixels"].split(' ')]
    area = 0
    for j, start_pixel in enumerate(fashion_rle[::2]):
#         sub_mask[start_pixel: start_pixel+fashion_rle[2*j+1]] = 1    
        area = fashion_rle[2*j+1] + area
    areas.append([area, row['ImageId'], row['Height'], row['Width'], row['ClassId'], row['AttributesIds']])
    
import pickle
with open('./areas.pk', 'wb') as f:
    pickle.dump(areas, f)
rates = []
for area in areas:
    rate = area[0] / (area[2]*area[3]) 
    rates.append(rate)
areas_pd = pd.DataFrame(rates)
areas_pd.describe(percentiles=[.25, .5, .6, .75])
areas_pd.plot(kind = 'hist',bins = 1000, figsize = (10,10),  fontsize = 10)
areas_pd.plot(kind = 'hist',bins = 1000, figsize = (10,10), logx= True, fontsize = 10)
lengths = []
for area in areas:
    length = np.sqrt((area[2]*area[3]))
    lengths.append(length)
lengths_pd = pd.DataFrame(lengths)
lengths_pd.plot(kind = 'hist',bins = 30, figsize = (10,10), logx= False, fontsize = 10)
lengths_pd.describe()