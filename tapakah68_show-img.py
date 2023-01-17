import pandas as pd
import random
from PIL import Image
path = '/kaggle/input/impressive-dataset/Impressive Dataset/'
df = pd.read_csv(path + '20k_vpechatlator_29_02_20.csv')
random_id = random.choice(df.index)
image_id = df.loc[random_id,'image_id']
impression = df.loc[random_id,'impression']
print(f'Index: {random_id}')
print(f'image_id: {image_id}')
print(f'impression: {impression}')
Image.open(f'{path}images/{image_id}.jpg')
random_id = random.choice(df.index)
image_id = df.loc[random_id,'image_id']
impression = df.loc[random_id,'impression']
print(f'Index: {random_id}')
print(f'image_id: {image_id}')
print(f'impression: {impression}')
Image.open(f'{path}images/{image_id}.jpg')
random_id = random.choice(df.index)
image_id = df.loc[random_id,'image_id']
impression = df.loc[random_id,'impression']
print(f'Index: {random_id}')
print(f'image_id: {image_id}')
print(f'impression: {impression}')
Image.open(f'{path}images/{image_id}.jpg')
random_id = random.choice(df.index)
image_id = df.loc[random_id,'image_id']
impression = df.loc[random_id,'impression']
print(f'Index: {random_id}')
print(f'image_id: {image_id}')
print(f'impression: {impression}')
Image.open(f'{path}images/{image_id}.jpg')
random_id = random.choice(df.index)
image_id = df.loc[random_id,'image_id']
impression = df.loc[random_id,'impression']
print(f'Index: {random_id}')
print(f'image_id: {image_id}')
print(f'impression: {impression}')
Image.open(f'{path}images/{image_id}.jpg')
