from fastai import *

from fastai.vision import *

import seaborn as sns

from PIL import Image

from matplotlib import pyplot as plt
path = Path('../input/aptos2019-blindness-detection/')
path.ls()
df = pd.read_csv(path/'train.csv')

df.head()
df_test = pd.read_csv(path/'test.csv')

df_test.head()
print(len(df))

print(len(df_test))
print(df.isna().sum()) 

print('-' * 20)

print(df_test.isna().sum())
df.diagnosis.value_counts()
# plot the value counts as histogram

b = sns.countplot(df['diagnosis'])

b.axes.set_title('Distribution of diagnosis', fontsize = 30)

b.set_xlabel('Diagnosis', fontsize = 20)

b.set_ylabel('Count', fontsize = 20)

plt.show()
im = Image.open("../input/aptos2019-blindness-detection/train_images/08b6e3240858.png")
print(im.format, im.size, im.mode)
im = Image.open("../input/aptos2019-blindness-detection/train_images/0ca0aee4d57e.png")
print(im.format, im.size, im.mode)
# plot the various sizes of images

def get_image_sizes(folder):

    image_list = (path/folder).ls()

    heights = []

    widths = []

    ids = []



    for image in image_list:

        im = Image.open(image)

        height, width = im.size

        heights.append(height)

        widths.append(width)

        ids.append(str(image)[-16:-4])

        

    return pd.DataFrame({'id_code': ids,

                         'height': heights,

                         'width': widths})
size_df = get_image_sizes('train_images')

size_df.head()
size_df_test = get_image_sizes('test_images')

size_df_test.head()
plt.hist(size_df['height'])
plt.hist(size_df_test['height'])
plt.hist(size_df['width'])
plt.hist(size_df_test['width'])
# plot the images from 0 and 4 to see the difference

df_0 = df[df['diagnosis'] == 0]

df_0.head()
df_4 = df[df['diagnosis'] == 4]

df_4.head()
data = (ImageList.from_df(df_0,path,folder='train_images',suffix='.png')

        .split_by_rand_pct(0.1, seed=42)

        .label_from_df()

        .transform(get_transforms(),size=128)

        .databunch()).normalize(imagenet_stats)
# add figsize argument

data.show_batch(rows=3)
data = (ImageList.from_df(df_4,path,folder='train_images',suffix='.png')

        .split_by_rand_pct(0.1, seed=42)

        .label_from_df()

        .transform(get_transforms(),size=128)

        .databunch()).normalize(imagenet_stats)
data.show_batch(rows=3)
data = (ImageList.from_df(df_4,path,folder='train_images',suffix='.png')

        .split_by_rand_pct(0.1, seed=42)

        .label_from_df()

        .transform([],size=128)

        .databunch()).normalize(imagenet_stats)
data.show_batch(rows=3, )
# in the next notebook work with various augmentations