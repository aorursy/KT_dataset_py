!pip install requests wget wordcloud
import os

import os.path as pth



import wget

from multiprocessing import Pool



import pandas as pd

from collections import Counter

from wordcloud import WordCloud

from matplotlib import pyplot  as plt
def get_metadata(url, base_path='./'):

    filename = url.split('/')[-1]

    full_filename = pth.join(base_path, filename)

    if pth.exists(full_filename):

        return full_filename, 1

    wget.download(url, out=base_path)

    ### If you can't use wget, you can use below blocked code

    ### But It's much slower than wget...

    # data = requests.get(url).data

    # with open(full_filename, 'wb') as f:

    #     f.write(data)

    return full_filename, 0
data_base_path = 'metadata'

metadata_path = pth.join(data_base_path, 'Metadata')

box_path = pth.join(data_base_path, 'Boxes')
metadata_url = 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'

train_box_url = 'https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv'

validation_box_url = 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv'

test_box_url = 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'



url_list = [metadata_url, train_box_url, validation_box_url, test_box_url]

path_list = [metadata_path] + [box_path]*3



os.makedirs(metadata_path, exist_ok=True)

os.makedirs(box_path, exist_ok=True)

pool = Pool(8)

for filename, status in pool.starmap(get_metadata, zip(url_list, path_list)):

    if status == 0:

        print(filename + ' is saved.')

    elif status == 1:

        print(filename + ' is already exist.')

    else:

        print('???')

pool.close()

pool.join()
label_filename = pth.join(metadata_path, 'class-descriptions-boxable.csv')

df = pd.read_csv(label_filename, header=None, index_col=None)

label_dict = dict(df.values)

dict(list(label_dict.items())[:10])
train_box_filename = pth.join(box_path, 'train-annotations-bbox.csv')

val_box_filename = pth.join(box_path, 'validation-annotations-bbox.csv')

test_box_filename = pth.join(box_path, 'test-annotations-bbox.csv')
df = pd.read_csv(train_box_filename)

labels = df['LabelName'].values

train_cnt = Counter(labels)

    

df = pd.read_csv(val_box_filename)

labels = df['LabelName'].values

val_cnt = Counter(labels)



df = pd.read_csv(test_box_filename)

labels = df['LabelName'].values

test_cnt = Counter(labels)
train_cnt.most_common(5), val_cnt.most_common(5), test_cnt.most_common(5)
train_cnt_dict = {label_dict[k]:v for k, v in train_cnt.items()}

val_cnt_dict = {label_dict[k]:v for k, v in val_cnt.items()}

test_cnt_dict = {label_dict[k]:v for k, v in test_cnt.items()}
wc = WordCloud(max_words=300

                , background_color='white'

#                 , width=1920, height=1080

#                 , mask=mask

#                 , color_func=MakeColor

                )

wc.generate_from_frequencies(train_cnt_dict)

# wc.to_file(pth.join(your_directory, wordcloud_filename))



plt.figure()

plt.axis("off")

plt.title('Train')

plt.imshow(wc, interpolation='bilinear')
wc = WordCloud(max_words=300

                , background_color='white'

                )

wc.generate_from_frequencies(val_cnt_dict)



plt.figure()

plt.axis("off")

plt.title('Validation')

plt.imshow(wc, interpolation='bilinear')
wc = WordCloud(max_words=300

                , background_color='white'

                )

wc.generate_from_frequencies(val_cnt_dict)



plt.figure()

plt.axis("off")

plt.title('Test')

plt.imshow(wc, interpolation='bilinear')