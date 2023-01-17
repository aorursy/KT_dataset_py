import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt



%matplotlib inline
test_dir = '/kaggle/input/landmark-retrieval-2020/test/'

train_dir = '/kaggle/input/landmark-retrieval-2020/train/'

index_dir = '/kaggle/input/landmark-retrieval-2020/index/'

train = pd.read_csv('/kaggle/input/landmark-retrieval-2020/train.csv')
train.head()
landmark_list = train['landmark_id'].unique()

print('There are ' + str(len(landmark_list)) + ' locations in train data')

print(landmark_list)
def print_landmark(df, num):

    folder = []

    for i in range(3):

        folder.append(df['id'][num][i])

        

    _path = train_dir + '{}/{}/{}/'.format(folder[0], folder[1], folder[2]) 

    file_name = df['id'][num] + '.jpg'

    image_path = _path + file_name

    

    img = Image.open(image_path, 'r')

    plt.figure()

    plt.title('landmark_id: ' + str(df['landmark_id'][num]))

    plt.imshow(img)

    print(file_name)
print_landmark(train, 2)