import os
import shutil
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageOps
from tqdm import tqdm
print(os.listdir('../input/flowers/flowers/'))
print(os.listdir('../input/flowers/flowers/sunflower/')[:10])
print(os.listdir('../input/flowers/flowers/dandelion/')[:10])
len(os.listdir('../input/flowers/flowers/sunflower/'))
len(os.listdir('../input/flowers/flowers/tulip/'))
len(os.listdir('../input/flowers/flowers/daisy/'))
len(os.listdir('../input/flowers/flowers/rose/'))
len(os.listdir('../input/flowers/flowers/dandelion/'))
734 + 984 + 769 + 784 + 1055
CLASS_NAME_SUNFLOWER = 'sunflower'
CLASS_NAME_TULIP = 'tulip'
CLASS_NAME_DAISY = 'daisy'
CLASS_NAME_ROSE = 'rose'
CLASS_NAME_DANDELION = 'dandelion'

CLASS_NAME_LIST = [
    CLASS_NAME_SUNFLOWER,
    CLASS_NAME_TULIP,
    CLASS_NAME_DAISY,
    CLASS_NAME_ROSE,
    CLASS_NAME_DANDELION,
]

DATASET_DIR_LIST = [
    '../input/flowers/flowers/%s/' % CLASS_NAME_SUNFLOWER,
    '../input/flowers/flowers/%s/' % CLASS_NAME_TULIP,
    '../input/flowers/flowers/%s/' % CLASS_NAME_DAISY,
    '../input/flowers/flowers/%s/' % CLASS_NAME_ROSE,
    '../input/flowers/flowers/%s/' % CLASS_NAME_DANDELION,
]
overall_file_path_list = []
for dataset_dir in DATASET_DIR_LIST:
    listdir = os.listdir(dataset_dir)
    for file_name in listdir:
        file_path = dataset_dir + file_name
        overall_file_path_list.append(file_path)
assert len(overall_file_path_list) == 4326
overall_file_path_list[:3]
file_path_list = []
file_name_list = []
width_list = []
height_list = []

for file_path in tqdm(overall_file_path_list):
    is_in = '.py' in file_path
    if is_in:
        continue
    is_in = '.pyc' in file_path
    if is_in:
        continue
    
    img = Image.open(file_path)
    width, height = img.size
    file_name = file_path.split('/')[-1]
    
    file_path_list.append(file_path)
    file_name_list.append(file_name)
    width_list.append(width)
    height_list.append(height)
    
    img.close()
img_meta_df = pd.DataFrame(
    columns=['file_path', 'file_name', 'width', 'height', 'class'],
    index=np.arange(0, len(file_path_list)))

img_meta_df.file_path = file_path_list
img_meta_df.file_name = file_name_list
img_meta_df.width = width_list
img_meta_df.height = height_list
def get_class_from_path(file_path):
    """
    対象のファイルパスに応じた、クラスのラベルを取得する。
    
    Parameters
    ----------
    file_path : str
        対象のファイルパス。
    
    Returns
    -------
    class_label : str
        クラスのラベルが設定される。e.g. 'sunflower'.
    """
    for class_name in CLASS_NAME_LIST:
        is_in = class_name in file_path
        if is_in:
            return class_name
    raise ValueError('Invalid file path : %s' % file_path)
img_meta_df['class'] = img_meta_df.file_path.apply(get_class_from_path)
img_meta_df[:3]
img_meta_df.width.max()
img_meta_df.height.max()
img_meta_df.width.min()
img_meta_df.height.min()
def get_short_side_px(width, height):
    """
    画像サイズの短辺を取得する。
    
    Parameters
    ----------
    width : int
        幅のピクセル。
    height : int
        高さのピクセル。
    
    Returns
    -------
    short_side_px : int
        短辺のピクセル。
    """
    if width < height:
        return width
    if height < width:
        return height
    return width
IMG_SIZE = 224
IMG_EXTENSION = 'jpg'
RESIZED_IMG_PATH = './resized/'
def get_resized_img_class_dir_path(class_name):
    """
    リサイズ後の画像を格納するためのディレクトリのパスを取得する。
    
    Parameters
    ----------
    class_name : str
        対象のクラス名。
    
    Returns
    -------
    dir_path : str
        対象のディレクトリパス。
    """
    dir_path = RESIZED_IMG_PATH + class_name + '/'
    return dir_path
def make_resized_img_class_dir(class_name):
    """
    リサイズ後の画像を格納するためのディレクトリを生成する。
    
    Parameters
    ----------
    class_name : str
        対象のクラス名。
    """
    dir_path = get_resized_img_class_dir_path(class_name=class_name)
    if os.path.exists(dir_path):
        return
    os.makedirs(dir_path)
def convert_to_cropped_center_img(img, crop_width, crop_height):
    """
    指定された画像の中央部分を用いて、指定のサイズにトリミングを行う。
    
    Parameters
    ----------
    img : Image
        トリミング対象の画像。
    crop_width : int
        トリミング後の画像幅。
    crop_height : int
        トリミング後の画像の高さ。
    
    Returns
    -------
    cropped_img : img
        トリミング後の画像。
    
    Notes
    -----
    返却値は、元の画像からのコピーとなる。
    """
    width, height = img.size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img
def convert_img_path_extension(img_path_or_name):
    """
    対象の画像パスもしくはファイル名の拡張子を、定義されている
    フォーマットに変換する。
    
    Parameters
    ----------
    img_path_or_name : str
        対象の画像パスもしくは画像のファイル名。
    
    Returns
    -------
    img_path_or_name : str
        変換後のパスもしくはファイル名。
    """
    img_path_or_name = '.'.join(img_path_or_name.split('.')[0:-1])
    img_path_or_name += '.%s' % IMG_EXTENSION
    return img_path_or_name
assert convert_img_path_extension(img_path_or_name='abc.png') == 'abc.' + IMG_EXTENSION
def save_reduced_size_img(file_path, file_name, class_name):
    """
    サイズを縮小した画像を保存する。
    
    Parameters
    ----------
    file_path : str
        リサイズ前のファイルのパス。
    file_name : str
        ファイル名。
    class_name : str
        クラス名。
        
    Returns
    -------
    保存されたファイルのパス。
    """
    file_name = convert_img_path_extension(
        img_path_or_name=file_name)
    dir_path = get_resized_img_class_dir_path(class_name=class_name)
    dest_file_path = dir_path + file_name
    
    img = Image.open(file_path)
    width, height = img.size
    short_side_px = get_short_side_px(width=width, height=height)
    resize_ratio = short_side_px / IMG_SIZE
    resized_width = int(width / resize_ratio)
    resized_height = int(height / resize_ratio)
    resized_img = img.resize(
        size=(resized_width, resized_height), resample=Image.BILINEAR)
    
    cropped_img = convert_to_cropped_center_img(
        img=resized_img, crop_width=IMG_SIZE, crop_height=IMG_SIZE)
    cropped_img.save(fp=dest_file_path, quality=95)
    
    img.close()
    resized_img.close()
    cropped_img.close()
    
    return dest_file_path
def save_pasted_img(file_path, file_name, class_name):
    """
    指定のサイズの画像に貼り付けが必要な画像（保存サイズよりも短辺が
    短い画像）に対しての保存処理を行う。
    
    Parameters
    ----------
    file_path : str
        貼り付け対象の画像のファイルのパス。
    file_name : str
        ファイル名。
    class_name : str
        クラス名。
        
    Returns
    -------
    保存されたファイルのパス。
    """
    file_name = convert_img_path_extension(
        img_path_or_name=file_name)
    dir_path = get_resized_img_class_dir_path(class_name=class_name)
    dest_file_path = dir_path + file_name
    
    canvas_img = Image.new(
        mode='RGB', size=(IMG_SIZE, IMG_SIZE), color='#ffffff')
    img = Image.open(file_path)
    width, height = img.size
    left = int(IMG_SIZE / 2 - width / 2)
    top = int(IMG_SIZE / 2 - height / 2)
    canvas_img.paste(img, box=(left, top))
    dir_path = get_resized_img_class_dir_path(class_name=class_name)
    
    canvas_img.save(fp=dest_file_path, quality=95)
    img.close()
    canvas_img.close()
    
    return dest_file_path
reduced_size_img_path_list = []
pasted_img_path_list = []
overall_path_list = []

for index, sr in tqdm(img_meta_df.iterrows()):
    file_path = sr['file_path']
    file_name = sr['file_name']
    width = sr['width']
    height = sr['height']
    short_side_px = get_short_side_px(width=width, height=height)
    class_name = sr['class']
    make_resized_img_class_dir(class_name=class_name)
    
    if short_side_px > IMG_SIZE:
        saved_file_path = save_reduced_size_img(
            file_path=file_path, file_name=file_name, class_name=class_name)
        reduced_size_img_path_list.append(saved_file_path)
        overall_path_list.append(saved_file_path)
        continue
    
    saved_file_path = save_pasted_img(
        file_path=file_path, file_name=file_name, class_name=class_name)
    pasted_img_path_list.append(saved_file_path)
    overall_path_list.append(saved_file_path)
img_meta_df['resized_img_path'] = overall_path_list
img_meta_df[:1]
len(reduced_size_img_path_list)
len(pasted_img_path_list)
def get_saved_img(img_path):
    """
    保存された画像単体を取得する。
    
    Parameters
    ----------
    img_path : str
        対象の画像のパス。
    
    Returns
    -------
    img : Image
        対象の画像。
    """
    img = Image.open(fp=img_path)
    return img
get_saved_img(img_path=reduced_size_img_path_list[0])
get_saved_img(img_path=reduced_size_img_path_list[1])
get_saved_img(img_path=pasted_img_path_list[0])
img = get_saved_img(img_path=pasted_img_path_list[0])
img.size
get_saved_img(img_path=pasted_img_path_list[1])
img = get_saved_img(img_path=pasted_img_path_list[1])
img.size
RANDOM_SEED = 42
shuffled_df = img_meta_df.sample(frac=1, random_state=RANDOM_SEED)
shuffled_df.drop_duplicates(subset=['file_name'], inplace=True)
shuffled_df.reset_index(drop=True, inplace=True)
TEST_DATA_NUM = 2000
TRAIN_DATA_NUM = len(shuffled_df) - TEST_DATA_NUM
TRAIN_DATA_NUM
len(shuffled_df)
shuffled_df[:5]
train_df = shuffled_df[:TRAIN_DATA_NUM]
test_df = shuffled_df[TRAIN_DATA_NUM:]
len(test_df)
assert len(train_df) == TRAIN_DATA_NUM
assert len(test_df) == TEST_DATA_NUM
TRAIN_IMG_DIR_PATH = './train_img/'
TEST_IMG_DIR_PATH = './test_img/'
if os.path.exists(TRAIN_IMG_DIR_PATH):
    shutil.rmtree(TRAIN_IMG_DIR_PATH)
if os.path.exists(TEST_IMG_DIR_PATH):
    shutil.rmtree(TEST_IMG_DIR_PATH)

if not os.path.exists(TRAIN_IMG_DIR_PATH):
    os.makedirs(TRAIN_IMG_DIR_PATH)
if not os.path.exists(TEST_IMG_DIR_PATH):
    os.makedirs(TEST_IMG_DIR_PATH)
for index, sr in train_df.iterrows():
    img_path = sr['resized_img_path']
    file_name = img_path.split('/')[-1]
    dest_path = TRAIN_IMG_DIR_PATH + file_name
    shutil.copy(src=img_path, dst=dest_path)
test_df[:1]
for index, sr in test_df.iterrows():
    img_path = sr['resized_img_path']
    file_name = img_path.split('/')[-1]
    dest_path = TEST_IMG_DIR_PATH + file_name
    shutil.copy(src=img_path, dst=dest_path)
assert len(train_df) == len(os.listdir(TRAIN_IMG_DIR_PATH))
assert len(test_df) == len(os.listdir(TEST_IMG_DIR_PATH))
len(train_df)
train_df[:1]
SUFFIX_MIRRORED = '_mirrored'
SUFFIX_LEFT_ROTATION = '_left_rotation'
SUFFIX_RIGHT_ROTATION = '_right_rotation'
train_data_list = train_df.to_dict(orient='record')
def save_mirrored_img(
        data_aug_meta_data_list, source_file_path, train_data_dict,
        probability=0.5):
    """
    指定された画像の左右反転画像の保存を行い、メタデータのリストへの
    追加を行う。
    
    Parameters
    ----------
    data_aug_meta_data_list : list of dicts
        メタデータの辞書の追加先となるリスト。
    source_file_path : str
        処理対象となるファイルのパス。
    train_data_dict : dict
        処理対象のファイルのメタデータを格納した辞書。この辞書をベースに
        新たに保存されるファイルのためのパス情報などが上書きされ、リストへ
        追加される。
    probability : float, default 0.5
        処理を行う確率。もし乱数が確率未満であれば、画像の生成処理をスキップする。
    
    Returns
    -------
    data_aug_meta_data_list : list of dicts
        データが追加された後のリスト。
    """
    random_val = np.random.random()
    if probability < random_val:
        return data_aug_meta_data_list
    
    data_aug_meta_data_dict = deepcopy(train_data_dict)
    file_name = source_file_path.split('/')[-1]
    dir_path = source_file_path.replace(file_name, '')
    file_name_except_ext = file_name.split('.')[0]
    dest_file_path = dir_path + file_name_except_ext + SUFFIX_MIRRORED + \
        '.%s'  % IMG_EXTENSION
    data_aug_meta_data_dict['resized_img_path'] = dest_file_path
    
    img = Image.open(fp=source_file_path)
    img = ImageOps.mirror(img)
    img.save(fp=dest_file_path, quality=95)
    img.close()
    
    data_aug_meta_data_list.append(data_aug_meta_data_dict)
    return data_aug_meta_data_list
np.random.random()
data_aug_meta_data_list = []
for train_data_dict in tqdm(train_data_list):
    data_aug_meta_data_list = save_mirrored_img(
        data_aug_meta_data_list=data_aug_meta_data_list,
        source_file_path=train_data_dict['resized_img_path'],
        train_data_dict=train_data_dict)
data_aug_meta_data_list[:1]
img = Image.open(data_aug_meta_data_list[0]['resized_img_path'])
img
img.close()
img = Image.open('./resized/tulip/3457017604_90e4de7480_m.jpg')
img
train_data_list.extend(data_aug_meta_data_list)
def save_rotated_img(
        data_aug_meta_data_list, source_file_path, train_data_dict, direction):
    """
    指定方向への回転画像の保存と、メタデータのリストへの追加を行う。
    
    Parameters
    ----------
    data_aug_meta_data_list : list of dicts
        メタデータの辞書の追加先となるリスト。
    source_file_path : str
        処理対象となるファイルのパス。
    train_data_dict : dict
        処理対象のファイルのメタデータを格納した辞書。この辞書をベースに
        新たに保存されるファイルのためのパス情報などが上書きされ、リストへ
        追加される。
    direction : str, 'left' or 'right'
        回転方向指定用の文字列。
    """
    
    data_aug_meta_data_dict = deepcopy(train_data_dict)
    file_name = source_file_path.split('/')[-1]
    dir_path = source_file_path.replace(file_name, '')
    file_name_except_ext = file_name.split('.')[0]
    if direction == 'left':
        suffix = SUFFIX_LEFT_ROTATION
    elif direction == 'right':
        suffix = SUFFIX_RIGHT_ROTATION
    dest_file_path = dir_path + file_name_except_ext + suffix + \
        '.%s'  % IMG_EXTENSION
    data_aug_meta_data_dict['resized_img_path'] = dest_file_path
    
    img = Image.open(fp=source_file_path)
    rgba_img = img.convert('RGBA')
    background_img = Image.new(mode='RGBA', size=(IMG_SIZE, IMG_SIZE), color='white')
    if direction == 'left':
        angle = -10
    elif direction =='right':
        angle = 10
    rotated_img = rgba_img.rotate(angle=angle, resample=Image.BICUBIC)
    composite_img = Image.composite(
        image1=rotated_img, image2=background_img, mask=rotated_img)
    composite_img = composite_img.convert('RGB')
    composite_img.save(fp=dest_file_path, quality=95)
    
    img.close()
    rgba_img.close()
    background_img.close()
    rotated_img.close()
    composite_img.close()
    
    data_aug_meta_data_list.append(data_aug_meta_data_dict)
    return data_aug_meta_data_list
data_aug_meta_data_list = []
for train_data_dict in tqdm(train_data_list):
    data_aug_meta_data_list = save_rotated_img(
        data_aug_meta_data_list=data_aug_meta_data_list,
        source_file_path=train_data_dict['resized_img_path'],
        train_data_dict=train_data_dict,
        direction='left')
    
    data_aug_meta_data_list = save_rotated_img(
        data_aug_meta_data_list=data_aug_meta_data_list,
        source_file_path=train_data_dict['resized_img_path'],
        train_data_dict=train_data_dict,
        direction='right')
img = Image.open(data_aug_meta_data_list[0]['resized_img_path'])
img
img = Image.open(data_aug_meta_data_list[1]['resized_img_path'])
img
train_data_list.extend(data_aug_meta_data_list)
len(train_data_list)
data_aug_train_df = pd.DataFrame(train_data_list)
shuffled_train_df = data_aug_train_df.sample(frac=1, random_state=RANDOM_SEED)
shuffled_train_df[:1]
sliced_train_df = shuffled_train_df.loc[:, ['class', 'resized_img_path']]
sliced_train_df.rename(columns={'resized_img_path': 'img_path'}, inplace=True)
sliced_train_df.to_csv('meta.csv', index=False, encoding='utf-8')
X_train_mm_path = './X_train.npy'
X_test_mm_path = './X_test.npy'
X_train_mm = np.memmap(
    X_train_mm_path, dtype=np.float16, mode='w+',
    shape=(len(sliced_train_df), IMG_SIZE, IMG_SIZE, 3))
sliced_train_df.reset_index(drop=True, inplace=True)
def set_img_arr_to_mm_arr(numpy_mm, img_path, index):
    """
    対象のmemmap配列に、対象の画像の配列の値を設定する。
    
    Parameters
    ----------
    numpy_mm : numpy.memmap
        値を設定するmemmap配列。
    img_path : str
        対象の画像のパス。
    index : int
        対象のデータのインデックス（0～）。
    """
    img = Image.open(fp=img_path)
    np_array = np.array(img)
    np_array = np_array.astype(np.float16, copy=False)
    np_array /= 255
    numpy_mm[index] = np_array
    img.close()
    del np_array
for index, sr in tqdm(sliced_train_df.iterrows()):
    img_path = sr['img_path']
    set_img_arr_to_mm_arr(numpy_mm=X_train_mm, img_path=img_path, index=index)
X_train_mm.shape
sliced_test_df = test_df.loc[:, ['class', 'resized_img_path']]
sliced_test_df.rename(columns={'resized_img_path': 'img_path'}, inplace=True)
sliced_test_df.to_csv('./test_meta.csv', index=False, encoding='utf-8')
X_test_mm = np.memmap(
    filename=X_test_mm_path, dtype=np.float16, mode='w+',
    shape=(len(sliced_test_df), IMG_SIZE, IMG_SIZE, 3))
sliced_test_df.reset_index(drop=True, inplace=True)
for index, sr in tqdm(sliced_test_df.iterrows()):
    img_path = sr['img_path']
    set_img_arr_to_mm_arr(
        numpy_mm=X_test_mm, img_path=img_path,
        index=index)
ls -l
shutil.rmtree('./resized/')
shutil.rmtree('./test_img/')
shutil.rmtree('./train_img/')
