import os

from typing import List



import cv2

import numpy as np

import pandas as pd

from keras.models import load_model

from keras.utils import Sequence

from tqdm import tqdm
SIZE = 224

HEIGHT = 137

WIDTH = 236



NUM_CLASSES_ROOT = 168

NUM_CLASSES_VOWEL = 11

NUM_CLASSES_CONSONANT = 7



PQT_PREFIX = '/kaggle/input/bengaliai-cv19'

MODEL_PREFIX = '/kaggle/input/bengali-serialized-models'

TRAIN_PQTS = [f'{PQT_PREFIX}/train_image_data_{i}.parquet' for i in range(4)]

TEST_PQTS = [f'{PQT_PREFIX}/test_image_data_{i}.parquet' for i in range(4)]

TRAIN_PP_FOLDER = f'train_pp'

TEST_PP_FOLDER = f'test_pp'



MODELS = ['resnet18', 'resnet34', 'densenet121', 'seresnet18']

MODEL_BATCH_SIZES = {

    'resnet18': 100,

    'resnet34': 50,

    'densenet121': 40,

    'seresnet18': 30,

}



os.makedirs(TRAIN_PP_FOLDER, exist_ok=True)

os.makedirs(TEST_PP_FOLDER, exist_ok=True)
def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax





def crop_resize(img0, size=SIZE, pad=16):

    # crop a box around pixels large than the threshold 

    # some images contain line at the sides

    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)

    # cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax, xmin:xmax]

    # remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax - xmin, ymax - ymin

    l = max(lx, ly) + pad

    # make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant')

    return cv2.resize(img, (size, size))





def preprocess(parquets, out_folder):

    for fname in parquets:

        df = pd.read_parquet(fname)

        data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

        for idx in tqdm(range(len(df))):

            name = df.iloc[idx, 0]

            img = (data[idx] * (255.0 / data[idx].max())).astype(np.uint8)

            img = crop_resize(img)

            cv2.imwrite(f'{out_folder}/{name}.png', img)
def get_files(folder_path: str) -> List[str]:

    files = []

    for fname in os.listdir(folder_path):

        fpath = os.path.join(folder_path, fname)

        if os.path.isfile(fpath):

            files.append(fpath)

    return sorted(files)





class ImageSequence(Sequence):

    

    def __init__(self, files: List[str], batch_size: int):

        self._files = files

        self._batch_size = batch_size

    

    def __len__(self) -> int:

        return int(np.ceil(len(self._files) / float(self._batch_size)))

    

    def __getitem__(self, idx: int) -> np.ndarray:

        xs = None

        batch_files = self._files[idx * self._batch_size:(idx + 1) * self._batch_size]

        for file_path in batch_files:

            arr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            arr = np.expand_dims(np.expand_dims(arr, axis=0), axis=3)

            xs = np.concatenate([xs, arr], axis=0) if xs is not None else arr

        xs = np.repeat(xs / 255.0, 3, axis=3)

        return xs
models = [load_model(f'{MODEL_PREFIX}/{m}.checkpoint', compile=False) for m in tqdm(MODELS)]
# preprocess(TRAIN_PQTS, TRAIN_PP_FOLDER)

preprocess(TEST_PQTS, TEST_PP_FOLDER)
# files = get_files(TRAIN_PP_FOLDER)

files = get_files(TEST_PP_FOLDER)
def suppress_non_max(a: np.ndarray) -> np.ndarray:

    return a * np.equal(a, np.repeat(np.expand_dims(np.max(a, axis=1), axis=1), a.shape[1], axis=1)).astype(np.float32)

r_sub_arr, v_sub_arr, c_sub_arr = None, None, None



for model_name, model in zip(MODELS, models):

    print(f'Generating submission using model {model_name}')

    

    batch_size=MODEL_BATCH_SIZES[model_name]

    seq = ImageSequence(files=files, batch_size=batch_size)

    (model_r_sub_arr, model_v_sub_arr, model_c_sub_arr) = model.predict_generator(seq, verbose=1)

    model_r_sub_arr = suppress_non_max(model_r_sub_arr)

    model_v_sub_arr = suppress_non_max(model_v_sub_arr)

    model_c_sub_arr = suppress_non_max(model_c_sub_arr)

    r_sub_arr = r_sub_arr + model_r_sub_arr if r_sub_arr is not None else model_r_sub_arr

    v_sub_arr = v_sub_arr + model_v_sub_arr if v_sub_arr is not None else model_v_sub_arr

    c_sub_arr = c_sub_arr + model_c_sub_arr if c_sub_arr is not None else model_c_sub_arr



r_sub_arr = np.argmax(r_sub_arr, axis=1)

v_sub_arr = np.argmax(v_sub_arr, axis=1)

c_sub_arr = np.argmax(c_sub_arr, axis=1)
submission = []





for i, fpath in enumerate(files):

    image_id = os.path.splitext(os.path.basename(fpath))[0]

    submission.append({'row_id': f'{image_id}_consonant_diacritic', 'target': c_sub_arr[i]})

    submission.append({'row_id': f'{image_id}_grapheme_root', 'target': r_sub_arr[i]})

    submission.append({'row_id': f'{image_id}_vowel_diacritic', 'target': v_sub_arr[i]})
sub_df = pd.DataFrame(submission, columns = ['row_id','target'])
sub_df.head()
sub_df.to_csv('submission.csv', index=False)