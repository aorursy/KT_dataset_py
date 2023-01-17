import os

import pickle

import random



import numpy as np

import tensorflow as tf

from tqdm  import tqdm

from PIL import Image, ImageFile

from scipy.io import savemat, loadmat

import matplotlib.pyplot as plt
# Utility script - parsing dataset config .pkl files and mAP computation

from roxfordparis_tools import configdataset, compute_map
# Adapted from: https://github.com/filipradenovic/revisitop/blob/master/python/example_process_images.py



def pil_loader(path):

    # to avoid crashing for truncated (corrupted images)

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # open path as file to avoid ResourceWarning 

    # (https://github.com/python-pillow/Pillow/issues/835)

    with open(path, 'rb') as f:

        img = Image.open(f)

        return img.convert('RGB')
# Adapted from: https://github.com/filipradenovic/revisitop/blob/master/python/example_process_images.py



def extract_features(test_dataset, cfg, model):

    """

    Generates file with serialized model outputs for each image from test_dataset.

    

    Arguments

    ---------

    test_dataset   : name of dataset of interest (roxford5k | rparis6k)

    cfg            : unserialized dataset config, containing annotation metadata

    model          : loaded Tensorflow baseline model object

    """

    

    print('>> Processing query images...', flush=True)

    Q = []

    for i in tqdm(np.arange(cfg['nq'])):

        qim = pil_loader(cfg['qim_fname'](cfg, i)).crop(cfg['gnd'][i]['bbx'])

        image_data = np.array(qim)

        image_tensor = tf.convert_to_tensor(image_data)

        Q.append(model(image_tensor)['global_descriptor'].numpy())

    Q = np.array(Q, dtype=np.float32)

    Q = Q.transpose()

    

    print('>> Processing index images...', flush=True)

    X = []

    for i in tqdm(np.arange(cfg['n'])):

        im = pil_loader(cfg['im_fname'](cfg, i))

        image_data = np.array(im)

        image_tensor = tf.convert_to_tensor(image_data)

        X.append(model(image_tensor)['global_descriptor'].numpy())

    X = np.array(X, dtype=np.float32)

    X = X.transpose()



    feature_dict = {'X': X, 'Q': Q}

    mat_save_path = "{}_delg_baseline.mat".format(test_dataset)

    print('>> Saving model outputs to: {}'.format(mat_save_path))

    savemat(mat_save_path, feature_dict)
# Adapted from: https://github.com/filipradenovic/revisitop/blob/master/python/example_evaluate.py



def run_evaluation(test_dataset, cfg, features_dir):

    ks = [1, 5, 10]

    gnd = cfg['gnd']

    features = loadmat(os.path.join(features_dir, '{}_delg_baseline.mat'.format(test_dataset)))



    Q = features['Q']

    X = features['X']

    sim = np.dot(X.T, Q)

    ranks = np.argsort(-sim, axis=0)



    # search for easy

    gnd_t = []

    for i in range(len(gnd)):

        g = {}

        g['ok'] = np.concatenate([gnd[i]['easy']])

        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])

        gnd_t.append(g)

    mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)



    # search for easy & hard

    gnd_t = []

    for i in range(len(gnd)):

        g = {}

        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])

        g['junk'] = np.concatenate([gnd[i]['junk']])

        gnd_t.append(g)

    mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)



    # search for hard

    gnd_t = []

    for i in range(len(gnd)):

        g = {}

        g['ok'] = np.concatenate([gnd[i]['hard']])

        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])

        gnd_t.append(g)

    mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)



    print('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))

    print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
data_root = '/kaggle/input/roxfordparis'

model_root = '/kaggle/input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model'
model = tf.saved_model.load(model_root).signatures['serving_default']
# Extract features for roxford5k & rparis6k datasets



should_extract_features = False



datasets = ['roxford5k', 'rparis6k']

for test_dataset in datasets:

    if should_extract_features:

        print('Processing dataset: {}'.format(test_dataset), flush=True)

        cfg = configdataset(test_dataset, data_root)

        extract_features(test_dataset, cfg, model)
# Evaluate on roxford5k & rparis6k datasets



datasets = ['roxford5k', 'rparis6k']

for test_dataset in datasets:

    print('Evaluating on dataset: {}'.format(test_dataset), flush=True)

    cfg = configdataset(test_dataset, data_root)

    run_evaluation(test_dataset, cfg, '/kaggle/input/delg-baseline-roxfordparis-output')

    print()
def get_random_image(test_dataset, cfg):

    query_image_id = random.choice(range(cfg['nq']))

    im = pil_loader(cfg['im_fname'](cfg, query_image_id))

    return query_image_id, im
def run_inference(pil_image, model):

    image_data = np.array(pil_image)

    image_tensor = tf.convert_to_tensor(image_data)

    return model(image_tensor)['global_descriptor'].numpy()
test_dataset = 'roxford5k' # (roxford5k | rparis6k)

cfg = configdataset(test_dataset, data_root)



im_id, im = get_random_image(test_dataset, cfg)

im_feats = run_inference(im, model)
features_dir = '/kaggle/input/delg-baseline-roxfordparis-output'



features = loadmat(os.path.join(features_dir, '{}_delg_baseline.mat'.format(test_dataset)))

X = features['X']

sim = np.dot(X.T, im_feats)

ranks = np.argsort(-sim, axis=0)
plt.imshow(np.array(im))

plt.title('Query Image')

plt.axis('off')
fig=plt.figure(figsize=(16, 12))

columns = 5

rows = 1

for i in range(1, columns*rows +1):

    img = np.array(pil_loader(cfg['im_fname'](cfg, ranks[i-1])))

    fig.add_subplot(rows, columns, i)

    plt.imshow(img)

    plt.axis('off')

print('Five most similar images from query dataset:')