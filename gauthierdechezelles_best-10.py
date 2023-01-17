# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import imageio
import os

from skimage.transform import resize


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('../input/lotofroads'):
    name_list = []
    for filename in filenames:
        name_list.append(filename)
    name_list.sort()
    predictions = np.array([resize(imageio.imread(os.path.join(dirname, filename)), (608, 608)) for filename in name_list])

print(predictions.shape)

#Slice into smaller 16x16 packs
sliced = np.array([arr.reshape(38, 16, -1, 16)
               .swapaxes(1,2)
               .reshape(-1, 16, 16) for arr in predictions])
print(sliced.shape)
avg = np.array([np.max(patch) for patch in sliced.reshape(-1,16,16)]).reshape(94, 38, 38)
print(avg.shape)
_10_quantile = np.quantile(avg.flatten(), 0.10)
print(_10_quantile)
_20_quantile = np.quantile(avg.flatten(), 0.20)
print(_20_quantile)
result = np.array([False if patch > _10_quantile else True for patch in avg.flatten()]).reshape(94, 38, 38)

from skimage import morphology
resultWithoutSmallCompenents = np.array([morphology.remove_small_objects(patch, min_size = 10) for patch in result])
def pretendant(image):
    result = np.full((38,38), False)
    for x in range(1,37):
        for y in range(1, 37):
            total = 0
            if image[x-1,y] == True: total+=1
            if image[x,y-1] == True: total+=1
            if image[x+1,y] == True: total+=1
            if image[x,y+1] == True: total+=1
            if total >= 2 and image[x,y] == False: result[x,y] = True
    return result

pretendants = [pretendant(im) for im in resultWithoutSmallCompenents]
res = np.zeros((94, 608, 608))
for im in range(len(resultWithoutSmallCompenents)):
    for i in range(38):
        for j in range(38):
            res[im, i*16:(i*16+16), j*16:(j*16+16)] = np.full((16, 16), resultWithoutSmallCompenents[im, i,j])
import matplotlib.pyplot as plt
num = 5
plt.imshow(pretendants[num])
plt.show()
plt.imshow(res[num])
plt.show()
plt.imshow(predictions[num])

from tqdm import tqdm

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > 0.5:
        return 1
    else:
        return 0

def mask_to_submission_strings(image):
    patch_size = 16
    im = image[0]
    img_number = image[1]
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, guesses):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for image in tqdm(guesses):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image))
            
combined = list(zip(res, [int(el.split("_")[1].split(".")[0]) for el in name_list]))
masks_to_submission("out.csv", combined)
print("done.")
