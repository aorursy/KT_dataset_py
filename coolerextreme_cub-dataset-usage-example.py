import torch

import matplotlib.pyplot as plt

from utils import get_cub_200_2011

from torchvision.utils import make_grid
train_set, train_loader = get_cub_200_2011(split='train_val', d_batch=64)

print(len(train_set), len(train_loader))
batch = next(iter(train_loader))
imgs, class_ids, caps = batch

print(imgs.size(), imgs.dtype)

print(class_ids.size(), class_ids.dtype)

print(caps)
plt.figure(figsize=(16, 16))

plt.imshow(make_grid(imgs, normalize=True, range=(-1, 1)).permute(1, 2, 0))
print('\n'.join([train_set.class_id_to_class_name[class_id] for class_id in class_ids.tolist()]))
print('\n'.join([train_set.decode_caption(cap) for cap in caps]))
!ls /kaggle/input/cub-200-2011/
imgs = torch.load('/kaggle/input/cub-200-2011/imgs_64x64.pth')

imgs.size(), imgs.dtype
metadata = torch.load('/kaggle/input/cub-200-2011/metadata.pth')

metadata.keys()
metadata['class_id_to_class_name']
metadata['word_id_to_word']
metadata['word_to_word_id']
metadata['img_id_to_encoded_caps']