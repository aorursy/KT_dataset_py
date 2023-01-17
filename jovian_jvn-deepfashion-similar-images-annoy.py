%matplotlib inline

%reload_ext autoreload

%autoreload 
import jovian

from fastai.vision import *

import pandas as pd

import numpy as np
# get_category_names

with open('list_category_cloth.txt', 'r') as f:

    categories = []

    for i, line in enumerate(f.readlines()):

        if i > 1:

            categories.append(line.split(' ')[0])
# get image category map

with open('list_category_image.txt', 'r') as f:

    images = []

    for i, line in enumerate(f.readlines()):

        if i > 1:

            images.append([word.strip() for word in line.split(' ') if len(word) > 0])
#get train, valid, test split

with open('list_eval_partition.txt', 'r') as f:

    images_partition = []

    for i, line in enumerate(f.readlines()):

        if i > 1:

            images_partition.append([word.strip() for word in line.split(' ') if len(word) > 0])
data_df = pd.DataFrame(images, columns=['images', 'category_label'])

partition_df = pd.DataFrame(images_partition, columns=['images', 'dataset'])
data_df['category_label'] = data_df['category_label'].astype(int)
data_df = data_df.merge(partition_df, on='images')
data_df['dataset'].value_counts()
data_df['category'] = data_df['category_label'].apply(lambda x: categories[int(x) - 1])
data_df['category_label'].nunique()

# seems like few labels were merged in Dress label
data_df.head()
from pathlib import Path

images_path = Path('/home/jupyter/deepFashion')
data_source = (ImageList.from_df(df=data_df, path=images_path, cols='images')

                    .split_by_idxs((data_df[data_df['dataset']=='train'].index), (data_df[data_df['dataset']=='val'].index))

                    .label_from_df(cols='category')

              )
tmfs = get_transforms()



data = data_source.transform(tmfs, size=224).databunch(bs=128).normalize(imagenet_stats)
test_data = ImageList.from_df(df=data_df[data_df['dataset'] == 'test'], path=images_path, cols='images')

data.add_test(test_data)
# To maintain the order of images in train data, turning off shuffle

# data.train_dl = data.train_dl = data.train_dl.new(shuffle=False)
data.show_batch()
data
from fastai.metrics import accuracy, top_k_accuracy



top_3_accuracy = partial(top_k_accuracy, k=3)

top_5_accuracy = partial(top_k_accuracy, k=5)
learner = cnn_learner(data, models.resnet152, metrics=[accuracy, top_3_accuracy, top_5_accuracy])

learner.model = torch.nn.DataParallel(learner.model)
learner.lr_find()

learner.recorder.plot()
learner.fit_one_cycle(10, max_lr=1e-03)
learner.save('resnet50-224-freezed')
# .module because DataParallel was used

model = learner.model.module
class Hook():

    "Create a hook on `m` with `hook_func`."

    def __init__(self, m:nn.Module, hook_func:HookFunc, is_forward:bool=True, detach:bool=True):

        self.hook_func,self.detach,self.stored = hook_func,detach,None

        f = m.register_forward_hook if is_forward else m.register_backward_hook

        self.hook = f(self.hook_fn)

        self.removed = False



    def hook_fn(self, module:nn.Module, input:Tensors, output:Tensors):

        "Applies `hook_func` to `module`, `input`, `output`."

        if self.detach:

            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()

            output = (o.detach() for o in output) if is_listy(output) else output.detach()

        self.stored = self.hook_func(module, input, output)



    def remove(self):

        "Remove the hook from the model."

        if not self.removed:

            self.hook.remove()

            self.removed=True



    def __enter__(self, *args): return self

    def __exit__(self, *args): self.remove()

        

def get_output(module, input_value, output):

    return output.flatten(1)



def get_input(module, input_value, output):

    return list(input_value)[0]



def get_named_module_from_model(model, name):

    for n, m in model.named_modules():

        if n == name:

            return m

    return None
linear_output_layer = get_named_module_from_model(model, '1.4')

linear_output_layer
# getting all images in train

train_valid_images_df = data_df[data_df['dataset'] != 'test']

inference_data_source = (ImageList.from_df(df=train_valid_images_df, path=images_path, cols='images')

                    .split_none()

                    .label_from_df(cols='category')

              )
inference_data = inference_data_source.transform(tmfs, size=224).databunch(bs=32).normalize(imagenet_stats)
inference_dataloader = inference_data.train_dl.new(shuffle=False)
import time

img_repr_map = {}



with Hook(linear_output_layer, get_output, True, True) as hook:

    start = time.time()

    for i, (xb, yb) in enumerate(inference_dataloader):

        bs = xb.shape[0]

        img_ids = inference_dataloader.items[i*bs: (i+1)*bs]

        result = model.eval()(xb)

        img_reprs = hook.stored.cpu().numpy()

        img_reprs = img_reprs.reshape(bs, -1)

        for img_id, img_repr in zip(img_ids, img_reprs):

            img_repr_map[img_id] = img_repr

        if(len(img_repr_map) % 12800 == 0):

            end = time.time()

            print(f'{end-start} secs for 12800 images')

            start = end
img_repr_df = pd.DataFrame(img_repr_map.items(), columns=['img_id', 'img_repr'])
img_repr_df.shape
img_repr_df['label'] = [inference_data.classes[x] for x in inference_data.train_ds.y.items[0:img_repr_df.shape[0]]]
img_repr_df['label_id'] = inference_data.train_ds.y.items[0:img_repr_df.shape[0]]
from scipy.spatial.distance import cosine



def get_similar_images(img_index, n=10):

    start = time.time()

    base_img_id, base_vector, base_label  = img_repr_df.iloc[img_index, [0, 1, 2]]

    cosine_similarity = 1 - img_repr_df['img_repr'].apply(lambda x: cosine(x, base_vector))

    similar_img_ids = np.argsort(cosine_similarity)[-11:-1][::-1]

    end = time.time()

    print(f'{end - start} secs')

    return base_img_id, base_label, img_repr_df.iloc[similar_img_ids]



def show_similar_images(similar_images_df):

    images = [open_image(img_id) for img_id in similar_images_df['img_id']]

    categories = [learner.data.train_ds.y.reconstruct(y) for y in similar_images_df['label_id']]

    return learner.data.show_xys(images, categories)
base_image, base_label, similar_images_df = get_similar_images(30000)
print(base_label)

print(base_image)

open_image(base_image)
show_similar_images(similar_images_df)
from sklearn.manifold import TSNE



img_repr_matrix = [list(x) for x in img_repr_df['img_repr'].values]

tsne = TSNE(n_components=3, verbose=10, init='pca', perplexity=30, n_iter=500, n_iter_without_progress=100)

tsne_results_3 = tsne.fit_transform(img_repr_matrix)
img_repr_df['tsne1'] = tsne_results_3[:,0]

img_repr_df['tsne2'] = tsne_results_3[:,1]

img_repr_df['tsne3'] = tsne_results_3[:,2]
img_repr_df.to_parquet('deepFashion_similar_images')
import plotly_express as px

px.scatter_3d(img_repr_df, x='tsne1', y='tsne2', z='tsne3', color='label')
from annoy import AnnoyIndex
f = len(img_repr_df['img_repr'][0])

t = AnnoyIndex(f, metric='euclidean')
for i, vector in enumerate(img_repr_df['img_repr']):

    t.add_item(i, vector)

_  = t.build(inference_data.c)
def get_similar_images_annoy(img_index):

    start = time.time()

    base_img_id, base_vector, base_label  = img_repr_df.iloc[img_index, [0, 1, 2]]

    similar_img_ids = t.get_nns_by_item(img_index, 13)

    end = time.time()

    print(f'{(end - start) * 1000} ms')

    return base_img_id, base_label, img_repr_df.iloc[similar_img_ids[1:]]
# 230000, 130000, 190000

base_image, base_label, similar_images_df = get_similar_images_annoy(149999)
print(base_label)

open_image(base_image)
similar_images_df
show_similar_images(similar_images_df)
jovian.commit()