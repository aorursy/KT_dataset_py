# python lang sepecific
from path import Path

# Data manipulation
import pandas
import numpy as np
import cv2

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
DATA = Path('../input')
DATA.listdir()
train_dir = DATA/'train'
train_fn = DATA/'train.csv'
train_labels = pandas.read_csv(train_fn)
train_labels.head()
print("Number of sample:", train_labels.shape[0])
label_id2name = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

label_name2id = {v: k for k, v in label_id2name.items()}

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_id2name[int(num)]
        row.loc[name] = 1
    return row
def show_img(im, figsize=None, ax=None, title=None):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if title:  ax.set_title(title)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax
# sample id
Id = train_labels['Id'].iloc[100]
Id
label = train_labels[train_labels['Id'] == Id]
label_id = label['Target'].values[0][0]
label_name = label_id2name[int(label_id)]
label_name
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    filters = ['green', 'red', 'blue', 'yellow']
    img = cv2.imread(train_dir/f'{Id}_{filters[i]}.png')
    show_img(img, ax=ax, title=f'{label_name}/ filter: {filters[i]}')
plt.show()
for key in label_id2name:
    train_labels[label_id2name[key]] = 0
train_labels = train_labels.apply(fill_targets, axis=1)
train_labels.head()
target_counts = train_labels.drop(["Id", "Target"], axis=1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(15, 15))
sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)
plt.show()
if "number_of_targets" in train_labels:
    train_labels = train_labels.drop(["number_of_targets"], axis=1)
train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"], axis=1).sum(axis=1)
count_prec = np.round(100 * train_labels["number_of_targets"].value_counts() / train_labels.shape[0], 2)
plt.figure(figsize=(20, 5))
sns.barplot(x=count_prec.index.values, y=count_prec.values, palette="Reds")
plt.xlabel("# target per image")
plt.ylabel("% of samples")
plt.show()
plt.figure(figsize=(15, 15))
sns.heatmap(train_labels[train_labels.number_of_targets > 1].drop(
    ["Id", "Target", "number_of_targets"], axis=1
).corr(), cmap="RdYlBu", vmin=-1, vmax=1)
plt.show()
def load_image(basepath, image_id):
    images = np.zeros((4, 512, 512))
    filters = ['green', 'red', 'blue', 'yellow']
    for i, f in enumerate(filters):
        images[i,:,:] = cv2.imread(basepath/f'{image_id}_{f}.png', cv2.IMREAD_GRAYSCALE) # cv2 return 3 same gray scale as 3 channel
    return images
    
def plot_image_row(image, subax, title):
    subax[0].imshow(image[0], cmap='Greens')
    subax[0].set_title(title)
    subax[1].imshow(image[1], cmap='Reds')
    subax[1].set_title('stained microtubules')
    subax[2].imshow(image[2], cmap='Blues')
    subax[2].set_title('stained nucles')
    subax[3].imshow(image[3], cmap='Oranges')
    subax[3].set_title('stained endoplasmatic reticulum')
    return subax

def make_title(sample_id):
    file_targets = train_labels.loc[train_labels.Id==sample_id, "Target"].values[0]
    title = " - "
    for n in file_targets:
        title += label_id2name[int(n)] + ' - '
    return title
your_choice = ["Lysosomes", "Endosomes"]
target_list = [label_name2id[name] for name in your_choice]
target_list
def check_subset(targets):
        return np.where(set(target_list).issuperset(set(targets)), 1, 0)
    
train_labels["check_col"] = train_labels.Target.apply(
            lambda l: check_subset(l)
        )

image_id = train_labels[train_labels.check_col==1].Id.values
img = load_image(train_dir, image_id[0])
img.shape
fig, ax = plt.subplots(2,4, figsize=(20,5*2))
for i, ax in enumerate(ax):
    img = load_image(train_dir, image_id[i])
    plot_image_row(img, ax, make_title(image_id[i]))
plt.show()
from fastai.conv_learner import *
from fastai.imports import *
from fastai.transforms import *
from fastai.dataset import *
from sklearn.metrics import fbeta_score
import warnings

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])
PATH = './'
INPUT = Path('../input')
TRAIN = INPUT/'train'
TEST = INPUT/'test'
LABELS = INPUT/'train.csv'
SAMPLE = INPUT/'sample_submission.csv'
metrics=[f2]
f_model = resnet18
def get_data(sz):
    tfms = tfms_from_model(f_model, sz)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg-v2')
train_names = list({f[:36] for f in os.listdir(TRAIN)})
test_names = list({f[:36] for f in os.listdir(TEST)})
tr_n, val_n = train_test_split(train_names, test_size=0.1, random_state=42)
len(tr_n), len(val_n), len(test_names)
def make_grb(path, id):
    filters = ['red', 'green', 'blue']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(str(path/f'{id}_{f}.png'), flags).astype(np.float32)/255
          for f in filters]
    return np.stack(img, axis=-1)
# img = [cv2.imread(TRAIN/f'{Id}_{f}.png', cv2.IMREAD_GRAYSCALE)/255 for f in filters]
class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.labels = pd.read_csv(LABELS).set_index('Id')
        self.labels['Target'] = [[int(i) for i in s.split()] for s in self.labels['Target']]
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        return make_grb(self.path, self.fnames[i])
    
    def get_y(self, i):
        labels = self.labels.loc[self.fnames[i]]['Target']
        return np.eye(len(label_id2name), dtype=np.float)[labels].sum(axis=0)
        
    @property
    def is_multi(self): return True # is multilabel classification
    
    @property
    def is_reg(self): return False
    
    def get_c(self): return len(label_id2name)
def get_data(sz, bs):
    stats = A([0.0808, 0.0530, 0.0550], [0.394, 0.321, 0.327])
    tfms = tfms_from_stats(stats, sz, crop_type=CropType.NO, tfm_y=TfmType.NO)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=1, classes=None)
    return md
bs = 16
sz = 256
md = get_data(sz,bs)

x,y = next(iter(md.trn_dl))
x.shape, y.shape
def display_imgs(x):
    columns = 4
    bs = x.shape[0]
    rows = min((bs+3)//4,4)
    fig=plt.figure(figsize=(columns*4, rows*4))
    for i in range(rows):
        for j in range(columns):
            idx = i+j*columns
            fig.add_subplot(rows, columns, idx+1)
            plt.axis('off')
            plt.imshow((x[idx,:,:,:3]*255).astype(np.int))
    plt.show()

display_imgs(np.asarray(md.trn_ds.denorm(x)))
plt.imshow(md.val_ds.denorm(to_np(x))[1]);
learn = ConvLearner.pretrained(f_model, md, metrics=metrics)
dir(learn)
learn.crit
learn.lr_find()
learn.sched.plot()
lr = 2e-2
learn.fit(lr, 1)
