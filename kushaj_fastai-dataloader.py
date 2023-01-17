!pip uninstall fastai -y
!pip install fastai
from fastai.vision.all import *
path = Path('../input/landmark-recognition-2020/')
Path.BASE_PATH = path
path.ls()
df = pd.read_csv(path/'train.csv', low_memory=False)
df.shape
def add_path(s):
    x = s[:3]
    return f'{x[0]}/{x[1]}/{x[2]}/{s}.jpg'
df['id'] = df['id'].apply(add_path)

df.head()
gld = DataBlock(blocks=(ImageBlock, CategoryBlock),
                splitter=RandomSplitter(),
                get_x=ColReader('id', pref='../input/landmark-recognition-2020/train/'),
                get_y=ColReader('landmark_id'),
                item_tfms=Resize(448),
                batch_tfms=aug_transforms(size=224))
dls = gld.dataloaders(df, bs=32)
dls.show_batch(max_n=9)