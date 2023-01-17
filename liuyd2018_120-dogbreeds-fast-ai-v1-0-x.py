%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
print(torch.cuda.is_available(), torch.backends.cudnn.enabled)
path_model='/kaggle/working/'

path_input="/kaggle/input/"

label_df = pd.read_csv(f"{path_input}labels.csv")

label_df.head()
label_df.shape
label_df.pivot_table(index='breed',aggfunc=len).sort_values('id',ascending=False)
data = ImageDataBunch.from_csv(

                      path_input,

                      folder='train',

                      valid_pct=0.2,

                      ds_tfms=get_transforms(flip_vert=True,max_rotate=20., max_zoom=1.1),

                      size=224,

                      test='/kaggle/input/test/test',

                      suffix='.jpg',

                      bs=64,

                      num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(8,8))
[print(len(e)) for e in [data.train_ds, data.valid_ds, data.test_ds]]
files = os.listdir(f'{path_input}train/')[:5]

img = plt.imread(f'{path_input}train/{files[0]}')

plt.imshow(img)
#learner = Learner(data, models.resnet50, metrics=[accuracy], )

learner = create_cnn(data,models.resnet50,metrics=[accuracy],model_dir=f'{path_model}')
learner.fit_one_cycle(3)
np.set_printoptions(precision=6, suppress=True)

test_result = learner.get_preds(ds_type=DatasetType.Test)
for i in range(0, 12):

    print(np.array(test_result[0][1][i*10:i*10+10]))
pd.options.display.float_format = '{:.6f}'.format

df = pd.DataFrame(np.array(test_result[0]))

df.columns = data.classes

df.head()
df.shape
# insert clean ids - without folder prefix and .jpg suffix - of images as first column

df.insert(0, "id", [e.name[:-4] for e in data.test_ds.x.items])
df.head()
df.to_csv(f"dog-breed-identification-submission.csv", index=False)