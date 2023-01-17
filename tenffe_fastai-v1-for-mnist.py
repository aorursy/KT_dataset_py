from fastai.vision import *

from fastai.callbacks import *

from tqdm import tqdm_notebook

from cv2 import cv2 as cv
cv.__version__
path = Path('../input')

path.ls()
train_df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'test.csv')

sub = pd.read_csv(path/'sample_submission.csv')
train_df.head()
test_df.head()
sub.head()
!ls /kaggle/working
working_dir = Path('/kaggle/working/mnist_data')

train_dir = (working_dir/'train')

test_dir = (working_dir/'test')
working_dir.mkdir(exist_ok=True)

train_dir.mkdir(exist_ok=True)

test_dir.mkdir(exist_ok=True)

working_dir.ls()
labels = train_df['label'].unique()

labels
for i in tqdm_notebook(labels):

    label_i_path = train_dir/f'{i}'

    label_i_path.mkdir(exist_ok=True)
def save_array2image(save_dir, index, array):

    plt.imsave(save_dir/f'{index}.jpg', array)
# len_train = len(train_df)

# for i in tqdm_notebook(range(len_train)):

#     row = train_df.iloc[i]

#     label = row['label'] 

#     image_array = row[1:]

#     image_array = np.reshape(np.array(image_array), (28, 28))

#     save_dir = train_dir/str(label)

#     save_array2image(save_dir, f'train_{i}', image_array)
!ls /kaggle/working/mnist_data/train
# len_test = len(test_df)

# for i in tqdm_notebook(range(len_test)):

#     image_array = test_df.iloc[i]

#     image_array = np.reshape(np.array(image_array), (28, 28))

#     save_dir = test_dir

#     save_array2image(save_dir, f'test_{i}', image_array)
!ls /kaggle/working/mnist_data/test
tfms = get_transforms(do_flip=False, max_rotate=20, max_zoom=1.2, max_lighting=0.1, max_warp=0.1)
test_src = ImageList.from_folder(extensions='.jpg', path='./mnist_data/test')

test_src
train_src = ImageList.from_folder(extensions='.jpg', path='./mnist_data/train')

train_src = train_src.split_by_rand_pct(0.2)

train_src = train_src.label_from_folder()

train_src = train_src.add_test(test_src)

train_src = train_src.transform(tfms)



train_data = train_src.databunch().normalize()

train_data
train_data.show_batch(rows= 5, ds_type=DatasetType.Test)
esc = partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.03, patience=3)

# smc = partial(SaveModelCallback, every='improvement', monitor='accuracy', name='best')
def train_for_leaner(learn):

    # freeze cycle

    learn.freeze()

    learn.lr_find()

    learn.recorder.plot(suggestion=True)

    lr_fr = learn.recorder.min_grad_lr

    

    learn.fit(3, lr_fr)

    

    # unfreeze cycle

    learn.unfreeze()

    learn.lr_find()

    learn.recorder.plot(suggestion=True)

    lr_un = learn.recorder.min_grad_lr

    

    learn.fit(5, slice(lr_un, lr_fr/10))

    learn.fit(10, slice(lr_un/3, lr_fr/30))

    return learn
learner_1 = cnn_learner(train_data, models.resnet101, metrics=accuracy, ps=0.6, wd = 0.005, callback_fns=esc).mixup().to_fp16()
# learner_1 = train_for_leaner(learner_1)
learner_1.show_results(ds_type=DatasetType.Valid)
pred_1, _ = learner_1.get_preds(ds_type = DatasetType.Test)

pred_1.shape
pred_1_cat = np.argmax(pred_1, axis=1)

pred_1_cat = pred_1_cat.reshape(-1, 1)
learner_2 = cnn_learner(train_data, models.densenet121, metrics=accuracy, ps=0.6, wd = 0.005).mixup().to_fp16()
# learner_2 = train_for_leaner(learner_2)
pred_2, _ = learner_2.get_preds(ds_type = DatasetType.Test)

pred_2_cat = np.argmax(pred_2, axis=1)

pred_2_cat = np.array(pred_2_cat.reshape(-1, 1))
pred = pred_1 * 0.5 + pred_2 * 0.5

pred_cat = np.argmax(pred, axis=1)

pred_cat = np.array(pred_cat.reshape(-1, 1))

pred_cat
sub['Label'] = pred_cat

sub.head()
sub.to_csv('submission_1.csv', index=False)
sub_test = pd.read_csv('submission_1.csv')

sub_test.head()