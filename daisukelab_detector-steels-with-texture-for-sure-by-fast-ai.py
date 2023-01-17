import fastai

from fastai.vision import *
PATH = Path('../input/severstal-steel-defect-detection/')

IMAGES = PATH/'train_images'



def prepare_SteelDefectDet(path):

    df = pd.read_csv(path/'train.csv')

    df['filename'] = df.ImageId_ClassId.map(lambda s: s.split('_')[0])

    return df



df = prepare_SteelDefectDet(PATH)

df.head()
imgs_tmpl = ['000789191.jpg','00d7ae946.jpg','01d590c5f.jpg','01e501f99.jpg','023353d24.jpg',\

              '031614d60.jpg','03395a3da.jpg','063b5dcbe.jpg','06a86ee90.jpg','07cb85a8d.jpg','07e8fca73.jpg',\

              '08e21ba66.jpg','047681252.jpg','092c1f666.jpg','0a3bbea4d.jpg','0a46cc4bf.jpg','0a65bd8d4.jpg',\

              '0a76ac9b8.jpg','0b3a0fabe.jpg','0b50b417a.jpg','0d0c21687.jpg','0d22de6d4.jpg','0e09ff3bd.jpg',\

              '0e3ade070.jpg','0d0c21687.jpg','0d22de6d4.jpg','0ef4bff49.jpg','0faa71251.jpg','0fac62a3e.jpg',\

              '100de36e9.jpg','109fbcecf.jpg','110e63bfa.jpg']

len(imgs_tmpl)
files = df.filename.unique()



df_trn = pd.DataFrame({'filename': [f for f in files if f not in imgs_tmpl][:50] + imgs_tmpl,

                       'label': ['0'] * 50 + ['1'] * len(imgs_tmpl)})

df_trn.head()
SZ = 384

data = ImageDataBunch.from_df(IMAGES, df=df_trn, ds_tfms=get_transforms(), bs=8, size=SZ)

data.show_batch()
learn = cnn_learner(data, models.resnet18, metrics=accuracy).mixup()

learn.fit_one_cycle(1)

learn.unfreeze()

learn.fit_one_cycle(5)

learn.show_results()
def prepare_test_ds_dl(data_path, files=None, labels=None,

                       tfms=None, img_size=None, extension='.jpg'):

    if files is None:

        files = [str(f) for f in data_path/('*'+extension)]

    if labels is None:

        labels = [Path(f).parent.name for f in files]

    # Once create data bunch

    tmp_data = ImageDataBunch.from_lists(data_path, files, labels, valid_pct=0, ds_tfms=tfms, size=img_size)

    # Create dataloader again so that it surely set `shuffle=False`

    dl = torch.utils.data.DataLoader(tmp_data.train_ds, batch_size=tmp_data.batch_size, shuffle=False)

    dl = DeviceDataLoader(dl, tmp_data.device)

    return tmp_data.train_ds, dl



test_ds, test_dl = prepare_test_ds_dl(IMAGES, files=[IMAGES/f for f in files],

                                      labels=['1' if f in imgs_tmpl else '0' for f in files], img_size=SZ)
def predict_ext_dl(model, data_loader):

    preds, ys = [], []

    for X, y in data_loader:

        with torch.no_grad():

            out = model(X).softmax(1).cpu().detach().numpy()

            preds.append(out)

        ys.append(y)

    preds = np.concatenate(preds)

    ys   = np.concatenate(ys)

    return preds, ys



preds, ys = predict_ext_dl(learn.model, test_dl)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

yhat = np.argmax(preds, axis=-1)

print(f'acc = {accuracy_score(yhat, ys)}')

print(f'precision = {precision_score(yhat, ys)}')

print(f'recall = {recall_score(yhat, ys)}')

print(f'F1 = {f1_score(yhat, ys)}')

print(f'AUC = {roc_auc_score(ys, yhat)}')
def plot_predictions(yhat):

    fig, axs = plt.subplots(3, 4, figsize=(24, 8))

    for i, (ax, x_i) in enumerate(zip(axs.flat, np.where(yhat)[0])):

        ax.imshow(plt.imread(IMAGES/files[x_i]))

        ax.set_title(f'{x_i}th: p(y=1)={preds[x_i, 1]:.2f} p(y=0)={preds[x_i, 0]:.2f}')



plot_predictions(yhat)
yhat_for_sure = [1 if pred[1] > 0.95 else 0 for pred in preds]

plot_predictions(yhat_for_sure)
img_tmpl = np.array(files)[np.where(yhat_for_sure)[0]]

print(len(img_tmpl))

with open('possible_train_texture_images.txt', 'w') as f:

    f.writelines('\n'.join(img_tmpl))
! cat possible_train_texture_images.txt