from fastai import *

from fastai.vision import *

DATAPATH = Path('/kaggle/input/Kannada-MNIST/')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def get_data_labels(csv,label):

    fileraw = pd.read_csv(csv)

    labels = fileraw[label].to_numpy()

    data = fileraw.drop([label],axis=1).to_numpy(dtype=np.float32).reshape((fileraw.shape[0],28,28))

    data = np.expand_dims(data, axis=1)

    return data, labels
train_data, train_labels = get_data_labels(DATAPATH/'train.csv','label')

test_data, test_labels = get_data_labels(DATAPATH/'test.csv','id')

other_data, other_labels = get_data_labels(DATAPATH/'Dig-MNIST.csv','label')

train_data.shape, train_labels.shape, test_data.shape, test_labels.shape, other_data.shape, other_labels.shape
plt.title(f'Training Label: {train_labels[2]}')

plt.imshow(train_data[2,0],cmap='gray');
np.random.seed(42)

ran_10_pct_idx = (np.random.random_sample(train_labels.shape)) < .1



train_90_labels = train_labels[np.invert(ran_10_pct_idx)]

train_90_data = train_data[np.invert(ran_10_pct_idx)]



valid_10_labels = train_labels[ran_10_pct_idx]

valid_10_data = train_data[ran_10_pct_idx]
class ArrayDataset(Dataset):

    "Dataset for numpy arrays based on fastai example: "

    def __init__(self, x, y):

        self.x, self.y = x, y

        self.c = len(np.unique(y))

    

    def __len__(self):

        return len(self.x)

    

    def __getitem__(self, i):

        return self.x[i], self.y[i]
train_ds = ArrayDataset(train_90_data,train_90_labels)

valid_ds = ArrayDataset(valid_10_data,valid_10_labels)

other_ds = ArrayDataset(other_data, other_labels)

test_ds = ArrayDataset(test_data, test_labels)
bs = 128

databunch = DataBunch.create(train_ds, valid_ds, test_ds=test_ds, bs=bs)
def conv2(ni,nf,stride=2,ks=3): return conv_layer(ni,nf,stride=stride,ks=ks)
smallConvolutional = nn.Sequential(

    conv2(1,8,ks=5),

    conv2(8,16),

    conv2(16,32),

    conv2(32, 16),

    conv2(16, 10),

    Flatten()

)
learn = Learner(databunch, smallConvolutional, loss_func = nn.CrossEntropyLoss(), metrics=[accuracy] )
learn.fit_one_cycle(8)
preds, ids = learn.get_preds(DatasetType.Test)

y = torch.argmax(preds, dim=1)
submission = pd.DataFrame({ 'id': ids,'label': y })
submission.to_csv(path_or_buf ="submission.csv", index=False)