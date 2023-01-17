import numpy as np
from matplotlib.image import imread

class ImageDatasetBuilder:
    """
    Given a DataFrame whose index is a set of image IDs (as with {train, test}.csv), returns featurized images.
    """
    def __init__(self, x_dim=100, y_dim=100, source='../data/train/images/', mask=False):
        """
        Builds the featurized image transform.
        
        x_dim: int
            The X dimension to crop the images to.
        y_dim: int
            The Y dimension to crop the images to.
        source: str
            Path to the folder containing the image files.
        mask: booleon
            If true, the underlying data is a mask. If false, the underlying data is RGB. If the data is RGB,
            we take just the R component and skip the GB, because the images are grayscale anyway.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.source = source
        self.mask = mask
        
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        r = np.asarray(
            list(
                map(lambda img_id: np.ravel(
                    imread(f'{self.source}/{img_id}.png')[:self.x_dim,:self.y_dim]
                ), X.index.values)
            )
        )
        return r if self.mask else r[:,::3]
class CropBoth:
    """
    Crops featurized images.
    """
    
    def __init__(self, ratio=0.9, cardinality=1):
        """
        ratio: float
            The percentage of the image to retain, expressed as a float ratio. Images are cropped in squares.
        cardinality: int
            The number of images to output. Each image will be a random crop of the original.
            
            If more than 1, this will result in multiplicatively more training cases created from the original 
            images.
        """
        self.ratio, self.cardinality = ratio, cardinality
        
    def fit(self, X, y=None):
        import math
        
        self.n_samples, self.n_pixels = X.shape[0], X.shape[1]
        self.img_dimension = int(self.n_pixels**(1/2))
        self.max_offset = math.floor((1 - self.ratio) * self.img_dimension)
        return self
    
    def transform(self, X, y=None):
        out_X = []
        out_y = []

        selection_size = int(self.ratio * self.img_dimension)
        
        for n in range(self.n_samples):
            for _ in range(self.cardinality):
                x_offset = np.random.randint(1, self.max_offset)
                y_offset = np.random.randint(1, self.max_offset)

                img_data = X[n]
                selection = np.ravel(img_data.reshape([self.img_dimension]*2)[x_offset:x_offset + selection_size,
                                                                              y_offset:y_offset + selection_size])
                out_X.append(selection)
                
                if y is not None:
                    img_data = y[n]
                    selection = np.ravel(img_data.reshape([self.img_dimension]*2)[x_offset:x_offset + selection_size,
                                                                                  y_offset:y_offset + selection_size])
                    out_y.append(selection)
        
        if y is not None:
            return np.asarray(out_X), np.asarray(out_y)
        else:
            return np.asarray(out_X)
class ScaleBoth:
    """
    Scales featurized images.
    """
    
    def __init__(self, shape=(128, 128)):
        self.shape = shape
        
    def fit(self, X, y=None):
        self.img_dimension = int(X.shape[1]**(1/2))
        return self
    
    def transform(self, X, y=None):
        from skimage.transform import rescale
        
        in_shape = [self.img_dimension]*2
        dim_mult = np.asarray(self.shape) / self.img_dimension
        
        X_out = np.asarray([np.ravel(rescale(x.reshape(in_shape), dim_mult)) for x in X])
        if y is not None:
            y_out = np.round(np.asarray([np.ravel(rescale(_y.reshape(in_shape), dim_mult)) for _y in y]))
        
        if y is not None:
            return X_out, y_out
        else:
            return X_out
class RefractBoth:
    """
    Refracts the edges of a featurized image.
    """
    
    def __init__(self, ratio=0.95):
        self.ratio = ratio
        
    def fit(self, X, y=None):
        self.img_dimension = int(X.shape[1]**(1/2))
        return self
    
    def transform(self, X, y=None):
        in_shape  = [self.img_dimension] * 2
        out_shape = [int(self.img_dimension * (1 + (1 - self.ratio) * 2))] * 2
        
        # Code from: https://stackoverflow.com/a/52472602/1993206
        def transformOne(x):
            x = x.reshape(in_shape)
            outy = outx = out_shape[0]
            
            iny, inx, *_ = x.shape
            iny -= 1; inx -= 1
            yoffs, xoffs = (outy - iny) // 2, (outx - inx) // 2

            Y, X = np.ogrid[:outy, :outx]
            out = x[np.abs((Y - yoffs + iny) % (2*iny) - iny), np.abs((X - xoffs + inx) % (2*inx) - inx)]
            return out
        
        X_out = np.asarray([np.ravel(transformOne(x)) for x in X])
        if y is not None:
            y_out = np.asarray([np.ravel(transformOne(_y)) for _y in y])
        
        if y is not None:
            return X_out, y_out
        else:
            return X_out
class FeaturizeDepth:
    """
    Extracts the depth feature (from depths.csv), which is later passed to the model.
    """
    
    def __init__(self):
        pass
    
    def transform(self, X):
        X_out = X.loc[:, 'z'].values[:, np.newaxis]
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit(X_out.astype(float)).transform(X_out.astype(float))
class Melt:
    """
    Transforms a flattened image matrix back into an image grid representation.
    
    This is necessary because the Conv2D layer in keras expects input in the form of a list of images with a certain number of 
    channels (RGB is three, this data is grayscale and thus only one). That is, it wants a four-dimensional matrix, whose axes 
    are record, x-value, y-value, channel. This transform mutates our data representation to match this format.
    """
    def __init__(self):
        pass
    
    def fit(self, X):        
        self.img_dimension = int(X.shape[1]**(1/2))
        return self

    def transform(self, X):
        return X.reshape((X.shape[0], self.img_dimension, self.img_dimension))[:,:,:,np.newaxis]
import pandas as pd

train = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
_depths = pd.read_csv("../input/depths.csv", index_col="id")
train = train.join(_depths)
test = _depths[~_depths.index.isin(train.index)]
del _depths


X = ImageDatasetBuilder(source='../input/train/images/').transform(train)
y = ImageDatasetBuilder(source='../input/train/masks/', mask=True).transform(train)
X_trans, y_trans = CropBoth(ratio=0.9, cardinality=1).fit(X, y).transform(X, y)
X_trans, y_trans = RefractBoth(ratio=0.9).fit(X_trans, y_trans).transform(X_trans, y_trans)
X_trans, y_trans = ScaleBoth(shape=(128, 128)).fit(X_trans, y_trans).transform(X_trans, y_trans)


X_trans = Melt().fit(X_trans).transform(X_trans)
y_trans = Melt().fit(y_trans).transform(y_trans)


X_feat = FeaturizeDepth().transform(train)


from sklearn.model_selection import train_test_split
X_train, X_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(X_trans, X_feat, y_trans, 
                                                                               test_size=0.15, random_state=42)
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


input_img = Input((X_trans.shape[1], X_trans.shape[2], 1), name='img')
input_features = Input((1,), name='feat')

c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2d_1') (input_img)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2d_2') (c1)
p1 = MaxPooling2D((2, 2), name='max_pooling2d_1') (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_3') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_4') (c2)
p2 = MaxPooling2D((2, 2), name='max_pooling2d_2') (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_5') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_6') (c3)
p3 = MaxPooling2D((2, 2), name='max_pooling2d_3') (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_7') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_8') (c4)
p4 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_4') (c4)

# Join features information in the depthest layer
f_repeat = RepeatVector(8*8, name='repeat_vector_1')(input_features)
f_conv = Reshape((8, 8, 1), name='reshape_1')(f_repeat)
p4_feat = concatenate([p4, f_conv], -1, name='concatenate_1')

c5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_9') (p4_feat)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_10') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='conv2d_transpose_1') (c5)
u6 = concatenate([u6, c4], name='concatenate_2')
c6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_11') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_12') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='conv2d_transpose_2') (c6)
u7 = concatenate([u7, c3], name='concatenate_3')
c7 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_13') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2d_14') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='conv2d_transpose_3') (c7)
u8 = concatenate([u8, c2], name='concatenate_4')
c8 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_15') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2d_16') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same', name='conv2d_transpose_4') (c8)
u9 = concatenate([u9, c1], axis=3, name='concatenate_5')
c9 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2d_17') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv2d_18') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid', name='conv2d_out') (c9)

clf = Model(inputs=[input_img, input_features], outputs=[outputs])
clf.compile(optimizer='adam', loss='binary_crossentropy')
clf.summary()
callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = clf.fit({'img': X_train, 'feat': X_feat_train}, y_train, batch_size=32, epochs=5,
                   callbacks=callbacks,
                   validation_data=({'img': X_test, 'feat': X_feat_test}, y_test))
pd.DataFrame(results.history).loc[:, ['val_loss', 'loss']].plot.line(figsize=(12, 6), fontsize=16, linewidth=5)
import seaborn as sns; sns.despine()
import matplotlib.pyplot as plt
plt.title("Validation and Training Loss Per Epoch", fontsize=24)
plt.ylabel("Epoch", fontsize=18)
plt.xlabel("Loss (Binary Cross-Entropy)", fontsize=18)
pass
np.random.seed(42)
r = np.random.randint(X_train.shape[0], size=10)
X_check = X_train[r]
y_check = y_train[r]
X_feat_check = X_feat[r]
y_predict = clf.predict({'img': X_check, 'feat': X_feat_check})
fig, axarr = plt.subplots(4, r.shape[0], figsize=(24, 9))

for n in range(r.shape[0]):
    axarr[0][n].imshow(X_check[n][:,:,0], cmap='gray')
    axarr[1][n].imshow(y_check[n][:,:,0], cmap='gray')
    axarr[2][n].imshow(np.round(y_predict[n][:,:,0]), cmap='gray')
    axarr[3][n].imshow(np.round(y_predict[n][:,:,0]) - y_check[n][:,:,0], cmap='RdYlGn', vmin=-1, vmax=1)
import pandas as pd

train = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
_depths = pd.read_csv("../input/depths.csv", index_col="id")
train = train.join(_depths)
test = _depths[~_depths.index.isin(train.index)]
del _depths


X_out = ImageDatasetBuilder(source='../input/test/images/').transform(test)
X_out_trans = ScaleBoth(shape=(128, 128)).fit(X_out).transform(X_out)
X_out_trans = Melt().fit(X_out_trans).transform(X_out_trans)

X_out_feat = FeaturizeDepth().transform(test)
# Free up RAM. Use %whos to see memory utilization.
del train
del test
del X_out
del X_trans
del y_trans
del X
del y
del X_test
del y_test
del X_train
del y_train
%time y_out_pred = np.round(clf.predict({'img': X_out_trans, 'feat': X_out_feat}))
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    _bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in _bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
y_out_pred = y_out_pred[:,:,:,0].reshape((y_out_pred.shape[0], 128**2))
y_out_pred = ScaleBoth(shape=(100, 100)).fit(y_out_pred).transform(y_out_pred)
y_out_pred = y_out_pred.reshape((y_out_pred.shape[0], 100, 100))
results = []
from tqdm import tqdm_notebook
for subarr in tqdm_notebook(y_out_pred):
    img = subarr
    result = RLenc(np.round(subarr))
    results.append(result)
import pandas as pd

train = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
_depths = pd.read_csv("../input/depths.csv", index_col="id")
train = train.join(_depths)
test = _depths[~_depths.index.isin(train.index)]
del _depths
del train

out = pd.DataFrame(results)
out = out.assign(id=test.index).rename(columns={0: 'rle_mask'})[['id', 'rle_mask']]
out.head()
out.set_index("id").to_csv("submission.csv")