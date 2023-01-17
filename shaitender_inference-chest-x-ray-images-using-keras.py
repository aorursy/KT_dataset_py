import matplotlib.pyplot as plt
import resnet
img_channels = 1
img_rows = 256
img_cols = 256
nb_classes = 2

DATASET_PATH = '../input/cxr-dataset'

# There are two classes of images that we will deal with
disease_cls = ['effusion', 'nofinding']

model_path = '../input/cxr-resnet-18-256x256/cxr_resnet_18_256x256.h5'
def preprocess_img(img, mode):
    img = (img - img.min())/(img.max() - img.min())
    img = rescale(img, 0.25, multichannel=True, mode='constant')
    
    if mode == 'train':
        if np.random.randn() > 0:
            img = datagen.random_transform(img)
    return img
effusion_path = os.path.join(DATASET_PATH, disease_cls[0], '*')
effusion = glob.glob(effusion_path)
effusion = io.imread(effusion[0])

normal_path = os.path.join(DATASET_PATH, disease_cls[1], '*')
normal = glob.glob(normal_path)
normal = io.imread(normal[0])

f, axes = plt.subplots(1, 2, sharey=True)
f.set_figwidth(10)

axes[0].imshow(effusion, cmap='gray')
axes[1].imshow(normal, cmap='gray')
val_model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)

#load the saved weights
val_model.load_weights(model_path)
effusion_path = os.path.join(DATASET_PATH, disease_cls[0], '*')
effusion = glob.glob(effusion_path)
effusion = io.imread(effusion[-8])
plt.imshow(effusion,cmap='gray')
img = preprocess_img(effusion[:, :, np.newaxis], 'validation')
val_model.predict(img[np.newaxis,:])

