%matplotlib inline

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV
import pandas as pd

from sklearn.model_selection import train_test_split
in_dir = '../input/dataset2-master/dataset2-master/images'
resource_dir = '/opt/conda/lib/python3.6/site-packages/kaggle_blood_cells'

df = pd.read_csv(os.path.join(resource_dir, 'stats.csv.py'))
# Plot Image
def plotImage(image_path):
    image = cv2.imread(image_path)  # BGR
    image = image[:, :, [2, 1, 0]]  # Reorder to RGB for Matplotlib display
    plt.imshow(image)
    return

plt.figure(figsize=(12,8))
plt.subplot(221)
plt.title('Lymphocyte'); plt.axis('off'); plotImage(os.path.join(in_dir, 'TRAIN/LYMPHOCYTE/_0_204.jpeg'))
plt.subplot(222)
plt.title('Monocyte'); plt.axis('off'); plotImage(os.path.join(in_dir, 'TRAIN/MONOCYTE/_0_9309.jpeg'))
plt.subplot(223)
plt.title('Neutrophil'); plt.axis('off'); plotImage(os.path.join(in_dir, 'TRAIN/NEUTROPHIL/_0_9742.jpeg'))
plt.subplot(224)
plt.title('Eosinophil'); plt.axis('off'); plotImage(os.path.join(in_dir, 'TRAIN/EOSINOPHIL/_5_907.jpeg'))
print('Training samples:')
train_dir = os.path.join(in_dir, "TRAIN")
num_samples = 0
for cell in os.listdir(train_dir):
    num_cells = len(os.listdir(os.path.join(train_dir, cell)))
    num_samples += num_cells
    print('Cell: {:15s}  num samples: {:d}'.format(cell, num_cells))
print('Total training samples: {:d}\n'.format(num_samples))

print('Test samples:')
test_dir = os.path.join(in_dir, "TEST")
num_samples = 0
for cell in os.listdir(test_dir):
    num_cells = len(os.listdir(os.path.join(test_dir, cell)))
    num_samples += num_cells
    print('Cell: {:15s}  num samples: {:d}'.format(cell, num_cells))
print('Total test samples: {:d}'.format(num_samples))
def plot_learning_curves(exp_id):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df[exp_id + '_epoch'][:200], df[exp_id + '_loss'][:200], label='Train', color='black')
    plt.plot(df[exp_id + '_epoch'][:200], df[exp_id + '_val_loss'][:200], label='Validation', color='blue')
    plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Loss -->')

    plt.subplot(1, 2, 2)
    plt.plot(df[exp_id + '_epoch'][:200], df[exp_id + '_acc'][:200], label='Train', color='black')
    plt.plot(df[exp_id + '_epoch'][:200], df[exp_id + '_val_acc'][:200], label='Validation', color='blue')
    plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Accuracy -->');
mean_img = np.load(os.path.join(resource_dir, 'mean_image_160x120.npy.py'))
std_img = np.load(os.path.join(resource_dir, 'std_image_160x120.npy.py'))
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1); plt.axis('off'); plt.title('Mean Image'); plt.imshow(mean_img.astype(np.uint8));
plt.subplot(1, 2, 2); plt.axis('off'); plt.title('Std  Image'); plt.imshow(std_img.astype(np.uint8));
plot_learning_curves('100_0')
plot_learning_curves('100_1')
plot_learning_curves('100_2')
plot_learning_curves('20_2')
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
plt.plot(df['100_4_epoch'][:200], df['100_4_loss'][:200], label='Type_0', color='blue')
plt.plot(df['100_4_epoch'], df['100_4_10_loss'], label='Type_1', color='green')
plt.plot(df['100_4_epoch'], df['100_4_11_loss'], label='Type_2', color='orange')
plt.plot(df['100_4_epoch'], df['100_4_12_loss'], label='Type_3', color='red')
plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Training Loss -->');

plt.subplot(2, 2, 2)
plt.plot(df['100_4_epoch'][:200], df['100_4_val_loss'][:200], label='Type_0', color='blue')
plt.plot(df['100_4_epoch'], df['100_4_10_val_loss'], label='Type_1', color='green')
plt.plot(df['100_4_epoch'], df['100_4_11_val_loss'], label='Type_2', color='orange')
plt.plot(df['100_4_epoch'], df['100_4_12_val_loss'], label='Type_3', color='red')
plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Validation Loss -->'); plt.ylim(0, 4);

plt.subplot(2, 2, 3)
plt.plot(df['100_4_epoch'][:200], df['100_4_acc'][:200], label='Type_0', color='blue')
plt.plot(df['100_4_epoch'], df['100_4_10_acc'], label='Type_1', color='green')
plt.plot(df['100_4_epoch'], df['100_4_11_acc'], label='Type_2', color='orange')
plt.plot(df['100_4_epoch'], df['100_4_12_acc'], label='Type_3', color='red')
plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Training Accuracy -->');

plt.subplot(2, 2, 4)
plt.plot(df['100_4_epoch'][:200], df['100_4_val_acc'][:200], label='Type_0', color='blue')
plt.plot(df['100_4_epoch'], df['100_4_10_val_acc'], label='Type_1', color='green')
plt.plot(df['100_4_epoch'], df['100_4_11_val_acc'], label='Type_2', color='orange')
plt.plot(df['100_4_epoch'], df['100_4_12_val_acc'], label='Type_3', color='red')
plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Validation Accuracy -->');
plt.figure(figsize=(16, 5))

plt.subplot(1, 2, 1)
plt.plot(df['20_2_epoch'][:200], df['20_2_loss'][:200], label='Dropout', color='red')
plt.plot(df['100_4_epoch'][:200], df['100_4_loss'][:200], label='Dropout+BN', color='green')
plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Training Loss -->');

plt.subplot(1, 2, 2)
plt.plot(df['20_2_epoch'][:200], df['20_2_val_loss'][:200], label='Dropout', color='red')
plt.plot(df['100_4_epoch'][:200], df['100_4_val_loss'][:200], label='Dropout+BN', color='green')
plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Validation Loss -->');
# function to draw plots
def plot_experiments(tag, labels):
    colors = ['blue', 'green', 'orange', 'red']
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    for i, label in enumerate(labels):
        plt.plot(df[tag + '_epoch'], df[tag + '_' + str(i) + '_loss'], label=label, color=colors[i])
    plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Training Loss -->');

    plt.subplot(2, 2, 2)
    for i, label in enumerate(labels):
        plt.plot(df[tag + '_epoch'], df[tag + '_' + str(i) + '_val_loss'], label=label, color=colors[i])    
    plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Validation Loss -->');

    plt.subplot(2, 2, 3)
    for i, label in enumerate(labels):
        plt.plot(df[tag + '_epoch'], df[tag + '_' + str(i) + '_acc'], label=label, color=colors[i])    
    plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Training Accuracy -->');

    plt.subplot(2, 2, 4)
    for i, label in enumerate(labels):
        plt.plot(df[tag + '_epoch'], df[tag + '_' + str(i) + '_val_acc'], label=label, color=colors[i])
    plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Validation Accuracy -->');
plot_experiments('100_5', ['Dropout=0.1', 'Dropout=0.2', 'Dropout=0.3']);
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(df['100_5_epoch'], df['100_5_0_loss'], label='Training', color='black')
plt.plot(df['100_5_epoch'], df['100_5_0_val_loss'], label='Validation', color='blue')
plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Loss -->'); plt.title('Dropout=0.1');

plt.subplot(1, 2, 2)
plt.plot(df['100_5_epoch'], df['100_5_1_loss'], label='Training', color='black')
plt.plot(df['100_5_epoch'], df['100_5_1_val_loss'], label='Validation', color='blue')
plt.legend(); plt.xlabel('Epochs -->'); plt.ylabel('Loss -->'); plt.title('Dropout=0.2');
plot_experiments('100_6', ['Adam', 'Adadelta', 'Adagrad', 'RMSProp']);
plot_experiments('100_7', ['Sigmoid', 'Tanh', 'ReLU']);
plot_experiments('100_8', ['Kernel:3x3', 'Kernel:5x5']);
class Data:
    def __init__(self, batch_size):
        self.in_ht, self.in_wd = 240, 320
        self.out_ht, self.out_wd = int(self.in_ht / 2), int(self.in_wd / 2)
        self.vld_portion = 0.1
        self.batch_size = {'TRAIN': batch_size, 'VALIDATION': batch_size, 'TEST': 1}
        self.in_dir = in_dir

        self.id2cell = pd.Series(os.listdir(os.path.join(self.in_dir, 'TRAIN')))
        self.cell2id = pd.Series(range(len(self.id2cell)), index=self.id2cell)

        self.x_trn_list, self.x_vld_list, self.y_trn, self.y_vld = self._get_names_labels(phase='TRAIN')
        self.x_tst_list, self.y_tst = self._get_names_labels(phase='TEST')
        self.steps_per_epoch = int(np.ceil(len(self.x_trn_list)/self.batch_size['TRAIN']))
        self.validation_steps = int(np.ceil(len(self.x_vld_list)/self.batch_size['TRAIN']))
        self.test_steps = int(np.ceil(len(self.x_tst_list)/self.batch_size['TEST']))        

        self.mean_img, self.std_img = self._get_stat_images()

    def _get_names_labels(self, phase):
        in_dir = os.path.join(self.in_dir, phase)
        if not os.path.exists(in_dir):
            raise IOError('Error: Directory {:s} does not exist.'.format(in_dir))

        x = list()
        labels = dict()
        for cell_id in self.id2cell.index:
            img_dir = os.path.join(in_dir, self.id2cell[cell_id])
            img_names = [a for a in os.listdir(img_dir) if a.endswith('.jpeg')]
            img_paths = [os.path.join(img_dir, img_name) for img_name in img_names]
            x += img_paths
            labels[cell_id] = np.zeros([len(img_paths), len(self.id2cell)], dtype=bool)  # One hot vector
            labels[cell_id][:, cell_id] = True

        y = np.concatenate([labels[a] for a in self.id2cell.index])
        
        if phase == 'TRAIN':
            trn_x_list, vld_x_list, y_trn, y_vld = \
                train_test_split(x, y, test_size=self.vld_portion, random_state=42, stratify=y, shuffle=True)

            return trn_x_list, vld_x_list, y_trn, y_vld
        else:
            return x, y

    def get_batch(self, phase):
        if phase == 'TRAIN':
            x_list = self.x_trn_list
            y = self.y_trn
        elif phase == 'VALIDATION':
            x_list = self.x_vld_list
            y = self.y_vld
        else:
            x_list = self.x_tst_list
            y = self.y_tst        

        # Allocated one-time memory for the batch
        x_batch = np.zeros((self.batch_size[phase], self.out_ht, self.out_wd, 3), dtype=float)
        y_batch = np.zeros((self.batch_size[phase], len(self.cell2id)), dtype=bool)

        src_idx = 0
        dst_idx = 0
        while True:
            img_path = x_list[src_idx]
            img = cv2.imread(img_path)
            if img is None:
                raise self.DataBatchError("Error: Can't open image: {:s}".format(img_path))

            img = cv2.resize(img, (self.out_wd, self.out_ht)).astype(float)

            # Normalize the image: Normalize each dimension
            img = (img - self.mean_img) / self.std_img            

            x_batch[dst_idx] = img
            y_batch[dst_idx] = y[src_idx]
            src_idx += 1
            dst_idx += 1

            if src_idx >= len(x_list):
                src_idx = 0

            if dst_idx >= self.batch_size[phase]:
                dst_idx = 0
                yield x_batch.copy(), y_batch.copy()

    def _get_stat_images(self):
        mean_img_path = os.path.join(resource_dir, 'mean_image_{:d}x{:d}.npy.py'.format(self.out_wd, self.out_ht))
        std_img_path = os.path.join(resource_dir, 'std_image_{:d}x{:d}.npy.py'.format(self.out_wd, self.out_ht))
        mean_img = np.load(mean_img_path)
        std_img = np.load(std_img_path)
        return mean_img, std_img
    
data = Data(batch_size=16)
model_path = os.path.join(resource_dir, '100_4_model_e1000.h5.py')
