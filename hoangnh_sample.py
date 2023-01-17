%load_ext autoreload

%autoreload 2

!pip install solt==0.1.9

!pip install seaborn==0.11



import os

import math

from collections import OrderedDict



import numpy as np

import matplotlib.pyplot as plt

from scipy.special import softmax

from sklearn.metrics import average_precision_score

from sklearn.model_selection import GroupKFold

import pandas as pd

import seaborn as sns

import cv2

import torch

import torch.nn as nn

from torch.nn import CrossEntropyLoss

from torch.utils.data import Dataset

from torch.optim import Adam

import torchvision



from tqdm.notebook import tqdm

import solt

import solt.transforms as slt
def to_cpu(x: torch.Tensor or torch.cuda.FloatTensor, required_grad=False, use_numpy=True):

    x_cpu = x



    if isinstance(x, torch.Tensor):

        if x.is_cuda:

            if use_numpy:

                x_cpu = x.detach().cpu().numpy()

            elif required_grad:

                x_cpu = x.cpu()

            else:

                x_cpu = x.cpu().required_grad_(False)

        elif use_numpy:

            x_cpu = x.detach().numpy()



    return x_cpu





def count_data_amounts_per_target(df, target_name):

    amount_list = []

    target_list = df[target_name].unique()

    

    for target in target_list:

        if target == np.nan:

            df_by_target = df[df[target_name].isnull()]

        else:

            df_by_target = df[df[target_name] == target]

        num_samples = len(df_by_target.index)

        amount_list.append(num_samples)

        print(f'There are {num_samples} {target}-class samples.')

    return amount_list





def display_random_samples(df, root_path):

    """Function to display random samples"""

    target_names = {1: 'Real', 0: 'Synthesized'}

    pos = 1

    n_pairs = 10

    fig, axs = plt.subplots(2, n_pairs, figsize=(15, 4), sharex=True, sharey=True)

    

    # Loop through targets (0, 1)

    for target in range(2):

        # Loop through pair indices

        for pair_i in range(n_pairs):

            # Select rows by and target

            df_tmp = df[df['Validity'] == target]

            # Choose a random index

            len_df = len(df_tmp.index)

            random_id = np.random.randint(0, len_df)

            # Obtain image path corresponding to the selected index

            image_path = df_tmp.iloc[random_id]['Filename']

            image_fullname = os.path.join(root_path, image_path)

            # Read image

            img = cv2.imread(image_fullname, cv2.IMREAD_GRAYSCALE)

            # Show image and format its subplot

            axs[target, pair_i].imshow(img, cmap='gray')

            axs[target, pair_i].set_axis_off()

            if pair_i == 0:

                axs[target, pair_i].set_title(f'{target_names[target]}')

            pos += 1



    plt.tight_layout()

    plt.show()





def collect_and_update_metrics(metrics_collector, new_record):

    # Collect data

    metrics_collector['loss'].append(new_record['new_loss'])

    np_preds = to_cpu(torch.argmax(new_record['new_preds'], dim=1))

    np_probs = to_cpu(torch.softmax(new_record['new_preds'], dim=1))[:, 1]

    np_labels = to_cpu(new_record['new_labels'])

    metrics_collector['preds'] += np_preds.tolist()

    metrics_collector['labels'] += np_labels.tolist()

    metrics_collector['probs'] += np_probs.tolist()



    # Calculate metrics

    _loss = np.mean(np.array(metrics_collector['loss']))

    _ap = average_precision_score(metrics_collector['labels'], metrics_collector['probs'])



    # Put metrics into the dictionary metrics for display in progress bar    

    metrics_display = OrderedDict()

    metrics_display['loss'] = _loss

    metrics_display['average_precision'] = _ap

    return metrics_collector, metrics_display
root = "/kaggle/input/yo-medicalai-challenge/chest_xray_vadility_classification"

img_root = os.path.join(root, "train_validation", "images")

input_meta = os.path.join(root, "train_validation", "challenge_all_meta_final.csv")
df_meta = pd.read_csv(input_meta)

df_meta.head(10)
display_random_samples(df_meta, img_root)
validity_targets = df_meta['Validity'].unique()



print(f'Validity targets: {validity_targets}')
g1 = sns.displot(df_meta, x='Validity', discrete=True)

g1.set(xticks=[0, 1])
def clean_metadata(df):

    # TODO:

    return df



df_meta = clean_metadata(df_meta)
g1 = sns.displot(df_meta, x='Validity', discrete=True)

g1.set(xticks=[0, 1])
n_fold = 5

splitter = GroupKFold(n_splits=n_fold)

splitter_iter = splitter.split(df_meta, df_meta['Validity'], groups=df_meta['ID'])

cv_folds = [(train_idx, val_idx) for (train_idx, val_idx) in splitter_iter]
# Training and validation indices

train_idx, val_idx = cv_folds[0]



# Training and validation samples

train_df = df_meta.iloc[train_idx]

val_df = df_meta.iloc[val_idx]
train_df.head(10)
# TODO: implement, for reproducibility of the folds
class DataFrameDataset(Dataset):

    """Dataset based on ``pandas.DataFrame``.

        

    Parameters

    ----------

    root : str

        Path to root directory of input data.

    meta_data : pandas.DataFrame

        Meta data of data and labels.

    transform : callable, optional

        Transformation applied to row of :attr:`meta_data` (the default is None, which means to do nothing).

    std_size: tuple

        The size that input images are standardized to (Default: (224, 224))

    mean: tuple

        Mean to normalize image intensities (Default: (0.485, 0.456, 0.406))

    std: tuple

        Standard deviation to normalize image intensities (Default: (0.229, 0.224, 0.225))

    

    Raises

    ------

    TypeError

        `root` must be `str`.

    TypeError

        `meta_data` must be `pandas.DataFrame`.

    

    """



    def __init__(self, root, meta_data, 

                 transform=None, 

                 std_size=(64, 64), 

                 mean=(0.5, 0.5, 0.5), 

                 std=(0.5, 0.5, 0.5)):

        if not isinstance(root, str):

            raise TypeError("`root` must be `str`")

        if not isinstance(meta_data, pd.DataFrame):

            raise TypeError("`meta_data` must be `pandas.DataFrame`, but found {}".format(type(meta_data)))

        self.root = root

        self.meta_data = meta_data        

        self.transform = transform        

        self.std_size = std_size

        self.mean = mean

        self.std = std

        

    def parse_item(self, root, entry, transform):

        # Read image from path

        img_fullname = os.path.join(root, entry['Filename'])

        img = cv2.imread(img_fullname, cv2.IMREAD_COLOR)        

                        

        stats = {'mean': self.mean, 'std': self.std}

        # Apply transformations into image

        trf_output = transform({'image': img}, return_torch=True, normalize=True, **stats)

        

        if 'Validity' in entry:

            # Training/validation regime

            return {'Filename': entry['Filename'], 'data': trf_output['image'], 'Validity': int(entry['Validity'])}

        else:

            # Testing regime (no info on Validity)

            return {'Filename': entry['Filename'], 'data': trf_output['image']}



    def __getitem__(self, index):

        """Get ``index``-th parsed item of :attr:`meta_data`.

        

        Parameters

        ----------

        index : int

            Index of row.

        

        Returns

        -------

        entry : dict

            Dictionary of `index`-th parsed item.

        """

        entry = self.meta_data.iloc[index]

        entry = self.parse_item(self.root, entry, self.transform)

        if not isinstance(entry, dict):

            raise TypeError("Output of `parse_item_cb` must be `dict`, but found {}".format(type(entry)))

        return entry



    def __len__(self):

        """Get length of `meta_data`"""

        return len(self.meta_data.index)
# Standard size

std_size = (64, 64)



train_transforms = solt.Stream([slt.Pad(std_size)])

valid_transforms = solt.Stream([slt.Pad(std_size)])

print(f'>>> Training transformations: <<<\n{train_transforms.to_yaml()}')

print(f'>>> Validation transformations: <<<\n{valid_transforms.to_yaml()}')
# Dataframe to Dataset

train_ds = DataFrameDataset(img_root, train_df, train_transforms, std_size=std_size)

valid_ds = DataFrameDataset(img_root, val_df, valid_transforms, std_size=std_size)
batch_size = 128

train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=8)

eval_loader = torch.utils.data.DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, num_workers=8)



print(f'train_loader is {train_loader}')

print(f'eval_loader is {eval_loader}')
arch_name = 'mobilenet_v2'

model = torch.hub.load('pytorch/vision', arch_name, pretrained=False)
# Check the depth of `features` and the details of `classifier`

print(f'Depth of features: {len(model.features)}.')

print(model.classifier)
maxdepth_channel_map = {16: 160, 17:160, 18: 320, 19: 1280}

new_maxdepth = 18

model.features = model.features[:new_maxdepth]

print(f'After cropping, depth of features: {len(model.features)}.')
new_in_features = maxdepth_channel_map[new_maxdepth]

model.classifier = nn.Linear(in_features=new_in_features, out_features=2, bias=True)

print(model.classifier)
def main_process(model, train_loader, eval_loader, device='cpu', n_epochs=10):

    model.to(device)

    lr = 1e-4

    wd = 1e-4

    loss_func = CrossEntropyLoss()

    optimizer = Adam(params=model.parameters(), lr=lr, weight_decay=wd)    

    for epoch_id in range(n_epochs):

        train_loop(model, loss_func, optimizer, train_loader, epoch_id, device)

        eval_loop(model, loss_func, eval_loader, epoch_id, device)
def train_loop(model, loss_func, optimizer, train_loader, epoch_id, device):

    metrics_collector = {'loss': [], 'probs': [], 'preds': [], 'labels': [], 'average_precision': None}

    # Tell the model that we are training it

    model.train(True)    

    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch_id}][Train]:")

    for batch_id, batch in enumerate(progress_bar):

        # Get sampled data and transfer them to the correct device        

        inputs = batch['data'].to(device)

        targets = batch['Validity'].to(device)

        

        # Forward through the model

        preds = model(inputs)        



        # Set gradients to 0

        optimizer.zero_grad()



        # Calculate loss        

        loss = loss_func(preds, targets)



        # Learn from the loss by applying backpropagation. This function will compute gradients 

        loss.backward()

        # Update the model's weights based on the computed gradients and other parameters of the optimizer

        optimizer.step()

        

        new_record = {'new_loss': loss.item(), 'new_preds': preds, 'new_labels': targets}

        metrics_collector, metrics = collect_and_update_metrics(metrics_collector, new_record)

        # Update metrics to progress bar

        metrics_display = {k:f'{metrics[k]:.03f}' for k in metrics}

        progress_bar.set_postfix(metrics_display)
def eval_loop(model, loss_func, eval_loader, epoch_id, device, save_model=True, return_df=False):

    global best_ap

    metrics_collector = {'loss': [], 'probs': [], 'preds': [], 'labels': [], 'average_precision': None}

    # Tell the model we are not training but evaluating it

    model.train(False) # or model.eval()

    # Init dictionary to store metrics

    metrics = OrderedDict()

    

    if return_df:

        validity_list = []

        filename_list = []

    n_batches = len(eval_loader)

    

    progress_bar = tqdm(eval_loader, total=n_batches, desc=f"Epoch [{epoch_id}][Eval]:")

    with torch.no_grad():

        for batch_id, batch in enumerate(progress_bar):

            # Get sampled data and transfer them to the correct device

            filenames = batch['Filename']            

            inputs = batch['data'].to(device)

            targets = batch['Validity'].to(device)            

            

            # Forward through the model

            preds = model(inputs)                    



            # Calculate loss        

            loss = loss_func(preds, targets)

            

            new_record = {'new_loss': loss.item(), 'new_preds': preds, 'new_labels': targets}

            metrics_collector, metrics = collect_and_update_metrics(metrics_collector, new_record)

            # Update metrics to progress bar

            metrics_display = {k:f'{metrics[k]:.03f}' for k in metrics}

            progress_bar.set_postfix(metrics_display)

            

            # Collect data for output dataframe 

            if return_df:

                probs = torch.softmax(preds, 1)[:, 1].cpu().detach().numpy()

                filename_list.extend(filenames)

                validity_list.extend(probs)



        # Store model based on balanced accuracy

        if metrics['average_precision'] > best_ap and save_model:

            model_filename = "best_checkpoint.pth"

            print(f'Improved average_precision from {best_ap} to {metrics["average_precision"]}. Saved to {model_filename}...')

            torch.save(model.state_dict(), model_filename)

            best_ap = metrics['average_precision']

            

    if return_df:

        results = pd.DataFrame(columns=['Filename', 'Validity'])

        results['Filename'] = filename_list

        results['Validity'] = validity_list

        return results    
device = "cuda"  # TODO: modify for running on `cpu`/`cuda`

n_epochs = 10

best_ap = -1.0



main_process(model,

             train_loader, 

             eval_loader, 

             device=device, 

             n_epochs=n_epochs)
def test_loop(model, loss_func, eval_loader, epoch_id, device, save_model=True, return_df=False):

    global best_ap

    # Tell the model we are not training but evaluating it

    model.train(False)  # or model.eval()

    

    # Init dictionary to store metrics    

    validity_list = []

    filename_list = []

    n_batches = len(eval_loader)

    

    progress_bar = tqdm(eval_loader, total=n_batches, desc=f"Epoch [{epoch_id}][Eval]:")

    with torch.no_grad():

        for batch_id, batch in enumerate(progress_bar):

            # Get sampled data and transfer them to the correct device

            filenames = batch['Filename']            

            inputs = batch['data'].to(device)

            

            # Forward through the model

            preds = model(inputs)                                

            

            # Collect data for output dataframe 

            probs = torch.softmax(preds, 1)[:, 1].cpu().detach().numpy()

            filename_list.extend(filenames)

            validity_list.extend(probs)        

            

    

    results = pd.DataFrame(columns=['Filename', 'Validity'])

    results['Filename'] = filename_list

    results['Validity'] = validity_list

    return results    
# Configurations

device = "cuda"  # TODO: modify for running on `cpu`/`cuda`

batch_size = 128

std_size = (64, 64)



# Create test dataloader

test_img_dir = os.path.join(root, "test/images")



output_dir = '/kaggle/working'

pretrained_model = os.path.join(output_dir, "best_checkpoint.pth")



test_img_fnames = os.listdir(test_img_dir)

test_df = pd.DataFrame.from_dict({"Filename": test_img_fnames})



test_transforms = solt.Stream([slt.Pad(std_size)])

test_ds = DataFrameDataset(test_img_dir, test_df, test_transforms, std_size=std_size)



test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=0)



# Get architecture

arch_name = 'mobilenet_v2'

model = torch.hub.load('pytorch/vision', arch_name, pretrained=False)

maxdepth_channel_map = {16: 160, 17:160, 18: 320, 19: 1280}

new_maxdepth = 18

model.features = model.features[:new_maxdepth]

print(f'After cropping, depth of features: {len(model.features)}.')

new_in_features = maxdepth_channel_map[new_maxdepth]

model.classifier = nn.Linear(in_features=new_in_features, out_features=2, bias=True)

print(model.classifier)



# Load trained weights

model.load_state_dict(torch.load(pretrained_model), strict=True)

model = model.to(device)



# Call main process

loss_func = CrossEntropyLoss()

results = test_loop(model, loss_func, test_loader, 0, device, save_model=False, return_df=True)
submission_fullname = os.path.join(output_dir, "submission.csv")

results.to_csv(submission_fullname, index=None)

display(results.head(10))