import itertools

import os

import os.path as op

import pickle

import random

import shutil

import time

import zipfile



from matplotlib import image

import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator

import numpy as np

import pandas as pd 

import torch

import torchvision

from torchvision import transforms, models





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



ROOT = "/kaggle"

BATCH_SIZE = 25



# Extract the training data:

with zipfile.ZipFile(op.join(ROOT, 'input/platesv2/plates.zip'), 'r') as zip_obj:

    zip_obj.extractall(op.join(ROOT, 'working'))

    



# Make a separate folder for the default training dataset:

os.mkdir(op.join(ROOT, 'working/plates/train/default'))

shutil.move(op.join(ROOT, 'working/plates/train/dirty'), 

            op.join(ROOT, 'working/plates/train/default/dirty')

           )

shutil.move(op.join(ROOT, 'working/plates/train/cleaned'), 

            op.join(ROOT, 'working/plates/train/default/cleaned')

           )



# Copy the modified training dataset into the working directory:

shutil.copytree(op.join(ROOT, 'input/added-data/train_no_background'),

                op.join(ROOT, 'working/plates/train/no_background')

               )

    

ground_truth = np.load(op.join(ROOT, 'input/added-data/ground_truth.npy'))
default_img_dir = op.join(ROOT, 'working/plates/train/default/cleaned')

no_background_img_dir = op.join(ROOT, 'working/plates/train/no_background/cleaned')

sample_images = [fname for fname in os.listdir(default_img_dir) if fname.endswith('.jpg')]

# I'll demonstrate just 4 examples.

sample_images = sample_images[:4]



fig, ax = plt.subplots(2, 4)

fig.set_size_inches(16, 6)

fig.set_facecolor('white')



for i, img_dir in enumerate([default_img_dir, no_background_img_dir]):

    for j, fname in enumerate(sample_images):

        img = image.imread(op.join(img_dir, fname))

        ax[i][j].imshow(img)

        img_type = 'default' if 'default' in img_dir else 'no_background'

        ax[i][j].set_title('%s, %s' % (fname, img_type))

        ax[i][j].get_xaxis().set_visible(False)

        ax[i][j].get_yaxis().set_visible(False)
def get_train_transforms(seed):

    """Returns transformations for Torch Dataloader."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

    aug_lvl_1 = transforms.Compose([           

        transforms.ColorJitter(

            brightness = 0.175,   

            contrast = 0.175,   

            saturation = 0.195,   

            hue = (0.1, 0.25)

        ),  

        transforms.RandomRotation(360),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        transforms.Lambda(

            lambda x: x[np.random.permutation(3), :, :])                            

    ])



    aug_lvl_2 = transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.75),                                

        transforms.TenCrop(224, vertical_flip=True),

        transforms.Lambda(

            lambda crops: torch.stack([aug_lvl_1(crop) for crop in crops]))

    ])    



    train_transforms = transforms.Compose([  

        transforms.Resize((224, 224)),

        transforms.TenCrop(224),

        transforms.Lambda(

            lambda crops: torch.stack([aug_lvl_2(crop) for crop in crops]))

    ])

    

    return train_transforms
# Multiplying labels:

def initialize_augmented_train_dataloader(dir_data, dir_transforms, batch_size, shuffle, num_workers): 

    """Initializes and returns a Torch Dataloader object with augmented data."""

    

    def create_target_set(variable, size_target, dtype = torch.long):

        """A helper function that returns a padded tensor."""

        target = torch.tensor([variable], dtype = dtype, requires_grad=False)

        target = torch.nn.functional.pad(target,

                                         (size_target//2, size_target//2 - 1),

                                         "constant",

                                         variable)



        return target

    

    data = []

    target = []



    # Create a dataset tensor:

    dataset = torchvision.datasets.ImageFolder(dir_data, dir_transforms)



    # Change the tensor's dimensions and multiply labels:

    for dt, trgt in dataset:

        data.append(dt.resize_(100, 3, 224, 224))

        target.append(create_target_set(trgt, 100))



    # Concatenate the tensors:

    dtst = list(

        zip(torch.cat(data , dim = 0),

            torch.cat(target, dim = 0))

        )

    

    # Create a dataloader object:

    dataloader = torch.utils.data.DataLoader(

        dtst,

        batch_size = batch_size,

        shuffle = shuffle,

        num_workers = num_workers

        )

    

    return dataloader
def initialize_model(model, lr, optimizer_class, scheduler_step_size, scheduler_gamma):

    """Initializes and returns objects of model, loss function, optimizer, and scheduler."""

    

    # Freeze all pretrained weights:

    for param in model.parameters():

        param.requires_grad = False  

    

    if model.__class__.__name__.startswith('ResNet'):

        model.fc = torch.nn.Linear(model.fc.in_features, 2)

    if model.__class__.__name__.startswith('VGG'):

        # The VGG's superstructure is taken from andrewkowalski's Super-augmentation.

        model.classifier = torch.nn.Sequential(   

            torch.nn.Linear(512 * 7 * 7, 8),      

            torch.nn.ReLU(True),                  

            torch.nn.Dropout(0.5),                

            torch.nn.Linear(8, 8),

            torch.nn.ReLU(True),

            torch.nn.Dropout(0.5),

            torch.nn.Linear(8, 2)

        )



    model = model.to(device)



    loss = torch.nn.CrossEntropyLoss()

    

    if optimizer_class.__name__ == 'Adam':

        optimizer = optimizer_class(model.parameters(), lr = lr, amsgrad=True)

    elif optimizer_class.__name__ == 'SGD':

        optimizer = optimizer_class(model.parameters(), lr = lr)



    scheduler = torch.optim.lr_scheduler.StepLR(

        optimizer,

        step_size = scheduler_step_size,

        gamma = scheduler_gamma

    )

    

    return model, loss, optimizer, scheduler
def train_model(model, loss, optimizer, scheduler, num_epochs, train_dataloader):

    """Returns a trained model."""

    for epoch in range(num_epochs):



        model.train()  



        # Iterate over data

        for inputs, labels in train_dataloader:

            inputs = inputs.to(device)

            labels = labels.to(device)



            optimizer.zero_grad()

            

            # Forward and backward

            preds = model(inputs)

            loss_value = loss(preds, labels)



            loss_value.backward()

            optimizer.step()



        scheduler.step()



    return model
def predict(trained_model, dataloader):

    """Returns predictions of a trained model."""

    predictions = []

    for inputs, labels, paths in dataloader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        with torch.set_grad_enabled(False):

            preds = trained_model(inputs)

        predictions.append(

            torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())



    predictions = np.concatenate(predictions).reshape(1, -1)

    return predictions
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]

        tuple_with_path = (original_tuple + (path,))

        return tuple_with_path

    

    

def get_test_dataloader():

    test_transforms = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])



    

    shutil.copytree(op.join(ROOT, 'working/plates/test'),

                    op.join(ROOT, 'working/plates/test/unknown'))

    test_dataset = ImageFolderWithPaths(op.join(ROOT, 'working/plates/test'),

                                        test_transforms)



    test_dataloader = torch.utils.data.DataLoader(

        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=BATCH_SIZE)

    return test_dataloader
MODELS_DICT = {'VGG11' : torchvision.models.vgg11(pretrained=True),

               'VGG13' : torchvision.models.vgg13(pretrained=True),

               'VGG16' : torchvision.models.vgg16(pretrained=True),

               'VGG19' : torchvision.models.vgg19(pretrained=True),

               'VGG11bn' : torchvision.models.vgg11_bn(pretrained=True),

               'VGG13bn' : torchvision.models.vgg13_bn(pretrained=True),

               'VGG16bn' : torchvision.models.vgg16_bn(pretrained=True),

               'VGG19bn' : torchvision.models.vgg19_bn(pretrained=True),

               'ResNet18' : torchvision.models.resnet18(pretrained=True),

               'ResNet34' : torchvision.models.resnet34(pretrained=True),

               'ResNet50' : torchvision.models.resnet50(pretrained=True),

               'ResNet101' : torchvision.models.resnet101(pretrained=True),

               'ResNet152' : torchvision.models.resnet152(pretrained=True),

               'ResNext50' : torchvision.models.resnext50_32x4d(pretrained=True),

               'ResNext101' : torchvision.models.resnext101_32x8d(pretrained=True),

              }



arch_names = list(MODELS_DICT.keys())



OPTIMIZERS_DICT = {'SGD' : torch.optim.SGD,

                   'Adam' : torch.optim.Adam

                  }



opt_names = list(OPTIMIZERS_DICT.keys())



# The seed and the dataset type should be at the top of the list, so that

# the number of times the training set is reloaded during the search is minimized.



params_list = [

    # Seed for random generation processes:

    [2, 42],

    

    # Using the default training images vs using images with background removed:

    ['default', 'no_background'],

    

    # Model's architecture to use for transfer learning:

    arch_names,

    

    # Learning rate:

    [1e-3, 1e-4],

    

    # Optimizer type:

    opt_names,

    

    # Scheduler step_size:

    [5, 10],

    

    # Scheduler gamma:

    [1e-1, 1e-2],

    

    # Number of epochs:

    [5, 10, 30],

]



params_combinations = list(itertools.product(*params_list))
def get_grid_predictions(params_combinations):

    prev_seed = None

    prev_train_data_type = None

    prediction_accumulator = np.zeros((1, len(ground_truth)))

    test_dataloader = get_test_dataloader()

    start_time = time.time()



    for i, params in enumerate(params_combinations, 1):



        (seed, train_data_type, arch_name,

         lr, opt_name, scheduler_step_size,

         scheduler_gamma, num_epochs

        ) = params



        # If either the seed or train_data_type is changed from previous iteration,

        # reload the training dataset.

        if seed != prev_seed or train_data_type != prev_train_data_type:

            train_dir = op.join(ROOT, 'working/plates/train', train_data_type)

            train_transforms = get_train_transforms(seed)

            train_dataloader = initialize_augmented_train_dataloader(

                train_dir,

                train_transforms,

                batch_size = BATCH_SIZE,

                shuffle = True,

                num_workers = BATCH_SIZE

            )



        model_pre_init = MODELS_DICT[arch_name]

        optimizer_pre_init = OPTIMIZERS_DICT[opt_name]

        model, loss, optimizer, scheduler = initialize_model(

            model_pre_init,

            lr,

            optimizer_pre_init,

            scheduler_step_size,

            scheduler_gamma

        )



        trained_model = train_model(

            model,

            loss,

            optimizer,

            scheduler,

            num_epochs,

            train_dataloader

        )



        trained_model.eval()

        test_predictions = predict(trained_model, test_dataloader)

        prediction_accumulator = np.concatenate(

            (prediction_accumulator, test_predictions))



        prev_seed = seed

        prev_train_data_type = train_data_type

        print('%d/%d %d %s' % (i, len(params_combinations), time.time() - start_time, params))



    return prediction_accumulator[1:]



# raw_predictions = get_grid_predictions(params_combinations)

# np.save(op.join(ROOT, 'working/predictions.npy'),

#         predictions

#         )
# A Numpy array of m rows by p columns.

# m is the number of models (2880),

# p is the number of parameters varied during grid search (8).

keys = np.load(op.join(ROOT, 

                       'input/added-data/predictions',

                       'keys.npy'

                      ),

               allow_pickle=True

              )



# A Numpy array of m rows by t columns.

# m is the number of models (2880),

# t is the number of test images (744).

raw_predictions = np.load(op.join(ROOT, 

                              'input/added-data/predictions',

                              'prediction_accumulator.npy'

                             )

                     )



# Sample output:

print(keys[0], '\n')

print(raw_predictions[0])
def accuracy(predictions, ground_truth):

    """

    Returns the proportion of true predictions made by multiple models.

    

    Input is n x m Numpy ndarray, where n is the number of models making predictions,

    m is the number of images for which the predictions were made.

    

    Output is n x 1 vector of accuracies.

    """

    return np.sum(predictions == ground_truth, axis=1) / len(ground_truth)
def get_accuracies_for_all_thresholds(predictions, ground_truth, thresholds):

    """

    Returns n x n_thr matrix of accuracies for n different models

    with n_thr different thresholds between 0 and 1.

    """



    accuracies = np.zeros((len(predictions), len(thresholds)))

    for j, t in enumerate(thresholds):

        accuracies[:, j] = accuracy(predictions > t, ground_truth)

    return accuracies
n_thr = 200

thresholds = np.linspace(0, 1, n_thr)



accuracies_for_all_thresholds = get_accuracies_for_all_thresholds(

    raw_predictions, ground_truth, thresholds)

accuracies_with_optimal_thresholds = np.max(

    accuracies_for_all_thresholds, axis=1)

optimal_thresholds_idx = np.argmax(

    accuracies_for_all_thresholds, axis=1)



optimal_thresholds = thresholds[optimal_thresholds_idx].reshape((-1, 1))

predictions = raw_predictions > optimal_thresholds
accuracies = accuracies_with_optimal_thresholds
n_bins = 46

arch_colors = ["#d16aab", "#92b540", "#58378a", "#cd9c2e", "#6d80d8",

               "#55913f", "#bf6fc7", "#58c07b", "#a63a6e", "#43c9b0",

               "#c55533", "#9e9743", "#c34c5f", "#c18241", "#a54536"]



fig = plt.figure(figsize=(12, 16), dpi=100, facecolor='w', edgecolor='k')



# Accuracy histogram (overall)

ax1 = fig.add_subplot(2, 1, 1)

bars, bins, _ = ax1.hist(accuracies, n_bins, edgecolor='white')

ylim = round(1.05 * bars.max())



ax1.set_ylim([0, ylim])

# Dumb baseline: every plate is dirty

ax1.plot([0.65322, 0.65322], [0, ylim], color='black', label='0.65322: all plates dirty')

# Baseline in PyTorch by Igor Slinko

ax1.plot([0.83064, 0.83064], [0, ylim], color='blue', label='0.83064: I. Slinko\'s template')

# Super-augmentation by Andrew Kowalski

ax1.plot([0.94623, 0.94623], [0, ylim], color='red', label='0.94623: A. Kowalski\'s Super-augmentation')



ax1.legend(bbox_to_anchor=[0.055, 0.995], loc='upper left')

ax1.set_title('Histogram of models\' accuracies')

ax1.set_xlabel('Accuracy')

ax1.set_ylabel('Number of models')

ax1.set_xticks(bins[::5])

ax1.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))

ax1.xaxis.set_minor_locator(AutoMinorLocator(5))



# Stacked accuracy histogram (by architecture):



ax2 = fig.add_subplot(2, 1, 2)



cumulative_bars = np.zeros(n_bins)

for arch_name, arch_color in zip(arch_names, arch_colors):

    filter_by_arch = keys[:, 2] == arch_name

    accs_of_arch = accuracies[filter_by_arch]

   

    bars, _, _ = ax2.hist(accs_of_arch, bins, 

                          bottom=cumulative_bars,

                          color=arch_color,

                          edgecolor='white',

                          label=arch_name)

    

    cumulative_bars += bars

    

handles, labels = ax2.get_legend_handles_labels()

# The bars are stacked bottom to top, and so should be the legend markers.

ax2.legend(handles[::-1], labels[::-1], bbox_to_anchor=[0.12, 0.95], loc='upper left')

ax2.set_ylim([0, ylim])

ax2.set_title('Histogram of models\' accuracies broke down by architecture type')

ax2.set_xlabel('Accuracy')

ax2.set_ylabel('Number of models')

ax2.set_xticks(bins[::5])

ax2.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))

ax2.xaxis.set_minor_locator(AutoMinorLocator(5))



plt.show()
print('Top 30 models by accuracy\n(OBT = Optimal Binarization Threshold)\n')

print('\033[1m\033[4m#'.ljust(11), 'Accuracy'.ljust(10), 'Model key'.ljust(56), 'OBT    \033[0m')



# Negate to sort in reverse order

top30pos = (-accuracies).argsort()[:30]

for i, pos in enumerate(top30pos, 1):

    # Highlight non-VGG13 with boldface:

    is_vgg13bn = keys[pos, 2] == 'VGG13bn'

    if not is_vgg13bn:

        print('\033[1m', end='')

    print(str(i).ljust(3), ('%.5f' % accuracies[pos]).ljust(10), str(keys[pos]).ljust(56), '%.5f' % optimal_thresholds[pos, 0],

          end='\033[0m\n' if not is_vgg13bn  else '\n')
# Initializations:

candidate_keys = np.zeros((len(arch_names), len(keys[0])), dtype='object')

candidate_thresholds = np.zeros(len(arch_names))

candidate_preds = np.zeros((len(arch_names), len(predictions[0])))

candidate_accs = np.zeros(len(arch_names))



print('Ensemble candidates\n(OBT = Optimal Binarization Threshold)\n')

print('\033[1m\033[4mAccuracy'.ljust(18), 'Model key'.ljust(56), 'OBT    \033[0m')



# Finding best model by accuracy in each architecture:

for i, arch_name in enumerate(arch_names):

    # Because the prior position is lost when filtering,

    # I have to filter ALL of those arrays together to be able 

    # to juxtapose them in the second part of the loop.

    filter_by_arch = keys[:, 2] == arch_name

    keys_of_arch = keys[filter_by_arch]

    thresholds_of_arch = optimal_thresholds[filter_by_arch]

    preds_of_arch = predictions[filter_by_arch]

    accs_of_arch = accuracies[filter_by_arch]

    

    idx_best_of_arch = accs_of_arch.argmax()

    

    candidate_keys[i] = keys_of_arch[idx_best_of_arch]

    candidate_thresholds[i] = thresholds_of_arch[idx_best_of_arch]

    candidate_preds[i] = preds_of_arch[idx_best_of_arch]

    candidate_accs[i] = accs_of_arch[idx_best_of_arch]

    print(('%.5f' % candidate_accs[i]).ljust(10),

          str(candidate_keys[i]).ljust(56),

          '%.5f' % candidate_thresholds[i]

         )
def ensemble_prediction(votes):

    """Returns binary predictions chosen by the majority vote."""

    vote_counts = np.sum(votes, axis=0)

    n_voters = votes.shape[0]

    return vote_counts > (n_voters // 2)



n_best = len(candidate_keys)

models_ids = list(range(n_best))

best_ensemble_pred = np.zeros(len(ground_truth))

best_ensemble_acc = 0



for k in range(1, n_best+1, 2):

    combinations_of_k_ids = itertools.combinations(models_ids, k)

    for id_combination in combinations_of_k_ids:

        # Fancy indexing does not work with tuples, but works with lists.

        id_combination = list(id_combination)

        votes = candidate_preds[id_combination]

        ensemble_pred = ensemble_prediction(votes)

        # Reshape to fit the syntax of the accuracy function:

        ensemble_pred = ensemble_pred.reshape(1, -1)

        ensemble_acc = accuracy(ensemble_pred, ground_truth)[0]

        # Remember the ensemble with the highest accuracy.

        if ensemble_acc > best_ensemble_acc:

            best_ensemble_acc = ensemble_acc

            best_ensemble_ids = id_combination

            best_ensemble_pred = ensemble_pred

            

            

print('Best ensemble accuracy: ',

      best_ensemble_acc,

      '\n'

     )



print('The ensemble consists of the following individual models:')

print('(OBT = Optimal Binarization Threshold)\n')

print('\033[1m\033[4mInd. accuracy'.ljust(22), 'Model key'.ljust(56), 'OBT    \033[0m')

for i in best_ensemble_ids:

    print(('%.5f' % candidate_accs[i]).ljust(14),

          str(candidate_keys[i]).ljust(56),

          '%.5f' % candidate_thresholds[i]

         )

    

ensemble_keys = candidate_keys[best_ensemble_ids]

ensemble_thresholds = candidate_thresholds[best_ensemble_ids].reshape(-1, 1)
def make_submission_file(prediction):

    """Writes predictions into CSV."""

    ids = ['%04d' % i for i in range(len(prediction))]



    submission_df = pd.DataFrame.from_dict({'id': ids,

                                            'label': prediction})

    submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred else 'cleaned')

    submission_df.set_index('id', inplace=True)

    submission_df.to_csv('submission.csv')



best_ensemble_pred = best_ensemble_pred.reshape(-1)

make_submission_file(best_ensemble_pred)

!rm -rf /kaggle/working/plates