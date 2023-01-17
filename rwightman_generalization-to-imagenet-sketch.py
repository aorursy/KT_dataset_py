%pip -q install timm
# Import packages, setup our logger and check if we have CUDA

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torchvision as tv

from torchvision import transforms

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

from PIL import Image



import timm

from timm.utils import *



import os

import shutil

import time

from collections import OrderedDict



setup_default_logging()



print('PyTorch version:', torch.__version__)

if torch.cuda.is_available():

    print('CUDA available')

    device='cuda'

else:

    print('WARNING: CUDA is not available')

    device='cpu'



BATCH_SIZE = 96
# a basic validation routine and runner that configures each model and loader



def validate(model, loader, criterion=None, device='cuda'):

    # metrics

    batch_time = timm.utils.AverageMeter()

    losses = AverageMeter()

    top1 = AverageMeter()

    top5 = AverageMeter()

    

    # for collecting per sample prediction/loss details

    losses_val = []

    top5_idx = []

    top5_val = []

    

    end = time.time()

    with torch.no_grad():

        for i, (input, target) in enumerate(loader):

            target = target.to(device)

            input = input.to(device)

            output = model(input)

            

            if criterion is not None:

                loss = criterion(output, target)

                if not loss.size():

                    losses.update(loss.item(), input.size(0))

                else:

                    # only bother collecting top5 we're also collecting per-example loss

                    output = output.softmax(1)

                    top5v, top5i = output.topk(5, 1, True, True)

                    top5_val.append(top5v.cpu().numpy())

                    top5_idx.append(top5i.cpu().numpy())

                    losses_val.append(loss.cpu().numpy())

                    losses.update(loss.mean().item(), input.size(0))

                

            prec1, prec5 = timm.utils.accuracy(output, target, topk=(1, 5))

            top1.update(prec1.item(), input.size(0))

            top5.update(prec5.item(), input.size(0))



            batch_time.update(time.time() - end)

            end = time.time()



            if i % 20 == 0:

                print('Test: [{0}/{1}]\t'

                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {rate_avg:.3f}/s) \t'

                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'

                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(

                    i, len(loader), batch_time=batch_time,

                    rate_avg=input.size(0) / batch_time.avg,

                    top1=top1, top5=top5))



    results = OrderedDict(

        top1=top1.avg, top1_err=100 - top1.avg,

        top5=top5.avg, top5_err=100 - top5.avg,

    )

    if criterion is not None:

        results['loss'] = losses.avg

    if len(top5_idx):

        results['top5_val'] = np.concatenate(top5_val, axis=0)

        results['top5_idx'] = np.concatenate(top5_idx, axis=0)

    if len(losses_val):

        results['losses_val'] = np.concatenate(losses_val, axis=0)

    print(' * Prec@1 {:.3f} ({:.3f}) Prec@5 {:.3f} ({:.3f})'.format(

       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results





def runner(model_args, dataset, device='cuda', collect_loss=False):

    model_name = model_args['model']

    model = timm.create_model(model_name, pretrained=True)

    model = model.to(device)

    model.eval()



    data_config = timm.data.resolve_data_config(model_args, model=model, verbose=True)



    loader = timm.data.create_loader(

        dataset,

        input_size=data_config['input_size'],

        batch_size=BATCH_SIZE,

        use_prefetcher=True,

        interpolation='bicubic',

        mean=data_config['mean'],

        std=data_config['std'],

        crop_pct=1.0, #data_config['crop_pct'],

        num_workers=2)



    criterion = None

    if collect_loss:

        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    results = validate(model, loader, criterion, device)

    

    # cleanup checkpoint cache to avoid running out of disk space

    shutil.rmtree(os.path.join(os.environ['HOME'], '.cache', 'torch', 'checkpoints'), True)

    

    # add some non-metric values for charting / comparisons

    results['model'] = model_name

    results['img_size'] = data_config['input_size'][-1]



    # create key to identify model in charts

    key = [model_name, str(data_config['input_size'][-1])]

    key = '-'.join(key)

    return key, results
# load the dataset

#dataset = tv.datasets.ImageFolder(root="../input/imagenet-sketch/sketch")

dataset = timm.data.Dataset("../input/imagenet-sketch/sketch")

assert len(dataset) == 50889
def show_img(ax, img):

    npimg = img.numpy()

    ax.imshow(np.transpose(npimg, (1,2,0)), interpolation='bicubic')



fig = plt.figure(figsize=(8, 16), dpi=100)

ax = fig.add_subplot('111')

num_images = 4*8

images = []

dataset.transform = transforms.Compose([

    transforms.Resize(320, Image.BICUBIC),

    transforms.CenterCrop(320),

    transforms.ToTensor()])

for i in np.random.permutation(np.arange(len(dataset)))[:num_images]:

    images.append(dataset[i][0])

   

grid_img = make_grid(images, nrow=4, padding=10, normalize=True, scale_each=True)

show_img(ax, grid_img)    

models = [

    dict(model='mobilenetv3_100'),

    dict(model='dpn68b'),

    dict(model='gluon_resnet50_v1d'),

    dict(model='efficientnet_b2'),

    dict(model='gluon_seresnext50_32x4d'),

    dict(model='dpn92'),

    dict(model='gluon_seresnext101_32x4d'),

    dict(model='inception_resnet_v2'),

    dict(model='pnasnet5large'),

    dict(model='tf_efficientnet_b5'),

    dict(model='ig_resnext101_32x8d'),

    dict(model='ig_resnext101_32x16d'),

    dict(model='ig_resnext101_32x32d'),

    dict(model='ig_resnext101_32x48d'),

]



# Run all the models through validation

results = OrderedDict()

for ma in models:

    mk, mr = runner(ma, dataset, device)

    results[mk] = mr



results_df = pd.DataFrame.from_dict(results, orient='index')

results_df.to_csv('./cached-results.csv')
# Setup the common charting elements

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]



names_all = list(results.keys())

top1_all = np.array([results[m]['top1'] for m in names_all])

top1_sort_ix = np.argsort(top1_all)

top1_sorted = top1_all[top1_sort_ix]

top1_names_sorted = np.array(names_all)[top1_sort_ix]



top5_all = np.array([results[m]['top5'] for m in names_all])

top5_sort_ix = np.argsort(top5_all)

top5_sorted = top5_all[top5_sort_ix]

top5_names_sorted = np.array(names_all)[top5_sort_ix]
fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.barh(top1_names_sorted, top1_sorted, color='lightcoral')



ax1.set_title('Top-1 by Model')

ax1.set_xlabel('Top-1 Accuracy (%)')

ax1.set_yticklabels(top1_names_sorted)

ax1.autoscale(True, axis='both')



acc_min = top1_sorted[0]

acc_max = top1_sorted[-1]

plt.xlim([math.ceil(acc_min - .3*(acc_max - acc_min)), math.ceil(acc_max)])



plt.vlines(plt.xticks()[0], *plt.ylim(), color='0.5', alpha=0.2, linestyle='--')

plt.show()



print('Results by top-1 accuracy:')

results_by_top1 = list(sorted(results.keys(), key=lambda x: results[x]['top1'], reverse=True))

for m in results_by_top1:

  print('  Model: {:30} Top-1 {:4.2f}, Top-5 {:4.2f}'.format(m, results[m]['top1'], results[m]['top5']))
# download ImageNet-1k resuls run on my model collection for top-1/top-5 comparisons

!wget -q https://raw.githubusercontent.com/rwightman/pytorch-image-models/master/results/results-all.csv

    

original_df = pd.read_csv('./results-all.csv', index_col=0)



original_results = original_df.to_dict(orient='index')
# some helpers for the dumbbell plots

import matplotlib.lines as mlines



def label_line_horiz(ax, line, label, color='0.5', fs=14, halign='center'):

    xdata, ydata = line.get_data()

    x1, x2 = xdata

    xx = 0.5 * (x1 + x2)

    text = ax.annotate(

        label, xy=(xx, ydata[0]), xytext=(0, 1), textcoords='offset points',

        size=fs, color=color, zorder=3,

        bbox=dict(boxstyle="round", fc="w", color='0.5'),

        horizontalalignment='center',

        verticalalignment='center')

    return text



def draw_line_horiz(ax, p1, p2, label, color='black'):

    l = mlines.Line2D(*zip(p1, p2), color=color, zorder=0)

    ax.add_line(l)

    label_line(ax, l, label)

    return l



def label_line_vert(ax, line, label, color='0.5', fs=14, halign='center'):

    xdata, ydata = line.get_data()

    y1, y2 = ydata

    yy = 0.5 * (y1 + y2)

    text = ax.annotate(

        label, xy=(xdata[0], yy), xytext=(0, 0), textcoords='offset points',

        size=fs, color=color, zorder=3,

        bbox=dict(boxstyle="round", fc="w", color='0.5'),

        horizontalalignment='center',

        verticalalignment='center')

    return text



def draw_line_vert(ax, p1, p2, label, color='black'):

    l = mlines.Line2D(*zip(p1, p2), color=color, zorder=0)

    ax.add_line(l)

    label_line_vert(ax, l, label)

    return l

fig = plt.figure()

ax1 = fig.add_subplot(111)



# draw the ImageNet-Sketch dots, we're sorted on this

ax1.scatter(x=top1_names_sorted, y=top1_sorted, s=64, c='lightcoral',marker="o", label='ImageNet-Sketch')



# draw the original ImageNet-1k validation dots

orig_top1 = [original_results[results[n]['model']]['top1'] for n in top1_names_sorted]

ax1.scatter(x=top1_names_sorted, y=orig_top1, s=64, c='steelblue', marker="o", label='ImageNet-1K')



for n, vo, vn in zip(top1_names_sorted, orig_top1, top1_sorted):

    draw_line_vert(ax1, (n, vo), (n, vn),

                   str(round(vo - vn, 2)), 'skyblue')



ax1.set_title('Top-1 Difference')

ax1.set_ylabel('Top-1 Accuracy (%)')

ax1.set_xlabel('Model')

yl, yh = ax1.get_ylim()

yl = 5 * ((yl + 1) // 5 + 1) 

yh = 5 * (yh // 5 + 1)

for y in plt.yticks()[0][1:-1]:

    ax1.axhline(y, 0.02, 0.98, c='0.5', alpha=0.2, linestyle='-.')

ax1.set_xticklabels(top1_names_sorted, rotation='-30', ha='left')

ax1.legend(loc='upper left')

plt.show()
fig = plt.figure()

ax1 = fig.add_subplot(111)



# draw the ImageNet-Sketch top-5 dots, we're sorted on this

ax1.scatter(x=top5_names_sorted, y=top5_sorted, s=64, c='lightcoral',marker="o", label='ImageNet-Sketch')



# draw the original ImageNet-1k validation dots

orig_top5 = [original_results[results[n]['model']]['top5'] for n in top5_names_sorted]

ax1.scatter(x=top5_names_sorted, y=orig_top5, s=64, c='steelblue', marker="o", label='ImageNet-1K')



for n, vo, vn in zip(top5_names_sorted, orig_top5, top5_sorted):

    draw_line_vert(ax1, (n, vo), (n, vn),

                   str(round(vo - vn, 2)), 'skyblue')



ax1.set_title('Top-5 Difference')

ax1.set_ylabel('Top-5 Accuracy (%)')

ax1.set_xlabel('Model')

yl, yh = ax1.get_ylim()

yl = 5 * ((yl + 1) // 5 + 1) 

yh = 5 * (yh // 5 + 1)

for y in plt.yticks()[0][2:-2]:

    ax1.axhline(y, 0.02, 0.98, c='0.5', alpha=0.2, linestyle='-.')

ax1.set_xticklabels(top5_names_sorted, rotation='-30', ha='left')

ax1.legend(loc='upper left')

plt.show()
print('Results by absolute accuracy gap between ImageNet-Sketch and original ImageNet top-1:')



gaps = {x: (results[x]['top1'] - original_results[results[x]['model']]['top1']) for x in results.keys()}

sorted_keys = list(sorted(results.keys(), key=lambda x: gaps[x], reverse=True))

for m in sorted_keys:

  print('  Model: {:30} {:4.2f}%'.format(m, gaps[m]))

print()



print('Results by relative accuracy gap between ImageNet-Sketch and original ImageNet top-1:')

gaps = {x: 100 * (results[x]['top1'] - original_results[results[x]['model']]['top1']) / original_results[results[x]['model']]['top1'] for x in results.keys()}

sorted_keys = list(sorted(results.keys(), key=lambda x: gaps[x], reverse=True))

for m in sorted_keys:

  print('  Model: {:30} {:4.2f}%'.format(m, gaps[m]))

print()
print('Results by relative accuracy gap between ImageNet-Sketch and original ImageNet top-5:')

gaps = {x: (results[x]['top5'] - original_results[results[x]['model']]['top5']) for x in results.keys()}

sorted_keys = list(sorted(results.keys(), key=lambda x: gaps[x], reverse=True))

for m in sorted_keys:

  print('  Model: {:30} {:4.2f}%'.format(m, gaps[m]))

print()



print('Results by relative accuracy gap between ImageNet-Sketch and original ImageNet top-5:')

gaps = {x: 100 * (results[x]['top5'] - original_results[results[x]['model']]['top5']) / original_results[results[x]['model']]['top5'] for x in results.keys()}

sorted_keys = list(sorted(results.keys(), key=lambda x: gaps[x], reverse=True))

for m in sorted_keys:

  print('  Model: {:30} {:4.2f}%'.format(m, gaps[m]))
# create mappings of label id to text and synset

!wget -q https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt

with open('./synset_words.txt', 'r') as f:

    split_lines = [l.strip().split(' ') for l in f.readlines()]

    id_to_synset = dict(enumerate([l[0] for l in split_lines]))

    id_to_text = dict(enumerate([' '.join(l[1:]) for l in split_lines]))
# re-run validation on just one model, this time collecting per-example losses and predictions

BATCH_SIZE=128

mk, mr = runner(dict(model='ig_resnext101_32x16d'), dataset, device, collect_loss=True)    
# a function to display images in a grid and ground truth vs predictions for specified indices

def show_summary(indices, dataset, nrows):

    col_scale = len(indices) // nrows

    top5_idx = mr['top5_idx'][indices]

    top5_val = mr['top5_val'][indices]



    images = []

    labels = []

    filenames = []



    dataset.transform = transforms.Compose([

        transforms.Resize(320, Image.BICUBIC),

        transforms.CenterCrop(320),

        transforms.ToTensor()])



    for i in indices:

        img, label = dataset[i]

        images.append(img)

        labels.append(label)

    filenames = dataset.filenames(list(indices), basename=True)



    fig = plt.figure(figsize=(8, 8 * col_scale), dpi=100)

    ax = fig.add_subplot('111')

    grid_best = make_grid(images, nrow=nrows, padding=10, normalize=True, scale_each=True)

    show_img(ax, grid_best)

    plt.show()



    summary = OrderedDict()

    for i, l in enumerate(labels):

        image_name = id_to_synset[i] + '/' + filenames[i]

        summary[image_name] = {}

        summary[image_name]['gt'] = id_to_text[l]

        summary[image_name]['predictions'] = [(100. * pv, id_to_text[pi]) for pi, pv in zip(top5_idx[i], top5_val[i])]

    return summary
nrows = 2

num_images = 10

best_idx = np.argsort(mr['losses_val'])[:num_images]

best_summary = show_summary(best_idx, dataset, nrows)
print('Best prediction ground truth vs predictions')

for k, v in best_summary.items():

    print('{} ground truth = {}'.format(k, v['gt']))

    print('Predicted:')

    for p, l in v['predictions']:

        if p > 2e-3:

            print('  {:.3f} {}'.format(p, l))

    print()
nrows = 2

num_images = 20

worst_idx = np.argsort(mr['losses_val'])[-num_images:][::-1]

worst_summary = show_summary(worst_idx, dataset, nrows)
print('Worst prediction ground truth vs predictions')

for k, v in worst_summary.items():

    print('{} ground truth = {}'.format(k, v['gt']))

    print('Predicted:')

    for p, l in v['predictions']:

        if p > 2e-3:

            print('  {:.3f} {}'.format(p, l))

    print()