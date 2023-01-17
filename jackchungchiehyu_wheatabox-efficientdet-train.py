! pip install --no-deps '/kaggle/input/timm-package/timm-0.1.26-py3-none-any.whl'
import sys

sys.path.insert(0, "/kaggle/input/timm-efficientdet-pytorch")

sys.path.insert(0, "/kaggle/input/omegaconf")

import os, time, random

from pathlib import Path

from glob import glob

from datetime import datetime

import numpy as np, pandas as pd

import PIL

import matplotlib.pyplot as plt

import numba

from numba import jit

from sklearn.model_selection import StratifiedKFold

import torch

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SequentialSampler, RandomSampler

import cv2

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchEval

from effdet.efficientdet import HeadNet



Path.ls = lambda x: list(x.iterdir())
SZ = 512
marking = pd.read_csv('/kaggle/input/global-wheat-detection/train.csv')

bboxs = np.stack(marking.bbox.apply(lambda r: np.fromstring(r[1:-1], sep=',')))

for i, column in enumerate(['x', 'y', 'w', 'h']): marking[column] = bboxs[:,i]

marking.drop(columns='bbox', inplace=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

df_folds = marking[['image_id']].copy()

df_folds.loc[:,'bbox_count'] = 1

df_folds = df_folds.groupby('image_id').count()

df_folds.loc[:,'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']

df_folds.loc[:,'stratify_group'] = np.char.add(

    df_folds['source'].values.astype(str),

    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str))

df_folds.loc[:,'fold'] = 0

for fold_number, (train_index, valid_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):

    df_folds.loc[df_folds.iloc[valid_index].index,'fold'] = fold_number
def get_train_transforms():

    return A.Compose(

        [A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),

         A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),

                  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],

                 p=0.9),

         A.ToGray(p=0.01),

         A.HorizontalFlip(p=0.5),

         A.VerticalFlip(p=0.5),

         A.Resize(height=SZ, width=SZ, p=1),

#          A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=.5), 

         A.Cutout(num_holes=80, max_h_size=SZ//20, max_w_size=SZ//20, fill_value=0, p=.5),

#          A.Blur(p=.5),

         ToTensorV2(p=1)],

        p=1,

        bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels'])

    )



def get_valid_transforms():

    return A.Compose(

        [A.Resize(height=SZ, width=SZ, p=1), ToTensorV2(p=1)], 

        p=1,

        bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0, label_fields=['labels']))
TRAIN_ROOT_PATH = '../input/global-wheat-detection/train/'



class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False):

        super().__init__()

        self.marking, self.image_ids = marking, image_ids

        self.transforms, self.test = transforms, test

        

    def __getitem__(self, index:int): 

        image_id = self.image_ids[index]

        if self.test or random.random() > 0.5: 

            image, boxes = self.load_image_and_boxes(index)

        else: 

            image, boxes = self.load_cutmix_image_and_boxes(index)

        labels = torch.ones((len(boxes),), dtype=torch.int64)

        if self.transforms:

            for i in range(10):

                sample = self.transforms(image=image, bboxes=boxes, labels=labels)

                if len(sample['bboxes']) > 0:

                    image = sample['image']

                    boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

                    boxes[:,[0,1,2,3]] = boxes[:,[1,0,3,2]]

                    break

        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        target['image_id'] = torch.tensor([index])

        return image, target, image_id

        

    def __len__(self) -> int: return len(self.image_ids)

    

    def load_image_and_boxes(self, index):

        image_id = self.image_ids[index]

        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.

        records = self.marking[self.marking['image_id']==image_id]

        boxes = records[['x', 'y', 'w', 'h']].values

        boxes[:,2] = boxes[:,0] + boxes[:,2]

        boxes[:,3] = boxes[:,1] + boxes[:,3]

        return image, boxes

        

    def load_cutmix_image_and_boxes(self, index, imsize=1024):

        w, h = imsize, imsize

        s = imsize // 2

        xc, yc = [int(random.uniform(.25*imsize, .75*imsize)) for _ in range(2)]

        indexes = [index] + [random.randint(0, len(self.image_ids)-1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)

        result_boxes = []

        for i, index in enumerate(indexes):

            image, boxes = self.load_image_and_boxes(index)

            if i == 0:

                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)

                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)

            elif i == 1:  # top right

                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc

                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h

            elif i == 2:  # bottom left

                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)

                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)

            elif i == 3:  # bottom right

                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)

                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            result_image[y1a:y2a,x1a:x2a] = image[y1b:y2b,x1b:x2b]

            padw, padh = x1a - x1b, y1a - y1b

            boxes[:,0] += padw; boxes[:,1] += padh

            boxes[:,2] += padw; boxes[:,3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)

        np.clip(result_boxes[:,0:], 0, 2*s, out=result_boxes[:,0:])

        result_boxes = result_boxes.astype(np.int32)

        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0]) * (result_boxes[:,3]-result_boxes[:,1]) > 0)]

        return result_image, result_boxes            
fold_number = 3



train_dataset = DatasetRetriever(

    image_ids=df_folds[df_folds.fold!=fold_number].index.values, 

    marking=marking, transforms=get_train_transforms(), test=False)



validation_dataset = DatasetRetriever(

    image_ids = df_folds[df_folds.fold==fold_number].index.values, 

    marking=marking, transforms=get_valid_transforms(), test=True)
image, target, image_id = train_dataset[8]

boxes = target['boxes'].cpu().numpy().astype(np.int32)

assert len(target['boxes']) == len(target['labels'])

numpy_image = image.permute(1, 2, 0).cpu().numpy()



fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:

    cv2.rectangle(numpy_image, (box[1], box[0]), (box[3], box[2]), (0, 1, 0), 2)

ax.set_axis_off()

ax.imshow(numpy_image);
class Data:

    def __init__(self, train_dl, valid_dl):

        self.train_dl, self.valid_dl = train_dl, valid_dl
bs = 4
def collate_fn(batch): return tuple(zip(*batch))



train_loader = torch.utils.data.DataLoader(

    train_dataset, 

    batch_size=bs,

    sampler=RandomSampler(train_dataset),

    pin_memory=False,

    drop_last=True,

    num_workers=2,

    collate_fn=collate_fn)



val_loader = torch.utils.data.DataLoader(

    validation_dataset,

    batch_size=bs,

    num_workers=2,

    shuffle=False,

    sampler=SequentialSampler(validation_dataset),

    pin_memory=False,

    collate_fn=collate_fn)



data = Data(train_loader, val_loader)
def get_benches():

    config = get_efficientdet_config('tf_efficientdet_d7')

    net = EfficientDet(config, pretrained_backbone=False)

    checkpoint = torch.load('/kaggle/input/efficientdet/efficientdet_d7-f05bf714.pth')

#     checkpoint = torch.load('/kaggle/input/efficientdet/efficientdet_d6-51cb0132.pth')

#     checkpoint = torch.load('/kaggle/input/efficientdet/efficientdet_d5-ef44aea8.pth')

    net.load_state_dict(checkpoint)

    config.num_classes = 1

    config.image_size = SZ

    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    return DetBenchTrain(net, config), DetBenchEval(net, config)
@jit(nopython=True)

def calculate_iou(gt, pr, form='pascal_voc') -> float:

    """Calculates the Intersection over Union.



    Args:

        gt: (np.ndarray[Union[int, float]]) coordinates of the ground-truth box

        pr: (np.ndarray[Union[int, float]]) coordinates of the prdected box

        form: (str) gt/pred coordinates format

            - pascal_voc: [xmin, ymin, xmax, ymax]

            - coco: [xmin, ymin, w, h]

    Returns:

        (float) Intersection over union (0.0 <= iou <= 1.0)

    """

    if form == 'coco':

        gt = gt.copy()

        pr = pr.copy()



        gt[2] = gt[0] + gt[2]

        gt[3] = gt[1] + gt[3]

        pr[2] = pr[0] + pr[2]

        pr[3] = pr[1] + pr[3]



    # Calculate overlap area

    dx = min(gt[2], pr[2]) - max(gt[0], pr[0]) + 1

    if dx < 0: return 0.0



    dy = min(gt[3], pr[3]) - max(gt[1], pr[1]) + 1

    if dy < 0: return 0.0



    overlap_area = dx * dy

    # Calculate union area

    union_area = (

            (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1) +

            (pr[2] - pr[0] + 1) * (pr[3] - pr[1] + 1) -

            overlap_area

    )

    return overlap_area / union_area



@jit(nopython=True)

def find_best_match(gts, pred, pred_idx, threshold = 0.5, form = 'pascal_voc', ious=None) -> int:

    """Returns the index of the 'best match' between the

    ground-truth boxes and the prediction. The 'best match'

    is the highest IoU. (0.0 IoUs are ignored).



    Args:

        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes

        pred: (List[Union[int, float]]) Coordinates of the predicted box

        pred_idx: (int) Index of the current predicted box

        threshold: (float) Threshold

        form: (str) Format of the coordinates

        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.



    Return:

        (int) Index of the best match GT box (-1 if no match above threshold)

    """

    best_match_iou = -np.inf

    best_match_idx = -1



    for gt_idx in range(len(gts)):

        

        if gts[gt_idx][0] < 0:

            # Already matched GT-box

            continue

        

        iou = -1 if ious is None else ious[gt_idx][pred_idx]



        if iou < 0:

            iou = calculate_iou(gts[gt_idx], pred, form=form)

            

            if ious is not None:

                ious[gt_idx][pred_idx] = iou



        if iou < threshold:

            continue



        if iou > best_match_iou:

            best_match_iou = iou

            best_match_idx = gt_idx

    return best_match_idx



@jit(nopython=True)

def calculate_precision(gts, preds, threshold = 0.5, form = 'coco', ious=None) -> float:

    """Calculates precision for GT - prediction pairs at one threshold.



    Args:

        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes

        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,

               sorted by confidence value (descending)

        threshold: (float) Threshold

        form: (str) Format of the coordinates

        ious: (np.ndarray) len(gts) x len(preds) matrix for storing calculated ious.



    Return:

        (float) Precision

    """

    n = len(preds)

    tp = 0

    fp = 0

    

    # for pred_idx, pred in enumerate(preds_sorted):

    for pred_idx in range(n):

        best_match_gt_idx = find_best_match(gts, preds[pred_idx], pred_idx,

                                            threshold=threshold, form=form, ious=ious)



        if best_match_gt_idx >= 0:

            # True positive: The predicted box matches a gt box with an IoU above the threshold.

            tp += 1

            # Remove the matched GT box

            gts[best_match_gt_idx] = -1



        else:

            # No match

            # False positive: indicates a predicted box had no associated gt box.

            fp += 1



    # False negative: indicates a gt box had no associated predicted box.

    fn = (gts.sum(axis=1) > 0).sum()

    return tp / (tp + fp + fn)





@jit(nopython=True)

def calculate_image_precision(gts, preds, thresholds = (0.5, ), form = 'coco') -> float:

    """Calculates image precision.



    Args:

        gts: (List[List[Union[int, float]]]) Coordinates of the available ground-truth boxes

        preds: (List[List[Union[int, float]]]) Coordinates of the predicted boxes,

               sorted by confidence value (descending)

        thresholds: (float) Different thresholds

        form: (str) Format of the coordinates



    Return:

        (float) Precision

    """

    n_threshold = len(thresholds)

    image_precision = 0.0

    

    ious = np.ones((len(gts), len(preds))) * -1

    # ious = None



    for threshold in thresholds:

        precision_at_threshold = calculate_precision(gts.copy(), preds, threshold=threshold,

                                                     form=form, ious=ious)

        image_precision += precision_at_threshold / n_threshold



    return image_precision



# Numba typed list!

iou_thresholds = numba.typed.List()

for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:

    iou_thresholds.append(x)
def single_image_score(prediction, score_threshold):

    '''

    Returns the precision for a single image.

    '''

    gt_boxes = prediction['gt_boxes'].copy()

    pred_boxes = prediction['pred_boxes'].copy()

    scores = prediction['scores'].copy()

    image_id = prediction['image_id']

    indexes = np.where(scores > score_threshold)

    pred_boxes = pred_boxes[indexes]

    scores = scores[indexes]

    image_precision = calculate_image_precision(gt_boxes, pred_boxes, thresholds=iou_thresholds, form='pascal_voc')

    return image_precision



def calculate_final_score(all_predictions, score_threshold):

    '''

    Returns the average precision for multiple images.

    '''

    final_scores = []

    for i in range(len(all_predictions)):

        image_precision = single_image_score(all_predictions[i], score_threshold)

        final_scores.append(image_precision)

    return np.mean(final_scores)
class AverageMeter:

    def __init__(self): self.reset()

        

    def reset(self): 

        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

        

    def update(self, val, n=1):

        self.val = val

        self.sum += n*val

        self.count += n

        self.avg = self.sum / self.count
def mixup_things(imgs, boxss, labelss, alpha=3.):

    '''

    Take a batch of samples and convert it into a batch of mixup samples.

    '''

    assert len(imgs) == len(boxss) == len(labelss)

    for i in range(len(imgs)):

        assert len(boxss[i]) == len(labelss[i])



    beta_dist = torch.distributions.Beta(alpha, alpha)

    

    batch_size = len(imgs)

    idxs = torch.arange(batch_size)

    idxs_shifted = torch.cat([torch.arange(1, batch_size), torch.arange(1)])



    imgs_mixup, boxss_mixup, labelss_mixup = [], [], []

    for i, j in zip(idxs, idxs_shifted):

        if i == j:

            img, boxs, labels = imgs[i], boxss[i], labelss[i]

        else:

            w = beta_dist.sample()

            img = torch.stack([w * imgs[i], (1 - w) * imgs[j]]).sum(dim=0)

            boxs = torch.cat([boxss[i], boxss[j]], dim=0)

            labels = torch.cat([labelss[i], labelss[j]], dim=0)

        imgs_mixup.append(img)

        boxss_mixup.append(boxs)

        labelss_mixup.append(labels)

    imgs_mixup = torch.stack(imgs_mixup, dim=0)



    assert len(imgs_mixup) == len(boxss_mixup) == len(labelss_mixup)

    for i in range(len(imgs_mixup)):

        assert len(boxss_mixup[i]) == len(labelss_mixup[i])

        

    return imgs_mixup, boxss_mixup, labelss_mixup
import warnings

warnings.filterwarnings('ignore')



class StopTrainException(Exception): pass



def organize_prediction(image_id, target, det):

    boxes = det.detach().cpu().numpy()[:,:4]

    scores = det.detach().cpu().numpy()[:,4]

    boxes[:,2] = boxes[:,2] + boxes[:,0]

    boxes[:,3] = boxes[:,3] + boxes[:,1]

    pred = {'pred_boxes':(2 * boxes).clip(min=0, max=1023).astype(int),

            'scores':scores,

            'gt_boxes':(2 * target['boxes'].cpu().numpy()).clip(min=0, max=1023).astype(int)[:,[1,0,3,2]],  # Need to change yxyx to xyxy for gt_boxes for metric calculation

            'image_id':image_id}

    return pred





class Fitter:

    def __init__(self, data, train_bench, eval_bench, device, base_dir='./', 

                 opt_func=torch.optim.AdamW, scheduler_func=torch.optim.lr_scheduler.OneCycleLR,

                 mixup = False,

                 verbose=True, verbose_step=True): 

        self.data, self.train_bench, self.eval_bench = data, train_bench, eval_bench

        self.opt_func, self.scheduler_func = opt_func, scheduler_func

        self.base_dir, self.device = base_dir, device

        self.mixup = mixup

        self.verbose, self.verbose_step = verbose, verbose_step

        if not os.path.exists(self.base_dir): os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'

        self.best_summary_score = 0.

        self.freeze(-1)

        self.log(f'Fitter prepared.  Device is {self.device}.')

    

    def _get_param_groups(self):

        return [{'params':self.train_bench.model.backbone.parameters()},

                {'params':self.train_bench.model.class_net.parameters()}]



    def freeze(self, idx_grp=None):

        if idx_grp is None:

            for p in self.train_bench.model.parameters(): 

                p.requires_grad = False

            return

        gs = self._get_param_groups()

        for g in gs[:idx_grp]: 

            for p in g['params']: 

                p.requires_grad = False

        for g in gs[idx_grp:]:

            for p in g['params']: 

                p.requires_grad = True

    

    def unfreeze(self):

        for p in self.train_bench.model.parameters(): 

            p.requires_grad = True

    

    def _get_batch_model_inputs(self, images, targets):

        images = torch.stack(images)

        images = images.to(self.device).float()

        boxes = [target['boxes'].to(self.device).float() for target in targets]

        labels = [target['labels'].to(self.device).float() for target in targets]

        return images, boxes, labels

    

    def lr_find(self, min_lr=1e-7, max_lr=1e-2):

        self.train_bench.eval()

        torch.save(self.train_bench.model.state_dict(), f'{self.base_dir}/tmp_lr_find.pth')

        self.train_bench.train()

        

        opt = self.opt_func(self._get_param_groups(), lr=min_lr)

        n_iter = len(self.data.train_dl)

        loss_min = 1e9

        lrs, losses = [], []

        for i, (images, targets, image_ids) in enumerate(self.data.train_dl):

            pos = i / n_iter

            lr = min_lr * (max_lr / min_lr)**pos

            print(f'lr = {lr:.5e}', end='\r')

            for g in opt.param_groups: g['lr'] += lr

            images, boxes, labels = self._get_batch_model_inputs(images, targets)

            opt.zero_grad()

            loss, _, _ = self.train_bench(images, boxes, labels)

            loss.backward()

            opt.step()

            loss = loss.item() / len(images)

            lrs.append(opt.param_groups[0]['lr']); losses.append(loss)

            if loss > 10 * loss_min: 

                break

            else: 

                loss_min = loss

        self.train_bench.model.load_state_dict(torch.load(f'{self.base_dir}/tmp_lr_find.pth'))

        self.lrs, self.losses = np.array(lrs), np.array(losses)

        return self.lrs, self.losses

    

    def plt_lr_find_results(self, skip_end=1):

        assert skip_end >=0

        if skip_end==0: lrs, losses = self.lrs[:], self.losses[:]

        else: lrs, losses = self.lrs[:-skip_end], self.losses[:-skip_end]

        plt.semilogx(lrs, losses);



    def _display_progress(self, dl, step=0, process_name='train', start_time=0, 

                          meter=None, metric_name='loss'):

        process_name = process_name[0].upper() + process_name[1:]

        if self.verbose:

            if step % self.verbose_step == 0:

                print(f'{process_name} Step {step}/{len(dl)}, ' + 

                      (f'{metric_name}: {meter.avg:.5f}, ' if (meter and metric_name) else '') + 

                      f'time: {time.time() - start_time:.5f}', end='\r')

        

    def fit(self, n_epochs=1, max_lr=1e-3):

        self.epoch, self.train_losses, self.valid_losses, self.valid_scores = 0, [], [], []

        self.optimizer = self.opt_func(self._get_param_groups())

        if isinstance(max_lr, int): max_lr = [max_lr / 10, max_lr]

        self.scheduler = self.scheduler_func(

            self.optimizer, max_lr=max_lr, epochs=n_epochs, steps_per_epoch=len(self.data.train_dl))

        train_loader, validation_loader = self.data.train_dl, self.data.valid_dl

        for e in range(n_epochs):

            if self.verbose:

                lr = self.optimizer.param_groups[0]['lr']

                timestamp = datetime.utcnow().isoformat()

                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()

            summary_loss = self.train_one_epoch(train_loader)

            self.train_losses.append(summary_loss.avg)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {time.time() - t:.5f}')

            self.save(f'{self.base_dir}/last-checkpoint.bin')

            

            t = time.time()

            summary_loss = self.validation(validation_loader)

            summary_score = self.calculate_metric(validation_loader)

            self.valid_losses.append(summary_loss.avg)

            self.valid_scores.append(summary_score.avg)

            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, summary_score: {summary_score.avg:.5f} time: {time.time() - t:.5f}')

            if summary_score.avg > self.best_summary_score:

                self.best_summary_score = summary_score.avg

                self.train_bench.eval()

                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

                for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]: os.remove(path)

            self.epoch += 1

    

    def single_image_metrics(self, val_loader, score_threshold=0.37):

        self.eval_bench.eval()

        preds = []

        t = time.time()

        for step, (images, targets, image_ids) in enumerate(val_loader):

            self._display_progress(dl=val_loader, step=step, process_name='metric', start_time=t)

            with torch.no_grad():

                images, boxes, labels = self._get_batch_model_inputs(images, targets)

                dets = self.eval_bench(images, torch.tensor(len(images) * [1]).float().to(self.device))

                for i in range(len(images)):

                    pred = organize_prediction(image_ids[i], targets[i], dets[i])

                    pred.update({'metric':single_image_score(pred, score_threshold)})

                    preds.append(pred)

        return preds

    

    def calculate_metric(self, val_loader, score_threshold=0.37):

        self.eval_bench.eval()

        summary_score = AverageMeter()

        t = time.time()

        for step, (images, targets, image_ids) in enumerate(val_loader):

            self._display_progress(dl=val_loader, step=step, process_name='metric', start_time=t,

                                   meter=summary_score, metric_name='score')

            with torch.no_grad():

                images, boxes, labels = self._get_batch_model_inputs(images, targets)

                dets = self.eval_bench(images, torch.tensor(len(images) * [1]).float().to(self.device))

                preds = []

                for i in range(len(images)):

                    preds.append(organize_prediction(image_ids[i], targets[i], dets[i]))

                score = calculate_final_score(preds, score_threshold)

                summary_score.update(score, len(images))

        return summary_score

    

    def validation(self, val_loader):

        self.train_bench.eval()

        summary_loss = AverageMeter()    

        t = time.time()

        for step, (images, targets, image_ids) in enumerate(val_loader):

            self._display_progress(dl=val_loader, step=step, process_name='Val', start_time=t,

                                   meter=summary_loss, metric_name='valid_loss')

            with torch.no_grad():

                images, boxes, labels = self._get_batch_model_inputs(images, targets)

                loss, _, _ = self.train_bench(images, boxes, labels)

                summary_loss.update(loss.detach().item(), len(images))

        return summary_loss

    

    def train_one_epoch(self, train_loader):

        self.train_bench.train()

        summary_loss = AverageMeter()

        t = time.time()

        for step, (images, targets, image_ids) in enumerate(train_loader):

            self._display_progress(dl=train_loader, step=step, meter=summary_loss, start_time=t, 

                                   process_name='train', metric_name='train_loss')

            images, boxes, labels = self._get_batch_model_inputs(images, targets)

            if self.mixup:

                images, boxes, labels = mixup_things(images, boxes, labels)

            self.optimizer.zero_grad()

            loss, _, _ = self.train_bench(images, boxes, labels)

            loss.backward()

            summary_loss.update(loss.detach().item(), len(images))

            self.optimizer.step()

            self.scheduler.step()

        return summary_loss

        

    def save(self, path):

        self.train_bench.eval()

        torch.save({'model_state_dict':self.train_bench.model.state_dict(),

                    'optimizer_state_dict':self.optimizer.state_dict(),

                    'scheduler_state_dict':self.scheduler.state_dict(),

                    'best_summary_score':self.best_summary_score,

                    'epoch':self.epoch}, path)

        

    def load(self, path):

        checkpoint = torch.load(path)

        self.train_bench.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = self.opt_func(self._get_param_groups())

        self.scheduler = self.scheduler_func(

            self.optimizer, max_lr=[1e-4, 1e-3], epochs=1, steps_per_epoch=len(self.data.train_dl))

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_summary_score = checkpoint['best_summary_score']

        self.epoch = checkpoint['epoch'] + 1

        

    def log(self, message):

        if self.verbose: print(message)

        with open(self.log_path, 'a+') as logger:

            logger.write(f'{message}\n')

            

    def plt_losses(self):

        _, ax = plt.subplots()

        ax.plot(self.train_losses, label='train')

        ax.plot(self.valid_losses, label='valid')

        ax.legend()

        ax.set_xlabel('epoch')

        ax.set_ylabel('loss')
def get_fitter():

    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    train_bench, eval_bench = get_benches()

    train_bench.to(device); eval_bench.to(device)

    return Fitter(data, train_bench, eval_bench, device=device, mixup=False)
fitter = get_fitter()
def plt_img_boxs(img, boxs=None, ax=None, figsize=(6, 6)):

    if ax is None:

        fig, ax = plt.subplots(figsize=figsize)

    if boxs is not None:

        for b in boxs:

            cv2.rectangle(img, (b[1], b[0]),(b[3], b[2]), (0, 1, 0), 2)

    ax.imshow(img)

    ax.axis('off')

    return fig, ax
# del imgs, boxss, labelss, targs, imgids
# imgs, targs, imgids = next(iter(fitter.data.train_dl))

# imgs, boxss, labelss = fitter._get_batch_model_inputs(imgs, targs)

# imgs, boxss, labelss = mixup_things(imgs, boxss, labelss, alpha=1.5)
# i = 3

# img, boxs, labels = imgs[i], boxss[i], labelss[i]

# img_np = img.permute(1, 2, 0).cpu().numpy().copy()

# boxs = boxs.cpu().numpy().astype(np.int32)

# fig, ax = plt_img_boxs(img_np, boxs)
%%time

lrs, losses = fitter.lr_find(max_lr=1e2)
fitter.plt_lr_find_results(skip_end=1)
%%time

fitter.fit(n_epochs=1, max_lr=[1e-4, 5e-4])
fitter.unfreeze()
%%time

lrs, losses = fitter.lr_find(max_lr=1e2)
fitter.plt_lr_find_results(skip_end=10)
%%time

fitter.fit(36, max_lr=[1e-4, 4e-4])
fitter.plt_losses()