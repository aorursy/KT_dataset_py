import torch
from torch import tensor
from torchvision.ops.boxes import box_iou
def align_coordinates(boxes):
    """Align coordinates (x1,y1) < (x2,y2) to work with torchvision `box_iou` op
    Arguments:
        boxes (Tensor[N,4])
    
    Returns:
        boxes (Tensor[N,4]): aligned box coordinates
    """
    x1y1 = torch.min(boxes[:,:2,],boxes[:, 2:])
    x2y2 = torch.max(boxes[:,:2,],boxes[:, 2:])
    boxes = torch.cat([x1y1,x2y2],dim=1)
    return boxes


def calculate_iou(gt, pr, form='pascal_voc'):
    """Calculates the Intersection over Union.

    Arguments:
        gt: (torch.Tensor[N,4]) coordinates of the ground-truth boxes
        pr: (torch.Tensor[M,4]) coordinates of the prdicted boxes
        form: (str) gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
        IoU values for every element in boxes1 and boxes2
    """
    if form == 'coco':
        gt = gt.clone()
        pr = pr.clone()

        gt[:,2] = gt[:,0] + gt[:,2]
        gt[:,3] = gt[:,1] + gt[:,3]
        pr[:,2] = pr[:,0] + pr[:,2]
        pr[:,3] = pr[:,1] + pr[:,3]

    gt = align_coordinates(gt)
    pr = align_coordinates(pr)
    
    return box_iou(gt,pr)
# These are the predicted boxes (and scores) from my locally trained model.
preds = tensor([[956, 409, 68, 85],
                  [883, 945, 85, 77],
                  [745, 468, 81, 87],
                  [658, 239, 103, 105],
                  [518, 419, 91, 100],
                  [711, 805, 92, 106],
                  [62, 213, 72, 64],
                  [884, 175, 109, 68],
                  [721, 626, 96, 104],
                  [878, 619, 121, 81],
                  [887, 107, 111, 71],
                  [827, 525, 88, 83],
                  [816, 868, 102, 86],
                  [166, 882, 78, 75],
                  [603, 563, 78, 97],
                  [744, 916, 68, 52],
                  [582, 86, 86, 72],
                  [79, 715, 91, 101],
                  [246, 586, 95, 80],
                  [181, 512, 93, 89],
                  [655, 527, 99, 90],
                  [568, 363, 61, 76],
                  [9, 717, 152, 110],
                  [576, 698, 75, 78],
                  [805, 974, 75, 50],
                  [10, 15, 78, 64],
                  [826, 40, 69, 74],
                  [32, 983, 106, 40]]).float()

targs = tensor([[954, 391,  70,  90],
       [660, 220,  95, 102],
       [ 64, 209,  76,  57],
       [896,  99, 102,  69],
       [747, 460,  72,  77],
       [885, 163, 103,  69],
       [514, 399,  90,  97],
       [702, 794,  97,  99],
       [721, 624,  98, 108],
       [826, 512,  82,  94],
       [883, 944,  79,  74],
       [247, 594, 123,  92],
       [673, 514,  95, 113],
       [829, 847, 102, 110],
       [ 94, 737,  92, 107],
       [588, 568,  75, 107],
       [158, 890, 103,  64],
       [744, 906,  75,  79],
       [826,  33,  72,  74],
       [601,  69,  67,  87]]).float()

scores = tensor([0.9932319, 0.99206185, 0.99145633, 0.9898089, 0.98906296, 0.9817738,
                   0.9799762, 0.97967803, 0.9771589, 0.97688967, 0.9562935, 0.9423076,
                   0.93556845, 0.9236257, 0.9102379, 0.88644403, 0.8808225, 0.85238415,
                   0.8472188, 0.8417798, 0.79908705, 0.7963756, 0.7437897, 0.6044758,
                   0.59249884, 0.5557045, 0.53130984, 0.5020239])
preds.shape,scores.shape,targs.shape
preds = preds[scores.argsort().flip(-1)]
iou_mat = calculate_iou(targs,preds,form='coco'); iou_mat[:4,:4]
gt_count, pr_count = iou_mat.shape
thresh = 0.5
iou_mat = iou_mat.where(iou_mat>thresh,tensor(0.)); iou_mat[:4,:4]
def get_mappings(iou_mat):
    mappings = torch.zeros_like(iou_mat)
    gt_count, pr_count = iou_mat.shape
    
    #first mapping (max iou for first pred_box)
    if not iou_mat[:,0].eq(0.).all():
        # if not a zero column
        mappings[iou_mat[:,0].argsort()[-1],0] = 1

    for pr_idx in range(1,pr_count):
        # Sum of all the previous mapping columns will let 
        # us know which gt-boxes are already assigned
        not_assigned = torch.logical_not(mappings[:,:pr_idx].sum(1)).long()

        # Considering unassigned gt-boxes for further evaluation 
        targets = not_assigned * iou_mat[:,pr_idx]

        # If no gt-box satisfy the previous conditions
        # for the current pred-box, ignore it (False Positive)
        if targets.eq(0).all():
            continue

        # max-iou from current column after all the filtering
        # will be the pivot element for mapping
        pivot = targets.argsort()[-1]
        mappings[pivot,pr_idx] = 1
    return mappings
mappings = get_mappings(iou_mat)
assert mappings.sum(1).le(1).all()
assert mappings.sum(0).le(1).all()
tp = mappings.sum(); tp
fp = mappings.sum(0).eq(0).sum(); fp
fn = mappings.sum(1).eq(0).sum(); fn
mAP = tp / (tp+fp+fn); mAP
def calculate_map(gt_boxes,pr_boxes,scores,thresh=0.5,form='pascal_voc'):
    # sorting
    pr_boxes = pr_boxes[scores.argsort().flip(-1)]
    iou_mat = calculate_iou(gt_boxes,pr_boxes,form) 
    
    # thresholding
    iou_mat = iou_mat.where(iou_mat>thresh,tensor(0.))
    
    mappings = get_mappings(iou_mat)
    
    # mAP calculation
    tp = mappings.sum()
    fp = mappings.sum(0).eq(0).sum()
    fn = mappings.sum(1).eq(0).sum()
    mAP = tp / (tp+fp+fn)
    
    return mAP
calculate_map(targs,preds,scores,form='coco')
calculate_map(targs,preds,scores,thresh=0.75,form='coco')
