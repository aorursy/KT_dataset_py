import numpy as np 
def calc_ap(preds,gt_boxes,iou_thresholds = np.linspace(0.5,0.75,6)):
    ap = 0.
    n = len(iou_thresholds)
    
    for iou_threshold in  iou_thresholds:
        
        #calculate iou
        inter_xmax = np.minimum(preds[:,None,2],gt_boxes[:,2])
        inter_ymax = np.minimum(preds[:,None,3],gt_boxes[:,3])
        inter_xmin = np.maximum(preds[:,None,0],gt_boxes[:,0])
        inter_ymin = np.maximum(preds[:,None,1],gt_boxes[:,1])

        w = (inter_xmax-inter_xmin+1).clip(0.)
        h = (inter_ymax-inter_ymin+1).clip(0.)

        inter_area = w*h

        gt_area = (gt_boxes[:,2]-gt_boxes[:,0]+1) * (gt_boxes[:,3]-gt_boxes[:,1]+1)

        preds_area = (preds[:,2]-preds[:,0]+1) * (preds[:,3]-preds[:,1]+1)
        
        #iou shape (28,20)
        #In matrix iou ,each row(prediction) has iou value with every gt
        iou = inter_area/(gt_area+preds_area[:,None]-inter_area)
        
        gt_to_match = np.zeros(gt_boxes.shape[0])
        
        #pick the one with highest iou with gt
        arg_max = np.argmax(iou,axis = -1)
        
        
        #if the highest iou is less than threshold, constider it as false positive 
        fp_idx = np.where(iou[np.arange(arg_max.shape[0]),arg_max]<iou_threshold)

        fp = len(fp_idx[0])
        
        #discard these false positive predictions 
        iou = np.delete(iou,fp_idx,axis = 0)
        
        #since we have deleted the boxes that are invalid,the rest woud just be counted as tp
        arg_max = np.argmax(iou,axis = -1)
        
        #However,for some cases ,there are more than one predictions match for one gt
        #we need to count extra ones as fp 
        tp_arg_max_filter  = np.unique(arg_max)
        
        gt_to_match[tp_arg_max_filter] = 1

        tp = len(tp_arg_max_filter)

        fp+=(len(arg_max)-tp)
        #check if there's box that no prediction match.
        fn = len(np.where(gt_to_match==0.)[0])

        ap+= tp / (tp + fp + fn)
        
    return ap/n
gt_boxes = np.array([[954, 391,  70,  90],
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
                    [601,  69,  67,  87]])
preds = np.array([[956, 409, 68, 85],
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
                [32, 983, 106, 40]]
                )
scores = np.array([0.9932319, 0.99206185, 0.99145633, 0.9898089, 0.98906296, 0.9817738,
                0.9799762, 0.97967803, 0.9771589, 0.97688967, 0.9562935, 0.9423076,
                0.93556845, 0.9236257, 0.9102379, 0.88644403, 0.8808225, 0.85238415,
                0.8472188, 0.8417798, 0.79908705, 0.7963756, 0.7437897, 0.6044758,
                0.59249884, 0.5557045, 0.53130984, 0.5020239])


scores_idx = np.argsort(scores)[::-1]
preds = preds[scores_idx]
preds[:,2] = preds[:,2] + preds[:,0]
preds[:,3] = preds[:,3] + preds[:,1]
gt_boxes[:,2] = gt_boxes[:,2] + gt_boxes[:,0]
gt_boxes[:,3] = gt_boxes[:,3] + gt_boxes[:,1]

calc_ap(preds,gt_boxes)