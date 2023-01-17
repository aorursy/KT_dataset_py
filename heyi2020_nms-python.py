import torch

import numpy as np



def nms(dets, thresh):

    x1 = dets[:, 1]

    y1 = dets[:, 0]

    x2 = dets[:, 3]

    y2 = dets[:, 2]

    scores = dets[:, 4]

    

    areas = (x2-x1+1)*(y2-y1+1)

#     print(areas)

    order = scores.sort(0,descending=True)[1] #按照置信度降序排列

    keep = []

    while order.size(0)>0:

        

        #保留置信度最大的框

        ind = order[0]

        keep.append(ind)

        

        x11 = x1[order[1:]].masked_fill(~torch.gt(x1[order[1:]],x1[ind]), x1[ind])

        y11 = y1[order[1:]].masked_fill(~torch.gt(y1[order[1:]],y1[ind]), y1[ind])

        x22 = x2[order[1:]].masked_fill(~torch.lt(x2[order[1:]],x2[ind]), x2[ind])

        y22 = y2[order[1:]].masked_fill(~torch.lt(y2[order[1:]],y2[ind]), y2[ind])

        

        

        

        intersec = (x22-x11+1).clamp(min=0)*(y22-y11+1).clamp(min=0)

        

#         print('intesec',intersec)

#         print('areas_',areas[order[1:]])

        

        ious = intersec/(areas[ind] + areas[order[1:]] - intersec)

#         print(ious)

        order = order[1:][ious<thresh]

    return keep

        

        



        

    
boxes=torch.Tensor([[100,100,210,210,0.72],

        [250,250,420,420,0.8],

        [220,220,320,330,0.92],

        [100,100,210,210,0.72],

        [230,240,325,330,0.81],

        [220,230,315,340,0.9]])

keep = nms(boxes,0.7)

print(keep)
import matplotlib.pyplot as plt

def plot_bbox(dets, c='k'):

    dets_copy = dets.numpy()

    x1 = dets_copy[:,0]

    y1 = dets_copy[:,1]

    x2 = dets_copy[:,2]

    y2 = dets_copy[:,3]

    

    

    plt.plot([x1,x2], [y1,y1], c)

    plt.plot([x1,x1], [y1,y2], c)

    plt.plot([x1,x2], [y2,y2], c)

    plt.plot([x2,x2], [y1,y2], c)

    plt.title("after nms")



plot_bbox(boxes,'k')   # before nms



keep = nms(boxes, thresh=0.7)

print(list(map(lambda x:x.item(),keep)))

# print(boxes[],boxes[[2,0]])

plot_bbox(boxes[list(map(lambda x:x.item(),keep))], 'r')# after nms