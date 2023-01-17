!ls /kaggle/input/ifashion-2020-boxes-w-h/boxes_w_h.pk
# import pandas as pd
# train_df = pd.read_csv('/kaggle/input/imaterialist-fashion-2020-fgvc7/train.csv')
# train_df.head()
# coding=utf-8
# k-means ++ for YOLOv2 anchors
import numpy as np

class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area


def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u


def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)


def init_centroids(boxes,n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = int(np.random.choice(boxes_num, 1))
    centroids.append(boxes[centroid_index])

    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0,n_anchors-1):

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance*np.random.random()

        for i in range(0,boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break

    return centroids


def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids): 
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        groups[group_index].append(box)
        loss += min_distance 
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors): 
        new_centroids[i].w /= len(groups[i])
        new_centroids[i].h /= len(groups[i])

    return new_centroids, groups, loss


def compute_centroids(pickle_dir,n_anchors,loss_convergence,grid_size,iterations_num,plus):

    boxes = []

    
    
    import pickle
    with open(pickle_dir,'rb') as pk:
        boxes_list = pickle.load(pk)
    for box_w_h in boxes_list:
        w = box_w_h[0]
        h = box_w_h[1]
        boxes.append(Box(0, 0, w, h))
       
    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while (True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        print("loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            print("iterations:",iterations)
            print("abs(old_loss - loss)",abs(old_loss - loss))
            return centroids
        old_loss = loss

        for centroid in centroids:
            print(centroid.w * grid_size, centroid.h * grid_size)

    # print result
    for centroid in centroids:
        print("k-means result：\n")
        print(centroid.w * grid_size, centroid.h * grid_size)
        
    return centroids
    

def compute_rate(centroids, comment):
    rates = []
    scales = []
    for box in centroids:
        rate = box.w/box.h 
        rates.append(rate)
        rates.sort()
        
        scale = box.w*box.h
        scales.append(scale)
        scales.sort()
        
    print(comment)
    print("rates: ", rates)
    print('scales: ', scales)
# label_path = "/raid/pengchong_data/Data/Lists/paul_train.txt"
n_anchors = 5
loss_convergence = 1e-6
grid_size = 1
iterations_num = 100
plus = 0
anchors5_1e6_100 = compute_centroids('/kaggle/input/ifashion-2020-boxes-w-h/boxes_w_h.pk',n_anchors,loss_convergence,grid_size,iterations_num,plus)
# label_path = "/raid/pengchong_data/Data/Lists/paul_train.txt"
n_anchors = 3
loss_convergence = 1e-6
grid_size = 1
iterations_num = 100
plus = 1
anchors3_1e6_100 = compute_centroids('/kaggle/input/ifashion-2020-boxes-w-h/boxes_w_h.pk',n_anchors,loss_convergence,grid_size,iterations_num,plus)
compute_rate(anchors3_1e6_100,'anchors3_1e6_100')
compute_rate(anchors5_1e6_100,'anchors5_1e6_100')
