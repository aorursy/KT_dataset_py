# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def iou(box, clusters):

    """

    计算一个ground truth边界盒和k个先验框(Anchor)的交并比(IOU)值。

    参数box: 元组或者数据，代表ground truth的长宽。

    参数clusters: 形如(k,2)的numpy数组，其中k是聚类Anchor框的个数

    返回：ground truth和每个Anchor框的交并比。

    """

    x = np.minimum(clusters[:, 0], box[0])

    y = np.minimum(clusters[:, 1], box[1])

    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:

        raise ValueError("Box has no area")

    intersection = x * y

    box_area = box[0] * box[1]

    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_





def avg_iou(boxes, clusters):

    """

    计算一个ground truth和k个Anchor的交并比的均值。

    """

    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])
def kmeans(boxes, k=9, dist=np.median):

    """

    利用IOU值进行K-means聚类

    参数boxes: 形状为(r, 2)的ground truth框，其中r是ground truth的个数

    参数k: Anchor的个数

    参数dist: 距离函数

    返回值：形状为(k, 2)的k个Anchor框

    """

    # 即是上面提到的r

    rows = boxes.shape[0]

    # 距离数组，计算每个ground truth和k个Anchor的距离

    distances = np.empty((rows, k))

    # 上一次每个ground truth"距离"最近的Anchor索引

    last_clusters = np.zeros((rows,))

    # 设置随机数种子

    np.random.seed()



    # 初始化聚类中心，k个簇，从r个ground truth随机选k个

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    # 开始聚类

    while True:

        # 计算每个ground truth和k个Anchor的距离，用1-IOU(box,anchor)来计算

        for row in range(rows):

            distances[row] = 1 - iou(boxes[row], clusters)

        # 对每个ground truth，选取距离最小的那个Anchor，并存下索引

        nearest_clusters = np.argmin(distances, axis=1)

        # 如果当前每个ground truth"距离"最近的Anchor索引和上一次一样，聚类结束

        if (last_clusters == nearest_clusters).all():

            break

        # 更新簇中心为簇里面所有的ground truth框的均值

        for cluster in range(k):

            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        # 更新每个ground truth"距离"最近的Anchor索引

        last_clusters = nearest_clusters



    return clusters
def cvt_csv(whd_df):

    # change dtype

    whd_df[['bbox_xmin', 'bbox_ymin', 'bbox_width', 'bbox_height']] = whd_df['bbox'].str.split(',', expand=True)

    whd_df['bbox_xmin'] = whd_df['bbox_xmin'].str.replace('[', '').astype(float)

    whd_df['bbox_ymin'] = whd_df['bbox_ymin'].str.replace(' ', '').astype(float)

    whd_df['bbox_width'] = whd_df['bbox_width'].str.replace(' ', '').astype(float)

    whd_df['bbox_height'] = whd_df['bbox_height'].str.replace(']', '').astype(float)



    # add xmax, ymax, and area columns for bounding box

    whd_df['bbox_xmax'] = whd_df['bbox_xmin'] + whd_df['bbox_width']

    whd_df['bbox_ymax'] = whd_df['bbox_ymin'] + whd_df['bbox_height']

    whd_df['bbox_area'] = whd_df['bbox_height'] * whd_df['bbox_width']

    whd_df['hw_ratio'] = whd_df['bbox_height'] / whd_df['bbox_width']



    return whd_df
train_csv = '/kaggle/input/global-wheat-detection/train.csv'

df = pd.read_csv(train_csv)

df = cvt_csv(df)
df
h = df['bbox_height']

w = df['bbox_width']
boxes = np.array([w, h]).T

boxes.shape
%time

cluster_w_h = kmeans(boxes)
w, h = np.split(cluster_w_h, 2, axis=1)

ratio = w / h

ratio = ratio.flatten()

ratio.sort()

ratio