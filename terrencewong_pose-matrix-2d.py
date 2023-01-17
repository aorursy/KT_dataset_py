input_pose = {

      "nose": {

        "x": 327.89340098007864,

        "y": 69.99157457559647,

        "score": 0.9365222454071045

      },

      "leftEye": {

        "x": 333.20123042231023,

        "y": 64.32421818551984,

        "score": 0.9083114862442017

      },

      "rightEye": {

        "x": 323.97788449992305,

        "y": 64.71316861066492,

        "score": 0.906348705291748

      },

      "leftEar": {

        "x": 342.36758165774137,

        "y": 65.5764081901479,

        "score": 0.7242887020111084

      },

      "rightEar": {

        "x": 316.4632065192513,

        "y": 66.71006369665031,

        "score": 0.5855812430381775

      },

      "leftShoulder": {

        "x": 363.50065695721173,

        "y": 96.95807999762417,

        "score": 0.9976296424865723

      },

      "rightShoulder": {

        "x": 306.9370521462482,

        "y": 98.03811477946346,

        "score": 0.9979366064071655

      },

      "leftElbow": {

        "x": 397.2907660111137,

        "y": 123.36682658923377,

        "score": 0.9930130243301392

      },

      "rightElbow": {

        "x": 282.42742472109586,

        "y": 135.51764322441312,

        "score": 0.9713993072509766

      },

      "leftWrist": {

        "x": 412.8602578121683,

        "y": 146.42606703143252,

        "score": 0.9441394805908203

      },

      "rightWrist": {

        "x": 276.6764075237772,

        "y": 141.20603945247854,

        "score": 0.9404780268669128

      },

      "leftHip": {

        "x": 346.161307791005,

        "y": 182.25992885185553,

        "score": 0.9988557696342468

      },

      "rightHip": {

        "x": 310.64364876954454,

        "y": 182.69368222792195,

        "score": 0.9984779953956604

      },

      "leftKnee": {

        "x": 388.00455773395043,

        "y": 228.69147455803704,

        "score": 0.9902427196502686

      },

      "rightKnee": {

        "x": 299.15315039261526,

        "y": 242.43479372779154,

        "score": 0.9965622425079346

      },

      "leftAnkle": {

        "x": 388.3440190190854,

        "y": 302.21759440223,

        "score": 0.9546959400177002

      },

      "rightAnkle": {

        "x": 286.0427179751189,

        "y": 316.6578940275673,

        "score": 0.9885237812995911

      },

      "neck": {

        "x": 335,

        "y": 97,

        "score": 0.9976296424865723

      },

      "hip": {

        "x": 328,

        "y": 182,

        "score": 0.9984779953956604

      }

    }
input_vector = []

labels=['leftShoulder','leftElbow','leftWrist','rightShoulder','rightElbow','rightWrist',

'leftHip','leftKnee','leftAnkle','rightHip','rightKnee','rightAnkle']



for k in labels:

    input_vector.append(input_pose[k]['x'])

    input_vector.append(input_pose[k]['y'])



import numpy as np

input_vector = np.array(input_vector)

print('input_vector', input_vector)


def normalize(v):

    xs = v[::2]

    ys = v[1::2]

    min_x, max_x = min(xs), max(xs)

    min_y, max_y = min(ys), max(ys)

    for i in range(0,len(v), 2):

        v[i] = (v[i] - min_x) / (max_x-min_x)

    for i in range(1, len(v), 2):

        v[i] = (v[i] - min_y) / (max_x-min_x)



normalize(input_vector)

print('normalized input_vector', input_vector)
f = np.load('/kaggle/input/pose2d80k3/poseMatrix.npz')

pose_mat = f['pose']

index_mat = f['index']



print('pose_mat shape', pose_mat.shape)
from scipy import spatial



ev = spatial.distance.cdist(pose_mat, [input_vector], 'euclidean').reshape(-1)

ev_i = ev.argsort()[:5]



# 打印前5名

for i in ev_i:

    print('Index', i, 'dist', ev[i], 'refer', index_mat[i][1], index_mat[i][0])

import matplotlib.pyplot as plt



for i in ev_i:

    v1 = pose_mat[i]

    plt.scatter(input_vector[::2], input_vector[1::2])

    plt.scatter(v1[::2], v1[1::2])

    plt.axis('equal')

    plt.gca().invert_yaxis()

    plt.legend(labels=['input vector', str(i)])

    plt.title(index_mat[i][0].split('-')[0] + ' ' + index_mat[i][1] + ' time:' + index_mat[i][3])

    plt.show()
cv = spatial.distance.cdist(pose_mat, [input_vector], 'cosine').reshape(-1)

cv_i = cv.argsort()[:5]



# 打印前5名

for i in cv_i:

    print('Index', i, 'dist', cv[i], 'refer', index_mat[i][1], index_mat[i][0])
for i in cv_i:

    v1 = pose_mat[i]

    plt.scatter(input_vector[::2], input_vector[1::2])

    plt.scatter(v1[::2], v1[1::2])

    plt.axis('equal')

    plt.gca().invert_yaxis()

    plt.legend(labels=['input vector', str(i)])

    plt.title(index_mat[i][0].split('-')[0] + ' ' + index_mat[i][1] + ' time:' + index_mat[i][3])

    plt.show()