from cvxpy import *
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import numpy as np
graph_32 = plt.imread("../input/lena32color.tiff")
graph_64 = plt.imread("../input/lena64color.tiff")
prediction_graph = np.zeros([64,64,3])
for i in np.arange(64):
    for j in np.arange(64):
        if j % 2 == 0 and i % 2 == 0:
            prediction_graph[i,j] = graph_32[int(i/2),int(j/2)]
            
prediction_graph = prediction_graph / 255

import tensorflow as tf
Known = prediction_graph.copy()
Known[Known > 0] = 1
def smooth(value, *args):
        value = expressions.expression.Expression.cast_to_const(value)
        rows, cols = value.shape
        args = list(map(expressions.expression.Expression.cast_to_const, args))
        values = [value] + list(args)
        diffs = []
        for mat in values:
            diffs += [
                mat[0:rows-1, 1:cols] - mat[0:rows-1, 0:cols-1],
                mat[1:rows, 0:cols-1] - mat[0:rows-1, 0:cols-1],
            ]
        length = diffs[0].shape[0]*diffs[1].shape[1]
        stacked = vstack([reshape(diff, (1, length)) for diff in diffs])
        return sum(norm(stacked, p=2, axis=0))
rows = 64
cols = 64
colors = 3
colors = 3

for i in range(colors):
    if i == 0:
        v = 0
        U_1 = Variable(shape=(rows, cols))

        obj = norm(U_1[:63,:] - U_1[1:,:],'fro') + 0.6*norm(U_1[:,:63] - U_1[:,1:],'fro') #+ 0.0165*smooth(U_1)
        constraints = [multiply(Known[:,:,i],U_1) == multiply(prediction_graph[:,:,i],Known[:,:,i])]
        prob = Problem(Minimize(obj),constraints)
        prob.solve(solver = SCS)
        print(prob.status)
    elif i == 1:
        v = 0
        U_2 = Variable(shape=(rows, cols))
        obj = norm(U_2[:63,:] - U_2[1:,:],'fro') + 0.6*norm(U_2[:,:63] - U_2[:,1:],'fro') #+ 0.0165*smooth(U_2)
        constraints = [multiply(Known[:,:,i],U_2) == multiply(prediction_graph[:,:,i],Known[:,:,i])]
        prob = Problem(Minimize(obj),constraints)
        prob.solve(solver = SCS)
        print(prob.status)
    else:
        v = 0
        U_3 = Variable(shape=(rows, cols))
        obj = norm(U_3[:63,:] - U_3[1:,:],'fro') + 0.6*norm(U_3[:,:63] - U_3[:,1:],'fro') #+ 0.0165*smooth(U_3)
        constraints = [multiply(Known[:,:,i],U_3) == multiply(prediction_graph[:,:,i],Known[:,:,i])]
        prob = Problem(Minimize(obj),constraints)
        prob.solve(solver = SCS)
        print(prob.status)
        
graph_rebuild = np.zeros((rows, cols, colors))
for i,j in enumerate([U_1.value,U_2.value,U_3.value]):
    graph_rebuild[:, :, i] = j
graph_rebuild[graph_rebuild > 1] = 1
graph_rebuild[graph_rebuild < 0] = 0

plt.imshow(graph_rebuild)
psnr = tf.image.psnr(graph_rebuild, graph_64/255, max_val=1)
with tf.Session() as sess:
    print(sess.run(psnr))
    