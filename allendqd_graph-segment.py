import numpy as np

import copy

from matplotlib import pyplot as plt

import h5py

from PIL import Image





def load_dataset():

	train_dataset = h5py.File('datasets/train_signs.h5', "r")

	train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features

	train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

	test_dataset = h5py.File('datasets/test_signs.h5', "r")

	test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features

	test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

	classes = np.array(test_dataset["list_classes"][:])  # the list of classes

	train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

	test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

	return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes





class edge:

	def __init__(self, node1, node2, weight):

		self.node1 = node1

		self.node2 = node2

		self.weight = weight



	def __lt__(self, other):

		if isinstance(other, edge):

			return self.weight < other.weight

		else:

			return NotImplemented





def cal_similarity(node1, node2, channel=3):  # 默认通道是最后一位 eg（64, 64, 3）

	temp = 0

	for i in range(channel):

		temp += (node1[i] - node2[i]) ** 2

	temp = np.sqrt(temp)

	return temp





def create_edges(image, neighborhood8=False):

	x_size, y_size, channel = image.shape

	edges = []

	for i in range(x_size):

		for j in range(y_size):

			if i + 1 < x_size:

				temp = cal_similarity(image[i, j, :], image[i + 1, j, :])

				e = edge(i * y_size + j, (i + 1) * y_size + j, temp)

				edges.append(e)

			if j + 1 < y_size:

				temp = cal_similarity(image[i, j, :], image[i, j + 1, :])

				e = edge(i * y_size + j, i * y_size + j + 1, temp)

				edges.append(e)

			if neighborhood8:

				if i + 1 < x_size and j + 1 < y_size:

					temp = cal_similarity(image[i, j, :], image[i + 1, j + 1, :])

					e = edge(i * y_size + j, (i + 1) * y_size + j + 1, temp)

					edges.append(e)

				if i - 1 > 0 and j + 1 < y_size:

					temp = cal_similarity(image[i, j, :], image[i - 1, j + 1, :])

					e = edge(i * y_size + j, (i - 1) * y_size + j + 1, temp)

					edges.append(e)

	edges = np.sort(edges)

	return edges





def create_trees(edges, shape, k):

	Intcs = np.zeros(shape, dtype='S8')

	for i in range(shape[0]):

		for j in range(shape[1]):

			Intcs[i, j] = str(i * shape[1] + j)

	# ind = np.where(Intcs == Intcs[0, 0])

	Intc_val = np.zeros(shape)  # 存intc的值

	C_val = np.ones(shape)

	print(len(edges))

	for i in range(len(edges)):

		x1 = edges[i].node1 // shape[1]

		x2 = edges[i].node2 // shape[1]

		y1 = edges[i].node1 % shape[1]

		y2 = edges[i].node2 % shape[1]

		# print(Intcs[x1, y1], '-=-=-')

		if Intcs[x1, y1] != Intcs[x2, y2]:

			index_name1 = np.where(Intcs == Intcs[x1, y1])

			index_name2 = np.where(Intcs == Intcs[x2, y2])

			if len(index_name1[0]) == 1 and len(index_name2[0]) == 1:

				Intcs[x2, y2] = Intcs[x1, y1]

				C_val[x1, y1] += 1

				C_val[x2, y2] += 1

				Intc_val[x1, y1] = edges[i].weight

				Intc_val[x2, y2] = edges[i].weight

			else:

				inc1 = Intc_val[x1, y1] + k / C_val[x1, y1]  # 可能有问题

				inc2 = Intc_val[x2, y2] + k / C_val[x2, y2]

				Inct_val_min = min(inc1, inc2)

				diff = edges[i].weight

				if diff <= Inct_val_min:



					Intcs[index_name2] = Intcs[x1, y1]

					C_val[index_name1] = C_val[x2, y2] + C_val[x1, y1] 

					C_val[index_name2] = C_val[index_name1[0][0], index_name1[1][0]]

					Intc_val[index_name1] = diff#max(diff, Intc_val[x1, y1])

					Intc_val[index_name2] = diff#max(diff, Intc_val[x1, y1])

	return Intcs, Intc_val, C_val





def test(image):

	image_copy = copy.copy(image)

	edges = create_edges(image, True)

	Intcs, Intc_val, C_val = create_trees(edges, (image.shape[0], image.shape[1]), 500)

	unique_intcs = np.unique(Intcs)

	print(unique_intcs.shape, unique_intcs[0:100])

	for i in range(len(unique_intcs)):

		index = np.where(Intcs == unique_intcs[i])

		# print(index[0], index[1])

		image_copy[index[0], index[1], :] = image[index[0][0], index[1][0], :]

	res = Image.fromarray(np.uint8(image_copy))

	res.save('../working/result.jpg')

	plt.imshow(image_copy)

	plt.show()





# im = Image.open('test_segment_graph.jpg', 'r')

im = Image.open('../input/test-graph-segment1/22.jpg', 'r')

im_aray = np.array(im)

print(im_aray.shape)

test(im_aray)
