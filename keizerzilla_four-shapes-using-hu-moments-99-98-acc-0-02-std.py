import os

import sys

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.svm import SVC as SVM

from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.neighbors import NearestCentroid as NC

from sklearn.model_selection import train_test_split as data_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
classes = {

	"circle"   : 1,

	"square"   : 2,

	"star"     : 3,

	"triangle" : 4

}



classifiers = {

	"NC"         : NC(),

	"LDA"        : LDA(),

	"QDA"        : QDA(),

	"SVM_linear" : SVM(kernel="linear"),

	"SVM_radial" : SVM(kernel="rbf")

}
def feature_extraction(data_file, dataset="../input/shapes/"):

	dump = []

	

	print("Extracting Hu moments...")

	

	for c, idx in classes.items():

		class_folder = dataset + "{}/".format(c)

		

		for f in os.listdir(class_folder):

			fpath = class_folder + f

			sample = int(f.replace(".png", ""))

			

			img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

			img = cv2.bitwise_not(img)

			hu = cv2.HuMoments(cv2.moments(img))

			

			for i in range(0, 7):

				hu[i] = -1 * np.sign(hu[i]) * np.log10(np.abs(hu[i]))

			

			hu = hu.reshape((1, 7)).tolist()[0] + [sample, idx]

			dump.append(hu)

		

		print(c, "ok!")



	cols = ["hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7", "sample", "class"]

	

	df = pd.DataFrame(dump, columns=cols)

	df.to_csv(data_file, index=None)

	

	print("Extraction done!")
data_file = "hu_moments.csv"



feature_extraction(data_file)
def classification(data_file, rounds=100, remove_disperse=[]):

	df = pd.read_csv(data_file)

	df = df.drop(["sample"], axis=1)

	

	if remove_disperse:

		df = df.drop(remove_disperse, axis=1)

	

	X = df.drop(["class"], axis=1)

	y = df["class"]

	

	ans = {key: {"score" : [], "sens" : [], "spec" : []}

	       for key, value in classifiers.items()}

	

	print("Classifying...")

	

	for i in range(rounds):

		X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.3)

		

		for name, classifier in classifiers.items():

			scaler = StandardScaler()

			scaler.fit(X_train)

			X_train = scaler.transform(X_train)

			X_test = scaler.transform(X_test)

			

			classifier.fit(X_train, y_train)

			score = classifier.score(X_test, y_test)

			

			ans[name]["score"].append(score)

		

	print("Classification done!")

	

	return ans
ans = classification(data_file)
def sumary(ans, title="Summary"):

	size = 70

	separator = "-"

	

	print(separator*size)

	print("SUMARY: {}".format(title))

	print(separator*size)

	print("CLASSIF\t\tMEAN\tMEDIAN\tMINV\tMAXV\tSTD")

	print(separator*size)

	

	for n in ans:

		m = round(np.mean(ans[n]["score"])*100, 2)

		med = round(np.median(ans[n]["score"])*100, 2)

		minv = round(np.min(ans[n]["score"])*100, 2)

		maxv = round(np.max(ans[n]["score"])*100, 2)

		std = round(np.std(ans[n]["score"])*100, 2)

		

		print("{:<16}{}\t{}\t{}\t{}\t{}".format(n, m, med, minv, maxv, std))

	

	print(separator*size)

	print()
sumary(ans)