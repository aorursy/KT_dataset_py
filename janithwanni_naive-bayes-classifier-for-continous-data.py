#iris species classifying using naive bayes

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import StratifiedKFold

import math



#open csv file

main_df = pd.read_csv("../input/Iris.csv")



#clean the data

print(main_df.head())

#categorizing 

#Iris-virginica as 0

#Iris-versicolor as 1

#Iris-setosa as 2



#check empty fields

print(main_df.isnull().sum())

#no null values

main_df["Species"] = main_df["Species"].map({

	"Iris-virginica":0,

	"Iris-versicolor":1,

	"Iris-setosa":2

	}).astype(int)



#visualizing the data

print(main_df.describe())

print(main_df.Species.value_counts())

plt.hist(

	main_df.SepalLengthCm.values,

	bins=30,

	normed=True,

	histtype="stepfilled",

	color='b',

	label='Sepal Length in cm',

	alpha=0.4

	)

plt.hist(

	main_df.SepalWidthCm.values,

	bins=30,

	normed=True,

	histtype="stepfilled",

	color='g',

	alpha=0.5,

	label='Sepal Width in cm'

	)

plt.hist(

	main_df.PetalLengthCm.values,

	bins=30,

	normed=True,

	histtype="stepfilled",

	color='r',

	alpha=0.6,

	label='Petal Length in cm'

	)

plt.hist(

	main_df.PetalWidthCm.values,

	bins=30,

	normed=True,

	histtype="stepfilled",

	color='m',

	alpha=0.7,

	label='Petal Width in cm'

	)

plt.xlabel("Iris Data")

plt.ylabel("Probability")

plt.title("Histogram")

plt.show()

plt.figure()

sns.set(color_codes=True)

sns.kdeplot(main_df.SepalLengthCm.values,shade=True,bw=1.2,color=sns.xkcd_rgb["denim blue"])

sns.kdeplot(main_df.SepalWidthCm.values,shade=True,bw=1.2,color=sns.xkcd_rgb["green"])

sns.kdeplot(main_df.PetalLengthCm.values,shade=True,bw=1.2,color=sns.xkcd_rgb["red"])

sns.kdeplot(main_df.PetalWidthCm.values,shade=True,bw=1.2,color=sns.xkcd_rgb["maroon"])

plt.show()

#plt.figure()

sns.pairplot(main_df.drop('Id',axis=1))

plt.show()



#generate the probabilities

#assume we are not building an all purpose NB model. we are just making a model for this scenario only.

#therefore we know the features and the classes



#calculating prior probabilties of classes

priors = {0:0.0,1:0.0,2:0.0}

for column in main_df.Species.value_counts(normalize=True).index:

	priors[column] = main_df.Species.value_counts(normalize=True)[column]

	#print(column)

print(priors)



evidenceLikelihoods = {

						"SepalLengthCm":{0:{

											"mean":main_df.loc[main_df["Species"] == 0,"SepalLengthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 0,"SepalLengthCm"].std()

											},

										 1:{

											"mean":main_df.loc[main_df["Species"] == 1,"SepalLengthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 1,"SepalLengthCm"].std()

											},

										 2:{

											"mean":main_df.loc[main_df["Species"] == 2,"SepalLengthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 2,"SepalLengthCm"].std()

											}},

						"SepalWidthCm":{0:{

											"mean":main_df.loc[main_df["Species"] == 0,"SepalWidthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 0,"SepalWidthCm"].std()

											},

										 1:{

											"mean":main_df.loc[main_df["Species"] == 1,"SepalWidthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 1,"SepalWidthCm"].std()

											},

										 2:{

											"mean":main_df.loc[main_df["Species"] == 2,"SepalWidthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 2,"SepalWidthCm"].std()

											}},

						"PetalLengthCm":{0:{

											"mean":main_df.loc[main_df["Species"] == 0,"PetalLengthCm"].mean(),

											"std":main_df.loc[main_df["Species"] ==0,"PetalLengthCm"].std()

											},

										 1:{

											"mean":main_df.loc[main_df["Species"] == 1,"PetalLengthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 1,"PetalLengthCm"].std()

											},

										 2:{

											"mean":main_df.loc[main_df["Species"] == 2,"PetalLengthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 2,"PetalLengthCm"].std()

											}},

						"PetalWidthCm":{0:{

											"mean":main_df.loc[main_df["Species"] == 0,"PetalWidthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 0,"PetalWidthCm"].std()

											},

										 1:{

											"mean":main_df.loc[main_df["Species"] == 1,"PetalWidthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 1,"PetalWidthCm"].std()

											},

										 2:{

											"mean":main_df.loc[main_df["Species"] == 2,"PetalWidthCm"].mean(),

											"std":main_df.loc[main_df["Species"] == 2,"PetalWidthCm"].std()

											}}

						}

print(evidenceLikelihoods)



#classifiying functions



#probability density function

def pdf(mean,std,x):

	ePart = math.pow(math.e, -(x - mean)**2 / (2 * std ** 2))

	return (1.0 / (math.sqrt(2 * math.pi) * std )) * ePart



#classifying function

def classify(x):

	#let x be the sepallength,width petallength,width

	clsprobs = []

	

	for i in range(3):

		clsprobs.append(pdf(

				evidenceLikelihoods["SepalLengthCm"][i]["mean"],

				evidenceLikelihoods["SepalLengthCm"][i]["std"],

				x["SepalLengthCm"]) *

			pdf(

				evidenceLikelihoods["SepalWidthCm"][i]["mean"],

				evidenceLikelihoods["SepalWidthCm"][i]["std"],

				x["SepalWidthCm"]) *

			pdf(

				evidenceLikelihoods["PetalLengthCm"][i]["mean"],

				evidenceLikelihoods["PetalLengthCm"][i]["std"],

				x["PetalLengthCm"]) *

			pdf(

				evidenceLikelihoods["PetalWidthCm"][i]["mean"],

				evidenceLikelihoods["PetalWidthCm"][i]["std"],

				x["PetalWidthCm"]) * priors[i])

	return np.argmax(clsprobs)



#cross validation



#f1 score calculating function

def getf1(cmdf):

	pvir = cmdf[0][0] / cmdf.iloc[0,:].sum(axis=0)

	pset = cmdf[2][2] / cmdf.iloc[2,:].sum(axis=0)

	pver = cmdf[1][1] / cmdf.iloc[1,:].sum(axis=0)

	print("pvir: %s pset : %s pver: %s" % (pvir,pset,pver))

	rvir = cmdf[0][0] / cmdf.iloc[:,0].sum(axis=0)

	rset = cmdf[2][2] / cmdf.iloc[:,2].sum(axis=0)

	rver = cmdf[1][1] / cmdf.iloc[:,1].sum(axis=0)

	print("rvir: %s rset : %s rver: %s" % (rvir,rset,rver))

	P = (pvir+pset+pver) / 3

	R = (rvir+rset+rver) / 3

	print("P: %s R: %s " % (P,R))

	print("F1: %s" %( (2 * P * R) / (P + R) ))

	return (2 * P * R) / (P + R)



X = main_df.drop(["Id","Species"],axis=1)

y = main_df["Species"]

n_splits = 10

skf = StratifiedKFold(n_splits=n_splits)

f1score = 0

cmdf = 0

f1score = 0

f1tot = 0



for train,test in skf.split(X,y):

	cm = {0:{0:0.0,1:0.0,2:0.0},1:{0:0.0,1:0.0,2:0.0},2:{0:0.0,1:0.0,2:0.0}}

	for elem in train:

		#print(X.iloc[elem],"train elem")

		pred = classify(X.iloc[elem])

		cm[pred][y.iloc[elem]] += 1



	print(pd.DataFrame.from_dict(cm))

	cmdf = pd.DataFrame.from_dict(cm)

	f1tot += getf1(cmdf)

	print("f1tot: %s" % (f1tot))

	cm = {0:{0:0.0,1:0.0,2:0.0},1:{0:0.0,1:0.0,2:0.0},2:{0:0.0,1:0.0,2:0.0}}

	for elem in test:

		#print(X.iloc[elem],"test elem")

		pred = classify(X.iloc[elem])

		cm[pred][y.iloc[elem]] += 1



	print(pd.DataFrame.from_dict(cm))

	cmdf = pd.DataFrame.from_dict(cm)

	f1tot += getf1(cmdf)

	print("f1tot: %s " % (f1tot))

	

f1score = f1tot / (n_splits * 2)

print(f1score)
