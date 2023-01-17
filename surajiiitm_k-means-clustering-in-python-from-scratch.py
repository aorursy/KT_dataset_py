# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rd

import matplotlib.pyplot as plt

from math import sqrt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/big-mart-sale/clustering.csv")

data.head()
X = data[["ApplicantIncome", "LoanAmount"]]

# Visualize data point

plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c="blue")

plt.xlabel("Applicant Income")

plt.ylabel("Loan Amount (In Thousands)")

plt.show()
# step 1 and 2



# number of centriod

K=3



# select random observation as a centriod 

Centroids = (X.sample(n=K))

plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c="blue")

plt.scatter(Centroids["ApplicantIncome"], Centroids["LoanAmount"], c="red")

plt.xlabel("AnnualIncome")

plt.ylabel("Loan Amount (In Thousand)")

plt.show()
Centroids
# Step 3 - Assign all the points to the closest cluster centroid

# Step 4 - Recompute centroids of newly formed clusters

# Step 5 - Repeat step 3 and 4



diff = 1

j=0



while(diff!=0):

    XD=X

    i=1

    for index1, row_c in Centroids.iterrows():

        ED=[]

        for index2, row_d in XD.iterrows():

            d1 = (row_c["ApplicantIncome"]-row_d["ApplicantIncome"])**2

            d2 = (row_c["LoanAmount"]-row_d["LoanAmount"])**2

            d = sqrt(d1+d2)

            ED.append(d)

        X[i] = ED

        i = i+1

    

    C = []

    for index, row in X.iterrows():

        min_dist=row[1]

        pos=1

        for i in range(K):

            if row[i+1] < min_dist:

                min_dist = row[i+1]

                pos = i+1

        C.append(pos)

    X["Cluster"]=C

    Centroids_new = X.groupby(["Cluster"]).mean()[["LoanAmount", "ApplicantIncome"]]

    if j == 0:

        diff = 1

        j = j+1

    else:

        diff = (Centroids_new['LoanAmount'] - Centroids['LoanAmount']).sum() + (Centroids_new['ApplicantIncome'] - Centroids['ApplicantIncome']).sum()

        print(diff.sum())

    Centroids = X.groupby(["Cluster"]).mean()[["LoanAmount","ApplicantIncome"]]
color=['blue','green','cyan']

for k in range(K):

    data=X[X["Cluster"]==k+1]

    plt.scatter(data["ApplicantIncome"],data["LoanAmount"],c=color[k])

plt.scatter(Centroids["ApplicantIncome"],Centroids["LoanAmount"],c='red')

plt.xlabel('Income')

plt.ylabel('Loan Amount (In Thousands)')

plt.show()