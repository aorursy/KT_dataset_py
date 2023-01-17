# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import csv

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline 





name='ehresp_2014.csv'

K=7



df=pd.read_csv("../input/ehresp_2014.csv")

df=df.drop('eeincome1', axis=1)

df=df.drop('euincome2', axis=1)

df=df.drop('exincome1', axis=1)



df=df[df["erincome"] > 0]



id_data=np.array(df['tucaseid'])

income_data=np.array(df['erincome'])



tmp_df=df.drop('tucaseid',axis=1)

tmp_df=tmp_df.drop('erincome',axis=1)

variable_data=np.array(tmp_df)



tmp_mean=np.mean(variable_data,axis=0)

tmp_sd=np.std(variable_data,axis=0)



variable_data=(variable_data-tmp_mean)/(tmp_sd+0.00001)



val=np.shape(variable_data)[1]

num=np.shape(income_data)[0]

cluster_id=np.zeros(num)

slope=np.zeros([val,K])

#intercept=np.zeros([val,K])

for i in range(num):

	cluster_id[i]=(i+K-3)%K
def update(slope):

	for k in range(K):

		target_list=list()

		for i in range(1,6):

			target_list.append(list(np.where((cluster_id==k) & (income_data==i))[0]))

		for i in range(val):



			num=0

			tmp=0

			for j in range(5):

				for l in range(len(target_list[j])):

					if cluster_id[target_list[j][l]]==k:

						num+=(j+1)

						tmp+=variable_data[target_list[j][l],i]

			if num>0:

				slope[i,k]=tmp/float(num)

	return [slope]





def allocation(cluster_id,slope,K):

	for l in range(cluster_id.shape[0]):

		income=income_data[l]

		tmp_attribute=np.zeros([val,K])

		for k in range(K):

			tmp_attribute[:,k]=income*slope[:,k]

		dist=np.zeros(K)

		tmp=variable_data[l,:]

		for k in range(K):

			dist[k]=np.sum(np.power(tmp.transpose()-tmp_attribute[:,k],2))

		cluster_id[l]=np.argmin(dist)

	return cluster_id



def total_loss(cluster_id,income_data,variable_data):

	loss_sum=0

	for l in range(int(cluster_id.shape[0])):

		income=income_data[l]

		tmp=variable_data[l,:]

		cluster=int(cluster_id[l])

		tmp_attribute=income*slope[:,cluster]

		loss_sum+=np.sum(np.power(tmp.transpose()-tmp_attribute,2))

	return loss_sum
Epoch=50

loss=np.zeros([Epoch])



for epoch in range(Epoch):

	[slope]=update(slope)

	cluster_id=allocation(cluster_id,slope,K)

	tmp_loss=total_loss(cluster_id,income_data,variable_data)

	loss[epoch]=tmp_loss

plt.plot(loss)

plt.xlabel("Epoch")

plt.ylabel("Train error")

clustering_summary=np.zeros([K,5])

for i in range(num):

    clustering_summary[int(cluster_id[i]),int(income_data[i]-1)]+=1

clustering_summary=pd.DataFrame(clustering_summary,columns=list('12345'))

clustering_summary

i=0

tmp=slope#.reshape(val)

tmp_slope_df=pd.DataFrame(tmp,index=tmp_df.columns)#,

tmp_slope_df.plot(kind='bar',figsize=(20, 6))

tmp_df.columns

limited_tmp=tmp_slope_df

limited_tmp=limited_tmp.drop("tulineno",axis=0)

limited_tmp=limited_tmp.drop("erspemch",axis=0)

limited_tmp=limited_tmp.drop("eugenhth",axis=0)

limited_tmp=limited_tmp.drop("eusnap",axis=0)

limited_tmp=limited_tmp.drop("eufinlwgt",axis=0)

#limited_tmp=limited_tmp.drop("",axis=0)

limited_tmp=limited_tmp.drop("erhhch",axis=0)

limited_tmp=limited_tmp.drop("ertpreat",axis=0)

limited_tmp=limited_tmp.drop("euexfreq",axis=0)

limited_tmp=limited_tmp.drop("euinclvl",axis=0)

limited_tmp=limited_tmp.drop("euwic",axis=0)

limited_tmp.plot(kind='bar',figsize=(20, 6))
