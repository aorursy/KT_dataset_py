from sklearn.neural_network import MLPClassifier

import pandas as pd

import numpy as np



data = pd.read_table('/kaggle/input/data-to-work-on-basic-examples/iris_data_tb.txt')

del data['Sample']

y = data.loc[:,'Type']

x = data.loc[:,'PW':'SL']

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(300,),early_stopping=True, verbose=1,activation='tanh')

model.fit(x_train,y_train)

print("Test score is: ",model.score(x_test,y_test))

print('class: ', model.classes_)
data = data.sort_values('Type')

data = data.reset_index(drop=True)

class1 = data.iloc[:50,:]

class2 = data.iloc[50:100,:]

class3 = data.iloc[100:150,:]

class1.head()
from sklearn.preprocessing import Normalizer



sc = Normalizer()

scaled_class1 = sc.fit_transform(class1)

scaled_class2 = sc.fit_transform(class2)

scaled_class3 = sc.fit_transform(class3)

#%%

scaled_class1 = pd.DataFrame(scaled_class1)

scaled_class2 = pd.DataFrame(scaled_class2)

scaled_class3 = pd.DataFrame(scaled_class3)

scaled_class1.head()
cov_class1 = scaled_class1.drop(scaled_class1.columns[[0]],axis=1).cov()

cov_class2 = scaled_class2.drop(scaled_class2.columns[[0]],axis=1).cov()

cov_class3 = scaled_class3.drop(scaled_class3.columns[[0]],axis=1).cov()

cov_class1.head()
mean_class1 = scaled_class1.mean()

mean_class2 = scaled_class2.mean()

mean_class3 = scaled_class3.mean()

mean_class1.head()
mean_3and4_c1 = scaled_class1.iloc[:,1:3].mean()

cov_3and4_c1 = scaled_class1.iloc[:,1:3].cov()



mean_4and5_c1 = scaled_class1.iloc[:,2:4].mean()

cov_4and5_c1 = scaled_class1.iloc[:,2:4].cov()



mean_5and6_c1 = scaled_class1.iloc[:,3:5].mean()

cov_5and6_c1 = scaled_class1.iloc[:,3:5].cov()



mean_3and5_c1 = scaled_class1.iloc[:,[1,3]].mean()

cov_3and5_c1 = scaled_class1.iloc[:,[1,3]].cov()



mean_3and6_c1 = scaled_class1.iloc[:,[1,4]].mean()

cov_3and6_c1 = scaled_class1.iloc[:,[1,4]].cov()



mean_4and6_c1 = scaled_class1.iloc[:,[2,4]].mean()

cov_4and6_c1 = scaled_class1.iloc[:,[2,4]].cov()



#%%

mean_3and4_c2 = scaled_class2.iloc[:,1:3].mean()

cov_3and4_c2 = scaled_class2.iloc[:,1:3].cov()



mean_4and5_c2 = scaled_class2.iloc[:,2:4].mean()

cov_4and5_c2 = scaled_class2.iloc[:,2:4].cov()



mean_5and6_c2 = scaled_class2.iloc[:,3:5].mean()

cov_5and6_c2 = scaled_class2.iloc[:,3:5].cov()



mean_3and5_c2 = scaled_class2.iloc[:,[1,3]].mean()

cov_3and5_c2 = scaled_class2.iloc[:,[1,3]].cov()



mean_3and6_c2 = scaled_class2.iloc[:,[1,4]].mean()

cov_3and6_c2 = scaled_class2.iloc[:,[1,4]].cov()



mean_4and6_c2 = scaled_class2.iloc[:,[2,4]].mean()

cov_4and6_c2 = scaled_class2.iloc[:,[2,4]].cov()



#%%

mean_3and4_c3 = scaled_class3.iloc[:,1:3].mean()

cov_3and4_c3 = scaled_class3.iloc[:,1:3].cov()



mean_4and5_c3 = scaled_class3.iloc[:,2:4].mean()

cov_4and5_c3 = scaled_class3.iloc[:,2:4].cov()



mean_5and6_c3 = scaled_class3.iloc[:,3:5].mean()

cov_5and6_c3 = scaled_class3.iloc[:,3:5].cov()



mean_3and5_c3 = scaled_class3.iloc[:,[1,3]].mean()

cov_3and5_c3 = scaled_class3.iloc[:,[1,3]].cov()



mean_3and6_c3 = scaled_class3.iloc[:,[1,4]].mean()

cov_3and6_c3 = scaled_class3.iloc[:,[1,4]].cov()



mean_4and6_c3 = scaled_class3.iloc[:,[2,4]].mean()

cov_4and6_c3 = scaled_class3.iloc[:,[2,4]].cov()
mean_4and5_c1.head()
cov_4and5_c2.head()
from matplotlib import pyplot as plt



fig,axs = plt.subplots(3,2)

fig.suptitle('Graph of feature correlations of all classes')

axs[0,0].scatter(scaled_class1.iloc[:,1],scaled_class1.iloc[:,2], label = 'c1')

axs[0,0].scatter(scaled_class2.iloc[:,1],scaled_class2.iloc[:,2],marker='x',label = 'c2')

axs[0,0].scatter(scaled_class3.iloc[:,1],scaled_class3.iloc[:,2], marker='v', label = 'c3')

axs[0,0].legend()

axs[0,0].set_title('3rd & 4th Columns(1st & 2nd Features)')





axs[0,1].scatter(scaled_class1.iloc[:,2],scaled_class1.iloc[:,3], label = 'c1')

axs[0,1].scatter(scaled_class2.iloc[:,2],scaled_class2.iloc[:,3],marker='x',label = 'c2')

axs[0,1].scatter(scaled_class3.iloc[:,2],scaled_class3.iloc[:,3], marker='v', label = 'c3')

axs[0,1].legend()

axs[0,1].set_title('4th & 5th Columns')



axs[1,0].scatter(scaled_class1.iloc[:,3],scaled_class1.iloc[:,4], label = 'c1')

axs[1,0].scatter(scaled_class2.iloc[:,3],scaled_class2.iloc[:,4],marker='x',label = 'c2')

axs[1,0].scatter(scaled_class3.iloc[:,3],scaled_class3.iloc[:,4], marker='v', label = 'c3')

axs[1,0].legend()

axs[1,0].set_title('5th & 6th Columns')



axs[1,1].scatter(scaled_class1.iloc[:,1],scaled_class1.iloc[:,3], label = 'c1')

axs[1,1].scatter(scaled_class2.iloc[:,1],scaled_class2.iloc[:,3],marker='x',label = 'c2')

axs[1,1].scatter(scaled_class3.iloc[:,1],scaled_class3.iloc[:,3], marker='v', label = 'c3')

axs[1,1].legend()

axs[1,1].set_title('3rd & 5th Columns')



axs[2,0].scatter(scaled_class1.iloc[:,1],scaled_class1.iloc[:,4], label = 'c1')

axs[2,0].scatter(scaled_class2.iloc[:,1],scaled_class2.iloc[:,4],marker='x',label = 'c2')

axs[2,0].scatter(scaled_class3.iloc[:,1],scaled_class3.iloc[:,4], marker='v', label = 'c3')

axs[2,0].legend()

axs[2,0].set_title('3rd & 6th Columns')



axs[2,1].scatter(scaled_class1.iloc[:,2],scaled_class1.iloc[:,4], label = 'c1')

axs[2,1].scatter(scaled_class2.iloc[:,2],scaled_class2.iloc[:,4],marker='x',label = 'c2')

axs[2,1].scatter(scaled_class3.iloc[:,2],scaled_class3.iloc[:,4], marker='v', label = 'c3')

axs[2,1].legend()

axs[2,1].set_title('4th & 6th Columns')