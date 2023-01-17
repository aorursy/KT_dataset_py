%matplotlib inline

%config InlineBackend.figure_format = 'svg'



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn import tree

from sklearn.manifold import TSNE

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



from scipy.stats import gaussian_kde



plt.style.use('ggplot')





data = pd.read_csv('../input/StudentsPerformance.csv')
data.isnull().sum()
data.duplicated().sum()
data.head()
data.shape
info_dict = {}

for i,item in enumerate(data.columns):

    if i < 5:

        info_list = list(set(list(data[item])))

        info_list.sort()

        info_dict[item]=info_list

info_dict
data_summary = data.describe()

data_summary.loc['skewness'] = data.skew()

data_summary.loc['kurtosis'] = data.kurtosis()

data_summary
xs = np.linspace(0,100,100)



fig = plt.figure(figsize=[8, 13.2])

fig.suptitle('Score Histograms and Kernel Density Estimations',fontsize=14,fontweight='bold')



ax1 = fig.add_subplot(311)

fig.subplots_adjust(top=0.945)

ax1.set_title('Math Score',fontsize=12)

ax1.set_ylabel('Number of Students')



density = gaussian_kde(data['math score'])

density.covariance_factor = lambda : .25

density._compute_covariance()

ax1.plot(xs,density(xs)*1000,color = '#4B4B4B')

ax1.hist(data['math score'],xs,color = '#FF3366')



ax2 = fig.add_subplot(312)

ax2.set_title('Reading Score',fontsize=12)

ax2.set_ylabel('Number of Students')



density = gaussian_kde(data['reading score'])

density.covariance_factor = lambda : .25

density._compute_covariance()

ax2.plot(xs,density(xs)*1000,color = '#4B4B4B')

ax2.hist(data['reading score'],xs,color = '#6666FF')



ax3 = fig.add_subplot(313)

ax3.set_title('Writing Score',fontsize=12)

ax3.set_ylabel('Number of Students')



density = gaussian_kde(data['writing score'])

density.covariance_factor = lambda : .25

density._compute_covariance()

ax3.plot(xs,density(xs)*1000,color = '#4B4B4B')

ax3.hist(data['writing score'],xs,color = '#FFFF33')



plt.show()

df_math = pd.DataFrame()

for item in info_dict.keys():

    for features in info_dict[item]:

        df_math[features]=data.loc[data[item]==features].describe()['math score']

df_math.loc['skewness'] = df_math.skew()

df_math.loc['kurtosis'] = df_math.kurtosis()

df_math['Total'] = data_summary['math score']

df_math
fig = plt.figure(figsize=[8, 20])

fig.suptitle('Kernel Density Estimations of Math Score for each Feature',fontsize=14,fontweight='bold')

fig.subplots_adjust(top=0.95)



color_list = ['#0000FF','#FF0000','#00FFFF','#FF00FF','#FFFF00','#00FF00']



for i,item in enumerate(info_dict.keys()):

    ax = fig.add_subplot(6,1,i+1)

    ax.set_title(item,fontsize=12)

    ax.set_ylabel('Probability Density')

    for ii,features in enumerate(info_dict[item]): 

        density = gaussian_kde(data.loc[data[item]==features]['math score'])

        density.covariance_factor = lambda : .3

        density._compute_covariance()

        ax.plot(xs,density(xs),color = color_list[ii])

    ax.legend(labels = info_dict[item], loc = 'best')

df_reading = pd.DataFrame()

for item in info_dict.keys():

    for features in info_dict[item]:

        df_reading[features]=data.loc[data[item]==features].describe()['reading score']

df_reading.loc['skewness'] = df_reading.skew()

df_reading.loc['kurtosis'] = df_reading.kurtosis()

df_reading['Total'] = data_summary['reading score']

df_reading
fig = plt.figure(figsize=[8, 20])

fig.suptitle('Kernel Density Estimations of Reading Score for each Feature',fontsize=14,fontweight='bold')

fig.subplots_adjust(top=0.95)



color_list = ['#0000FF','#00FF00','#FF0000','#00FFFF','#FF00FF','#FFFF00']



for i,item in enumerate(info_dict.keys()):

    ax = fig.add_subplot(6,1,i+1)

    ax.set_title(item,fontsize=12)

    ax.set_ylabel('Probability Density')

    for ii,features in enumerate(info_dict[item]): 

        density = gaussian_kde(data.loc[data[item]==features]['reading score'])

        density.covariance_factor = lambda : .3

        density._compute_covariance()

        ax.plot(xs,density(xs),color = color_list[ii])

    ax.legend(labels = info_dict[item], loc = 'best')
df_writing = pd.DataFrame()

for item in info_dict.keys():

    for features in info_dict[item]:

        df_writing[features]=data.loc[data[item]==features].describe()['writing score']

df_writing.loc['skewness'] = df_writing.skew()

df_writing.loc['kurtosis'] = df_writing.kurtosis()

df_writing['Total'] = data_summary['writing score']

df_writing
fig = plt.figure(figsize=[8, 20])

fig.suptitle('Kernel Density Estimations of Writing Score for each Feature',fontsize=14,fontweight='bold')

fig.subplots_adjust(top=0.95)



color_list = ['#0000FF','#00FF00','#FF0000','#00FFFF','#FF00FF','#FFFF00']



for i,item in enumerate(info_dict.keys()):

    ax = fig.add_subplot(6,1,i+1)

    ax.set_title(item,fontsize=12)

    ax.set_ylabel('Probability Density')

    for ii,features in enumerate(info_dict[item]): 

        density = gaussian_kde(data.loc[data[item]==features]['writing score'])

        density.covariance_factor = lambda : .3

        density._compute_covariance()

        ax.plot(xs,density(xs),color = color_list[ii])

    ax.legend(labels = info_dict[item], loc = 'best')
X = data[['math score','reading score','writing score']].values.tolist()

Y = data[['gender']].values.tolist()



X_embedded = TSNE(n_components=2).fit_transform(X)

x_min, x_max = np.min(X, 0), np.max(X, 0)

X_normalized = (X - x_min) / (x_max - x_min) 



plt.figure(figsize=[8,7])



blue_dot_x = []

blue_dot_y = []

red_dot_x = []

red_dot_y = []



for i in range(X_normalized.shape[0]):

    if Y[i] == ['male']:

        blue_dot_x.append(X_normalized[i,0])

        blue_dot_y.append(X_normalized[i,1])

    else:

        red_dot_x.append(X_normalized[i,0])

        red_dot_y.append(X_normalized[i,1])

plt.scatter(blue_dot_x, blue_dot_y, color = '#0000FF', label = 'male')

plt.scatter(red_dot_x, red_dot_y, color = '#FF0000',  label = 'female')

plt.xlabel("Dimension 1")

plt.ylabel("Dimension 2")

plt.title('t_SNE plot with perplexity 30')

plt.legend()

plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 

                                        test_size=0.1, random_state=42)



clf = tree.DecisionTreeClassifier(max_depth = 6)

clf = clf.fit(X_train,Y_train)



print('The accuracy on training set is %6.4f' % clf.score(X_train,Y_train))

print('The accuracy on testing set is %6.4f' %clf.score(X_test, Y_test))