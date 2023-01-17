# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/FullData.csv')
df.head()
# Start to filtering numeric values

str_list = []

for colname, colvalue in df.items():

    if type(colvalue[1]) == str:

        str_list.append(colname)

num_list = df.columns.difference(str_list)
# Defining a data frame with numeric values

df_num = df[num_list]
df_num = df_num.fillna(value=0,axis=1)
df_num.head()
X = df_num.values

X_std = StandardScaler().fit_transform(X)
f, ax = plt.subplots(figsize=(12,10))

plt.title('Player Correlation of Specialties Features #1')

sns.heatmap(df_num[df_num.columns[:10]].astype(float).corr(),linewidths=0.25,vmax=1.0,square=True,cmap='YlGnBu',linecolor='black',annot=True)
f, ax = plt.subplots(figsize=(12,10))

plt.title('Player Correlation of Specialties Features #2')

sns.heatmap(df_num[df_num.columns[10:20]].astype(float).corr(),linewidths=0.25,vmax=1.0,square=True,cmap='YlGnBu',linecolor='black',annot=True)
f, ax = plt.subplots(figsize=(12,10))

plt.title('Player Correlation of Specialties Features #3')

sns.heatmap(df_num[df_num.columns[30:40+1]].astype(float).corr(),linewidths=0.25,vmax=1.0,square=True,cmap='YlGnBu',linecolor='black',annot=True)
mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key=lambda x: x[0], reverse=True)

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize=(10,5))

plt.bar(range(len(var_exp)),var_exp,alpha=0.3333,align='center',label='individual explained variance',color='g')

plt.step(range(len(cum_var_exp)),cum_var_exp,where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.show()
pca = PCA(n_components=9)

x_9d = pca.fit_transform(X_std)
plt.figure(figsize=(9,7))

plt.scatter(x_9d[:,0],x_9d[:,1],c='goldenrod',alpha=0.5)

plt.ylim(-10,30)

plt.show()
kmeans = KMeans(n_clusters=3)

X_clustered = kmeans.fit_predict(x_9d)

LABEL_COLOR_MAP = {0:'r',1:'g',2:'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

plt.figure(figsize=(7,7))

plt.scatter(x_9d[:,0],x_9d[:,2],c=label_color,alpha=0.5)

plt.show()
df_num.plot(y='Ball_Control',x='Age',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Age',figsize=(12,8))
# Validating an hypothesis with a range for a spotted range

df_ball_control = df[df['Ball_Control'] >= 60][df['Ball_Control'] <= 65]
df_ball_control_and_age = df[df['Age'] >= 20][df['Age'] <= 25]
df_ball_control_and_age[['Name','Ball_Control','Age','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Short_Pass',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Short_Pass',figsize=(12,8))
df_ball_control_and_short_pass = df[df['Short_Pass'] >= 60][df['Short_Pass'] <= 65]
df_ball_control_and_short_pass[['Name','Ball_Control','Short_Pass','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Long_Pass',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Long_Pass',figsize=(12,8))
df_ball_control_and_long_pass = df[df['Long_Pass'] >= 60][df['Long_Pass'] <= 70]
df_ball_control_and_long_pass[['Name','Ball_Control','Long_Pass','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Crossing',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Crossing',figsize=(12,8))
df_ball_control_and_crossing = df_ball_control[df['Crossing'] > 60][df['Crossing'] <= 65]
df_ball_control_and_crossing[['Name','Ball_Control','Crossing','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Interceptions',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Interceptions',figsize=(12,8))
df_ball_control_and_interceptions = df_ball_control[df['Interceptions'] > 70][df['Interceptions'] <= 75]
df_ball_control_and_interceptions[['Name','Ball_Control','Interceptions','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Jumping',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Jumping',figsize=(12,8))
df_ball_control_and_jumping = df_ball_control[df['Jumping'] >= 70][df['Jumping'] <= 75]
df_ball_control_and_jumping[['Name','Ball_Control','Jumping','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Heading',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Heading',figsize=(12,8))
df_ball_control_and_heading = df_ball_control[df['Heading'] >= 40][df['Heading'] <= 60]
df_ball_control_and_heading[['Name','Ball_Control','Heading','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Volleys',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Volleys',figsize=(12,8))
df_ball_control_and_volleys = df_ball_control[df['Volleys'] >= 45][df['Volleys'] <= 65]
df_ball_control_and_volleys[['Name','Ball_Control','Volleys','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Marking',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Marking',figsize=(12,8))
df_ball_control_and_marking = df_ball_control[df['Marking'] >= 60][df['Marking'] <= 65]
df_ball_control_and_marking[['Name','Ball_Control','Marking','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Strength',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Strength',figsize=(12,8))
df_ball_control_and_strength = df_ball_control[df['Strength'] >= 55][df['Strength'] <= 65]
df_ball_control_and_strength[['Name','Ball_Control','Strength','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Stamina',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Stamina',figsize=(12,8))
df_ball_control_and_stamina = df_ball_control[df['Stamina'] >= 70][df['Stamina'] <= 75]
df_ball_control_and_stamina[['Name','Ball_Control','Stamina','Nationality']].sort_values(by='Ball_Control',ascending=False).head()
df_num.plot(y='Ball_Control',x='Speed',kind='hexbin',gridsize=35,sharex=False,

            colormap='cubehelix',title='Hexbin of Ball_Control and Speed',figsize=(12,8))
df_ball_control_and_speed = df_ball_control[df['Speed'] >= 60][df['Speed'] <= 70]
df_ball_control_and_speed[['Name','Ball_Control','Speed','Nationality']].sort_values(by='Ball_Control',ascending=False).head()