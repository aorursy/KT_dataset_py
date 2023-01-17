import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('../input/training.csv')

df.head()
fig = plt.figure(figsize=(8, 8)).gca(projection='3d')

fig.scatter(df['s1_speed'], df['s2_speed'], df['s3_speed'])

fig.set_xlabel('S1')

fig.set_ylabel('S2')

fig.set_zlabel('S3')

fig.set_title('Speed')

plt.show()
fig = plt.figure(figsize=(8, 8)).gca(projection='3d')

fig.scatter(df['s1_dir'], df['s2_dir'], df['s3_dir'])

fig.set_xlabel('S1')

fig.set_ylabel('S2')

fig.set_zlabel('S3')

fig.set_title('Direction')

plt.show()
def pol2rec(p, theta):

    return (p * np.cos(np.pi * theta/180.), p * np.sin(np.pi * theta/180.))



def rec2pol(x, y):

    theta = 180. / np.pi * np.arctan(y/x)

    theta[np.where(x < 0)[0]] += 180.

    theta[np.where(theta < 0)[0]] += 360.

    return (np.sqrt(x * x + y * y), theta)
xp = df[['s1_speed', 's1_dir', 's2_speed', 's2_dir']].values.T

yp = df[['s3_speed', 's3_dir']].values.T
x_data = np.concatenate([pol2rec(xp[0], xp[1]), 

                          pol2rec(xp[2], xp[3])], axis=0).T

y_data = np.concatenate([pol2rec(yp[0], yp[1])], axis=0).T
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.25)

model = LinearRegression()

model.fit(x_train, y_train)
R = np.sqrt(model.score(x_val, y_val))

print(R)
y_hat = model.predict(x_data)

speed_hat, dir_hat = rec2pol(y_hat[:,0], y_hat[:,1])
fig = plt.figure(figsize=(8,8)).gca(projection='3d')

fig.scatter(df['s1_speed'], df['s2_speed'], df['s3_speed'], c='b')

fig.scatter(df['s1_speed'], df['s2_speed'], speed_hat, c='r')

fig.set_xlabel('S1')

fig.set_ylabel('S2')

fig.set_zlabel('S3')

fig.set_title('Speed')

plt.show()
fig = plt.figure(figsize=(8, 8)).gca(projection='3d')

fig.scatter(df['s1_dir'], df['s2_dir'], df['s3_dir'], c='b')

fig.scatter(df['s1_dir'], df['s2_dir'], dir_hat, c='r')

fig.set_xlabel('S1')

fig.set_ylabel('S2')

fig.set_zlabel('S3')

fig.set_title('Direction')

plt.show()
ix1 = np.where(np.abs(speed_hat) > 1)[0]

ix1.shape
fig = plt.figure(figsize=(8, 8)).gca(projection='3d')

fig.scatter(df['s1_dir'], df['s2_dir'], df['s3_dir'], c='b')

fig.scatter(df['s1_dir'].values[ix1], df['s2_dir'].values[ix1], dir_hat[ix1], c='r')

fig.set_xlabel('S1')

fig.set_ylabel('S2')

fig.set_zlabel('S3')

fig.set_title('Direction, for wind speeds > 1')

plt.show()
tdf = pd.read_csv('../input/test.csv')

txp = tdf[['s1_speed', 's1_dir', 's2_speed', 's2_dir']].values.T

x_test = np.concatenate([pol2rec(txp[0], txp[1]), 

                          pol2rec(txp[2], txp[3])], axis=0).T
y_pred = model.predict(x_test)

tdf['s3_speed'], tdf['s3_dir'] = rec2pol(y_pred[:,0], y_pred[:,1])

tdf.to_csv('submission.csv')