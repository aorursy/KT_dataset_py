# this study requires ipyvolume
!pip install ipyvolume
import math
import pandas as pd
import numpy as np
import ipyvolume as ipv
import tensorflow as tf
import datetime
# 2 groups of values in 3D
number_of_rows = 20**2  # must be a square of an integer

# parabolic design of ds_group1 (positive values)
radius = np.linspace(0, 6, int(math.sqrt(number_of_rows)))
theta = np.linspace(0, 2*math.pi, int(math.sqrt(number_of_rows)))
radius, theta = np.meshgrid(radius, theta)
# Parametrise it
X = radius*np.cos(theta)
Y = radius*np.sin(theta)
Z = radius**2
X = X.reshape(number_of_rows, 1)
Y = Y.reshape(number_of_rows, 1)
Z = Z.reshape(number_of_rows, 1)

dfX = pd.DataFrame(X, columns=['x'])
dfY = pd.DataFrame(Y, columns=['y'])
dfZ = pd.DataFrame(Z, columns=['z'])

ds_group1 = pd.concat([dfX, dfY, dfZ], axis=1)

# potatoe design of ds_group2 (negative values)
ds_group2 = np.random.randn(number_of_rows, 3) + np.array([0, 0, 20])

# stack the features
array_data_set = np.vstack([ds_group1, ds_group2])
# transform to DataFrame
features = pd.DataFrame(array_data_set, columns=['x', 'y', 'z'])

# create categories for ds_group1
category1 = pd.DataFrame(np.full((number_of_rows, 1), 'danger'),
                         columns=['target'])

# create categories for ds_group2
category2 = pd.DataFrame(np.full((number_of_rows, 1), 'safe'),
                         columns=['target'])

# concat the categories into targets DataFrame
targets = pd.concat([category1, category2], ignore_index=True)
colored = list()
for i in range(number_of_rows):
    colored.append((30, 100, 0))
for i in range(number_of_rows):
    colored.append((0, 30, 0))
ipv.figure()
fea = ipv.scatter(features['x'], features['y'], features['z'], color=colored,
                  size=1, marker='box')
# s = ipv.scatter(X, Y, Z,   size=3,  marker='sphere')

ipv.show()
df_full = pd.concat([features, targets], axis=1)
df_full['target'] = pd.Categorical(df_full['target'])
cat_dict = dict(enumerate(df_full['target'].cat.categories))
df_full['target'] = df_full.target.cat.codes
print('cat_dict = ', cat_dict)
target = df_full.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df_full.values, target.values))
# Define training set
train_dataset = dataset.shuffle(len(df_full)).batch(2)
# define the model
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, input_dim=3, activation='sigmoid'),
        tf.keras.layers.Dense(4, activation='sigmoid'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
# train the model
model = get_compiled_model()
model.fit(train_dataset, epochs=100, verbose=1)
number_of_values = 10
array_to_classify = np.random.randn(number_of_values, 3) + np.array([1, 1, 12])
df_to_classify = pd.DataFrame(array_to_classify, columns=['x', 'y', 'z'])
prediction = model.predict(df_to_classify, verbose=1)
# go into an activation function
predi=[]
for predict_sigmoid in prediction:
    predi = predi + [np.round(tf.sigmoid(predict_sigmoid).numpy())]
df_predi = pd.DataFrame(predi, columns=['prediction'])
df_results=pd.concat([df_to_classify,df_predi], axis=1)
df_results
# volume size
side = 30
a = np.linspace(-10, 10, side)
b = np.linspace(-10, 10, side)
c = np.linspace(-1, 40, side)
cubic_cloud_points_arr = np.empty([0], dtype=np.float32)
for x in a:
    for y in b:
        for z in c:
            cubic_cloud_points_arr = np.append(cubic_cloud_points_arr, [[x, y, z]])

cubic_cloud_points_arr = cubic_cloud_points_arr.reshape(side**3, 3)
df_cubic_cloud_points = pd.DataFrame(cubic_cloud_points_arr, columns=['x', 'y', 'z'], dtype=np.float32)

# pass it into the model
predicted_cubic_cloud_arr = model.predict(df_cubic_cloud_points)

# activate with sigmoid
sigmoid_predicted_cubic_cloud_arr = tf.sigmoid(predicted_cubic_cloud_arr).numpy()
# keep only values at the decision border
df_predicted = pd.DataFrame(sigmoid_predicted_cubic_cloud_arr, 
                       columns=['weight'], dtype=np.float32)

df_cubic_before_refine = pd.concat([df_cubic_cloud_points, df_predicted], axis=1, ignore_index=False)
# let's slenderize the data to keep only the ones at the decision obrder

def refine(df, size):
    df = df.sort_values(by=['weight'])
    df['weight'] = df['weight']-0.5
    df = df.sort_values(by=['weight'])
    df['weight'] = np.abs(df['weight'])
    df = df.sort_values(by=['weight'])
    df = df.reset_index()

    df = df[0:size]
    df = df.drop(['index'], axis=1)
    return df

df_decision_surf = df_cubic_before_refine
df_decision_surf = refine(df_decision_surf, 80)
draw_hull = True
if draw_hull:
    from scipy.spatial import ConvexHull
    hull_arr = np.array(df_decision_surf)
    points_3d = hull_arr
    hull_3d = ConvexHull(points_3d)
ipv.figure()
test_serie = ipv.scatter(df_full['x'], df_full['y'], df_full['z'], color=colored,
                         size=3, marker='sphere')

# uncomment to plot only the points of the decision surface 
# decision_surf_p = ipv.scatter(df_decision_surf['x'],
#                              df_decision_surf['y'],
#                              df_decision_surf['z'],
#                              color='blue', size=0.5, marker='box')

if draw_hull:
    decision_surf_v = ipv.plot_trisurf(points_3d[:, 0], points_3d[:, 1],
                                       points_3d[:, 2],
                                       triangles=hull_3d.simplices,
                                       color='blue')
    decision_surf_v.color = [0.95, 0.7, 0.8, 0.05]
    decision_surf_v.material.transparent = True
ipv.show()
