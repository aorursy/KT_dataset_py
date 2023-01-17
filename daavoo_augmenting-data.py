import sys

sys.path.append("../input")
import h5py

import numpy as np

import pandas as pd

from voxelgrid import VoxelGrid

from matplotlib import pyplot as plt
%matplotlib inline

plt.rcParams['image.interpolation'] = None

plt.rcParams['image.cmap'] = 'gray'
with h5py.File("../input/train_small.h5", "r") as hf:    



    a = hf["1"]

    

    digit = (a["img"][:], a["points"][:], a.attrs["label"]) 





plt.title("DIGIT: " + str(digit[2]))

plt.imshow(digit[0])
original = VoxelGrid(digit[1], x_y_z=[16,16,16])
def Rx(angle, degrees=True):

    """ 

    """

    if degrees:

        

        cx = np.cos(np.deg2rad(angle))

        sx = np.sin(np.deg2rad(angle))

        

    else:

        

        cx = np.cos(angle)

        sx = np.sin(angle)

        

    Rx = np.array(

    [[1  , 0  , 0  ],

     [0  , cx , sx ],

     [0  , -sx, cx]]

    )

    

    return Rx
def Ry(angle, degrees=True):

    

    if degrees:

        

        cy = np.cos(np.deg2rad(angle))

        sy = np.sin(np.deg2rad(angle))

        

    else:

        

        cy = np.cos(angle)

        sy = np.sin(angle)

        

    Ry = np.array(

    [[cy , 0  , -sy],

     [0  , 1  , 0  ],

     [sy , 0  , cy]]

    )

    

    return Ry
def Rz(angle, degrees=True):

        

    if degrees:

        

        cz = np.cos(np.deg2rad(angle))

        sz = np.sin(np.deg2rad(angle))

        

    else:

        

        cz = np.cos(angle)

        sz = np.sin(angle)

        

    Rz = np.array(

    [[cz , sz , 0],

     [-sz, cz , 0],

     [0  , 0  , 1]]

    )

        

    return Rz
original.plot()
rotated_z = VoxelGrid(digit[1] @ Rz(60), x_y_z=[16,16,16])
rotated_z.plot()
rotated_y = VoxelGrid(digit[1] @ Ry(90), x_y_z=[16,16,16])
rotated_y.plot()
def add_noise(xyz, strength=0.25):

    std = xyz.std(0) * strength

    noise = np.zeros_like(xyz)

    for i in range(3):

        noise[:,i] += np.random.uniform(-std[i], std[i], xyz.shape[0])

    return xyz + noise  

    

    
with_nosie = VoxelGrid(add_noise(digit[1]), x_y_z=[16,16,16])
with_nosie.plot()
save = False
if save: # grab a coffe and wait

    with h5py.File("../input/train_small.h5", "r") as hf:

        size = len(hf.keys())



        out = []        

        for i in range(size):

            if i % 200 == 0:

                print(i, "\t processed")

                

            original_cloud = hf[str(i)]["points"][:]

            label = hf[str(i)].attrs["label"]

            

            voxelgrid = VoxelGrid(original_cloud, x_y_z=[16, 16, 16])



            vector = voxelgrid.vector.reshape(-1) / np.max(voxelgrid.vector)

            

            out.append(vector.tolist().append(label))  



            s_x = np.random.normal(0, 90)

            s_y = np.random.normal(0, 90)

            s_z = np.random.normal(0, 180)



            cloud = original_cloud @ Rz(s_z) @ Ry(s_y) @ Rx(s_x)



            cloud = add_noise(cloud)



            voxelgrid = VoxelGrid(cloud, x_y_z=[16, 16, 16])



            vector = voxelgrid.vector.reshape(-1) / np.max(voxelgrid.vector)



            out.append(vector.tolist().append(label))

            

        print("[DONE]")
train = pd.DataFrame(out)
train.to_csv("train_16_16_16.csv", index=False)
if save: # grab a coffe and wait

    with h5py.File("../input/test_small.h5", "r") as hf:

        size = len(hf.keys())



        out = []        

        for i in range(size):

            if i % 200 == 0:

                print(i, "\t processed")

                

            original_cloud = hf[str(i)]["points"][:]

            label = hf[str(i)].attrs["label"]

            

            voxelgrid = VoxelGrid(original_cloud, x_y_z=[16, 16, 16])



            vector = voxelgrid.vector.reshape(-1) / np.max(voxelgrid.vector)

            

            out.append(vector.tolist().append(label))  



            s_x = np.random.normal(0, 90)

            s_y = np.random.normal(0, 90)

            s_z = np.random.normal(0, 180)



            cloud = original_cloud @ Rz(s_z) @ Ry(s_y) @ Rx(s_x)



            cloud = add_noise(cloud)



            voxelgrid = VoxelGrid(cloud, x_y_z=[16, 16, 16])



            vector = voxelgrid.vector.reshape(-1) / np.max(voxelgrid.vector)



            out.append(vector.tolist().append(label))

            

        print("[DONE]")
test = pd.DataFrame(out)
test.to_csv("test_16_16_16.csv", index=False)