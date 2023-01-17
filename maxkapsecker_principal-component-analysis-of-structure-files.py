# General libraries

import pandas as pd

import numpy as np



# Plotting and Visualization Library

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt



# Math Libraries

from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull
# Load the main train and test data

train = pd.read_csv("../input/train.csv")

molecule_name = train.molecule_name.unique()
def get_xyz_data(filepath):

    """A function to load the 3D points contained in the xyz-Files to a pandas DataFrame.



    Args:

        filename (string): Path to the xyz-file



    Returns:

        pandas DataFrame: A table in which the first three columns correspond to the point's

                          xyz-coordinates and the fourth column the atom abbreviation



    Note:

        This function is adapted from Tony Y: https://www.kaggle.com/tonyyy



    """

    A = pd.DataFrame(columns=list('ABCD'))

    with open(filepath) as f:

        k = 0

        for line in f.readlines():

            x = line.split()

            if k > 1:

                newrowA = pd.DataFrame([[x[1],x[2],x[3],x[0]]], columns=list('ABCD'))

                A = A.append(newrowA)

            k = k + 1

    return A
def plot_pca(index, showConvexHull=False):

    """A function that projects the 3D points onto its principal components using PCA algorithm.

       Further, the points (in 3D and 2D) are drawn while the target scalar_coupling_constant is

       represented as the thickness of the linking line between the points.

       

    Args:

        index (int): An index that picks from the molecule_name array a molecule name

        showConvexHull(bool): An optional value. If set to true the 2D plots will include the convex hull

                              around the point configuration.

    """

    

    fn = "../input/structures/{}.xyz".format(molecule_name[index])

    train_xyz = get_xyz_data(fn)

    temp = train[train.molecule_name == molecule_name[index]]

    color_dict = { 'C':'black', 'H':'blue', 'O':'red', 'N':'green' }

    

    minimalValue = np.min(temp.scalar_coupling_constant)

    maximalValue = np.max(temp.scalar_coupling_constant)

    # TODO: One can probably find a much better scaling for the thickness of the lines

    thickness = 3*temp.scalar_coupling_constant.values / maximalValue

    

    matrix = train_xyz

    colour = matrix["D"]

    matrix = matrix[["A","B","C"]].values

    matrix = matrix.astype(float)

    

    pca = PCA(n_components=3)

    X_r = pca.fit(matrix).transform(matrix)

    

    df_ = pd.DataFrame(np.round(X_r,2))

        

    x = np.array(matrix[:,0])

    y = np.array(matrix[:,1])

    z = np.array(matrix[:,2])



    fig = plt.figure(figsize=(20,20))

    

    ax = fig.add_subplot(221, projection='3d')

    ax.scatter(x, y, z, c=[color_dict[i] for i in colour], marker='o', s=70)

    ax.set_xlabel('X Coordinate')

    ax.set_ylabel('Y Coordinate')

    ax.set_zlabel('Z Coordinate')

    ax.set_title(molecule_name[index])

    

    ax = fig.add_subplot(222)

    plt.scatter(X_r[:, 0], X_r[:, 1], color=[color_dict[i] for i in colour], alpha=.8, lw=1, s=70)

    if(showConvexHull):

        hull = ConvexHull(X_r[:,[0,1]])

        volume_1 = hull.volume

        plt.plot(X_r[hull.vertices,0], X_r[hull.vertices,1], 'r--', lw=1)

    for k in range(0,len(temp)):

        a = temp.iloc[k,2]

        b = temp.iloc[k,3]

        plt.plot(X_r[[a, b],0], X_r[[a, b],1], 'b', lw=thickness[k])

    plt.title(molecule_name[index])

    ax.set_xlabel('First Principal Component')

    ax.set_ylabel('Second Principal Component')

    ax.set_title(molecule_name[index])

    

    ax = fig.add_subplot(223)

    plt.scatter(X_r[:, 0], X_r[:, 2], color=[color_dict[i] for i in colour], alpha=.8, lw=1, s=70)

    if(showConvexHull):

        hull = ConvexHull(X_r[:,[0,2]])

        volume_2 = hull.volume

        plt.plot(X_r[hull.vertices,0], X_r[hull.vertices,2], 'r--', lw=1)

    for k in range(0,len(temp)):

        a = temp.iloc[k,2] #'atom_index_0']

        b = temp.iloc[k,3] #'atom_index_1']

        plt.plot(X_r[[a, b],0], X_r[[a, b],2], 'b', lw=thickness[k])

    plt.title(molecule_name[index])

    ax.set_xlabel('First Principal Component')

    ax.set_ylabel('Third Principal Component')

    ax.set_title(molecule_name[index])

    

    ax = fig.add_subplot(224)

    plt.scatter(X_r[:, 1], X_r[:, 2], color=[color_dict[i] for i in colour], alpha=.8, lw=1, s=70)

    if(showConvexHull):

        hull = ConvexHull(X_r[:,[1,2]])

        volume_3 = hull.volume

        plt.plot(X_r[hull.vertices,1], X_r[hull.vertices,2], 'r--', lw=1)

    for k in range(0,len(temp)):

        a = temp.iloc[k,2]

        b = temp.iloc[k,3]

        plt.plot(X_r[[a, b],1], X_r[[a, b],2], 'b', lw=thickness[k])

    plt.title(molecule_name[index])  

    ax.set_xlabel('Second Principal Component')

    ax.set_ylabel('Third Principal Component')

    ax.set_title(molecule_name[index])

    

    plt.show()

    

    print("On the first PC are approx. " + str(len(df_[0].unique())) + " distinct coordinates with atoms")

    print("On the second PC are approx. " + str(len(df_[1].unique())) + " distinct coordinates with atoms")

    print("On the third PC are approx. " + str(len(df_[2].unique())) + " distinct coordinates with atoms")

    

    if(showConvexHull):

        print("")

        print("Area covered by the first and second principal component: " + str(volume_1))

        print("Area covered by the first and third principal component: " + str(volume_2))

        print("Area covered by the second and third principal component: " + str(volume_3))
index = [345,654,1000,1337,2789,10000]
plot_pca(index=index[0])
plot_pca(index=index[1])
plot_pca(index=index[2])
plot_pca(index=index[3])
plot_pca(index=index[4])
plot_pca(index=index[5])