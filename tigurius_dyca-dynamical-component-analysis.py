#import some standard libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp # for generalized eigenproblem
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
def DyCA(data, eig_threshold = 0.98):
    "Given data of the form (time, sensors) returns the DyCA projection and projected data with eigenvalue threshold eig_threshold"
    derivative_data = np.gradient(data,axis=0,edge_order=1) #get the derivative of the data
    time_length = data.shape[0] #for time averaging
    #construct the correlation matrices
    C0 = np.matmul(data.transpose(), data) / time_length 
    C1 = np.matmul(derivative_data.transpose(), data) / time_length    
    C2 = np.matmul(derivative_data.transpose(), derivative_data) / time_length   
    #solve generalized eigenproblem
    eigvalues, eigvectors = sp.linalg.eig(np.matmul(np.matmul(C1,np.linalg.inv(C0)),np.transpose(C1)), C2)
    eigvectors = eigvectors[:,np.array(eigvalues > eig_threshold) &  np.array(eigvalues <= 1)] # eigenvalues > 1 are artifacts of singularity of C0
    if eigvectors.shape[1] > 0:
        C3 = np.matmul(np.linalg.inv(C1), C2)
        proj_mat = np.concatenate((eigvectors, np.apply_along_axis(lambda x: np.matmul(C3,x), 0, eigvectors)),axis=1)
    else:
        raise ValueError('No generalized eigenvalue fulfills threshold!')        
    return proj_mat, np.matmul(data,proj_mat)
eeg_data = pd.read_csv("../input/seizure.csv")[["F12","FT10","TP10","F8","T8","P8","O2","FP2","F$","C4","P4","Fz","Cz","Pz","F11","FT9","TP9","F7","T7","P7","O1","FP1","F3","C3","P3"]]
eeg_data.plot(legend=None);
pca = PCA(n_components=4)
pca.fit(eeg_data)
pca_eeg = pca.transform(eeg_data)
#plot projected time-series
plt.figure(1);
plt.plot(pca_eeg);
#plot in sections of the 4D phase space
fig = plt.figure(figsize=(15,14))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(pca_eeg[:,0], pca_eeg[:,1], pca_eeg[:,2])
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot(pca_eeg[:,0], pca_eeg[:,1], pca_eeg[:,3])
ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot(pca_eeg[:,1], pca_eeg[:,2], pca_eeg[:,3])
ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot(pca_eeg[:,0], pca_eeg[:,2], pca_eeg[:,3])
plt.show();
proj, proj_eeg = DyCA(eeg_data, eig_threshold=0.9)
#plot projected time-series
plt.figure(1);
plt.plot(proj_eeg);
#plot in sections of the 4D phase space
fig = plt.figure(figsize=(15,14))
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot(proj_eeg[:,0], proj_eeg[:,1], proj_eeg[:,2])
ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot(proj_eeg[:,0], proj_eeg[:,1], proj_eeg[:,3])
ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot(proj_eeg[:,1], proj_eeg[:,2], proj_eeg[:,3])
ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot(proj_eeg[:,0], proj_eeg[:,2], proj_eeg[:,3])
plt.show();
def visualizeProjection(proj, fig,sub1,sub2,nsub):
    # some parameters
    N = 500             # number of points for interpolation
    xy_center = [3,4]   # center of the plot
    radius = 3.5          # radius    
    #eeg electrode dictionary
    eeg_dict = { "FP1" : [2,7.75], "FP2" : [4, 7.75], "F11" : [-0.5,5.75], "F7" : [0.75,5.75], "F3" : [1.75,5.75], "Fz" : [3,5.75], "F$" : [4.25,5.75], "F8" : [5.25,5.75], "F12" : [6.5,5.75],"FT9" : [-1,4.75], "FT10" : [7,4.75], "T7" : [0.5,3.75] , "T8" : [5.5,3.75], "C3" : [1.75,3.75] , "Cz" : [3,3.75], "C4" : [4.25,3.75], "TP9" : [-1,2.75], "TP10" : [7,2.75],"P7" : [0.5,1.75], "P4" : [4.25,1.75], "P3" : [1.75,1.75], "Pz" : [3,1.75], "P8" : [5.25,1.75],"O1" : [2,0.25], "O2" :[4,0.25] }
    electrodes = ["F12","FT10","TP10","F8","T8","P8","O2","FP2","F$","C4","P4","Fz","Cz","Pz","F11","FT9","TP9","F7","T7","P7","O1","FP1","F3","C3","P3"]
    x,y = [],[]
    for i in electrodes:
        x.append(eeg_dict[i][0])
        y.append(eeg_dict[i][1])    

    projData = np.asfarray(proj,float)   
    z = projData    
    xi = np.linspace(-8, 16, N)
    yi = np.linspace(-8, 16, N)
    zi = sp.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')    
    dr = xi[1] - xi[0]
    for i in range(N):
        for j in range(N):
            r = np.sqrt((xi[i] - xy_center[0])**2 + (yi[j] - xy_center[1])**2)
            if (r - dr/2) > radius:
                zi[j,i] = "nan"    
    # make figure
    ax = fig.add_subplot(sub1,sub2,nsub, aspect = 1)            
    # use different number of levels for the fill and the lines
    CS = ax.contourf(xi, yi, zi, 60, cmap = plt.cm.jet, zorder = 1)#,levels=np.linspace(-220, 170, 100),)            
    # draw a circle
    # change the linewidth to hide the 
    circle = mpl.patches.Circle(xy = xy_center, radius = radius, edgecolor = "k", facecolor = "none")
    ax.add_patch(circle)            
    # make the axis invisible 
    for loc, spine in ax.spines.items():
        # use ax.spines.items() in Python 3
        spine.set_linewidth(0)                
    # remove the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # add two ears
    circle = mpl.patches.Ellipse(xy = [-0.5,4], width = 1, height = 2.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)
    circle = mpl.patches.Ellipse(xy = [6.5,4], width = 1, height = 2.0, angle = 0, edgecolor = "k", facecolor = "w", zorder = 0)
    ax.add_patch(circle)            
    # set axes limits
    ax.set_xlim(-2.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
       
fig = plt.figure(figsize=(15,14))
visualizeProjection(proj[:,0],fig,1,4,1)
visualizeProjection(proj[:,1],fig,1,4,2)
visualizeProjection(proj[:,2],fig,1,4,3)
visualizeProjection(proj[:,3],fig,1,4,4)

fig = plt.figure(figsize=(15,14))
visualizeProjection(pca.components_[0,:],fig,1,4,1)
visualizeProjection(pca.components_[1,:],fig,1,4,2)
visualizeProjection(pca.components_[2,:],fig,1,4,3)
visualizeProjection(pca.components_[3,:],fig,1,4,4)