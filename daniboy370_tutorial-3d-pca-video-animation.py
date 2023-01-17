import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib import animation

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

import matplotlib.animation as animation

from matplotlib.patches import FancyArrowPatch

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





url = '../input/breast-cancer-wisconsin-data/data.csv'

df = pd.read_csv( url )



rndperm = np.random.permutation(df.shape[0])

D = df.iloc[rndperm, :]



## Print the number of rows in the data set

df_rows, df_cols = df.shape

print('Table size : {} x {}'.format(df_rows, df_cols) )



ix_mn = [*range(0, 10, 1)]



def arr_shift(arr, shift):

    return [i+shift for i in arr]



class_feat = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal dim.']



le = LabelEncoder()                 # label encoding

X, y = D.iloc[:, 2:], D[['diagnosis']]

y = y.rename(columns={'diagnosis': 'Diagnosis'})

y = le.fit_transform( y['Diagnosis'].values )

if isinstance(y, pd.DataFrame):

    y = y.values.ravel()

    

X_mn = X.iloc[:, ix_mn]
# **************************************************************** #

# ------------------------ 2D PCA scatter ------------------------ #

# **************************************************************** #



from sklearn.preprocessing import StandardScaler



# In general, it's a good idea to scale the data prior to PCA.

scaler = StandardScaler()

scaler.fit(X_mn)

X_mn = scaler.transform(X_mn)

pca = PCA()

x_new = pca.fit_transform(X_mn)



def PCA_scatter(X, coeff, y_M, labels):

    

    n = coeff.shape[0]

    xs, ys = X[:,0], X[:,1]             # zs = X[:,2]

    scalex, scaley = 1.0/(xs.max() - xs.min()) , 1.0/(ys.max() - ys.min())

    # scalez = 1.0/(zs.max() - zs.min()) <-- 3D of Z

    

    # ---------- Scatter color by class ----------- #

    plt.scatter(xs[y_M] * scalex, ys[y_M] * scaley, c = 'orange', alpha=0.5) 

    plt.scatter(xs[1-y_M==True] * scalex, ys[1-y_M==True] * scaley, c = 'blue', alpha=0.5)

    # plt.scatter(xs * scalex, ys * scaley, c = 'blue')



    for i in range(n):

        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'r',alpha = 0.5)

        if labels is None:

            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')

        else:

            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', fontsize=12, ha = 'center', va = 'center')

    

    plt.xlabel("PC{}".format(1))

    plt.ylabel("PC{}".format(2))

    plt.grid(linestyle='-', linewidth=0.5)



#Call the function. Use only the 2 PCs

y_M, y_B = y==1, y==0           # Logical statement for Benign indication



pca_i = 2



PCA_scatter(x_new[:, 0:pca_i], np.transpose(pca.components_[0:pca_i, :]), y_M, class_feat)

plt.rcParams['figure.figsize'] = (12, 6)

plt.show()
D.iloc[:, 2:].head(10)        # Show ten first samples (after random shuffling)
# ******************************************************** #

# ------------- PCA : Utilization functions -------------- #

# ******************************************************** #



# --------------- Dimensionality Reduction --------------- #

def PCA_reduction(X, PC_num):

    scaler = StandardScaler()

    scaler.fit(X)

    X = scaler.transform(X)

    pca = PCA()                                            # Perform PCA transformation

    X_pca = pca.fit_transform(X)[:, 0:PC_num]              # Low dim : Projected  instances

    max_Var = np.transpose(pca.components_[0:PC_num, :])   # Direction of maximum variance

    return X_pca, max_Var





class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):

        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)

        self._verts3d = xs, ys, zs



    def draw(self, renderer):

        xs3d, ys3d, zs3d = self._verts3d

        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)

        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        FancyArrowPatch.draw(self, renderer)



# ------------------- PCA Visualization ------------------ #

def Scatter_3D(X_pca, max_Var, y, labels):

    

    y_M, y_B = y==1, y==0           # Logical statement for Benign indication

    xs, ys, zs = X_pca[:,0], X_pca[:,1], X_pca[:,2]

    s_x, s_y, s_z = 1.0/(xs.max() - xs.min()), 1.0/(ys.max() - ys.min()), 1.0/(zs.max() - zs.min())

    

    # -------------- Scatter color by class -------------- #

    

    ax.scatter(xs[y_M]*s_x, ys[y_M]*s_y, zs[y_M]*s_z, s=42, c='orange', alpha=0.35) 

    ax.scatter(xs[y_B]*s_x, ys[y_B]*s_y, zs[y_B]*s_z, s=42, c='blue',   alpha=0.35)

    n = max_Var.shape[0]



    for i in range(n):

        mean_x, mean_y, mean_z = max_Var[i,0], max_Var[i,1], max_Var[i,2]

        a = Arrow3D([mean_x, 0.0], [mean_y, 0.0], [mean_z, 0.0], mutation_scale=15, lw=3, arrowstyle="<|-", color="r")

        ax.add_artist(a)



        if labels is None:

            ax.text(max_Var[i,0]* 1.15, max_Var[i,1] * 1.15, max_Var[i,2] * 1.15, "Var"+str(i+1), color = 'g', fontsize=14, ha = 'center', va = 'center')

        else:

            ax.text(max_Var[i,0]* 1.15, max_Var[i,1] * 1.15, max_Var[i,2] * 1.15, labels[i],      color = 'g', fontsize=14, ha = 'center', va = 'center')





# --------------- Normalize data structure --------------- #

def self_Normalize( X ):

    X_n = (X-X.mean())/(X.max(axis=0)-X.min(axis=0))

    return X_n

# ******************************************************** #

# ----------------- Data Visualization ------------------- #

# ******************************************************** #





# ---------- Static : Setting of the figure -------------- #

def initialize_figure(pca_exp_mn):

    fig = plt.figure(figsize=(10, 6))

    ax = Axes3D(fig)

    ax.xaxis.pane.fill, ax.yaxis.pane.fill, ax.zaxis.pane.fill = False, False, False

    ax.legend(['Malignant', 'Benign'], fontsize=15, loc='best')

    ax.set_xlabel('PC-1 : %.2f [%%]'%pca_1, fontsize=13)

    ax.set_ylabel('PC-2 : %.2f [%%]'%pca_2, fontsize=13)

    ax.set_zlabel('PC-3 : %.2f [%%]'%pca_3, fontsize=13)

    return ax, fig







# --------- Static : points at constant location --------- #

def init():

    Scatter_3D(X_pca, max_Var, y, class_feat)

    return fig,





# ----------- Dynamic : define desired motion ------------ #

def animate(i):

    ''' 

    input i : number of frames 

    Total_frame : defines length of footage

    [elev, azim] : parameters of 3D point of view

    '''

    thres = 200

    if i > thres:

        j = i-thres

    else:

        j = 0



    # Explain on motion preferences

    Elev = 45 - i/6 + 2*j/6

    Azim = -120+i/2

    ax.view_init(elev=Elev, azim=Azim)

    

    frame_freq = 20

    if (i%frame_freq - Total_frame%10) == 0:

        print('Remaining frames : ', Total_frame-i)

    return fig,





# -------------- Implement PCA on the data --------------- #

PC_num = 3                     # Dimensionallity reduction to 3D

pca_mn = PCA()

X_mn = self_Normalize(X_mn)

pca_mn.fit( X_mn )

pca_exp_mn = pca_mn.explained_variance_ratio_

pca_1, pca_2, pca_3 = pca_exp_mn[0]*100, pca_exp_mn[1]*100, pca_exp_mn[2]*100

X_pca, max_Var = PCA_reduction(X_mn, PC_num)





# ------------- Processing of data before PCA ------------ #

y_M, y_B = y==1, y==0      # Logical statement for Benign indication

xs, ys, zs = X_pca[:,0], X_pca[:,1], X_pca[:,2]

s_x, s_y, s_z = 1.0/(xs.max() - xs.min()), 1.0/(ys.max() - ys.min()), 1.0/(zs.max() - zs.min())

ax, fig = initialize_figure(pca_exp_mn)





# ------------- Parameter for video footage -------------- #

Total_frame = 101

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Total_frame, interval=20, blit=True)

anim.save('PCA_vid.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

print('Finished ! Download video from : "../input/output/"')
# @title



plt.rcParams['figure.figsize'] = (9, 5)

class_full =  class_feat + ['Diagnosis']



# Initialize the PCA method

pca_mn = PCA()



# Mean Dataset

pca_mn.fit(X_mn)

pca_exp_mn = pca_mn.explained_variance_ratio_

t_solo = [*range(1, X_mn.shape[1]+1)]





def PCA_plot(t, pca_exp):

    # ---------------------------------------------- #

    # Instantiate the prinicipal (LHS) plot

    pca_cum = np.cumsum(pca_exp)

    fig, ax1 = plt.subplots()

    color = 'tab:blue'



    ax1.grid(color='b', ls = '-.', lw = 0.25)

    ax1.set_xlabel('n-th component', fontsize=16)

    ax1.set_ylabel('Explained Variance Ratio (EVR)', color=color, fontsize=17)

    ax1.plot(t, pca_exp, 'bo', color=color, markersize=7)

    ax1.plot(t, pca_exp, '--', color=color, linewidth=2.5)

    ax1.tick_params(axis="x", labelsize=12)

    ax1.tick_params(axis="y", labelsize=12)



    # ---------------------------------------------- #

    # Instantiate a second axes that shares the same x-axis

    ax2 = ax1.twinx()  

    color = 'tab:green'



    ax2.set_ylabel('Cumulative EVR', color=color, fontsize=17)  # we already handled the x-label with ax1

    ax2.plot(t, pca_cum, 'go', color=color, markersize=7)

    ax2.plot(t, pca_cum, '--', color=color, linewidth=2.5)

    ax2.tick_params(axis="y", labelsize=12)

    t_score, t_loc = pca_cum[2], pca_cum[2]*1.025

    ax2.annotate('%.2f '%(t_score*100)+'[%]', fontsize=18, xy =(3, t_loc), xytext =(3, t_loc*1.06), arrowprops = dict(facecolor ='green', shrink = 0.05),) 



    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.xticks(t)

    plt.show()



PCA_plot(t_solo, pca_exp_mn)