import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
try:
    import umap
except:
    !pip install umap-learn
    import umap
x = [1, 5, 10, 20]
y1 = [67.5, 70.5, 80.2, 88.1]
y2 = [45.1, 50.2, 56.2, 60.6]
y3 = [55.1, 60.2, 61.2, 66.6]
y4 = [25.2, 30.1, 32.4, 33.9]

y_std1 = np.random.randn(len(x))*3
y_std2 = np.random.randn(len(x))*5
y_std3 = np.random.randn(len(x))*5
y_std4 = np.random.randn(len(x))*2

z1 = [65.5, 67.5, 75.2, 79.1]
z2 = [41.1, 45.2, 46.2, 50.6]
z3 = [45.1, 50.2, 51.2, 56.6]
z4 = [25.2, 29.1, 32.4, 32.9]

z_std1 = np.random.randn(len(x))*3
z_std2 = np.random.randn(len(x))*5
z_std3 = np.random.randn(len(x))*5
z_std4 = np.random.randn(len(x))*2
label_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}

title_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}

legend_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 16,
}
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
ax.plot(x, y1, linewidth=2.0, color='tab:red', 
        marker='^', markersize=10, markeredgecolor='white', markeredgewidth=1.2, 
        label='SleepDPC')
ax.fill_between(x, y1 - y_std1, y1 + y_std1, color='tab:red', alpha=0.2)

ax.plot(x, y2, linewidth=2.0, color='tab:orange', 
        marker='d', markersize=10, markeredgecolor='white', markeredgewidth=1.2,
        label='PCA')
ax.fill_between(x, y2 - y_std2, y2 + y_std2, color='tab:orange', alpha=0.2)

ax.plot(x, y3, linewidth=2.0, color='tab:cyan', 
        marker='p', markersize=10, markeredgecolor='white', markeredgewidth=1.2, 
        label='AE')
ax.fill_between(x, y3 - y_std3, y3 + y_std3, color='tab:cyan', alpha=0.2)

ax.plot(x, y4, linewidth=2.0, color='tab:olive', 
        marker='o', markersize=10, markeredgecolor='white', markeredgewidth=1.2, 
        label='Random Weights')
ax.fill_between(x, y4 - y_std4, y4 + y_std4, color='tab:olive', alpha=0.2)

ax.set_xlabel('percentage of labeled data', label_font)
ax.set_ylabel('accuracy', label_font)

ax.set_title('SleepEDF', title_font)


# ax.legend(prop=legend_font, loc='auto', frameon=True, columnspacing=3, ncol=3)

ax = fig.add_subplot(122)
ax.plot(x, z1, linewidth=2.0, color='tab:red', 
        marker='^', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(x, z1 - z_std1, z1 + z_std1, color='tab:red', alpha=0.2)

ax.plot(x, z2, linewidth=2.0, color='tab:orange', 
        marker='d', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(x, z2 - z_std2, z2 + z_std2, color='tab:orange', alpha=0.2)

ax.plot(x, z3, linewidth=2.0, color='tab:cyan', 
        marker='p', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(x, z3 - z_std3, z3 + z_std3, color='tab:cyan', alpha=0.2)

ax.plot(x, z4, linewidth=2.0, color='tab:olive', 
        marker='o', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(x, z4 - z_std4, z4 + z_std4, color='tab:olive', alpha=0.2)

ax.set_xlabel('percentage of labeled data', label_font)
ax.set_ylabel('accuracy', label_font)

ax.set_title('ISRUC', title_font)

bb = (0.06, 0.9, 
      fig.subplotpars.right-fig.subplotpars.left, 1.0)

lg = fig.legend(prop=legend_font, loc='lower center', frameon=True, bbox_to_anchor=bb, 
           columnspacing=3, ncol=4, mode='expand')
# fig.tight_layout()
fig.subplots_adjust(top=0.8)
fig.savefig('rep_comp.pdf', pad_inches=0.0, bbox_inches="tight", bbox_extra_artists=(lg,))
fig.show()
x = [1, 5, 10, 20]
y1 = [67.5, 70.5, 80.2, 88.1]
y2 = [45.1, 50.2, 56.2, 60.6]
y3 = [55.1, 60.2, 61.2, 66.6]
y4 = [25.2, 30.1, 32.4, 33.9]

y_std1 = np.random.randn(len(x))*3
y_std2 = np.random.randn(len(x))*5
y_std3 = np.random.randn(len(x))*5
y_std4 = np.random.randn(len(x))*2

z1 = [65.5, 67.5, 75.2, 79.1]
z2 = [41.1, 45.2, 46.2, 50.6]
z3 = [45.1, 50.2, 51.2, 56.6]
z4 = [25.2, 29.1, 32.4, 32.9]

z_std1 = np.random.randn(len(x))*3
z_std2 = np.random.randn(len(x))*5
z_std3 = np.random.randn(len(x))*5
z_std4 = np.random.randn(len(x))*2
label_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}

title_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}

legend_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}
fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(121)
ax.plot(x, y1, linewidth=2.0, color='tab:red', 
        marker='^', markersize=10, markeredgecolor='white', markeredgewidth=1.2, 
        label='SleepDPC')
ax.fill_between(x, y1 - y_std1, y1 + y_std1, color='tab:red', alpha=0.2)

ax.plot(x, y2, linewidth=2.0, color='tab:purple', 
        marker='d', markersize=10, markeredgecolor='white', markeredgewidth=1.2,
        label='SVM')
ax.fill_between(x, y2 - y_std2, y2 + y_std2, color='tab:purple', alpha=0.2)

ax.plot(x, y3, linewidth=2.0, color='tab:pink', 
        marker='p', markersize=10, markeredgecolor='white', markeredgewidth=1.2, 
        label='HHT-CNN')
ax.fill_between(x, y3 - y_std3, y3 + y_std3, color='tab:pink', alpha=0.2)

ax.plot(x, y4, linewidth=2.0, color='tab:brown', 
        marker='o', markersize=10, markeredgecolor='white', markeredgewidth=1.2, 
        label='DeepSleepNet')
ax.fill_between(x, y4 - y_std4, y4 + y_std4, color='tab:brown', alpha=0.2)

ax.set_xlabel('percentage of labeled data', label_font)
ax.set_ylabel('accuracy', label_font)

ax.set_title('SleepEDF', title_font)


# ax.legend(prop=legend_font, loc='auto', frameon=True, columnspacing=3, ncol=3)

ax = fig.add_subplot(122)
ax.plot(x, z1, linewidth=2.0, color='tab:red', 
        marker='^', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(x, z1 - z_std1, z1 + z_std1, color='tab:red', alpha=0.2)

ax.plot(x, z2, linewidth=2.0, color='tab:purple', 
        marker='d', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(x, z2 - z_std2, z2 + z_std2, color='tab:purple', alpha=0.2)

ax.plot(x, z3, linewidth=2.0, color='tab:pink', 
        marker='p', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(x, z3 - z_std3, z3 + z_std3, color='tab:pink', alpha=0.2)

ax.plot(x, z4, linewidth=2.0, color='tab:brown', 
        marker='o', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(x, z4 - z_std4, z4 + z_std4, color='tab:brown', alpha=0.2)

ax.set_xlabel('percentage of labeled data', label_font)
ax.set_ylabel('accuracy', label_font)

ax.set_title('ISRUC', title_font)

bb = (0.06, 0.9, 
      fig.subplotpars.right-fig.subplotpars.left, 1.0)

lg = fig.legend(prop=legend_font, loc='lower center', frameon=True, bbox_to_anchor=bb, 
           columnspacing=3, ncol=4, mode='expand')
# fig.tight_layout()
fig.subplots_adjust(top=0.8)
fig.savefig('sup_comp.pdf', pad_inches=0.0, bbox_inches="tight", bbox_extra_artists=(lg,))
fig.show()
x1 = ['3/4', '1/2', '1/4']
x2 = [5, 10, 20, 30]
x3 = [4, 8, 16, 32, 64, 128, 256, 512]

y1 = np.random.randn(3)*10
y2 = np.random.randn(4)*10
y3 = np.random.randn(8)*10

z1 = np.random.randn(3)*10 + 1.2
z2 = np.random.randn(4)*10 - 0.5
z3 = np.random.randn(8)*10 + 3
label_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}

title_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}

legend_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}
fig = plt.figure(figsize=(16,3))
ax = fig.add_subplot(131)
ax.plot(x1, y1, linewidth=2.0, color='tab:red', label = 'Pre-training', 
        marker='d', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.plot(x1, z1, linewidth=2.0, color='tab:blue', label = 'Evaluation',
        marker='p', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.set_title('Prediction Steps', title_font)

ax = fig.add_subplot(132)
ax.plot(x2, y2, linewidth=2.0, color='tab:red', 
        marker='d', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.plot(x2, z2, linewidth=2.0, color='tab:blue', 
        marker='p', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.set_title('Window Length', title_font)

ax = fig.add_subplot(133)
ax.plot(x3, y3, linewidth=2.0, color='tab:red', 
        marker='d', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.plot(x3, z3, linewidth=2.0, color='tab:blue', 
        marker='p', markersize=10, markeredgecolor='white', markeredgewidth=1.2)
ax.set_title('Representation Dimension', title_font)

bb = (0.06, 0.95, 
      fig.subplotpars.right-fig.subplotpars.left, 1.0)

lg = fig.legend(prop=legend_font, loc='lower center', frameon=True, bbox_to_anchor=bb,
           columnspacing=3, ncol=2)
fig.subplots_adjust(top=0.9)
fig.savefig('param.pdf', pad_inches=0.0, bbox_inches="tight", bbox_extra_artists=(lg,))
fig.show()
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.tri as tri

plt.style.use('seaborn-paper')
rep1 = np.random.randn(1000, 128)
rep2 = np.random.randn(500, 128)
rep3 = np.random.randn(800, 128)
rep4 = np.random.randn(750, 128)
rep5 = np.random.randn(600, 128)
reducer = umap.UMAP(random_state=42)
emb1 = reducer.fit_transform(rep1)
emb2 = reducer.fit_transform(rep2)
emb3 = reducer.fit_transform(rep3)
emb4 = reducer.fit_transform(rep4)
emb5 = reducer.fit_transform(rep5)
label_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}

title_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}

legend_font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
}
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(20, 4), sharex=True, sharey=True)

xi = np.linspace(-2.1, 2.1, 1000)
yi = np.linspace(-2.1, 2.1, 1000)

ax1.scatter(emb1[:,0], emb1[:,1], color='tab:red', alpha=0.4, label='W')
triang = tri.Triangulation(emb1[:,0], emb1[:,1])
interpolator = tri.LinearTriInterpolator(triang, np.ones_like(len(emb1)))
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)
ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
ax1.axis('off')

ax2.scatter(emb2[:,0], emb2[:,1], color='tab:blue', alpha=0.4, label='N1')
ax2.axis('off')

ax3.scatter(emb3[:,0], emb3[:,1], color='tab:olive', alpha=0.4, label='N2')
ax3.axis('off')

ax4.scatter(emb4[:,0], emb4[:,1], color='tab:cyan', alpha=0.4, label='N3')
ax4.axis('off')

ax5.scatter(emb5[:,0], emb5[:,1], color='tab:purple', alpha=0.4, label='R')
ax5.axis('off')

fig.tight_layout()
fig.legend(prop=legend_font, frameon=True, loc='lower center', ncol=5, columnspacing=4)
data = np.load('/kaggle/input/sleepedf-lite-0/SC4001E0.npz')
data = np.stack([data['eeg_fpz_cz'], data['eeg_pz_oz']], axis=1)
plt.plot(data[0][0])
from scipy import fftpack
def get_fft_values(y, f_s):
    N = len(y)
    f_values = np.linspace(0.0, f_s/2.0, N//2)
    fft_values_ = fft(y)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values
y = data[0][0]
f_s=100
X = fftpack.fft(y)
freqs = fftpack.fftfreq(len(y)) * f_s

fig, ax = plt.subplots()

ax.stem(freqs, np.abs(X))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(0, f_s / 2)
# ax.set_ylim(-5, 110)