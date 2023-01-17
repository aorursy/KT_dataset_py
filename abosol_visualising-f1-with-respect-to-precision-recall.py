import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
precision_min, precision_max = 0, 1
recall_min, recall_max = 0, 1
xx, yy = np.meshgrid(np.arange(precision_min, precision_max, 0.1),
                     np.arange(recall_min, recall_max, 0.1))
Z = np.array([2.0*r*p/(r + p) for p, r in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.ylabel('recall')
plt.xlabel('precision')
plt.title('F1')
plt.colorbar()
plt.show()
