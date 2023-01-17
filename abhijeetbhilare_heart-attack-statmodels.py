import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
import statsmodels.stats.tests.test_influence
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv", index_col=0)
df.head()
y = df.target
x = df.drop(columns=["target"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
res = GLM(y_train, x_train,
          family=families.Binomial()).fit(attach_wls=True, atol=1e-10)
print(res.summary())
pred = np.array(res.predict(x_test), dtype=float)
table = np.histogram2d(y_test, pred, bins=2)[0]
table
print("Statmodel Acc : ", (table[0,0] + table[1,1])/(table[0,0] + table[1,1]+table[1,0] + table[0,1]))
ax = sns.heatmap(table, linewidth=0.5)
plt.show()
infl = res.get_influence(observed=False)
summ_df = infl.summary_frame()
summ_df.sort_values('cooks_d', ascending=False)[:10]
infl.plot_influence()
infl.plot_index(y_var='cooks', threshold=2 * infl.cooks_distance[0].mean())
infl.plot_index(y_var='resid', threshold=1)
