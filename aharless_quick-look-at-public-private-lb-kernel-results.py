import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import statsmodels.formula.api as sm
df = pd.read_csv("../input/porto-seguro-public-kernel-results/portoKernels.csv",index_col=0)

df.head()
df.set_index('votes').drop(['reported public score','adjusted'],axis=1).sort_index(ascending=False)
X = df.rename(columns={'public score':'pubscore','private score':'privscore'})

sm.ols(formula="difference ~ pubscore+votes", data=X).fit().summary()
sm.ols(formula="difference ~ votes", data=X).fit().summary()
sm.ols(formula="privscore ~ pubscore+votes", data=X).fit().summary()
plt.scatter(X.votes,X.difference)

plt.axis([0, 250, -.003, .012])

plt.show()
plt.scatter(X.pubscore,X.difference)

plt.axis([0, .3, -.003, .012])

plt.show()
# More detail on the right side

plt.scatter(X.pubscore,X.difference)

plt.axis([.2, .3, -.003, .012])

plt.show()
sm.ols(formula="privscore ~ pubscore", data=X).fit().summary()
plt.scatter(X.pubscore,X.privscore)

plt.axis([0, .3, 0, .3])

plt.show()
# More detail on the right side

plt.scatter(X.pubscore,X.privscore)

plt.axis([.2, .3, .2, .3])

plt.show()
# Even more detail on the far right side

plt.scatter(X.pubscore,X.privscore)

plt.axis([.270, .292, .270, .292])

plt.show()
# Yet even more detail on the very far right side

plt.scatter(X.pubscore,X.privscore)

plt.axis([.278, .288, .278, .291])

plt.show()
# Yet still even more detail on the extreme far right side

plt.scatter(X.pubscore,X.privscore)

plt.axis([.282, .288, .282, .291])

plt.show()