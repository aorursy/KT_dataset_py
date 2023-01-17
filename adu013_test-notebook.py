import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
df = pd.DataFrame(np.random.randn(5,10))
df
plt.plot(df)
plt.show()
