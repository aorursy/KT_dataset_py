import numpy as np
import pandas as pd

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
train.head()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
titanic_df['Sex'].unique()
sns.factorplot('Sex',data=titanic_df,kind='count',hue='Pclass')

