import itertools

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



datass = pd.read_csv('../input/winequality-red.csv')

datass.groupby(['quality']).count().plot(kind='bar',legend=False,color='blue', figsize=(6,6));



#datass.plot.scatter(x='citric acid', y='pH', s=2, c='red')



#data['quality'].value_counts().plot(kind = 'hist',bins = 6,figsize = (10,12))

import itertools

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

datass = pd.read_csv('../input/winequality-red.csv')

(sns.FacetGrid(datass, hue = 'quality', size = 5, palette = 'Reds')

 .map(plt.scatter, "citric acid", "pH", s = 25)

.add_legend());
import itertools

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

datass = pd.read_csv('../input/winequality-red.csv')

(sns.FacetGrid(datass, hue = 'quality', size = 5, palette = 'Reds')

 .map(plt.scatter, "residual sugar", "alcohol", s = 25)

.add_legend());
import itertools

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

datass = pd.read_csv('../input/winequality-red.csv')

(sns.FacetGrid(datass, hue = 'quality', size = 5, palette = 'Reds')

 .map(plt.scatter, "sulphates", "pH", s = 25)

.add_legend());
import itertools

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

datass = pd.read_csv('../input/winequality-red.csv')

datass.groupby(["quality"])["citric acid", "sulphates", "alcohol"].mean()