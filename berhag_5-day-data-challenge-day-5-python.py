import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import chisquare, chi2_contingency, chi2
cereal = pd.read_csv('../input/cereal.csv')

cereal.head(10)
cereal.describe(include = ['O'])
cereal['shelf'] = cereal['shelf'].apply(str)

cereal.describe(include = ['O'])
cereal_tab = pd.crosstab(cereal.mfr, cereal.shelf, margins = True)

cereal_tab.columns =  ['shelf1', 'shelf2', 'shelf3', 'total_rows']

cereal_tab.index =  ['A', 'G', 'K', 'N', 'P', 'Q', 'R', 'total_cols']

cereal_tab
cereal_Observed = cereal_tab.iloc[0:7,0:3]

cereal_Observed
cereal.groupby('mfr', as_index=False).agg({"shelf": "count"})
cereal.groupby('shelf', as_index=False).agg({"mfr": "count"})
cereal_expected =  np.outer(cereal_tab["total_rows"][0:7], cereal_tab.loc["total_cols"][0:3]) / 77

cereal_expected = pd.DataFrame(cereal_expected)

cereal_expected.columns =  ['shelf1', 'shelf2', 'shelf3']

cereal_expected.index =  ['A', 'G', 'K', 'N', 'P', 'Q', 'R']

cereal_expected
chi2_contingency(observed = cereal_Observed)
# Find the critical value for 95% confidence where  df = number of categories  - 1

# the df is calculated (7-1)*(3-1) = 12

critical = chi2.ppf(q = 0.95, df = 12)

print()

print("Critical value {:.3f}".format(critical))

chisquare(cereal_Observed,f_exp=cereal_expected)
x = np.arange(len(cereal_Observed.index))

width = 0.3

plt.figure(figsize=(12,8))

plt.bar(x, cereal_Observed['shelf1'], width, align='center',

        alpha=0.5, edgecolor = 'black', 

        label = 'shelf1',

       color = 'b')

plt.bar(x+width, cereal_Observed['shelf2'], width, align='center',

        alpha=0.5, edgecolor = 'black', 

        label = 'shelf2',

       color = 'g')

plt.bar(x+2*width, cereal_Observed['shelf3'], width, align='center',

        alpha=0.5, edgecolor = 'black', 

        label = 'shelf3',

       color = 'r')



plt.xticks(x+width, cereal_Observed.index, fontsize=18 )

plt.ylabel('Cereals count from each producer', fontsize=18 )

plt.legend(fontsize=18)

plt.title('Manufacturers of cereals ', fontsize=18)

 

plt.show()