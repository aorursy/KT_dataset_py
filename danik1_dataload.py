%pylab inline  
%precision 6
import pandas as pd
import matplotlib.pylab as plt
import sklearn as skl
pd.options.display.max_colwidth=100
np.set_printoptions(linewidth=140,edgeitems=10)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
rcParams['figure.figsize'] = (8.0, 5.0)
D = pd.read_json('train.json')
D = D[['cuisine','ingredients']]
D.head()