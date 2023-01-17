import numpy as np

import pandas as pd 
fundamentals = pd.read_csv("../input/fundamentals.csv", index_col=0)
fundamentals.head()
%matplotlib inline
# pandas works...

fundamentals['Accounts Receivable'].head(10).plot()
from ggplot import (ggplot, aes, geom_density)
# ...ggplot flashes the plot, but then doesn't work after that.

(

    ggplot(fundamentals, aes(x='Accounts Payable', y='Accounts Receivable')) +

    geom_density()

)