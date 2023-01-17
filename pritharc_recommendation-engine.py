!pip install dexplot
!pip install chart_studio
!pip install pandas-profiling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
import dexplot as dxp
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

import seaborn as sns
from pandas_profiling import ProfileReport
import pandas_profiling
import plotly.express as px
df = pd.read_csv('../input/sales-modified/summer-products-with-rating-and-performance_2020-08.csv')
df.describe()
pandas_profiling.ProfileReport(df)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings; warnings.simplefilter('ignore')
%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def get_recommendations(id):    
    sales=pd.read_csv("../input/sales-modified/summer-products-with-rating-and-performance_2020-08.csv")
    sales.head(2)
    
    orders_for_product = sales[sales.product_id == 548 ].Order.unique()
    
    relevant_orders = sales[sales.Order.isin(orders_for_product)]
    
    accompanying_products_by_order = relevant_orders[relevant_orders.product_id != 548]
    num_instance_by_accompanying_product = accompanying_products_by_order.groupby(548)[548].count().reset_index(name='instances')
    
    num_orders_for_product = orders_for_product.size
    product_instances = pd.DataFrame(num_instance_by_accompanying_product)
    product_instances['frequency'] = product_instances['instances']/num_orders_for_product
    
    recommended_products = pd.DataFrame(product_instances.sort_values('frequency', ascending=False).head(3))
    
    products = pd.read_csv("../input/sales-modified/summer-products-with-rating-and-performance_2020-08.csv")
    recommended_products = pd.merge(recommended_products, products, on='product_id')