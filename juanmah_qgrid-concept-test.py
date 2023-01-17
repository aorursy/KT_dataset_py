# !conda install -y nodejs

# !jupyter labextension install qgrid
# !jupyter nbextension enable --py --sys-prefix widgetsnbextension

# !jupyter nbextension enable --py --sys-prefix qgrid
import qgrid

from sklearn.datasets import load_iris

import pandas as pd



data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)

qgrid_widget = qgrid.show_grid(df, show_toolbar=True)

qgrid_widget