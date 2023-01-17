!pip install dabl

import dabl
import pandas as pd

train_data = pd.read_csv('../input/titanic/train.csv')
dabl.detect_types(train_data)
dabl.plot(train_data, target_col="Survived")
train_data.corr().style.background_gradient(cmap='Oranges')