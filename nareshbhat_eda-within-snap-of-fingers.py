%%capture

!pip install sweetviz
import pandas as pd

import sweetviz

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train.head()
report = sweetviz.analyze([train,'train'],target_feat='Survived')

report.show_html('report.html')