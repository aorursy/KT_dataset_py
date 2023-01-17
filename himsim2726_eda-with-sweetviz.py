!pip install sweetviz
import pandas as pd
import sweetviz as sv
train = pd.read_csv('../input/titanic/train.csv')
train.head(5)
result = sv.analyze([train, "Train"],target_feat='Survived')
result.show_html('Report.html')