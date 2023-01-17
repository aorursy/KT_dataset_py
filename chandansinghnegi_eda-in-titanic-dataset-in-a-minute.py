!pip install sweetviz
import pandas as pd

import sweetviz
train=pd.read_csv("../input/titanic/train (1).csv")

test=pd.read_csv("../input/titanic/test (1).csv")
train.head()
final_report=sweetviz.analyze([train,"TRAIN"],target_feat="Survived")
final_report.show_html("report.html")