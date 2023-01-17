# Not present already, must install with pip

# This is how you can send commands to the command line

!pip install autoviz
import pandas as pd

data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv', index_col = 0)

data.head()
data.info()
from autoviz.AutoViz_Class import AutoViz_Class

av = AutoViz_Class()
## Finally the 1 line you have been waiting for

df = av.AutoViz('../input/data-analyst-jobs/DataAnalyst.csv', dfte=data)