import pandas as pd
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.tail(-10)
!pip install sweetviz #Fist Install the library
import sweetviz as sv

sweet_report = sv.analyze(df)
sweet_report.show_html('sweet_report.html') #This step will generate the report and save it in a file named “sweet_report.html” which is user-defined.
!pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
df = AV.AutoViz('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
import pandas as pd

from pandas_profiling import ProfileReport
design_report = ProfileReport(df)

design_report.to_file(output_file='python_profiling_report.html')