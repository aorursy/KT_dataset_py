import pandas as pd
cdc_2003 = pd.read_csv("../input/education_sugar_cdc_2003.csv")
cdc_2003.set_index("State", inplace=True)
cdc_2003.head()
for column in cdc_2003.columns:
  cdc_2003[column]=cdc_2003[column].str.replace(r"\(.*\)","")
  cdc_2003[column]=pd.to_numeric(cdc_2003[column])
cdc_2003.head()
  
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__) # requires version >= 1.9.0
import cufflinks as cf
cf.go_offline()
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=False)

cdc_2003.iplot(
    title="Crude prevalence* of sugar-sweetened beverage† consumption ≥1 time/day among adults, by employment status, education, and state.  2003, CDC",
                    xTitle="State",
                    yTitle="Sugar-sweetened beverage consumption ≥1 time/day",
    kind='bar', filename='cufflinks/grouped-bar-chart')