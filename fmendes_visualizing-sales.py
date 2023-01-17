import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

sales = pd.read_csv("../input/OfficeSupplies.csv")
sales.head()
sales_plot = sales.groupby( "Rep" ).sum().plot( kind='bar' )
sales_plot.set_xlabel( "Sales Rep" )
sales_plot.set_ylabel( "Qty & Price" )