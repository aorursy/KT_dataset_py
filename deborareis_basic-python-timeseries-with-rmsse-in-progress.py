# Import calendar dataset and show first lines
import pandas as pd
calend = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
calend.head()
# amount of lines, columns
calend.shape
# Years from most 2011 to middle 2016
calend.year.value_counts()
# Events, Holidays
calend.event_name_1.value_counts()
calend.event_type_1.value_counts()
calend.event_name_2.value_counts()
calend.event_type_2.value_counts()
import pandas as pd
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')
sales.head()
sales.shape
# Sales per State
sales.state_id.value_counts()
# Products per category
sales.cat_id.value_counts()
sales.dept_id.value_counts()
sales.store_id.value_counts()
sales.describe()
# Plot max sales per day
