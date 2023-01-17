import numpy as np
import pandas as pd
la_df=pd.read_csv('../input/whats-happening-la-calendar-dataset.csv')
la_df.head()
la_df.columns
la_df.info()
la_df.nunique()
la_df['Age Groupings'].unique()
by_age=la_df.groupby(['Age Groupings'])['Event Name'].count()
by_age=by_age.reset_index()
by_age.plot.bar(x='Age Groupings', y='Event Name')
la_df['start_month'] = la_df['Event Date & Time Start'].astype(str).str[5:7]
events_by_month=la_df.groupby(['start_month'])['Event Name'].count()
events_by_month=events_by_month.reset_index().loc[1:]
events_by_month.plot.bar(x='start_month', y='Event Name')