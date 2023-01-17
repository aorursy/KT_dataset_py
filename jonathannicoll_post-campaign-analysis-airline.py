import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
data = pd.read_csv('../input/post-campaign-analysis/Promo Transactions Data-original.csv')
print(data.info())
data.head()
data['Miles_Received_as_a_Bonus'] = data['Miles_Received_as_a_Bonus'].fillna(0)
data['Member Behavior'].value_counts()
data['COUNTRY'].value_counts()
data.head(2)
columns_to_drop = ['Unique_Transaction_ID', 'Member_ID', 'Transaction_Date', 'COUNTRY', 'Enrollment Year']
data = data.drop(columns = columns_to_drop)
data.head(3)
summary = data.groupby('Promotion_Flag').agg(transactions = ('Member Behavior', 'count'),\
                                   miles_purchased = ('Miles_Purchased', 'sum'),\
                                   miles_received = ('Miles_Received_as_a_Bonus', 'sum'))
summary['avg_miles_purchased'] = (summary['miles_purchased'] / summary['transactions']).round(0)
summary['bonus%'] = (summary['miles_received'] / summary['miles_purchased'] * 100.0).round(2)
summary






