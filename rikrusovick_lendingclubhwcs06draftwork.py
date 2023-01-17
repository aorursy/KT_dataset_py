import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

lending_club_data = pd.read_csv('../input/lendingclub-issued-loans/lc_loan.csv', low_memory=False)
pd.set_option('display.max_columns', 74)
lending_club_data.head()

lending_club_data.shape
lending_club_data.dropna(axis=1, inplace=True) #Dropping NANs
lending_club_data.head()
lending_club_data.shape
lending_club_data.loan_status
lending_club_data.loan_status.value_counts().plot()
lending_club_data.loan_status.value_counts().plot(kind='barh', title='Loan Status')
plt.show()
