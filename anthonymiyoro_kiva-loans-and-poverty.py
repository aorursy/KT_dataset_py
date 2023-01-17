import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# the command below means that the output of multiple commands in a cell will be output at once.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# the command below tells jupyter to display up to 100 columns, this keeps everything visible
pd.set_option('display.max_columns', 100)
pd.set_option('expand_frame_repr', True)
# We begin by importing all the data
df_kiva_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/'+'kiva_loans.csv', low_memory=False)
df_loan_theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/'+'loan_theme_ids.csv', low_memory=False)
df_loan_coords = pd.read_csv('../input/additional-kiva-snapshot/'+'loan_coords.csv', low_memory=False)
# We can now preview the data...
# We can see that both df_kiva_loans and df_loan_theme_ids share similar ID numbers within the id column while the ids in df_loan_coords are stored in the loan_id column
df_kiva_loans.head(n=3)
df_loan_theme_ids.head(n=3)
df_loan_coords.head(n=3)
#We then rename column titles so that they are all single words
#This makes it easier when working with python
df_loan_theme_ids = df_loan_theme_ids.rename(index=str, columns={"Loan Theme ID": "LoanThemeID",  "Loan Theme Type": "LoanThemeType", "Partner ID": "PartnerID"})
# We can now confirm that the titles have changed
df_loan_theme_ids.head(n=3)
df_loan_coords.head(n=3)
# We can then merge all the loans in the df_kiva_loans with df_loan_theme_ids

dfa = df_loan_theme_ids.drop_duplicates(subset=['id'])
dfb = df_kiva_loans.drop_duplicates(subset=['id'])

new_df = pd.merge(dfa, dfb, how='inner', on='id')

new_df.head(n=3)
new_df.shape

# We then add the location data to the combined loans dataset using an inner merge
dfa = df_loan_coords.drop_duplicates(subset=['loan_id'])
dfb = new_df.drop_duplicates(subset=['id'])

new_df = pd.merge(dfa, dfb, how='inner', left_on='loan_id', right_on='id')

new_df.head(n=3)
new_df.shape

# We can then create a new dateaframe that has isolated data by country
kenyan_loans_df = new_df[(new_df.country == 'Kenya')]
# After merging the individual loan data with the location data, we'll now select the loans that are located in Kenya.

