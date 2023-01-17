import pandas as pd

md = pd.read_csv('/kaggle/input/mock-drafts-2020-features/MockDraftStats.csv')

pd.set_option('display.max_rows', None)
md = md.sort_values(by=['Mean'])

md