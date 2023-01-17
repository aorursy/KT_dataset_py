import pandas as pd

csv_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter01/Dataset/overall_topten_2012-2013.csv'
csv_df = pd.read_csv(csv_url, skiprows=1)
csv_df
tsv_url = 'https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter01/Dataset/overall_topten_2012-2013.tsv'
tsv_df = pd.read_csv(tsv_url, skiprows=1, sep='\t')
tsv_df
xlsx_url = 'https://github.com/PacktWorkshops/The-Data-Science-Workshop/blob/master/Chapter01/Dataset/overall_topten_2012-2013.xlsx?raw=true'
xlsx_df = pd.read_excel(xlsx_url)
xlsx_df
xlsx_df1 = pd.read_excel(xlsx_url, skiprows=1, sheet_name=1)
xlsx_df1