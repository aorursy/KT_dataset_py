#Task 1: Read CSV file using Pandas
import pandas as pd
file=r'../input/experiment4/Experiment4.csv'
df=pd.read_csv(file)
print(df)
#Task 1: Read Excel file using pandas
import pandas as pd
file=r'../input/experiment4excel/Experiment4(excel).xlsx'
df=pd.read_excel(file)
print(df)
#Task 1: Read JSON files using Pandas
import pandas as pd
file=r'../input/experiment4json/Experiment4(json).json'
df=pd.read_json(file)
print(df)