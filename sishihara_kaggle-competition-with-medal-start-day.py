import pandas as pd





# Dataset don't contain the info of competitions which havn't ended yet.

Competitions = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv')

Competitions['Date'] = pd.to_datetime(Competitions['EnabledDate']).dt.date
df = Competitions.query('CanQualifyTiers')[['Title', 'Date']].sort_values('Date').reset_index(drop=True)

df['numDay'] = [d.days for d in (df['Date'] - df['Date'][0])]

df['diffNumDay'] = df['numDay'].diff(1)

df.head()
df['diffNumDay'].plot(figsize=(15, 6))
df.query('diffNumDay > 30')