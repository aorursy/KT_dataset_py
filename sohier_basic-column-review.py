import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/microdados_enem_2016_coma.csv", nrows=5, encoding='iso-8859-1')
for col in df.columns:
    print(col)
