import pandas as pd

demodata_dirty = pd.read_csv("../input/dsm-beuth-edl-demodata-dirty.csv")
demodata_dirty
# Set pandas index as id and increment.
demodata_dirty['id'] = demodata_dirty.index
demodata_dirty['id'] = demodata_dirty['id'] + 1
demodata_dirty
# Replace (stupid predict) all NAN's in the gender column with the value of the following row.
demodata_dirty['gender'].fillna(method = 'bfill', inplace=True)
demodata_dirty
# Clean age column.
# dtype should be numeric!
demodata_dirty['age'] = pd.to_numeric(demodata_dirty['age'], downcast='unsigned', errors='coerce')
demodata_dirty['age'].fillna(-1, inplace=True)

# Set invalid values to mean.
mask = demodata_dirty['age'] < 0
column_name = 'age'
demodata_dirty.loc[mask, column_name] = round(demodata_dirty['age'].mean())

demodata_dirty
# Drop empty rows
demodata_dirty.dropna(axis=0, inplace=True)
demodata_dirty
# Drop duplicate entries.
demodata_dirty.drop_duplicates(subset=['fullname', 'email'])