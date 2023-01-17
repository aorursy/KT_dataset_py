import pandas as pd



# Create pandas dataframe from target input csv

impatient_df = pd.read_csv('../input/inpatientCharges.csv')
def clean_inpatient_data(df):

    # Preparing a dict with old columns as keys, with new columns as values

    old_cols = df.columns.values

    new_cols = {}

    for i in range(len(old_cols)):

        new_cols[old_cols[i]] = old_cols[i].strip().replace(' ', '_').lower()

    # Renaming all columns names via pandas rename() function

    df = df.rename(columns=new_cols)

    

    # Parsing financial values as floats

    df['average_covered_charges'] = df['average_covered_charges'].str[1:].astype(float)

    df['average_total_payments'] = df['average_total_payments'].str[1:].astype(float)

    df['average_medicare_payments'] = df['average_medicare_payments'].str[1:].astype(float)



    return df



# Cleaned dataframe

cleaned_impatient_df = clean_inpatient_data(impatient_df)
print(impatient_df.columns.values)
print(cleaned_impatient_df.columns.values)
print(impatient_df[[' Average Covered Charges ', 

                    ' Average Total Payments ', 

                    'Average Medicare Payments'

                   ]].head())
print(cleaned_impatient_df[['average_covered_charges',

                            'average_total_payments',

                            'average_medicare_payments'

                           ]].head())
# cleaned_impatient_df.to_csv('data/inpatient_charges.csv', index=False)