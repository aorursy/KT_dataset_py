import pandas as pd

data = pd.read_csv('../input/diabetic_data.csv')
def get_vals_to_encode(in_series, freq_thresh):
    n_rows = in_series.shape[0]
    if in_series.nunique() <= 4:
        out = [in_series.value_counts().idxmax()]
    else:
        out = [val for val, val_count in in_series.value_counts().iteritems()
                        if (val_count > freq_thresh) and (val_count < (n_rows - freq_thresh))]
    return out

def my_encoder(in_df, freq_thresh=1000):
    out_df = pd.DataFrame()
    for cname, cdata in in_df.iteritems():
        vals_to_encode = get_vals_to_encode(cdata, freq_thresh)
        for val in vals_to_encode:
                new_name = cname + '_' + val
                out_df[new_name] = cdata == val
    return out_df

categoricals = data.select_dtypes(include=object)
encoded_data = my_encoder(categoricals, 2500)
original_numeric = data.select_dtypes(include='number')
output = pd.concat([original_numeric, encoded_data], axis=1)

# readmitted is prediction target. Flip it from readmitted_no to readmitted because that's conventional
output['readmitted'] = 1-output.readmitted_NO
output.drop(['readmitted_NO'], axis=1, inplace=True)

# remove ID variables, and others that are hard to interpret because we don't have good labels
output.drop(['encounter_id', 'patient_nbr', 'admission_type_id',
             'discharge_disposition_id', 'admission_source_id'], axis=1, inplace=True)

output = output.sample(25000, random_state=1)
output.to_csv('readmission_data_for_modeling.csv', index=False)